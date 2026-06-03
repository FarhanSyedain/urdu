"""Training loop, soft-label loss, evaluation, and checkpointing."""
from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .dataset import EmotionDataset, load_raw, make_collate
from .model import EmotionTransformer
from .tokenizer import BPETokenizer
from .utils import set_seed, get_device, ensure_dir, count_params, save_json


# --------------------------------------------------------------------------- #
# loss + metrics
# --------------------------------------------------------------------------- #
def soft_cross_entropy(logits, target):
    """-sum_i q_i log p_i  (cross-entropy against a soft target distribution)."""
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def _kl(p, q, eps=1e-8):
    return (p * ((p + eps).log() - (q + eps).log())).sum(-1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    n = ce = kl = js = top1 = 0
    for batch in loader:
        ids = batch["ids"].to(device); mask = batch["mask"].to(device)
        target = batch["target"].to(device)
        logits = model(ids, mask)
        probs = logits.softmax(-1)
        b = ids.size(0); n += b
        ce += soft_cross_entropy(logits, target).item() * b
        m = 0.5 * (probs + target)
        js += (0.5 * _kl(probs, m) + 0.5 * _kl(target, m)).sum().item()
        kl += _kl(target, probs).sum().item()
        top1 += (probs.argmax(-1) == target.argmax(-1)).sum().item()
    return {"loss": ce / n, "kl": kl / n, "js": js / n, "top1_acc": top1 / n, "n": n}


# --------------------------------------------------------------------------- #
# checkpoint io
# --------------------------------------------------------------------------- #
def save_checkpoint(model, cfg: Config, vocab_size: int, path: Path) -> None:
    ensure_dir(Path(path).parent)
    torch.save({
        "state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "n_classes": model.n_classes,
        "task": cfg.task,
        "labels": cfg.labels(),
        "labels_ur": cfg.labels_ur(),
        "model_cfg": {
            "d_model": cfg.d_model, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
            "d_ff": cfg.d_ff, "max_len": cfg.max_len, "dropout": cfg.dropout,
            "pooling": cfg.pooling,
        },
    }, path)


def load_checkpoint(path: Path, device=None):
    ckpt = torch.load(path, map_location=device or "cpu")
    cfg = Config()
    cfg.task = ckpt.get("task", "emotion")
    mc = ckpt["model_cfg"]
    for k, v in mc.items():
        setattr(cfg, k, v)
    model = EmotionTransformer(cfg, ckpt["vocab_size"], ckpt["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    if device is not None:
        model.to(device)
    model.eval()
    labels = ckpt.get("labels", cfg.labels())
    labels_ur = ckpt.get("labels_ur", cfg.labels_ur())
    return model, cfg, labels, labels_ur


# --------------------------------------------------------------------------- #
# full training pipeline
# --------------------------------------------------------------------------- #
def get_or_train_tokenizer(texts, cfg: Config) -> BPETokenizer:
    if Path(cfg.tokenizer_path).exists():
        print(f"[tokenizer] loading {cfg.tokenizer_path}")
        return BPETokenizer.load(cfg.tokenizer_path)
    print("[tokenizer] training BPE from scratch ...")
    tok = BPETokenizer().train(texts, vocab_size=cfg.vocab_size,
                               min_pair_freq=cfg.min_pair_freq)
    tok.save(cfg.tokenizer_path)
    return tok


def fit(cfg: Config, data_path: Path) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"[device] {device}")

    rows = load_raw(Path(data_path), cfg)
    print(f"[data] {len(rows)} usable tweets from {data_path}")
    texts = [t for t, _ in rows]
    tok = get_or_train_tokenizer(texts, cfg)

    rng = random.Random(cfg.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * cfg.val_split))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    collate = make_collate(BPETokenizer.PAD)
    train_ds = EmotionDataset(train_rows, tok, cfg)
    val_ds = EmotionDataset(val_rows, tok, cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=cfg.num_workers)

    model = EmotionTransformer(cfg, tok.vocab_size, n_classes=len(cfg.labels())).to(device)
    print(f"[model] task={cfg.task} | classes={len(cfg.labels())} | "
          f"{count_params(model):,} params | vocab={tok.vocab_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, cfg.epochs * len(train_loader))
    warmup = int(cfg.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best, best_state, bad = float("inf"), None, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run = 0.0
        for batch in train_loader:
            ids = batch["ids"].to(device); mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            opt.zero_grad()
            logits = model(ids, mask)
            loss = soft_cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step()
            run += loss.item() * ids.size(0)
        train_loss = run / max(1, len(train_ds))
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val})
        print(f"epoch {epoch:02d} | train {train_loss:.4f} | val {val['loss']:.4f} "
              f"| top1 {val['top1_acc']:.3f} | JS {val['js']:.4f}")

        if val["loss"] < best - 1e-4:
            best, bad = val["loss"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[early-stop] no val improvement for {cfg.patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(model, cfg, tok.vocab_size, cfg.ckpt_path)
    save_json({"history": history, "config": cfg.to_dict()},
              Path(cfg.output_dir) / f"{cfg.task}_train_log.json")
    print(f"[done] best val loss {best:.4f} -> {cfg.ckpt_path}")
    return {"history": history, "best_val_loss": best}
