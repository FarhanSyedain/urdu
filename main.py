#!/usr/bin/env python3
"""Urdu Emotion Transformer — command line interface.

Examples
--------
    python main.py train                         # train on the bundled sample
    python main.py train --data data/raw/urdu_emotion.csv --epochs 40
    python main.py predict --text "مجھے بہت ڈر لگ رہا ہے"
    python main.py graph   --text "اس کے جھوٹ پر مجھے نفرت اور غصہ ہے"
    python main.py demo                           # train (if needed) + a few examples
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from urdu_emotion.config import Config, DEFAULT


def apply_overrides(cfg: Config, args) -> Config:
    if getattr(args, "task", None):
        cfg.task = args.task
    for attr in ["epochs", "batch_size", "lr", "vocab_size", "d_model", "n_layers",
                 "n_heads", "device", "label_smoothing", "graph_top_k", "seed", "max_len"]:
        v = getattr(args, attr, None)
        if v is not None:
            setattr(cfg, attr, v)
    if getattr(args, "use_lexicon", False):
        cfg.use_lexicon_prior = True
    cfg.apply_task_paths()          # namespace tokenizer/checkpoint by task
    return cfg


def cmd_prepare(args):
    from urdu_emotion.dataset import load_raw
    from urdu_emotion.train import get_or_train_tokenizer
    cfg = apply_overrides(DEFAULT, args)
    labels = cfg.labels()
    data = Path(args.data or cfg.sample_csv)
    rows = load_raw(data, cfg)
    counts = {e: 0 for e in labels}
    multi = 0
    for _, mh in rows:
        if sum(mh) > 1:
            multi += 1
        for e, v in zip(labels, mh):
            counts[e] += int(v > 0)
    print(f"[data] task={cfg.task} | {len(rows)} rows | multi-label: {multi}")
    print(f"[data] per-{cfg.task} counts:")
    for e in labels:
        print(f"   {e:<12} {counts[e]}")
    get_or_train_tokenizer([t for t, _ in rows], cfg)
    print(f"[ok] tokenizer at {cfg.tokenizer_path}")


def cmd_train(args):
    from urdu_emotion.train import fit
    cfg = apply_overrides(DEFAULT, args)
    data = Path(args.data or cfg.sample_csv)
    fit(cfg, data)


def cmd_predict(args):
    from urdu_emotion.infer import Predictor, format_distribution
    cfg = apply_overrides(DEFAULT, args)
    pred = Predictor(cfg)
    dist, _ = pred.predict(args.text)
    print(f"\nTweet: {args.text}\nFuzzy {pred.task} distribution:")
    print(format_distribution(dist))


def cmd_graph(args):
    from urdu_emotion.infer import Predictor
    from urdu_emotion.cognitive_graph import build_and_export, print_summary
    cfg = apply_overrides(DEFAULT, args)
    pred = Predictor(cfg)
    stem = Path(args.out) if args.out else Path(cfg.output_dir) / ("graph_" + _slug(args.text))
    res = build_and_export(pred, args.text, cfg, stem)
    print_summary(res["graph"])
    print("\nExported:")
    for k, v in res["outputs"].items():
        print(f"  {k}: {v}")


def cmd_demo(args):
    from urdu_emotion.infer import Predictor, format_distribution
    from urdu_emotion.cognitive_graph import build_and_export, print_summary
    from urdu_emotion.train import fit
    cfg = apply_overrides(DEFAULT, args)
    if not Path(cfg.ckpt_path).exists():
        print("[demo] no checkpoint found — training on the sample first ...")
        fit(cfg, Path(args.data or cfg.sample_csv))
    pred = Predictor(cfg)
    if pred.task == "sentiment":
        examples = [
            "یہ فلم بہت شاندار تھی، مجھے بہت پسند آئی",
            "بکواس پروڈکٹ، پیسے بالکل ضائع ہو گئے",
            "ٹھیک ہے، کوئی خاص بات نہیں تھی",
            "service bohat achi thi, highly recommend karta hun",
        ]
    else:
        examples = [
            "اس کے جھوٹ پر مجھے نفرت اور غصہ دونوں ہیں",
            "نتیجہ آ گیا اور میں پاس ہو گیا، کتنی خوشی ہے",
            "رات کے اندھیرے میں اکیلا پھنس گیا، بہت خوف ہے",
            "خوشخبری اچانک ملی، خوشی اور حیرت ساتھ ساتھ",
        ]
    for i, text in enumerate(examples):
        dist, _ = pred.predict(text)
        print("\n" + "=" * 70)
        print(f"Tweet: {text}\n{format_distribution(dist)}")
        stem = Path(cfg.output_dir) / f"demo_{i}"
        res = build_and_export(pred, text, cfg, stem)
        print_summary(res["graph"])
        print("Exported:", ", ".join(f"{k}={v}" for k, v in res["outputs"].items()))


def _slug(text: str) -> str:
    s = re.sub(r"\s+", "_", text.strip())
    return (s[:24] or "tweet")


def build_parser():
    p = argparse.ArgumentParser(description="Urdu Emotion Transformer")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--task", choices=["emotion", "sentiment"], default=None,
                        help="label space (default: emotion)")
        sp.add_argument("--data", type=str, default=None, help="CSV/XLSX path (default: sample)")
        sp.add_argument("--max-len", dest="max_len", type=int, default=None)
        sp.add_argument("--device", type=str, default=None, help="auto|cpu|mps|cuda")
        sp.add_argument("--epochs", type=int, default=None)
        sp.add_argument("--batch-size", dest="batch_size", type=int, default=None)
        sp.add_argument("--lr", type=float, default=None)
        sp.add_argument("--vocab-size", dest="vocab_size", type=int, default=None)
        sp.add_argument("--d-model", dest="d_model", type=int, default=None)
        sp.add_argument("--n-layers", dest="n_layers", type=int, default=None)
        sp.add_argument("--n-heads", dest="n_heads", type=int, default=None)
        sp.add_argument("--label-smoothing", dest="label_smoothing", type=float, default=None)
        sp.add_argument("--graph-top-k", dest="graph_top_k", type=int, default=None)
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--use-lexicon", dest="use_lexicon", action="store_true")

    sp = sub.add_parser("prepare", help="inspect data + train tokenizer"); common(sp)
    sp.set_defaults(func=cmd_prepare)
    sp = sub.add_parser("train", help="train the model"); common(sp)
    sp.set_defaults(func=cmd_train)
    sp = sub.add_parser("predict", help="fuzzy emotion distribution for one tweet"); common(sp)
    sp.add_argument("--text", required=True); sp.set_defaults(func=cmd_predict)
    sp = sub.add_parser("graph", help="build a cognitive graph for one tweet"); common(sp)
    sp.add_argument("--text", required=True)
    sp.add_argument("--out", default=None, help="output path stem")
    sp.set_defaults(func=cmd_graph)
    sp = sub.add_parser("demo", help="train (if needed) + run example tweets"); common(sp)
    sp.set_defaults(func=cmd_demo)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
