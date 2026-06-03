"""Inference: load a trained checkpoint and assign a fuzzy emotion distribution."""
from __future__ import annotations

from pathlib import Path

import torch

from .config import Config
from .tokenizer import BPETokenizer
from .train import load_checkpoint
from .utils import get_device


class Predictor:
    """Holds the model + tokenizer so the CLI and the cognitive graph can share it."""

    def __init__(self, cfg: Config):
        self.device = get_device(cfg.device)
        (self.model, model_cfg,
         self.emotions, self.labels_ur) = load_checkpoint(cfg.ckpt_path, self.device)
        # adopt the model's task + structural params; keep runtime graph params
        cfg.task = model_cfg.task
        cfg.max_len = model_cfg.max_len
        cfg.d_model = model_cfg.d_model
        cfg.pooling = model_cfg.pooling
        self.cfg = cfg
        self.task = model_cfg.task
        self.tok = BPETokenizer.load(cfg.tokenizer_path)

    def encode(self, text: str):
        return self.tok.encode(text, add_special=True, max_len=self.cfg.max_len)

    def predict(self, text: str):
        """Return (distribution dict {emotion: prob}, Encoding)."""
        enc = self.encode(text)
        ids = torch.tensor([enc.ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        probs = self.model.predict_proba(ids, mask)[0].tolist()
        dist = {e: float(p) for e, p in zip(self.emotions, probs)}
        return dist, enc


def format_distribution(dist: dict[str, float], top: int | None = None) -> str:
    items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    if top:
        items = items[:top]
    width = 24
    lines = []
    for emo, p in items:
        bar = "█" * int(round(p * width))
        lines.append(f"  {emo:<10} {p:6.1%}  {bar}")
    return "\n".join(lines)
