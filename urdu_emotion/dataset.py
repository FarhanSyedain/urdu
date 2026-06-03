"""Dataset loading + batching (label-space agnostic: emotion or sentiment).

Reads CSV / TSV / XLSX and handles two schemas automatically:
  (A) one text column + N binary label columns        (multi-label one-hot)
  (B) one text column + one column of label names      (e.g. "Sentiment")

Both convert to a soft target distribution via soft_labels.build_soft_label.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import Config
from .preprocess import normalize, pre_tokenize
from .soft_labels import build_soft_label, labels_to_multihot
from .tokenizer import BPETokenizer

_TEXT_CANDIDATES = ["tweet", "text", "tweet text", "content", "sentence", "review"]
_MULTILABEL_CANDIDATES = ["labels", "label", "emotion", "emotions", "sentiment", "polarity"]


def _read_table(path: Path):
    import pandas as pd
    p = str(path).lower()
    if p.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if p.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _find_text_col(df, cfg: Config) -> str:
    if cfg.text_col in df.columns:
        return cfg.text_col
    lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in _TEXT_CANDIDATES:
        if cand in lower:
            return lower[cand]
    obj = [c for c in df.columns if df[c].dtype == object]
    if not obj:
        raise ValueError(f"No text column found in columns: {list(df.columns)}")
    return max(obj, key=lambda c: df[c].astype(str).str.len().mean())


def _detect_label_scheme(df, cfg: Config):
    """Return ('multihot', {col: label}) or ('multilabel_col', col_name)."""
    aliases, labels = cfg.aliases(), cfg.labels()
    if cfg.multilabel_col and cfg.multilabel_col in df.columns:
        return "multilabel_col", cfg.multilabel_col
    if cfg.label_cols:
        return "multihot", {c: aliases.get(str(c).lower(), c) for c in cfg.label_cols}

    # auto: columns whose *name* maps to a canonical label  -> binary multihot
    mapping = {}
    for c in df.columns:
        canon = aliases.get(str(c).strip().lower())
        if canon in labels:
            mapping[c] = canon
    if len(mapping) >= 2:
        return "multihot", mapping

    # else: a single column whose *name* says it holds the labels
    lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in _MULTILABEL_CANDIDATES:
        if cand in lower:
            return "multilabel_col", lower[cand]
    raise ValueError(
        "Could not detect labels. Set Config.label_cols or Config.multilabel_col. "
        f"Columns seen: {list(df.columns)}"
    )


def _parse_label_cell(cell, aliases, labels) -> list[str]:
    if cell is None:
        return []
    s = str(cell)
    for sep in [",", ";", "|", "/"]:
        s = s.replace(sep, ",")
    out = []
    for part in s.split(","):
        canon = aliases.get(part.strip().lower())
        if canon in labels:
            out.append(canon)
    return out


def load_raw(path: Path, cfg: Config) -> list[tuple[str, list[float]]]:
    """Read a table and return [(text, multihot[K]), ...] for the active task."""
    df = _read_table(Path(path))
    df = df.dropna(how="all")
    text_col = _find_text_col(df, cfg)
    scheme, info = _detect_label_scheme(df, cfg)
    aliases, labels = cfg.aliases(), cfg.labels()

    rows: list[tuple[str, list[float]]] = []
    for _, r in df.iterrows():
        text = str(r[text_col])
        if not text.strip() or text.strip().lower() == "nan":
            continue
        if scheme == "multihot":
            present = [lab for col, lab in info.items() if float(r.get(col, 0) or 0) > 0]
        else:
            present = _parse_label_cell(r[info], aliases, labels)
        multihot = labels_to_multihot(present, labels)
        if sum(multihot) == 0:
            continue                      # skip rows with no recognized label
        rows.append((text, multihot))
    if not rows:
        raise ValueError(f"No usable rows parsed from {path}")
    return rows


class EmotionDataset(Dataset):
    """Pre-tokenizes texts and pre-computes soft targets for fast iteration."""

    def __init__(self, rows: list[tuple[str, list[float]]],
                 tokenizer: BPETokenizer, cfg: Config):
        self.cfg = cfg
        labels = cfg.labels()
        use_lex = cfg.use_lexicon_prior and cfg.task == "emotion"
        self.examples = []
        for text, multihot in rows:
            enc = tokenizer.encode(text, add_special=True, max_len=cfg.max_len)
            if len(enc) <= 1:             # only [CLS] -> empty text, skip
                continue
            tokens = pre_tokenize(normalize(text))
            target = build_soft_label(
                multihot, tokens,
                smoothing=cfg.label_smoothing,
                use_lexicon=use_lex,
                lexicon_weight=cfg.lexicon_weight,
                label_list=labels,
            )
            self.examples.append({"ids": enc.ids, "target": target, "text": text})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        ex = self.examples[i]
        return {
            "ids": torch.tensor(ex["ids"], dtype=torch.long),
            "target": torch.tensor(ex["target"], dtype=torch.float),
        }


def make_collate(pad_id: int):
    def collate(batch):
        maxlen = max(len(b["ids"]) for b in batch)
        ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), maxlen), dtype=torch.bool)  # True = real token
        targets = torch.stack([b["target"] for b in batch])
        for i, b in enumerate(batch):
            n = len(b["ids"])
            ids[i, :n] = b["ids"]
            mask[i, :n] = True
        return {"ids": ids, "mask": mask, "target": targets}
    return collate
