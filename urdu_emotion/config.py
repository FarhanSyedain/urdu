"""Central configuration for the Urdu Emotion Transformer.

Everything tunable lives here as a single dataclass so experiments are reproducible.
Override any field from the CLI (see main.py) or by editing the DEFAULT instance.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Emotion label space
# --------------------------------------------------------------------------- #
# Canonical order used EVERYWHERE (model output index <-> emotion). The dataset
# loader maps its own column names onto this order, so changing column spellings
# in the raw data never silently scrambles the labels.
EMOTIONS: list[str] = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# Urdu display names for the cognitive graph (emotion nodes).
EMOTIONS_UR: dict[str, str] = {
    "anger": "غصہ",
    "disgust": "نفرت",
    "fear": "خوف",
    "happiness": "خوشی",
    "sadness": "اداسی",
    "surprise": "حیرت",
}

# Synonyms the raw CSV might use for each canonical emotion (lower-cased).
EMOTION_ALIASES: dict[str, str] = {
    "anger": "anger", "angry": "anger", "غصہ": "anger",
    "disgust": "disgust", "نفرت": "disgust",
    "fear": "fear", "خوف": "fear", "ڈر": "fear",
    "happiness": "happiness", "happy": "happiness", "joy": "happiness", "خوشی": "happiness",
    "sadness": "sadness", "sad": "sadness", "اداسی": "sadness", "غم": "sadness",
    "surprise": "surprise", "surprised": "surprise", "حیرت": "surprise",
}

# --------------------------------------------------------------------------- #
# Sentiment label space (3-class) — used by the urdu_tweets_sentiment_10k data
# --------------------------------------------------------------------------- #
SENTIMENTS: list[str] = ["negative", "neutral", "positive"]
SENTIMENTS_UR: dict[str, str] = {
    "negative": "منفی", "neutral": "غیر جانبدار", "positive": "مثبت",
}
SENTIMENT_ALIASES: dict[str, str] = {
    "negative": "negative", "neg": "negative", "-1": "negative", "منفی": "negative",
    "neutral": "neutral", "neu": "neutral", "0": "neutral", "غیر جانبدار": "neutral",
    "positive": "positive", "pos": "positive", "1": "positive", "مثبت": "positive",
}

# Registry of selectable tasks. `Config.task` picks one; everything downstream
# (loader, soft labels, model head, cognitive graph) reads the active space.
LABEL_SPACES: dict[str, dict] = {
    "emotion": {"labels": EMOTIONS, "ur": EMOTIONS_UR, "aliases": EMOTION_ALIASES},
    "sentiment": {"labels": SENTIMENTS, "ur": SENTIMENTS_UR, "aliases": SENTIMENT_ALIASES},
}

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---- paths ---------------------------------------------------------- #
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"
    sample_csv: Path = ROOT / "data" / "sample_tweets.csv"
    output_dir: Path = ROOT / "outputs"
    ckpt_path: Path = ROOT / "outputs" / "model.pt"
    tokenizer_path: Path = ROOT / "outputs" / "tokenizer.json"

    # Which label space to use: "emotion" (6-class multi-label) or "sentiment"
    # (3-class). Drives the loader, soft labels, model head, and cognitive graph.
    task: str = "emotion"

    # Columns in the CSV. `text_col` holds the tweet; labels are either N binary
    # emotion columns (multi-label) OR one column of comma-separated label names.
    text_col: str = "tweet"
    label_cols: tuple[str, ...] | None = None   # auto-detected if None
    multilabel_col: str | None = None           # set to use a single "labels" column

    # ---- tokenizer (trained from scratch on the corpus) ----------------- #
    vocab_size: int = 6000
    min_pair_freq: int = 2          # ignore byte-pairs rarer than this while training

    # ---- model (transformer encoder, built from scratch) --------------- #
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    max_len: int = 64               # tweets are short; covers >99% after tokenizing
    dropout: float = 0.1
    pooling: str = "cls"            # "cls" or "mean"

    # ---- soft / fuzzy labels ------------------------------------------- #
    # multi-hot -> distribution, then blend toward uniform by `label_smoothing`.
    label_smoothing: float = 0.05
    use_lexicon_prior: bool = False  # blend a small Urdu emotion lexicon signal
    lexicon_weight: float = 0.15

    # ---- training ------------------------------------------------------- #
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 30
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    val_split: float = 0.1
    patience: int = 6               # early-stopping patience (epochs)
    seed: int = 42
    device: str = "auto"            # "auto" -> mps if available else cpu
    num_workers: int = 0

    # ---- cognitive graph ----------------------------------------------- #
    graph_top_k: int = 8            # max concept-word nodes to keep
    graph_emotion_threshold: float = 0.08   # emotion node kept if prob >= this
    graph_edge_threshold: float = 0.05      # prune weak concept->emotion edges
    graph_attribution: str = "grad_x_attn"  # "grad_x_attn" | "attention" | "grad"
    graph_use_rollout: bool = True          # attention rollout vs last-layer only

    # ---- active label space (driven by `task`) ------------------------- #
    def labels(self) -> list[str]:
        return LABEL_SPACES[self.task]["labels"]

    def labels_ur(self) -> dict[str, str]:
        return LABEL_SPACES[self.task]["ur"]

    def aliases(self) -> dict[str, str]:
        return LABEL_SPACES[self.task]["aliases"]

    def apply_task_paths(self) -> "Config":
        """Namespace tokenizer/checkpoint by task so emotion & sentiment artifacts
        don't collide. Call after `task` is finalized (e.g. after CLI overrides)."""
        self.tokenizer_path = self.output_dir / f"{self.task}_tokenizer.json"
        self.ckpt_path = self.output_dir / f"{self.task}_model.pt"
        return self

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


DEFAULT = Config()
