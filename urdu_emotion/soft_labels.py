"""Turn hard / multi-label annotations into a *soft* probability distribution.

This is what makes the label "fuzzy" rather than fixed. It is label-space agnostic:
the caller passes the active label list (6 emotions, or 3 sentiments).

    multi-hot {anger, disgust}  ->  normalize  ->  [.5, .5, 0, 0, 0, 0]
                                ->  smooth(eps) ->  [.46, .46, .02, .02, .02, .02]

The optional Urdu emotion lexicon prior only applies to the emotion task (sentiment
has no word-level lexicon here); it is ignored when its words don't match the labels.
"""
from __future__ import annotations

# Small illustrative Urdu seed lexicon (emotion task only; extend with NRC-Urdu).
SEED_LEXICON: dict[str, list[str]] = {
    "anger":     ["غصہ", "غصے", "ناراض", "لعنت", "جھگڑا", "نفرت"],
    "disgust":   ["نفرت", "کراہت", "گھن", "بیزار", "مکروہ", "گندہ"],
    "fear":      ["ڈر", "خوف", "ڈرا", "خوفزدہ", "دہشت", "گھبرا"],
    "happiness": ["خوش", "خوشی", "مبارک", "شکر", "مسکرا", "ہنسی"],
    "sadness":   ["اداس", "غم", "دکھ", "آنسو", "افسوس", "رو"],
    "surprise":  ["حیران", "حیرت", "اچانک", "تعجب", "واہ"],
}
_WORD2EMO: dict[str, str] = {w: e for e, ws in SEED_LEXICON.items() for w in ws}


def normalize_dist(vec: list[float]) -> list[float]:
    s = sum(vec)
    if s <= 0:
        return [1.0 / len(vec)] * len(vec)
    return [v / s for v in vec]


def multihot_to_soft(multihot: list[float], smoothing: float = 0.05) -> list[float]:
    """multi-hot -> probability distribution, smoothed toward uniform by `smoothing`."""
    k = len(multihot)
    base = normalize_dist([float(v) for v in multihot])
    if smoothing <= 0:
        return base
    return [(1.0 - smoothing) * b + smoothing / k for b in base]


def lexicon_dist(tokens: list[str], label_list: list[str]) -> list[float] | None:
    """Distribution implied by lexicon words present in the text, aligned to
    `label_list`, or None if nothing matches (e.g. for the sentiment task)."""
    idx = {e: i for i, e in enumerate(label_list)}
    counts = [0.0] * len(label_list)
    hit = False
    for w in tokens:
        e = _WORD2EMO.get(w)
        if e is not None and e in idx:
            counts[idx[e]] += 1.0
            hit = True
    return normalize_dist(counts) if hit else None


def build_soft_label(multihot: list[float], tokens: list[str] | None = None, *,
                     smoothing: float = 0.05, use_lexicon: bool = False,
                     lexicon_weight: float = 0.15,
                     label_list: list[str] | None = None) -> list[float]:
    dist = multihot_to_soft(multihot, smoothing)
    if use_lexicon and tokens is not None and label_list is not None:
        lx = lexicon_dist(tokens, label_list)
        if lx is not None:
            dist = normalize_dist(
                [(1.0 - lexicon_weight) * d + lexicon_weight * l for d, l in zip(dist, lx)]
            )
    return dist


def labels_to_multihot(labels, label_list: list[str]) -> list[float]:
    """A set/list of label names -> multi-hot vector aligned to `label_list`."""
    idx = {e: i for i, e in enumerate(label_list)}
    vec = [0.0] * len(label_list)
    for lab in labels:
        if lab in idx:
            vec[idx[lab]] = 1.0
    return vec
