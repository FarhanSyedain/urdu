"""Urdu tweet preprocessing.

Urdu is written in the Arabic script (Nastaliq). Raw tweets mix Arabic-form and
Urdu-form code points for the *same* letter, carry optional diacritics (harakat),
and contain URLs / @mentions / #hashtags / emojis. We normalize all of that to a
canonical form so the tokenizer sees one spelling per word.

Tokenization later operates on logical character order, so right-to-left display
is irrelevant here — we never reverse the string.
"""
from __future__ import annotations

import re
import unicodedata

# --- placeholder tokens for tweet artifacts (kept as single units) ---------- #
URL_TOKEN = "<url>"
USER_TOKEN = "<user>"
NUM_TOKEN = "<num>"

# --- Arabic-form -> Urdu-form letter unification ---------------------------- #
# Map look-alike Arabic code points to their standard Urdu equivalents.
_CHAR_MAP = {
    "ي": "ی",  # ARABIC YEH      ي -> FARSI YEH   ی
    "ى": "ی",  # ALEF MAKSURA    ى -> FARSI YEH   ی
    "ك": "ک",  # ARABIC KAF      ك -> KEHEH       ک
    "ة": "ہ",  # TEH MARBUTA     ة -> HEH GOAL    ہ
    "ۃ": "ہ",  # TEH MARBUTA GOAL ۃ -> HEH GOAL   ہ
    "ه": "ہ",  # ARABIC HEH      ه -> HEH GOAL    ہ
    "ۀ": "ہ",  # HEH WITH YEH ABOVE ۀ -> HEH GOAL ہ
    "أ": "ا",  # ALEF WITH HAMZA ABOVE أ -> ALEF  ا
    "إ": "ا",  # ALEF WITH HAMZA BELOW إ -> ALEF  ا
    "ؤ": "و",  # WAW WITH HAMZA  ؤ -> WAW         و
    "ئ": "ی",  # YEH WITH HAMZA  ئ -> FARSI YEH   ی
    "ـ": "",        # TATWEEL (kashida) ـ -> removed
}
_CHAR_TABLE = {ord(k): v for k, v in _CHAR_MAP.items()}

# --- diacritics / harakat to strip ------------------------------------------ #
_DIACRITICS = re.compile(
    "["
    "ؐ-ؚ"   # Arabic sign / honorifics
    "ً-ٟ"   # fathatan..wavy hamza below (tanwin, harakat, shadda...)
    "ٰ"          # superscript alef
    "ۖ-ۜ"   # small high quranic marks
    "۟-ۨ"
    "۪-ۭ"
    "]"
)

# Zero-width chars that hurt tokenization. NOTE: ZWNJ (U+200C) is intentionally
# KEPT — it carries real orthographic meaning in Urdu ligatures.
_ZEROWIDTH = re.compile("[​‎‏‪-‮⁦-⁩﻿]")

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_LATIN_NUM = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_AR_NUM = re.compile(r"[٠-٩۰-۹]+")  # Arabic-Indic & Eastern digits
_MULTISPACE = re.compile(r"\s+")

# Emoji ranges (surround with spaces so each emoji becomes its own token).
_EMOJI = re.compile(
    "(["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0001F000-\U0001F0FF"
    "])"
)

# Urdu punctuation we keep but split off as separate tokens.
_PUNCT = re.compile(r"([،؛؟۔!?.,:;\"'()\[\]{}«»…—–-])")


def unify_chars(text: str) -> str:
    """Map Arabic-form code points to Urdu forms and drop tatweel."""
    return text.translate(_CHAR_TABLE)


def strip_diacritics(text: str) -> str:
    return _DIACRITICS.sub("", text)


def normalize(text: str, *, keep_emoji: bool = True) -> str:
    """Full normalization pipeline. Returns a clean, single-spaced string."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = unicodedata.normalize("NFC", text)
    # Roman-Urdu (Latin) is case-folded so "Acha"/"acha" share a token; Urdu
    # (Arabic) script is caseless, so this is a no-op there.
    text = text.lower()
    text = _ZEROWIDTH.sub("", text)
    text = _URL.sub(f" {URL_TOKEN} ", text)
    text = _MENTION.sub(f" {USER_TOKEN} ", text)
    text = _HASHTAG.sub(r" \1 ", text)              # keep hashtag *word*, drop '#'

    text = unify_chars(text)
    text = strip_diacritics(text)

    text = _LATIN_NUM.sub(f" {NUM_TOKEN} ", text)
    text = _AR_NUM.sub(f" {NUM_TOKEN} ", text)

    if keep_emoji:
        text = _EMOJI.sub(r" \1 ", text)
    else:
        text = _EMOJI.sub(" ", text)

    text = _PUNCT.sub(r" \1 ", text)
    text = _MULTISPACE.sub(" ", text).strip()
    return text


def pre_tokenize(text: str) -> list[str]:
    """Whitespace split of normalized text into pre-tokens (words/symbols)."""
    if not text:
        return []
    return text.split(" ")
