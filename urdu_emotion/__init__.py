"""Urdu Emotion Transformer.

A from-scratch transformer encoder that reads Urdu (Nastaliq) tweets and assigns a
*fuzzy* emotion label (a soft probability distribution over emotions) instead of a
single hard class, and produces a per-tweet **cognitive graph** linking the salient
concept-words of the tweet to the emotions they evoke.
"""

__version__ = "0.1.0"
