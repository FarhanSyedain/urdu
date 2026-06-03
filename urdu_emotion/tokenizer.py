"""A from-scratch Byte-Pair-Encoding (BPE) tokenizer for Urdu.

Pure Python — no sentencepiece / HF-tokenizers dependency. Trained on our own
corpus so the subword vocabulary is tuned to Urdu Nastaliq. It also records, for
every subword, which *original word* it came from (``word_ids``); the cognitive
graph needs that to merge subwords back into readable concept-words.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .preprocess import normalize, pre_tokenize, URL_TOKEN, USER_TOKEN, NUM_TOKEN
from .utils import save_json, load_json

END = "</w>"  # marks end-of-word so BPE can distinguish "ہی" in-word vs word-final


@dataclass
class Encoding:
    ids: list[int]
    tokens: list[str]
    word_ids: list[int]          # -1 for special tokens (e.g. [CLS])

    def __len__(self) -> int:
        return len(self.ids)


class BPETokenizer:
    SPECIALS = ["<pad>", "<unk>", "<cls>", "<sep>", URL_TOKEN, USER_TOKEN, NUM_TOKEN]
    PAD, UNK, CLS, SEP = 0, 1, 2, 3

    def __init__(self) -> None:
        self.merges: list[tuple[str, str]] = []
        self.ranks: dict[tuple[str, str], int] = {}
        self.token_to_id: dict[str, int] = {}
        self.vocab: list[str] = []
        self._cache: dict[str, list[str]] = {}
        for tok in self.SPECIALS:                 # reserve special-token ids first
            self._add_token(tok)

    # ------------------------------------------------------------------ #
    # vocab helpers
    # ------------------------------------------------------------------ #
    def _add_token(self, tok: str) -> int:
        if tok not in self.token_to_id:
            self.token_to_id[tok] = len(self.vocab)
            self.vocab.append(tok)
        return self.token_to_id[tok]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def train(self, texts: Iterable[str], vocab_size: int = 6000,
              min_pair_freq: int = 2, verbose: bool = True) -> "BPETokenizer":
        specials = set(self.SPECIALS)

        # 1. word frequencies over normalized, pre-tokenized corpus
        word_freqs: dict[str, int] = defaultdict(int)
        for text in texts:
            for w in pre_tokenize(normalize(text)):
                if w and w not in specials:
                    word_freqs[w] += 1

        # 2. seed vocab with every character; initial per-word symbol splits
        splits: dict[str, list[str]] = {}
        for word in word_freqs:
            syms = list(word)
            syms[-1] = syms[-1] + END
            splits[word] = syms
            for s in syms:
                self._add_token(s)

        # 3. initial pair stats + reverse index (pair -> words containing it)
        pair_freqs: dict[tuple[str, str], int] = defaultdict(int)
        where: dict[tuple[str, str], set[str]] = defaultdict(set)

        def index_word(word: str, freq: int) -> None:
            syms = splits[word]
            for i in range(len(syms) - 1):
                p = (syms[i], syms[i + 1])
                pair_freqs[p] += freq
                where[p].add(word)

        def deindex_word(word: str, freq: int) -> None:
            syms = splits[word]
            for i in range(len(syms) - 1):
                p = (syms[i], syms[i + 1])
                pair_freqs[p] -= freq
                where[p].discard(word)

        for word, freq in word_freqs.items():
            index_word(word, freq)

        # 4. greedily merge the most frequent pair until the vocab is full
        target_merges = max(0, vocab_size - self.vocab_size)
        rng = range(target_merges)
        if verbose:
            try:
                from tqdm import tqdm
                rng = tqdm(rng, desc="BPE merges")
            except Exception:
                pass

        for _ in rng:
            if not pair_freqs:
                break
            best = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[best] < min_pair_freq:
                break
            a, b = best
            merged = a + b
            self.merges.append(best)
            self.ranks[best] = len(self.merges) - 1
            self._add_token(merged)

            for word in list(where[best]):
                freq = word_freqs[word]
                deindex_word(word, freq)             # pull old pairs out of the index
                syms = splits[word]
                out, i = [], 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                splits[word] = out
                index_word(word, freq)               # re-index with merged symbol

        if verbose:
            print(f"[tokenizer] vocab_size={self.vocab_size} merges={len(self.merges)}")
        self._cache.clear()
        return self

    # ------------------------------------------------------------------ #
    # encoding
    # ------------------------------------------------------------------ #
    def _bpe_word(self, word: str) -> list[str]:
        if word in self.token_to_id and word in set(self.SPECIALS):
            return [word]
        if word in self._cache:
            return self._cache[word]
        syms = list(word)
        if not syms:
            return []
        syms[-1] = syms[-1] + END
        while len(syms) > 1:
            best_rank, best_i = None, -1
            for i in range(len(syms) - 1):
                r = self.ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            syms[best_i:best_i + 2] = [syms[best_i] + syms[best_i + 1]]
        self._cache[word] = syms
        return syms

    def encode(self, text: str, add_special: bool = True,
               max_len: int | None = None) -> Encoding:
        ids: list[int] = []
        toks: list[str] = []
        wids: list[int] = []
        if add_special:
            ids.append(self.CLS); toks.append("<cls>"); wids.append(-1)

        for wi, word in enumerate(pre_tokenize(normalize(text))):
            for sub in self._bpe_word(word):
                ids.append(self.token_to_id.get(sub, self.UNK))
                toks.append(sub)
                wids.append(wi)

        if max_len is not None and len(ids) > max_len:
            # keep [CLS] + the first (max_len-1) subwords
            ids, toks, wids = ids[:max_len], toks[:max_len], wids[:max_len]
        return Encoding(ids, toks, wids)

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i in (self.PAD, self.CLS, self.SEP):
                continue
            out.append(self.vocab[i] if 0 <= i < len(self.vocab) else "<unk>")
        return "".join(out).replace(END, " ").strip()

    @staticmethod
    def word_from_subwords(subs: list[str]) -> str:
        """Join subwords of one word back into a readable string (drop END mark)."""
        return "".join(subs).replace(END, "")

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        save_json({
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
        }, path)

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        data = load_json(path)
        tok = cls.__new__(cls)
        tok.vocab = list(data["vocab"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.vocab)}
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.ranks = {tuple(m): i for i, m in enumerate(tok.merges)}
        tok._cache = {}
        return tok
