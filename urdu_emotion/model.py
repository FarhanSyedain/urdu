"""Transformer encoder, built from scratch.

We implement scaled dot-product attention, multi-head splitting, the position-wise
feed-forward, residual + pre-LayerNorm wiring, and the pooling head ourselves
(rather than nn.Transformer / nn.MultiheadAttention) so that:
  * the architecture is transparent and "from scratch", and
  * every layer's attention matrix is captured and returned for the cognitive graph.

Output is a length-K logit vector; softmax(logits) is the fuzzy emotion distribution.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config, EMOTIONS


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h = n_heads
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_mask=None):
        # x: [B,T,D]   key_mask: [B,T] bool, True = real token
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)  # [B,H,T,dh]
        k = self.k(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)     # [B,H,T,T]
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        attn = self.drop(attn)
        out = attn @ v                                             # [B,H,T,dh]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out), attn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)); x + FFN(LN(x))."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_mask=None):
        a, attn = self.attn(self.ln1(x), key_mask)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, attn


class EmotionTransformer(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int, n_classes: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes or len(EMOTIONS)
        d = cfg.d_model
        self.tok_emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, d)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(d, self.n_classes))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()

    def encode(self, ids, mask=None, need_weights=False,
               inputs_embeds=None):
        """Returns (hidden[B,T,D], attentions or None).

        Pass `inputs_embeds` instead of `ids` to attribute gradients to token
        embeddings (used by the cognitive graph).
        """
        if inputs_embeds is None:
            x = self.tok_emb(ids)
        else:
            x = inputs_embeds
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).clamp_max(self.cfg.max_len - 1)
        x = x * math.sqrt(self.cfg.d_model) + self.pos_emb(pos)[None]
        x = self.emb_drop(x)
        attns = [] if need_weights else None
        for layer in self.layers:
            x, attn = layer(x, mask)
            if need_weights:
                attns.append(attn)
        return self.norm(x), attns

    def pool(self, hidden, mask=None):
        if self.cfg.pooling == "mean" and mask is not None:
            m = mask.unsqueeze(-1).float()
            return (hidden * m).sum(1) / m.sum(1).clamp_min(1e-6)
        return hidden[:, 0]                      # [CLS]

    def forward(self, ids=None, mask=None, need_weights=False, inputs_embeds=None):
        hidden, attns = self.encode(ids, mask, need_weights, inputs_embeds)
        logits = self.head(self.pool(hidden, mask))
        if need_weights:
            return logits, attns
        return logits

    @torch.no_grad()
    def predict_proba(self, ids, mask=None):
        self.eval()
        logits = self.forward(ids, mask)
        return logits.softmax(dim=-1)

    # token embeddings exposed for gradient attribution in the cognitive graph
    def embed_tokens(self, ids):
        return self.tok_emb(ids)
