"""Per-tweet cognitive graph: concept-word  -->  emotion.

How an edge weight is built (default mode ``grad_x_attn``):

  attention  : token salience s_t  = how much [CLS] attends to token t
               (attention rollout across all layers, heads averaged).
  gradient   : per-emotion attribution a_{t,e} = |∂logit_e/∂emb_t · emb_t|
               = first-order contribution of token t to emotion e.
  edge(w,e)  = p_e · Σ_{t∈w} ( s_t · â_{t,e} )

So attention decides *which words matter*; the gradient *routes* each salient word
to the specific emotion(s) it pushes the prediction toward. Subwords are merged
back into whole concept-words; stopwords / punctuation / placeholders are dropped.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .config import Config
from .preprocess import URL_TOKEN, USER_TOKEN, NUM_TOKEN
from .tokenizer import BPETokenizer
from .utils import save_json, ensure_dir

# small Urdu stop / function word list (kept short on purpose)
STOPWORDS = {
    "یہ", "وہ", "اور", "کہ", "سے", "کو", "میں", "کا", "کی", "کے", "ہے", "ہیں",
    "نے", "پر", "ایک", "تو", "بھی", "ہو", "تھا", "تھی", "تھے", "گا", "گی", "گے",
    "ہی", "جو", "اس", "اپنے", "اپنی", "کر", "کیا", " کیا", "ان", "ہم", "تم", "آپ",
}
_SKIP_TOKENS = {URL_TOKEN, USER_TOKEN, NUM_TOKEN, "<cls>", "<sep>", "<pad>", "<unk>"}
_PUNCT = set("،؛؟۔!?.,:;\"'()[]{}«»…—–-")


# --------------------------------------------------------------------------- #
# attention salience
# --------------------------------------------------------------------------- #
def attention_rollout(attns) -> torch.Tensor:
    """Abnar & Zuidema rollout. attns: list of [1,H,T,T] -> salience [T]."""
    T = attns[0].shape[-1]
    eye = torch.eye(T)
    result = eye.clone()
    for A in attns:
        a = A[0].mean(0).detach().cpu()          # average heads -> [T,T]
        a = a + eye                              # residual connection
        a = a / a.sum(-1, keepdim=True)
        result = a @ result
    return result[0]                             # [CLS] row -> attention to each token


def last_layer_salience(attns) -> torch.Tensor:
    return attns[-1][0].mean(0).detach().cpu()[0]


# --------------------------------------------------------------------------- #
# core computation
# --------------------------------------------------------------------------- #
def compute_cognitive_graph(predictor, text: str, cfg: Config) -> dict:
    model, tok, device = predictor.model, predictor.tok, predictor.device
    model.eval()

    enc = predictor.encode(text)
    ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)

    # forward through token embeddings so we can differentiate w.r.t. them
    emb = model.embed_tokens(ids).detach().clone().requires_grad_(True)
    logits, attns = model(inputs_embeds=emb, mask=mask, need_weights=True)
    probs = logits.softmax(-1)[0]
    pdet = probs.detach().cpu()                   # detached copy for python-side use
    dist = {e: float(pdet[i]) for i, e in enumerate(predictor.emotions)}

    T = ids.shape[1]
    salience = (attention_rollout(attns) if cfg.graph_use_rollout
                else last_layer_salience(attns)).clone()
    salience[0] = 0.0                            # drop [CLS] self-attention
    salience = salience.clamp_min(0)
    if salience.sum() > 0:
        salience = salience / salience.sum()

    # per-emotion token attribution for emotions above threshold
    kept_emotions = [i for i, e in enumerate(predictor.emotions)
                     if dist[e] >= cfg.graph_emotion_threshold]
    if not kept_emotions:                        # always keep the argmax
        kept_emotions = [int(pdet.argmax())]

    attr = {}                                    # emotion_idx -> normalized [T]
    for e in kept_emotions:
        g = torch.autograd.grad(logits[0, e], emb, retain_graph=True)[0][0]  # [T,D]
        a = (g * emb[0]).sum(-1).abs().detach().cpu()                        # [T]
        a[0] = 0.0
        attr[e] = a / a.sum() if a.sum() > 0 else a

    # ---- aggregate subword tokens into concept words ---------------------- #
    words: dict[int, dict] = {}
    for pos, wid in enumerate(enc.word_ids):
        if wid < 0:
            continue
        words.setdefault(wid, {"positions": [], "subs": []})
        words[wid]["positions"].append(pos)
        words[wid]["subs"].append(enc.tokens[pos])

    concept_nodes, edges = [], []
    for wid, w in words.items():
        surface = BPETokenizer.word_from_subwords(w["subs"])
        if surface in _SKIP_TOKENS or surface in STOPWORDS:
            continue
        if all(ch in _PUNCT for ch in surface) or surface.strip() == "":
            continue
        w_sal = float(sum(salience[p] for p in w["positions"]))
        if w_sal <= 0:
            continue
        out_edges = []
        for e in kept_emotions:
            emo = predictor.emotions[e]
            if cfg.graph_attribution == "attention":
                weight = dist[emo] * w_sal
            elif cfg.graph_attribution == "grad":
                weight = dist[emo] * float(sum(attr[e][p] for p in w["positions"]))
            else:  # grad_x_attn (default)
                weight = dist[emo] * float(sum(salience[p] * attr[e][p]
                                               for p in w["positions"]))
            if weight > 0:
                out_edges.append((emo, weight))
        if out_edges:
            concept_nodes.append({"word": surface, "salience": w_sal})
            for emo, weight in out_edges:
                edges.append({"source": surface, "target": emo, "weight": weight})

    # ---- prune: top-k concepts, threshold edges --------------------------- #
    concept_nodes.sort(key=lambda n: n["salience"], reverse=True)
    keep_words = {n["word"] for n in concept_nodes[: cfg.graph_top_k]}
    concept_nodes = [n for n in concept_nodes if n["word"] in keep_words]

    if edges:
        max_w = max(e["weight"] for e in edges) or 1.0
        for e in edges:
            e["weight_norm"] = e["weight"] / max_w
        edges = [e for e in edges
                 if e["source"] in keep_words and e["weight_norm"] >= cfg.graph_edge_threshold]

    emotion_nodes = [{"emotion": predictor.emotions[e],
                      "emotion_ur": predictor.labels_ur.get(predictor.emotions[e], ""),
                      "prob": dist[predictor.emotions[e]]} for e in kept_emotions]

    return {
        "text": text,
        "task": getattr(predictor, "task", "emotion"),
        "distribution": dist,
        "concepts": concept_nodes,
        "emotions": emotion_nodes,
        "edges": edges,
        "attribution_mode": cfg.graph_attribution,
    }


# --------------------------------------------------------------------------- #
# networkx + exporters
# --------------------------------------------------------------------------- #
def to_networkx(graph: dict):
    import networkx as nx
    G = nx.DiGraph()
    for c in graph["concepts"]:
        G.add_node(c["word"], kind="concept", salience=round(c["salience"], 4))
    for em in graph["emotions"]:
        G.add_node(em["emotion"], kind="emotion", prob=round(em["prob"], 4),
                   label_ur=em["emotion_ur"])
    for e in graph["edges"]:
        G.add_edge(e["source"], e["target"], weight=round(e["weight_norm"], 4))
    return G


def print_summary(graph: dict) -> None:
    task = graph.get("task", "emotion")
    urmap = {n["emotion"]: n["emotion_ur"] for n in graph["emotions"]}
    print(f"\nTweet: {graph['text']}")
    print(f"Fuzzy {task} distribution:")
    for emo, p in sorted(graph["distribution"].items(), key=lambda kv: -kv[1]):
        bar = "█" * int(round(p * 24))
        print(f"  {emo:<12} {p:6.1%}  {bar}")
    print(f"\nCognitive graph  (concept --> {task}, mode={graph['attribution_mode']}):")
    by_emo: dict[str, list] = {}
    for e in sorted(graph["edges"], key=lambda x: -x["weight_norm"]):
        by_emo.setdefault(e["target"], []).append((e["source"], e["weight_norm"]))
    if not by_emo:
        print("  (no salient concept->label edges)")
    for emo, lst in by_emo.items():
        ur = urmap.get(emo, "")
        chain = "  ".join(f"{w} ({wt:.2f})" for w, wt in lst)
        print(f"  {emo} [{ur}]  <--  {chain}")


def export_json(graph: dict, path: Path) -> None:
    save_json(graph, path)


def export_graphml(graph: dict, path: Path) -> None:
    import networkx as nx
    ensure_dir(Path(path).parent)
    nx.write_graphml(to_networkx(graph), path)


def export_html(graph: dict, path: Path) -> bool:
    """Interactive HTML via pyvis (renders Urdu RTL natively). Returns success."""
    try:
        from pyvis.network import Network
    except Exception:
        return False
    ensure_dir(Path(path).parent)
    # cdn_resources="in_line" embeds vis-network JS/CSS directly in the file, so the
    # HTML is self-contained (works offline, no sibling lib/ folder).
    net = Network(height="650px", width="100%", directed=True, bgcolor="#111",
                  font_color="#eee", cdn_resources="in_line")
    net.barnes_hut(gravity=-6000, spring_length=160)
    for c in graph["concepts"]:
        net.add_node(c["word"], label=c["word"], shape="dot",
                     size=10 + 40 * c["salience"], color="#4fc3f7", title=f"salience {c['salience']:.3f}")
    for em in graph["emotions"]:
        lbl = f"{em['emotion_ur']}\n{em['emotion']}"
        net.add_node(em["emotion"], label=lbl, shape="box",
                     size=20 + 40 * em["prob"], color="#ff8a65", title=f"p={em['prob']:.2f}")
    for e in graph["edges"]:
        net.add_edge(e["source"], e["target"], value=e["weight_norm"],
                     title=f"{e['weight_norm']:.2f}")
    net.write_html(str(path), notebook=False, open_browser=False)
    return True


# --------------------------------------------------------------------------- #
# static PNG (with best-effort Urdu shaping)
# --------------------------------------------------------------------------- #
def _has_arabic(s: str) -> bool:
    return any(0x0600 <= ord(c) <= 0x06FF or 0xFB50 <= ord(c) <= 0xFEFF for c in s)


def _shape_urdu(text: str, reshape: bool) -> str:
    """Reshape Urdu into joined presentation forms + apply bidi, if the chosen
    font supports presentation forms. matplotlib has no OpenType shaper, so this
    pre-shaping is what makes Urdu render joined and right-to-left in the PNG."""
    if not reshape:
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def _pick_urdu_font():
    """Return (FontProperties or None, use_reshaper).

    arabic_reshaper emits *presentation-form* code points (U+FB50.., U+FE70..),
    which Nastaliq fonts (Noto Nastaliq Urdu) deliberately omit — they expect a
    shaping engine. So for the PNG we prefer a font that actually contains those
    glyphs (Geeza Pro / Damascus / Arial Unicode MS ...); only then do we reshape.
    Falls back to a base-form font (unshaped) rather than rendering tofu boxes.
    """
    try:
        from matplotlib import font_manager as fm
        from matplotlib.ft2font import FT2Font
    except Exception:
        return None, False

    avail = {f.name for f in fm.fontManager.ttflist}
    pres_fonts = ["Geeza Pro", "Damascus", "Arial Unicode MS", "Al Bayan",
                  "DecoType Naskh", "Nadeem", "Baghdad", "KufiStandardGK",
                  "Noto Naskh Arabic", "Amiri", "Scheherazade"]
    base_fonts = ["Noto Nastaliq Urdu", "Jameel Noori Nastaleeq", "Noto Naskh Arabic"]

    def has_presentation(path):
        try:
            return FT2Font(path).get_char_index(0xFEDB) != 0   # KAF initial form
        except Exception:
            return False

    for name in pres_fonts:                      # joined + RTL (best)
        if name in avail:
            path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            if has_presentation(path):
                return fm.FontProperties(fname=path), True
    for name in base_fonts:                      # unshaped but no tofu
        if name in avail:
            path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            return fm.FontProperties(fname=path), False
    return None, False


def export_png(graph: dict, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    ensure_dir(Path(path).parent)
    font, reshape = _pick_urdu_font()

    concepts = graph["concepts"]
    emotions = graph["emotions"]
    if not concepts or not emotions:
        return False

    def ypos(n):
        return [i - (n - 1) / 2 for i in range(n)]

    cy = ypos(len(concepts)); ey = ypos(len(emotions))
    cpos = {c["word"]: (0.0, y) for c, y in zip(concepts, cy)}
    epos = {e["emotion"]: (1.0, y * (len(concepts) / max(1, len(emotions)))) for e, y in zip(emotions, ey)}

    fig, ax = plt.subplots(figsize=(9, max(4, 0.7 * max(len(concepts), len(emotions)))))
    for e in graph["edges"]:
        if e["source"] in cpos and e["target"] in epos:
            x0, y0 = cpos[e["source"]]; x1, y1 = epos[e["target"]]
            ax.plot([x0, x1], [y0, y1], color="#4fc3f7",
                    lw=0.5 + 4 * e["weight_norm"], alpha=0.4 + 0.5 * e["weight_norm"], zorder=1)

    for c in concepts:
        x, y = cpos[c["word"]]
        ax.scatter([x], [y], s=120 + 600 * c["salience"], color="#0288d1", zorder=2)
        # Roman-Urdu concepts are Latin -> use the default font (Arabic font lacks Latin)
        if _has_arabic(c["word"]):
            ax.text(x - 0.04, y, _shape_urdu(c["word"], reshape), ha="right",
                    va="center", fontsize=13, fontproperties=font)
        else:
            ax.text(x - 0.04, y, c["word"], ha="right", va="center", fontsize=12)
    for em in emotions:
        x, y = epos[em["emotion"]]
        ax.scatter([x], [y], s=300 + 600 * em["prob"], color="#ff7043", marker="s", zorder=2)
        # Urdu name (Urdu font) and Latin metric (default font) rendered separately,
        # since an Arabic-only font lacks Latin glyphs and vice-versa.
        ax.text(x + 0.05, y + 0.16, _shape_urdu(em["emotion_ur"], reshape), ha="left",
                va="center", fontsize=14, fontproperties=font)
        ax.text(x + 0.05, y - 0.16, f"{em['emotion']} {em['prob']:.0%}", ha="left",
                va="center", fontsize=10, color="#444")

    ax.set_xlim(-0.6, 1.6); ax.axis("off")
    ax.set_title(f"Cognitive graph: concept → {graph.get('task', 'emotion')}", fontsize=13)
    import warnings
    with warnings.catch_warnings():               # font coverage handled deliberately
        warnings.filterwarnings("ignore", message="Glyph .* missing from font")
        fig.tight_layout()                        # tight_layout also triggers a draw
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def build_and_export(predictor, text: str, cfg: Config, out_stem: Path) -> dict:
    graph = compute_cognitive_graph(predictor, text, cfg)
    out_stem = Path(out_stem)
    export_json(graph, out_stem.with_suffix(".json"))
    results = {"json": str(out_stem.with_suffix(".json"))}
    for name, fn, suffix in [("graphml", export_graphml, ".graphml"),
                             ("html", export_html, ".html"),
                             ("png", export_png, ".png")]:
        try:
            ok = fn(graph, out_stem.with_suffix(suffix))
            if ok is not False:
                results[name] = str(out_stem.with_suffix(suffix))
        except Exception as ex:  # an optional exporter must never abort the others
            results[f"{name}_error"] = str(ex)
    return {"graph": graph, "outputs": results}
