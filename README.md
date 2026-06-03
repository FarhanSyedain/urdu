# Urdu Emotion Transformer — fuzzy labels + cognitive graphs

A transformer **built from scratch** (in PyTorch) that reads **Urdu (Nastaliq) tweets**
and, instead of a single hard emotion class, assigns a **fuzzy emotion distribution**
(a soft probability over six emotions). For any tweet it also produces a **cognitive
graph** linking the tweet's salient **concept-words → the emotions they evoke**.

```
Urdu tweet ─▶ preprocess (Urdu normalization, emoji/URL/mention handling)
          ─▶ BPE tokenizer (trained from scratch on the corpus)
          ─▶ Transformer ENCODER from scratch (MHSA + FFN + LayerNorm; attention captured)
          ─▶ [CLS] pooling ─▶ linear head ─▶ softmax  =  FUZZY emotion distribution
                                              │
   target = multi-label set  ──normalize──▶ soft distribution  (soft cross-entropy)
                                              │
          attention rollout  ×  per-emotion gradient attribution
                                              ▼
                          COGNITIVE GRAPH:  concept-word ──▶ emotion  edges
```

Emotions: **anger, disgust, fear, happiness, sadness, surprise** (Ekman set).

### Two tasks (pick with `--task`)

The label space is configurable — the transformer, BPE tokenizer, soft-label head and
cognitive graph are all label-agnostic:

| `--task` | classes | data |
|---|---|---|
| `emotion` (default) | 6 — anger/disgust/fear/happiness/sadness/surprise (multi-label → soft) | `data/sample_tweets.csv`, or the 6k PeerJ corpus |
| `sentiment` | 3 — negative / neutral / positive | `urdu_tweets_sentiment_10k.xlsx` (10k Urdu + Roman-Urdu) |

Artifacts are namespaced per task (`outputs/<task>_model.pt`, `outputs/<task>_tokenizer.json`),
so the two models never collide. Roman-Urdu (Latin script) is case-folded and tokenized
alongside Nastaliq; the cognitive graph renders Latin and Urdu concept-words correctly.

---

## Why "fuzzy" labels

Most Urdu emotion corpora are *multi-label* (a tweet may carry several emotions) or
single-label. We convert that into a **soft target distribution**:

```
{anger, disgust}  ─normalize─▶  [.5, .5, 0, 0, 0, 0]
                  ─smooth ε──▶  [.46, .46, .02, .02, .02, .02]
```

The model is trained with **soft cross-entropy** `L = −Σ qᵢ log pᵢ` (KL to the target),
so its softmax output is a genuine graded "partial" emotion label, e.g.

```
sadness  62%  ███████████████
fear     21%  █████
anger    10%  ██
...
```

Optionally (`--use-lexicon`) a small Urdu emotion lexicon blends in a prior so even
single-label rows get a plausible graded target. See `urdu_emotion/soft_labels.py`.

## How the cognitive graph is built

For one tweet (`urdu_emotion/cognitive_graph.py`):

| signal | meaning |
|---|---|
| **attention** | token salience `sₜ` = how much `[CLS]` attends to token *t* (attention **rollout** across all layers, heads averaged) |
| **gradient** | per-emotion attribution `a₍ₜ,ₑ₎ = |∂logitₑ/∂embₜ · embₜ|` = first-order contribution of token *t* to emotion *e* |

Edge weight `edge(word, e) = pₑ · Σ_{t∈word} ( sₜ · â₍ₜ,ₑ₎ )`.

So **attention picks which words matter**, and **the gradient routes each word to the
specific emotion(s) it drives**. Subwords are merged back into whole words; stopwords,
punctuation and placeholders are dropped. Modes: `grad_x_attn` (default), `attention`,
`grad`.

Graphs export to **interactive HTML** (pyvis — renders Urdu RTL natively in a browser),
**PNG**, **JSON**, and **GraphML**, plus a terminal summary.

---

## Setup

Requires Python ≥ 3.11 (tested on 3.11; torch ≥ 2.12 also supports 3.14). Apple Silicon
uses the **MPS** backend automatically.

```bash
uv venv --python 3.11 .venv          # or: python3.11 -m venv .venv
uv pip install -r requirements.txt   # or: .venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

## Quick start (bundled sample, no download needed)

```bash
python main.py prepare      # inspect data + train the BPE tokenizer
python main.py train        # train on data/sample_tweets.csv
python main.py demo         # train if needed, then run example tweets + graphs
```

Predict / graph a single tweet:

```bash
python main.py predict --text "مجھے بہت ڈر لگ رہا ہے"
python main.py graph   --text "اس کے جھوٹ پر مجھے نفرت اور غصہ ہے"
# -> outputs/graph_*.html  (open in a browser),  .png, .json, .graphml
```

## Using the real dataset (6,043 tweets)

```bash
python download_data.py                                   # fetches the public repo
python main.py train --data data/raw/urdu_emotion.csv --epochs 40
```

Source: Bilal et al., *Multi-label emotion classification of Urdu tweets*, PeerJ CS 2022 —
[github.com/Noman712/Mutilabel_Emotion_Detection_Urdu](https://github.com/Noman712/Mutilabel_Emotion_Detection_Urdu).
The loader auto-detects either six binary emotion columns **or** one comma-separated
label column, so most schemas work without edits (override with `Config.label_cols` /
`Config.multilabel_col` if needed).

## Sentiment task (10k Urdu + Roman-Urdu)

Trained on `urdu_tweets_sentiment_10k.xlsx` (3-class, ~3.3k each; 59% Roman Urdu, mostly
reviews):

```bash
python main.py train   --task sentiment --data /path/to/urdu_tweets_sentiment_10k.xlsx \
                       --epochs 30 --max-len 96 --batch-size 64
python main.py predict --task sentiment --text "بکواس پروڈکٹ، پیسے ضائع ہو گئے"
python main.py graph   --task sentiment --text "service bohat achi thi, highly recommend"
python main.py demo    --task sentiment
```

A from-scratch 4.7M-param model reaches **~80% validation accuracy** (3-class, chance = 33%)
and the cognitive graph routes concept-words → sentiment, e.g. `achi`/`highly`/`recommend`
→ positive. The loader auto-detects the `Tweet Text` + `Sentiment` columns and reads `.xlsx`
directly (needs `openpyxl`, already in requirements if you re-install).

---

## Project layout

```
main.py                     CLI: prepare | train | predict | graph | demo
download_data.py            fetch + stage the public dataset (run outside a sandbox)
data/sample_tweets.csv      synthetic Urdu sample (runs end-to-end immediately)
urdu_emotion/
  config.py                 all hyperparameters + emotion label space
  preprocess.py             Urdu normalization (Arabic↔Urdu chars, diacritics, tweets)
  tokenizer.py              from-scratch BPE (+ subword→word map for the graph)
  soft_labels.py            multi-hot → soft distribution (+ optional lexicon prior)
  dataset.py                CSV loading (schema auto-detect) + padding/masks
  model.py                  transformer encoder from scratch (attention exposed)
  train.py                  soft-CE loss, metrics (CE/KL/JS/top-1), early stop, ckpt
  infer.py                  Predictor: tweet → fuzzy distribution
  cognitive_graph.py        attention×gradient → concept→emotion graph + exporters
```

## Configuration

Everything tunable lives in `urdu_emotion/config.py` (`Config` dataclass). Common
overrides are exposed on the CLI: `--epochs --batch-size --lr --vocab-size --d-model
--n-layers --n-heads --device --label-smoothing --graph-top-k --use-lexicon`.

## Notes & limitations

- **From scratch** means the architecture *and* tokenizer are trained on this corpus —
  with only ~6k tweets a small model (default 4 layers / 256 dim) is appropriate. For
  higher accuracy you can later swap in a pretrained encoder; the soft-label and
  cognitive-graph code is independent of how the encoder is built.
- Urdu rendering in the **PNG** needs a Nastaliq font (`Noto Nastaliq Urdu`) plus
  `arabic-reshaper` + `python-bidi`; without them the **HTML** graph still renders Urdu
  perfectly in a browser.
- Gradient attribution is first-order (fast, faithful enough for visualization); swap in
  Integrated Gradients in `cognitive_graph.py` for smoother attributions if desired.
```
