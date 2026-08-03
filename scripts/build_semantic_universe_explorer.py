"""
scripts/build_semantic_universe_explorer.py
-------------------------------------------
Emits assets/semantic_universe_explorer.html: a single self-contained
static page you can drop straight into any static webhost (GitHub Pages,
Netlify, S3, etc.).

Corpus reality
==============
Every document in this corpus is a Global Impact Report from one of five
direct competitors in the semiconductor-manufacturing-equipment sector:
KLA, Applied Materials, ASML, Lam Research, and TEL (Tokyo Electron).
The interesting axis therefore is NOT "which industry does this chunk
belong to" (all five are the same industry) but "which chunks and which
docs sit apart from the consensus their four competitors have converged
to."  The viewer is built around that question.

What the viewer supports
========================
  * 2D / 3D toggle of the corpus in PCA space
  * six color modes: company, prose-tabular score, digit density,
    cross-corpus outlierness (distance from grand centroid), in-doc
    typicality (distance from own doc centroid), and unsupervised GMM
    cluster
  * filters that HIDE non-matching chunks (as opposed to selection
    spotlight which only DIMS them):
      - content type: All / Prose / Ambiguous / Tabular
      - dual-handle digit-density range slider
      - per-company include/exclude checkboxes
      - "outliers only" toggle with a percentile slider
  * hover for chunk snippet + metadata
  * CLICK on a point to spotlight its top-K nearest neighbors in the
    original 384-dim embedding space (all other points fade to gray)
  * a curated "Salient Exhibits" panel: hand-picked chunks that make a
    specific point about how these five competitors compare
  * an explainer strip that says what the geometry reveals and what it
    cannot prove
  * a footer linking back to the repo / notebook

Inputs
======
  assets/chunks.parquet    (from scripts/build_visuals.py --cache)
  assets/embeddings.npy    unit-norm 384-dim MiniLM embeddings

Output
======
  assets/semantic_universe_explorer.html    (~5-8 MB, standalone)

Usage
=====
  python scripts/build_semantic_universe_explorer.py
  # then just open the HTML, or serve the assets/ folder statically.
"""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"          # what GitHub Pages publishes
CACHE = ROOT / "assets"       # build cache: parquet + npy live here
DOCS.mkdir(parents=True, exist_ok=True)

OUT_HTML = DOCS / "semantic_universe_explorer.html"
OUT_EXAMPLES_HTML = DOCS / "explorer_examples.html"
CHUNKS_PARQUET = CACHE / "chunks.parquet"
EMBEDDINGS_NPY = CACHE / "embeddings.npy"

REPO_URL = "https://github.com/emo7bf/SUE"           # placeholder, safe to edit

TOP_NEIGHBORS = 50           # how many nearest neighbors to precompute per chunk
TABLE_DIGIT_THRESHOLD = 0.18  # digit-density threshold that names a chunk "tabular"

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    "#393b79", "#e6550d", "#31a354", "#756bb1", "#636363",
]

_DIGIT_RE = re.compile(r"\d")


# ----------------------------------------------------------------- data

def load_cached():
    if not CHUNKS_PARQUET.exists() or not EMBEDDINGS_NPY.exists():
        raise SystemExit(
            "Missing cached artifacts. Run first:\n"
            "  python scripts/build_visuals.py --cache"
        )
    df = pd.read_parquet(CHUNKS_PARQUET).reset_index(drop=True)
    X = np.load(EMBEDDINGS_NPY).astype(np.float32)
    if len(df) != len(X):
        raise SystemExit(
            f"chunks.parquet has {len(df)} rows but embeddings.npy has "
            f"{len(X)}; regenerate both from build_visuals.py --cache"
        )
    return df, X


def digit_density(text: str) -> float:
    if not text:
        return 0.0
    return len(_DIGIT_RE.findall(text)) / max(1, len(text))


def compute_prose_axis(X: np.ndarray, dd: np.ndarray) -> np.ndarray:
    is_table = dd > TABLE_DIGIT_THRESHOLD
    if is_table.sum() < 5 or (~is_table).sum() < 5:
        return PCA(n_components=1, random_state=0).fit(X).components_[0]
    v = X[is_table].mean(0) - X[~is_table].mean(0)
    return v / max(1e-12, np.linalg.norm(v))


def compute_doc_centroids(df: pd.DataFrame, X: np.ndarray):
    """One centroid per (company, doc) pair.  Returns (C, cmeta, ci)."""
    keys = list(zip(df["company"], df["doc"]))
    order: dict = {}
    for k in keys:
        if k not in order:
            order[k] = len(order)
    ci = np.fromiter((order[k] for k in keys), dtype=np.int64, count=len(keys))
    C = np.zeros((len(order), X.shape[1]), dtype=X.dtype)
    counts = np.zeros(len(order), dtype=np.int64)
    for j, c in enumerate(ci):
        C[c] += X[j]
        counts[c] += 1
    C = C / counts[:, None]
    cmeta = [
        {"company": k[0], "doc": k[1], "n_chunks": int(counts[v])}
        for k, v in order.items()
    ]
    return C, cmeta, ci


def compute_company_centroids(df: pd.DataFrame, X: np.ndarray):
    """One centroid per company.  Returns (Cco, cometa, coi) where coi[j]
    is the row of Cco that chunk j belongs to."""
    companies = df["company"].tolist()
    order: dict = {}
    for c in companies:
        if c not in order:
            order[c] = len(order)
    coi = np.fromiter((order[c] for c in companies),
                      dtype=np.int64, count=len(companies))
    Cco = np.zeros((len(order), X.shape[1]), dtype=X.dtype)
    counts = np.zeros(len(order), dtype=np.int64)
    for j, c in enumerate(coi):
        Cco[c] += X[j]
        counts[c] += 1
    Cco = Cco / counts[:, None]
    cometa = [{"company": k, "n_chunks": int(counts[v])} for k, v in order.items()]
    return Cco, cometa, coi


def content_type_of(dd_val: float) -> str:
    """Three-way label used by the content-type filter."""
    if dd_val >= TABLE_DIGIT_THRESHOLD:
        return "tabular"
    if dd_val <= 0.05:
        return "prose"
    return "ambiguous"


def top_neighbors(X: np.ndarray, k: int) -> np.ndarray:
    """Return (N, k) int32 array of top-k neighbor row indices (excluding self)
    computed by cosine similarity on unit-norm vectors."""
    N = len(X)
    out = np.zeros((N, k), dtype=np.int32)
    # process in blocks to keep memory bounded
    block = 512
    for start in range(0, N, block):
        end = min(N, start + block)
        S = X[start:end] @ X.T          # (b, N)
        # exclude self
        for r in range(end - start):
            S[r, start + r] = -np.inf
        idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
        # sort the top-k for a stable UI
        rows = np.arange(end - start)[:, None]
        sorted_order = np.argsort(-S[rows, idx], axis=1)
        idx = idx[rows, sorted_order]
        out[start:end] = idx
        print(f"  neighbors {end}/{N}")
    return out


def gmm_cluster_labels(X: np.ndarray, subspace_dims: int = 20) -> np.ndarray:
    p = PCA(n_components=subspace_dims, random_state=0).fit_transform(X)
    gm = GaussianMixture(n_components=2, covariance_type="full",
                         random_state=0).fit(p)
    return gm.predict(p).astype(np.int32)


# ---------------------------------------------------------- exhibits

def _dist_from_competitor_consensus(X: np.ndarray, coi: np.ndarray,
                                    Cco: np.ndarray) -> np.ndarray:
    """
    For every chunk, distance from the mean of the OTHER companies\u2019
    embeddings (the "four-competitor consensus centroid" if there are
    five companies).  A large value means: this chunk sits far from
    what its competitors are, on average, saying.

    Uses the identity  mean_of_others = (S - n_own * C_own) / (N - n_own)
    where S is the sum over all chunks in the corpus.
    """
    N_co = Cco.shape[0]
    # per-company chunk counts and sums, reconstructed from Cco
    # (we already have Cco = mean per company; to get the sum we\u2019d need
    # counts; recompute here for numerical safety)
    counts = np.zeros(N_co, dtype=np.int64)
    for c in coi:
        counts[c] += 1
    sums = Cco * counts[:, None]
    total_sum = sums.sum(0)
    total_count = counts.sum()

    out = np.zeros(len(X), dtype=np.float32)
    for c in range(N_co):
        others_sum = total_sum - sums[c]
        others_count = total_count - counts[c]
        if others_count == 0:
            continue
        mu_others = others_sum / others_count
        rows = np.flatnonzero(coi == c)
        out[rows] = np.linalg.norm(X[rows] - mu_others, axis=1)
    return out


def curate_exhibits(df: pd.DataFrame, X: np.ndarray, dd: np.ndarray,
                    prose_axis: np.ndarray, ci: np.ndarray, C: np.ndarray,
                    coi: np.ndarray, Cco: np.ndarray, companies: List[str],
                    cluster: np.ndarray) -> List[dict]:
    """
    Hand-curated exhibits framed around the corpus\u2019 actual structure:
    five direct competitors, one industry.  Each exhibit points at a
    specific chunk that answers a specific comparative question.
    """
    proj = X @ prose_axis
    global_mean = X.mean(0)
    dist_global = np.linalg.norm(X - global_mean, axis=1)
    dist_others = _dist_from_competitor_consensus(X, coi, Cco)

    def _entry(i: int, title: str, note: str) -> dict:
        return {
            "index": int(i),
            "title": title,
            "note": note,
            "company": str(df.iloc[i]["company"]),
            "doc":     str(df.iloc[i]["doc"]),
            "chunk":   int(df.iloc[i]["chunk_id"]),
        }

    exhibits: List[dict] = []

    # ---- 1. Per-company off-consensus chunk (one per firm).
    # Leads the list per user request: these answer "what is firm X saying
    # that its four competitors aren't?"
    for c_idx, c_name in enumerate(companies):
        rows = np.flatnonzero(coi == c_idx)
        if len(rows) == 0:
            continue
        i = int(rows[np.argmax(dist_others[rows])])
        exhibits.append(_entry(
            i,
            f"{c_name}: most off-consensus passage",
            f"The paragraph in {c_name}\u2019s report that reads least "
            "like anything its four competitors wrote."))

    # ---- 2. The sector\u2019s average sentence (single grand-centroid pick).
    # Kept narrative-only so we get something readable, not a table.
    narrative_mask = (proj < 0) & (dd < 0.03)
    if narrative_mask.any():
        cand = np.flatnonzero(narrative_mask)
        i = int(cand[np.argmin(dist_global[cand])])
        exhibits.append(_entry(
            i, "The sector\u2019s average sentence",
            "So central that it could have come from any of the five "
            "reports. This is what \u201Cconsensus\u201D looks like."))

    # ---- 3. Per-company \u201Cmost typical\u201D chunk (one per firm).
    # The counterpart to the off-consensus exhibits: the chunk closest
    # to firm X\u2019s own centroid, i.e. the paragraph that best
    # represents how X sounds on a normal page.
    for c_idx, c_name in enumerate(companies):
        rows = np.flatnonzero(coi == c_idx)
        if len(rows) == 0:
            continue
        d = np.linalg.norm(X[rows] - Cco[c_idx], axis=1)
        i = int(rows[np.argmin(d)])
        exhibits.append(_entry(
            i,
            f"{c_name}: most typical passage",
            f"The paragraph closest to the center of {c_name}\u2019s "
            "report. What their writing sounds like on an average page."))

    # ---- 4. Best cross-competitor near-twin (single strongest example).
    # Kept because it\u2019s the corpus\u2019 most compelling storytelling
    # artifact. Runner-ups are surfaced on the companion page only.
    twin_candidates = []
    for c_idx in range(len(companies)):
        rows = np.flatnonzero((coi == c_idx) & (dd < TABLE_DIGIT_THRESHOLD))
        if len(rows) == 0:
            continue
        same_firm_cols = np.flatnonzero(coi == c_idx)
        sims = X[rows] @ X.T
        sims[:, same_firm_cols] = -np.inf
        best_pair = np.unravel_index(np.argmax(sims), sims.shape)
        score = float(sims[best_pair])
        twin_candidates.append((score, int(rows[best_pair[0]]), int(best_pair[1])))
    twin_candidates.sort(reverse=True)
    presented = set()
    twin_pairs: List[tuple] = []
    for score, i_own, i_other in twin_candidates:
        if i_own in presented or i_other in presented:
            continue
        presented.update({i_own, i_other})
        twin_pairs.append((i_own, i_other, score))
        if len(twin_pairs) >= 3:
            break

    if twin_pairs:
        i_own, i_other, _ = twin_pairs[0]
        co_own = df.iloc[i_own]["company"]
        co_other = df.iloc[i_other]["company"]
        exhibits.append(_entry(
            i_own,
            f"Near-twin: {co_own} \u2194 {co_other}",
            f"Two firms saying nearly the same thing. Click to spotlight "
            f"the twin passage inside {co_other}\u2019s report."))

    # ---- 5. Prose \u2194 tabular axis extremes.
    i = int(np.argmax(proj))
    exhibits.append(_entry(
        i, "Most tabular passage",
        "The densest wall of numbers in the whole corpus."))

    i = int(np.argmin(proj))
    exhibits.append(_entry(
        i, "Most narrative passage",
        "Pure prose, almost no digits. The purest \u201Cvoice\u201D "
        "chunk in the corpus."))

    # ---- 6. Runner-up near-twins for the companion page.
    for k, (i_own, i_other, _) in enumerate(twin_pairs[1:], start=2):
        co_own = df.iloc[i_own]["company"]
        co_other = df.iloc[i_other]["company"]
        exhibits.append(_entry(
            i_own,
            f"Near-twin #{k}: {co_own} \u2194 {co_other}",
            f"Another cross-competitor pair with near-identical content."))

    # ---- 7. GMM cluster reps.
    for cl in [0, 1]:
        rows = np.flatnonzero(cluster == cl)
        if len(rows) == 0:
            continue
        mu = X[rows].mean(0)
        d = np.linalg.norm(X[rows] - mu, axis=1)
        i = int(rows[np.argmin(d)])
        exhibits.append(_entry(
            i, f"Group {chr(ord('A') + cl)} center",
            "The chunk at the middle of one of two groups the encoder "
            "finds on its own, without being told about prose or tables."))

    curate_exhibits._twin_pairs = twin_pairs  # type: ignore[attr-defined]
    return exhibits


# ---------------------------------------------------------- HTML template

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semantic Universe Explorer &mdash; SUE</title>
<link rel="preconnect" href="https://cdn.plot.ly">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root {
    --bg: #f7f7f8;
    --panel: #ffffff;
    --border: #d9d9de;
    --text: #1e1e22;
    --muted: #55575c;
    --accent: #a45a1e;
    --accent2: #4a6a8a;
    --highlight: #fff5e0;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg);
               color: var(--text); font-family: -apple-system, "Segoe UI",
               Roboto, "Helvetica Neue", Arial, sans-serif;
               font-size: 14px; line-height: 1.45; }
  header { padding: 18px 28px 10px; border-bottom: 1px solid var(--border);
           background: var(--panel); }
  header h1 { font-size: 22px; margin: 0 0 4px; color: var(--text); }
  header h1 .accent { color: var(--accent); }
  header .subtitle { color: var(--muted); font-size: 13.5px; margin: 0; }
  header .links { margin-top: 6px; font-size: 12.5px; }
  header .links a { color: var(--accent2); margin-right: 14px;
                    text-decoration: none; border-bottom: 1px dotted var(--accent2); }
  header .links a:hover { color: var(--accent); border-bottom-color: var(--accent); }

  .app { display: grid; grid-template-columns: 240px 1fr 320px;
         gap: 14px; padding: 14px; max-width: 1600px; margin: 0 auto; }
  aside.controls {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 16px;
    max-height: calc(100vh - 220px); overflow-y: auto;
  }
  aside.right-col { display: flex; flex-direction: column; gap: 12px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 14px;
  }
  main.viz { display: flex; flex-direction: column; gap: 12px; }

  .ctrl-group { margin-bottom: 18px; }
  .ctrl-group h3 { font-size: 12px; text-transform: uppercase;
                   letter-spacing: 0.06em; color: var(--muted);
                   margin: 0 0 6px; }
  .ctrl-group label { display: block; margin: 3px 0; cursor: pointer; }
  .ctrl-group input[type=radio] { margin-right: 6px; }
  .ctrl-group select, .ctrl-group input[type=range] { width: 100%; }
  .k-value { display: inline-block; font-weight: bold; color: var(--accent);
             margin-left: 4px; }
  .sel-btn { flex: 1; padding: 6px 8px; font-size: 12.5px;
             border: 1px solid var(--border); background: #f1f1f4;
             color: var(--text); border-radius: 4px; cursor: pointer;
             font-family: inherit; }
  .sel-btn:hover:not(:disabled) { background: var(--highlight);
                                   border-color: var(--accent); }
  .sel-btn:disabled { opacity: 0.45; cursor: not-allowed; }

  #plot { flex-grow: 1; min-height: 640px;
          background: var(--panel); border: 1px solid var(--border);
          border-radius: 8px; }

  .panel h2 { font-size: 14px; margin: 0 0 8px;
              color: var(--accent); border-bottom: 1px solid var(--border);
              padding-bottom: 6px; text-transform: uppercase;
              letter-spacing: 0.05em; }
  .panel .meta { font-size: 12.5px; color: var(--muted); margin: 3px 0; }
  .panel .snippet {
    font-size: 13px; background: var(--highlight);
    padding: 8px 10px; border-left: 3px solid var(--accent);
    border-radius: 4px; margin: 8px 0 0;
    white-space: pre-wrap;
    /* clamp to ~6 lines with an ellipsis; no scrollbar */
    display: -webkit-box; -webkit-line-clamp: 6; -webkit-box-orient: vertical;
    overflow: hidden; text-overflow: ellipsis;
    line-height: 1.35;
  }
  .panel .empty { color: #a0a0a0; font-style: italic; font-size: 12.5px; }
  .k-block { margin-top: 10px; padding-top: 8px;
             border-top: 1px dashed var(--border); }
  .k-block .k-label { font-size: 12.5px; color: var(--muted);
                      display: block; margin-bottom: 4px; }
  .k-block input[type=range] { width: 100%; }

  .exhibit {
    border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 9px; margin: 5px 0; cursor: pointer;
    background: #fbfbfd; transition: background 0.1s ease;
  }
  .exhibit:hover { background: var(--highlight); }
  .exhibit.active { background: var(--highlight); border-color: var(--accent); }
  .exhibit .etitle { font-weight: bold; font-size: 12.3px;
                     color: var(--accent2);
                     white-space: nowrap; overflow: hidden;
                     text-overflow: ellipsis; }
  .exhibit .enote { font-size: 11.6px; color: var(--muted);
                    margin-top: 1px; font-style: italic;
                    white-space: nowrap; overflow: hidden;
                    text-overflow: ellipsis; }
  .see-more { display: block; text-align: right; margin-top: 6px;
              font-size: 11.8px; color: var(--accent2);
              text-decoration: none;
              border-bottom: 1px dotted var(--accent2);
              width: fit-content; margin-left: auto; }
  .see-more:hover { color: var(--accent); border-bottom-color: var(--accent); }

  @media (max-width: 1000px) {
    .app { grid-template-columns: 1fr; }
    aside.controls, aside.inspector { max-height: none; }
  }
</style>
</head>
<body>

<header>
  <h1>Semantic Universe Explorer <span class="accent">&mdash; SUE</span></h1>
  <p class="subtitle">
    An interactive corpus observatory for a specific comparative
    question: <b>five direct competitors in semiconductor-manufacturing
    equipment</b> \u2014 Applied Materials, ASML, KLA, Lam Research, and
    TEL (Tokyo Electron) \u2014 all publish Global Impact / ESG reports on
    the same topics every year. How comparable are they? Which chunks and
    which docs sit apart from the four-competitor consensus? Every dot
    below is one ~900-character chunk. Filter, color, click, and read.
  </p>
  <p class="links">
    <a href="explorer_examples.html">Longer walk-through &amp; storytelling examples \u2192</a>
    <a href="__REPO_URL__" target="_blank">GitHub repository</a>
  </p>
</header>

<div class="app">

<aside class="controls">
  <div class="ctrl-group" id="selection-controls">
    <h3>Selection</h3>
    <div style="display:flex;gap:6px;">
      <button id="undo-selection" class="sel-btn" disabled
        title="Go back to the previously selected point">\u21A9 Undo</button>
      <button id="clear-selection" class="sel-btn" disabled
        title="Deselect and show all points">Show all</button>
    </div>
  </div>

  <div class="ctrl-group">
    <h3>View</h3>
    <label><input type="radio" name="view" value="3d" checked> 3D projection</label>
    <label><input type="radio" name="view" value="2d"> 2D projection</label>
  </div>

  <div class="ctrl-group">
    <h3>Color by</h3>
    <label><input type="radio" name="color" value="company" checked> Company</label>
    <label><input type="radio" name="color" value="prose_tabular"> Prose \u2194 tabular score</label>
    <label><input type="radio" name="color" value="digit_density"> Digit density (raw)</label>
    <label><input type="radio" name="color" value="cross_corpus_out"> Cross-corpus outlierness</label>
    <label><input type="radio" name="color" value="doc_typicality"> In-doc typicality</label>
    <label><input type="radio" name="color" value="cluster"> Unsupervised cluster (GMM)</label>
  </div>

  <div class="ctrl-group">
    <h3>Filter \u2014 content type</h3>
    <label><input type="radio" name="content" value="all" checked> All chunks</label>
    <label><input type="radio" name="content" value="prose"> Only prose (digits &lt; 5%)</label>
    <label><input type="radio" name="content" value="ambiguous"> Only ambiguous (5\u201318%)</label>
    <label><input type="radio" name="content" value="tabular"> Only tabular (digits &gt; 18%)</label>
    <div style="margin-top:8px;font-size:11.5px;color:var(--muted);">
      Or narrow further with a range:
    </div>
    <div style="display:flex;gap:6px;align-items:center;margin-top:4px;">
      <input type="range" id="dd-min" min="0" max="40" value="0" style="flex:1;">
      <span id="dd-min-val" class="k-value">0%</span>
    </div>
    <div style="display:flex;gap:6px;align-items:center;margin-top:2px;">
      <input type="range" id="dd-max" min="0" max="40" value="40" style="flex:1;">
      <span id="dd-max-val" class="k-value">40%</span>
    </div>
  </div>

  <div class="ctrl-group">
    <h3>Filter \u2014 companies</h3>
    <div id="company-filters"></div>
  </div>

  <div class="ctrl-group">
    <h3>Filter \u2014 outliers only</h3>
    <label>
      <input type="checkbox" id="outliers-only"> Show only the top
      <span class="k-value" id="outliers-pct-val">15%</span> most atypical
    </label>
    <input type="range" id="outliers-pct" min="1" max="50" value="15" style="margin-top:6px;">
    <div style="font-size:11.5px;color:var(--muted);margin-top:2px;">
      By cross-corpus outlierness. Reveals which chunks/docs sit apart
      from what the sector has converged to.
    </div>
  </div>

  <div class="ctrl-group">
    <h3>Selection display</h3>
    <label><input type="checkbox" id="dim-others" checked> Gray out unrelated points</label>
  </div>
</aside>

<main class="viz">
  <div id="plot"></div>
</main>

<aside class="right-col">
  <div class="panel" id="inspector-panel">
    <h2>Selected point</h2>
    <div id="inspector-body">
      <div class="empty">Click any point on the plot, or pick a Cool
        example below.</div>
    </div>
  </div>

  <div class="panel" id="cool-examples-panel">
    <h2>Cool examples</h2>
    <div id="exhibits-list"></div>
    <a class="see-more" href="explorer_examples.html">See all &amp; storytelling walk-through \u2192</a>
  </div>
</aside>

</div>

<script>
const DATA = __DATA_JSON__;
const PLOTLY_CONFIG = { displaylogo: false, responsive: true,
  modeBarButtonsToRemove: ['toImage','sendDataToCloud','autoScale2d'] };

// ---------- state ----------
let state = {
  view: '3d',
  color: 'company',
  selection: null,       // index into DATA.chunks
  K: 15,
  dimOthers: true,
  // filters
  content: 'all',
  ddMin: 0,   // percent
  ddMax: 40,  // percent
  companyOn: {},   // company name -> true/false; initialized below
  outliersOnly: false,
  outliersPct: 15,
};

// ---------- helpers ----------
function colorValues(mode) {
  // Returns { values: [...], type, title, [cscale, cmin, cmax, levels] }
  if (mode === 'company') {
    const vals = DATA.company_of.map(i => DATA.companies[i]);
    return { values: vals, type: 'categorical', title: 'Company',
             levels: DATA.companies };
  }
  if (mode === 'prose_tabular') {
    return { values: DATA.prose_tabular_score, type: 'continuous',
             title: 'Prose \u2194 tabular projection',
             cscale: 'RdBu', cmin: -DATA.pt_absmax, cmax: DATA.pt_absmax };
  }
  if (mode === 'digit_density') {
    return { values: DATA.digit_density, type: 'continuous',
             title: 'Digit density', cscale: 'Purples',
             cmin: 0.0, cmax: 0.35 };
  }
  if (mode === 'cross_corpus_out') {
    return { values: DATA.dist_from_others, type: 'continuous',
             title: 'Distance from 4-competitor consensus',
             cscale: 'Reds', cmin: 0.0, cmax: DATA.dist_from_others_max };
  }
  if (mode === 'doc_typicality') {
    return { values: DATA.dist_own_centroid, type: 'continuous',
             title: 'Distance from own doc centroid',
             cscale: 'Viridis', cmin: 0.0, cmax: DATA.dist_own_max };
  }
  if (mode === 'cluster') {
    const vals = DATA.cluster_of.map(i => `Cluster ${String.fromCharCode(65 + i)}`);
    const seen = [];
    for (const x of DATA.cluster_of) {
      const lbl = `Cluster ${String.fromCharCode(65 + x)}`;
      if (!seen.includes(lbl)) seen.push(lbl);
    }
    seen.sort();
    return { values: vals, type: 'categorical', title: 'Unsupervised cluster',
             levels: seen };
  }
  return { values: [], type: 'categorical', title: '', levels: [] };
}

function palette(i) { return DATA.palette[i % DATA.palette.length]; }

// Memoised hover-text cache: computed once on page load. Rebuilding
// 3.7k HTML-escaped snippet strings on every redraw was the primary
// cause of the click-freeze reported by the user.
const _hoverCache = new Array(DATA.chunks.length);
function hoverTextFor(idx) {
  let s = _hoverCache[idx];
  if (s !== undefined) return s;
  const c = DATA.chunks[idx];
  const snip = c.text.slice(0, 200).replace(/\\n/g, ' ');
  const dd = (DATA.digit_density[idx] * 100).toFixed(1);
  s = `<b>${escapeHtml(c.company)}</b><br>`
    + `${escapeHtml(c.doc)} \u00b7 chunk ${c.chunk_id}`
    + ` \u00b7 digits ${dd}%<br>`
    + `${escapeHtml(snip)}\u2026`;
  _hoverCache[idx] = s;
  return s;
}
// Warm the cache on page load, off the main click path.
setTimeout(() => {
  for (let i = 0; i < DATA.chunks.length; i++) hoverTextFor(i);
}, 0);

// ---------- filter logic ----------
function passesFilters(i) {
  // company checkboxes
  const co = DATA.companies[DATA.company_of[i]];
  if (state.companyOn[co] === false) return false;
  // content-type + digit-density range
  const ddPct = DATA.digit_density[i] * 100;
  if (state.content === 'prose'     && ddPct >= 5.0) return false;
  if (state.content === 'ambiguous' && (ddPct < 5.0 || ddPct > 18.0)) return false;
  if (state.content === 'tabular'   && ddPct <= 18.0) return false;
  if (ddPct < state.ddMin || ddPct > state.ddMax) return false;
  // outliers-only
  if (state.outliersOnly) {
    const cutoff = DATA.outlier_cutoffs[state.outliersPct - 1];
    if (DATA.dist_from_others[i] < cutoff) return false;
  }
  return true;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, m => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[m]));
}

// ---------- plot builders ----------
function buildTraces() {
  const c = colorValues(state.color);
  const is3d = state.view === '3d';
  const coords = is3d ? DATA.coords_3d : DATA.coords_2d;

  // spotlight mask (always includes the selected chunk itself so it can
  // still be located even when filters would otherwise hide it)
  const spotlight = new Set();
  if (state.selection !== null) {
    spotlight.add(state.selection);
    for (const nb of DATA.neighbors[state.selection].slice(0, state.K)) {
      spotlight.add(nb);
    }
  }
  const dimming = state.selection !== null && state.dimOthers;

  // build the set of chunk indices that survive the filter panel;
  // spotlight always survives so a click never causes its target to vanish
  const N = DATA.chunks.length;
  const visible = new Array(N);
  let visibleCount = 0;
  for (let i = 0; i < N; i++) {
    if (passesFilters(i) || spotlight.has(i)) {
      visible[i] = true;
      visibleCount++;
    } else {
      visible[i] = false;
    }
  }
  const traces = [];

  if (c.type === 'categorical') {
    for (const lvl of c.levels) {
      const idx = [];
      for (let i = 0; i < c.values.length; i++) {
        if (c.values[i] === lvl && visible[i]) idx.push(i);
      }
      if (idx.length === 0) continue;
      const dimIdx = dimming ? idx.filter(i => !spotlight.has(i)) : [];
      const brightIdx = dimming ? idx.filter(i => spotlight.has(i)) : idx;
      const col = palette(c.levels.indexOf(lvl));

      if (dimIdx.length) {
        traces.push(_scatterTrace(dimIdx, coords, is3d, {
          color: '#dddddd', size: 2.4, opacity: 0.15,
          name: lvl, showlegend: false, noHover: true,
        }));
      }
      if (brightIdx.length) {
        traces.push(_scatterTrace(brightIdx, coords, is3d, {
          color: col, size: 3.4, opacity: dimming ? 0.95 : 0.75,
          name: lvl, showlegend: true,
        }));
      }
    }
  } else {
    // continuous
    const idxAll = [];
    for (let i = 0; i < N; i++) if (visible[i]) idxAll.push(i);
    const dimIdx = dimming ? idxAll.filter(i => !spotlight.has(i)) : [];
    const brightIdx = dimming ? idxAll.filter(i => spotlight.has(i)) : idxAll;

    if (dimIdx.length) {
      traces.push(_scatterTrace(dimIdx, coords, is3d, {
        color: '#dddddd', size: 2.4, opacity: 0.15,
        name: 'other', showlegend: false, noHover: true,
      }));
    }
    if (brightIdx.length) {
      traces.push({
        type: is3d ? 'scatter3d' : 'scattergl',
        x: brightIdx.map(i => coords[i][0]),
        y: brightIdx.map(i => coords[i][1]),
        ...(is3d ? {z: brightIdx.map(i => coords[i][2])} : {}),
        mode: 'markers',
        marker: {
          size: is3d ? 3.4 : 6,
          color: brightIdx.map(i => c.values[i]),
          colorscale: c.cscale, cmin: c.cmin, cmax: c.cmax,
          showscale: true,
          colorbar: {title: {text: c.title, font: {size: 10}},
                     len: 0.6, thickness: 12},
          opacity: dimming ? 0.95 : 0.8,
          line: {width: 0},
        },
        text: brightIdx.map(hoverTextFor),
        hovertemplate: '%{text}<extra></extra>',
        customdata: brightIdx,
        showlegend: false,
      });
    }
  }

  // spotlight ring (draws the selected chunk larger and outlined)
  if (state.selection !== null) {
    const sel = state.selection;
    traces.push({
      type: is3d ? 'scatter3d' : 'scattergl',
      x: [coords[sel][0]], y: [coords[sel][1]],
      ...(is3d ? {z: [coords[sel][2]]} : {}),
      mode: 'markers',
      marker: {size: is3d ? 8 : 14, color: '#a45a1e',
               line: {width: 2, color: '#000'}, symbol: is3d ? 'diamond' : 'star'},
      name: 'selected',
      hoverinfo: 'skip', showlegend: false,
    });
    // neighbors as a distinct outline layer
    const nbrs = DATA.neighbors[sel].slice(0, state.K);
    traces.push({
      type: is3d ? 'scatter3d' : 'scattergl',
      x: nbrs.map(i => coords[i][0]),
      y: nbrs.map(i => coords[i][1]),
      ...(is3d ? {z: nbrs.map(i => coords[i][2])} : {}),
      mode: 'markers',
      marker: {size: is3d ? 5 : 9, color: 'rgba(0,0,0,0)',
               line: {width: 1.5, color: '#a45a1e'}},
      name: `top ${state.K} neighbors`,
      text: nbrs.map(hoverTextFor),
      hovertemplate: '%{text}<extra></extra>',
      customdata: nbrs,
      showlegend: false,
    });
  }

  return traces;
}

function _scatterTrace(idx, coords, is3d, opts) {
  const base = {
    type: is3d ? 'scatter3d' : 'scattergl',
    x: idx.map(i => coords[i][0]),
    y: idx.map(i => coords[i][1]),
    ...(is3d ? {z: idx.map(i => coords[i][2])} : {}),
    mode: 'markers',
    marker: {size: is3d ? opts.size : opts.size * 1.7,
             color: opts.color, opacity: opts.opacity, line: {width: 0}},
    customdata: idx,
    name: opts.name, showlegend: opts.showlegend,
    legendgroup: opts.name,
  };
  if (opts.noHover) {
    base.hoverinfo = 'skip';
  } else {
    base.text = idx.map(hoverTextFor);
    base.hovertemplate = '%{text}<extra></extra>';
  }
  return base;
}

function buildLayout() {
  const is3d = state.view === '3d';
  const ev = DATA.explained_variance;
  const base = {
    autosize: true, height: 640,
    margin: {l: 0, r: 0, t: 30, b: 0},
    hovermode: 'closest',
    legend: {itemsizing: 'constant', font: {size: 10}, y: 0.98},
    title: {text: '', font: {size: 12}},
    uirevision: 'keep',   // preserve camera/zoom across react()
  };
  if (is3d) {
    return { ...base, scene: {
      xaxis: {title: `PC 1 (${(ev[0]*100).toFixed(1)}%)`, backgroundcolor: '#fbfbfd'},
      yaxis: {title: `PC 2 (${(ev[1]*100).toFixed(1)}%)`, backgroundcolor: '#fbfbfd'},
      zaxis: {title: `PC 3 (${(ev[2]*100).toFixed(1)}%)`, backgroundcolor: '#fbfbfd'},
      camera: {eye: {x: 1.6, y: 1.6, z: 0.9}},
    }};
  }
  return { ...base,
    xaxis: {title: `PC 1 (${(ev[0]*100).toFixed(1)}%)`, zeroline: false},
    yaxis: {title: `PC 2 (${(ev[1]*100).toFixed(1)}%)`, zeroline: false},
  };
}

function redraw() {
  Plotly.react('plot', buildTraces(), buildLayout(), PLOTLY_CONFIG);
}

// ---------- selection / inspector ----------
// Selection history: undo pops back to the previous selection. null means
// "no selection" (i.e. the base "show all" state).
const selectionHistory = [];
const HISTORY_MAX = 50;

function selectIndex(idx) {
  if (idx === state.selection) return;   // no-op click
  selectionHistory.push(state.selection);
  if (selectionHistory.length > HISTORY_MAX) selectionHistory.shift();
  state.selection = idx;
  if (typeof stopAutoRotate === 'function') stopAutoRotate();
  updateInspector();
  updateSelectionButtons();
  redraw();
}

function undoSelection() {
  if (selectionHistory.length === 0) return;
  state.selection = selectionHistory.pop();
  document.querySelectorAll('.exhibit').forEach(n => n.classList.remove('active'));
  updateInspector();
  updateSelectionButtons();
  redraw();
}

function clearSelection() {
  if (state.selection === null && selectionHistory.length === 0) return;
  selectionHistory.length = 0;
  state.selection = null;
  document.querySelectorAll('.exhibit').forEach(n => n.classList.remove('active'));
  updateInspector();
  updateSelectionButtons();
  redraw();
}

function updateSelectionButtons() {
  document.getElementById('undo-selection').disabled =
    selectionHistory.length === 0;
  document.getElementById('clear-selection').disabled =
    state.selection === null && selectionHistory.length === 0;
}

function updateInspector() {
  const el = document.getElementById('inspector-body');
  if (state.selection === null) {
    el.innerHTML = '<div class="empty">Click any point on the plot, or '
      + 'pick a Cool example below.</div>';
    return;
  }
  const c = DATA.chunks[state.selection];
  const pt = DATA.prose_tabular_score[state.selection].toFixed(2);
  const dd = (DATA.digit_density[state.selection] * 100).toFixed(1);
  const cco = DATA.dist_from_others[state.selection].toFixed(2);
  const kMax = DATA.neighbors[state.selection].length;
  el.innerHTML = `
    <div class="meta"><b>${escapeHtml(c.company)}</b> \u00b7
      ${escapeHtml(c.doc)} \u00b7 chunk ${c.chunk_id}</div>
    <div class="meta">digits ${dd}% \u00b7 prose\u2194tab ${pt}
      \u00b7 off-consensus ${cco}</div>
    <div class="snippet">${escapeHtml(c.text)}</div>
    <div class="k-block">
      <label for="k-slider" class="k-label">Show
        <span class="k-value" id="k-value">${state.K}</span>
        nearest neighbors of this point</label>
      <input type="range" id="k-slider" min="5" max="${kMax}" value="${state.K}">
    </div>
  `;
  // Wire the slider each time the inspector is rebuilt.
  const slider = document.getElementById('k-slider');
  slider.addEventListener('input', (e) => {
    state.K = parseInt(e.target.value, 10);
    document.getElementById('k-value').textContent = state.K;
    redraw();
  });
}

// ---------- exhibits ----------
const EXHIBITS_IN_PANEL = 6;   // remainder shown on the companion page
function renderExhibits() {
  const el = document.getElementById('exhibits-list');
  const items = DATA.exhibits.slice(0, EXHIBITS_IN_PANEL).map((e, k) => `
    <div class="exhibit" data-idx="${e.index}" data-k="${k}">
      <div class="etitle">${escapeHtml(e.title)}</div>
      <div class="enote">${escapeHtml(e.note)}</div>
    </div>
  `).join('');
  el.innerHTML = items;
  el.querySelectorAll('.exhibit').forEach(node => {
    node.addEventListener('click', () => {
      el.querySelectorAll('.exhibit').forEach(n => n.classList.remove('active'));
      node.classList.add('active');
      const idx = parseInt(node.dataset.idx, 10);
      selectIndex(idx);
    });
  });
}

// ---------- wire up controls ----------
document.querySelectorAll('input[name=view]').forEach(el => {
  el.addEventListener('change', () => {
    state.view = document.querySelector('input[name=view]:checked').value;
    redraw();
  });
});
document.querySelectorAll('input[name=color]').forEach(el => {
  el.addEventListener('change', () => {
    state.color = document.querySelector('input[name=color]:checked').value;
    redraw();
  });
});
document.getElementById('dim-others').addEventListener('change', (e) => {
  state.dimOthers = e.target.checked;
  redraw();
});
document.getElementById('undo-selection').addEventListener('click', undoSelection);
document.getElementById('clear-selection').addEventListener('click', clearSelection);

// content-type radios
document.querySelectorAll('input[name=content]').forEach(el => {
  el.addEventListener('change', () => {
    state.content = document.querySelector('input[name=content]:checked').value;
    redraw();
  });
});

// digit-density range sliders (dual handle: min & max)
function clampDDRange() {
  // enforce min <= max
  if (state.ddMin > state.ddMax) {
    // whichever slider moved last, push the other one
    state.ddMax = state.ddMin;
    document.getElementById('dd-max').value = state.ddMax;
  }
  document.getElementById('dd-min-val').textContent = state.ddMin + '%';
  document.getElementById('dd-max-val').textContent = state.ddMax + '%';
}
document.getElementById('dd-min').addEventListener('input', (e) => {
  state.ddMin = parseInt(e.target.value, 10);
  clampDDRange();
  redraw();
});
document.getElementById('dd-max').addEventListener('input', (e) => {
  state.ddMax = parseInt(e.target.value, 10);
  clampDDRange();
  redraw();
});

// per-company checkboxes
function renderCompanyFilters() {
  const el = document.getElementById('company-filters');
  const html = DATA.companies.map((c, i) => {
    const col = palette(i);
    return `
      <label style="display:flex;align-items:center;gap:6px;">
        <input type="checkbox" class="co-cb" data-co="${escapeHtml(c)}" checked>
        <span style="display:inline-block;width:10px;height:10px;
          background:${col};border-radius:2px;flex-shrink:0;"></span>
        <span style="font-size:12.5px;">${escapeHtml(c)}</span>
      </label>`;
  }).join('');
  el.innerHTML = html;
  el.querySelectorAll('.co-cb').forEach(cb => {
    state.companyOn[cb.dataset.co] = true;
    cb.addEventListener('change', () => {
      state.companyOn[cb.dataset.co] = cb.checked;
      redraw();
    });
  });
}

// outliers-only toggle + percentile slider
document.getElementById('outliers-only').addEventListener('change', (e) => {
  state.outliersOnly = e.target.checked;
  redraw();
});
document.getElementById('outliers-pct').addEventListener('input', (e) => {
  state.outliersPct = parseInt(e.target.value, 10);
  document.getElementById('outliers-pct-val').textContent = state.outliersPct + '%';
  if (state.outliersOnly) redraw();
});

// ---------- initial render ----------
renderCompanyFilters();
renderExhibits();
Plotly.newPlot('plot', buildTraces(), buildLayout(), PLOTLY_CONFIG);
const plotEl = document.getElementById('plot');
plotEl.on('plotly_click', (e) => {
  if (!e || !e.points || !e.points.length) return;
  const pt = e.points[0];
  if (pt.customdata !== undefined && pt.customdata !== null) {
    stopAutoRotate();
    selectIndex(pt.customdata);
  }
});

// ---------- orbital auto-rotate (3D only, until user interacts) ----------
let autoRotate = state.view === '3d';
let autoRotateT = 0;
const AUTOROTATE_RADIUS = 2.2;
const AUTOROTATE_Z = 0.9;
function autoRotateTick() {
  if (!autoRotate || state.view !== '3d') return;
  autoRotateT += 0.006;   // slow drift
  Plotly.relayout('plot', {
    'scene.camera.eye.x': AUTOROTATE_RADIUS * Math.cos(autoRotateT),
    'scene.camera.eye.y': AUTOROTATE_RADIUS * Math.sin(autoRotateT),
    'scene.camera.eye.z': AUTOROTATE_Z,
  });
}
const autoRotateHandle = setInterval(autoRotateTick, 33);
function stopAutoRotate() { autoRotate = false; }
plotEl.addEventListener('mousedown', stopAutoRotate);
plotEl.addEventListener('wheel',     stopAutoRotate);
plotEl.addEventListener('touchstart', stopAutoRotate);
// switching to 2D also stops the (irrelevant) rotation loop
document.querySelectorAll('input[name=view]').forEach(el => {
  el.addEventListener('change', stopAutoRotate);
});
</script>

</body>
</html>
"""


# ---------------------------------------------------------- companion page

_EXAMPLES_CSS = """
:root { --bg:#f7f7f8; --panel:#fff; --border:#d9d9de; --text:#1e1e22;
        --muted:#55575c; --accent:#a45a1e; --accent2:#4a6a8a;
        --highlight:#fff5e0; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
       font-size:15px; line-height:1.55; }
.wrap { max-width: 900px; margin: 0 auto; padding: 32px 24px 60px; }
header a.back { color: var(--accent2); text-decoration: none;
                border-bottom: 1px dotted var(--accent2); }
h1 { font-size: 28px; margin: 8px 0 4px; }
h1 .accent { color: var(--accent); }
h2 { font-size: 20px; color: var(--accent); margin: 36px 0 8px;
     border-bottom: 1px solid var(--border); padding-bottom: 6px; }
h3 { font-size: 16px; margin: 20px 0 4px; color: var(--accent2); }
p { margin: 8px 0; }
.subtle { color: var(--muted); font-size: 13.5px; }
.card { background: var(--panel); border: 1px solid var(--border);
        border-radius: 8px; padding: 14px 18px; margin: 14px 0; }
.card h3 { margin-top: 0; }
.twin { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.twin .side { background: var(--highlight); border-left: 3px solid var(--accent);
              padding: 10px 12px; border-radius: 4px; font-size: 14px;
              white-space: pre-wrap; }
.twin .side .who { font-weight: bold; color: var(--accent2);
                    font-size: 12.5px; margin-bottom: 6px;
                    text-transform: uppercase; letter-spacing: 0.05em; }
.snippet { background: var(--highlight); border-left: 3px solid var(--accent);
           padding: 10px 12px; border-radius: 4px; font-size: 14px;
           white-space: pre-wrap; margin: 8px 0; }
.meta { color: var(--muted); font-size: 12.5px; margin: 4px 0; }
dl.modes dt { font-weight: bold; color: var(--accent2); margin-top: 12px; }
dl.modes dd { margin: 2px 0 8px 0; }
@media (max-width: 720px) { .twin { grid-template-columns: 1fr; } }
"""


def _render_examples_page(df: pd.DataFrame, exhibits: List[dict],
                          twin_pairs: List[tuple], companies: List[str]) -> str:
    def _get_text(i: int) -> str:
        return escape(str(df.iloc[i]["text"])[:1400])

    def _get_meta(i: int) -> str:
        row = df.iloc[i]
        return f"{escape(str(row['company']))} \u00b7 {escape(str(row['doc']))} \u00b7 chunk {int(row['chunk_id'])}"

    # ---- twin cards
    twin_cards = []
    for k, (i_own, i_other, score) in enumerate(twin_pairs, start=1):
        row_own = df.iloc[i_own]
        row_other = df.iloc[i_other]
        twin_cards.append(f"""
<div class="card">
  <h3>Near-twin #{k}: {escape(str(row_own['company']))} \u2194 {escape(str(row_other['company']))}</h3>
  <p class="subtle">Cosine similarity in the 384-D embedding space: {score:.3f}.
     Two different firms, different reports, but the encoder places these two
     passages almost on top of each other.</p>
  <div class="twin">
    <div class="side"><div class="who">{escape(str(row_own['company']))}</div>{escape(str(row_own['text'])[:900])}</div>
    <div class="side"><div class="who">{escape(str(row_other['company']))}</div>{escape(str(row_other['text'])[:900])}</div>
  </div>
</div>
""")
    twin_section = "\n".join(twin_cards) if twin_cards else "<p class=\"subtle\">No twin pairs curated yet.</p>"

    # ---- all exhibits in full
    exhibit_cards = []
    for e in exhibits:
        i = int(e["index"])
        exhibit_cards.append(f"""
<div class="card">
  <h3>{escape(e['title'])}</h3>
  <p class="subtle">{escape(e['note'])}</p>
  <div class="meta">{_get_meta(i)}</div>
  <div class="snippet">{_get_text(i)}</div>
</div>
""")
    exhibits_section = "\n".join(exhibit_cards)

    companies_str = ", ".join(companies[:-1]) + f", and {companies[-1]}" if len(companies) > 1 else companies[0]

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SUE \u2014 walk-through &amp; storytelling examples</title>
<style>{_EXAMPLES_CSS}</style>
</head><body>
<div class="wrap">
<header><a class="back" href="semantic_universe_explorer.html">\u2190 back to the interactive viewer</a>
<h1>SUE \u2014 <span class="accent">walk-through</span> &amp; storytelling examples</h1>
<p class="subtle">A slower, static companion to the interactive Semantic
Universe Explorer. Five direct competitors in
semiconductor-manufacturing equipment ({escape(companies_str)}) all publish
Global Impact / ESG reports every year. This page is the guided tour of
what the geometry of their reports actually looks like.</p></header>

<h2>What each color mode means</h2>
<dl class="modes">
<dt>Company</dt>
<dd>Every dot colored by which of the five firms authored the underlying
    passage. Useful for eyeballing whether any single firm produces a
    visually distinct region of the space.</dd>
<dt>Prose \u2194 tabular score</dt>
<dd>A number for each chunk from \u201Cvery narrative\u201D (blue) through
    zero (mixed) to \u201Cvery tabular\u201D (red). Computed as the
    projection onto the axis that connects the average narrative-chunk
    location to the average tabular-chunk location. The direction that
    the fixed classifier <em>would</em> read as \u201Cnumbers vs.
    words.\u201D</dd>
<dt>Digit density (raw)</dt>
<dd>The unlearned control: what percentage of the chunk\u2019s characters
    are numerals. Purples = more numbers. Whenever the prose\u2194tabular
    score and digit density paint the same picture, that\u2019s your sign
    that the encoder\u2019s \u201Csemantics\u201D on this axis is really
    just numerals.</dd>
<dt>Cross-corpus outlierness</dt>
<dd>For every chunk, how far is it from the average of the OTHER four
    competitors\u2019 embeddings? Red = far. This is the answer to
    \u201Cwhich passages of firm X are doing something the other four
    aren\u2019t?\u201D</dd>
<dt>In-doc typicality</dt>
<dd>For every chunk, how far is it from the center of its own document?
    Yellow = close to the doc\u2019s center of gravity, viridis-blue =
    far. Useful for spotting chunks that read very unlike the rest of
    their own report.</dd>
<dt>Unsupervised cluster (GMM)</dt>
<dd>A 2-component Gaussian mixture fit on the top-20 principal components,
    without ever being told about prose or tables. If it recovers the
    same partition anyway, that\u2019s evidence the split is real.</dd>
</dl>

<h2>Two firms, one sentence: cross-competitor near-twins</h2>
<p>These are the most compelling examples in the corpus for a
non-technical reader. Two direct competitors, different reports, and
the sentence encoder puts their passages almost on top of each other.
The <em>content</em> is nearly identical; only the letterhead is
different.</p>
{twin_section}

<h2>Every curated example, in full</h2>
{exhibits_section}

<p class="subtle" style="margin-top:32px">\u2190 <a href="semantic_universe_explorer.html">back to the interactive viewer</a></p>
</div></body></html>
"""


# ---------------------------------------------------------- main

def main():
    print("Loading cached corpus ...")
    df, X = load_cached()
    print(f"  {len(df):,} chunks x {X.shape[1]}-dim embeddings")

    print("Computing 2D + 3D PCA ...")
    p2 = PCA(n_components=2, random_state=0).fit(X)
    p3 = PCA(n_components=3, random_state=0).fit(X)
    Z2 = p2.transform(X).astype(np.float32)
    Z3 = p3.transform(X).astype(np.float32)
    ev = p3.explained_variance_ratio_.tolist()
    print(f"  3D explained variance {sum(ev):.1%}")

    print("Computing digit density + prose/tabular axis ...")
    dd = df["text"].map(digit_density).to_numpy(dtype=np.float32)
    prose_axis = compute_prose_axis(X, dd).astype(np.float32)
    pt_score = (X @ prose_axis).astype(np.float32)
    pt_score = pt_score - float(pt_score.mean())
    pt_absmax = float(np.percentile(np.abs(pt_score), 98))

    print("Computing per-doc + per-company centroids ...")
    C, cmeta, ci = compute_doc_centroids(df, X)
    Cco, cometa, coi = compute_company_centroids(df, X)
    # Alphabetise company order (case-insensitive) so it matches the
    # subtitle and so the sidebar checkboxes render in a predictable order.
    order = np.argsort([c["company"].lower() for c in cometa])
    Cco = Cco[order]
    cometa = [cometa[k] for k in order]
    remap = {int(old): new for new, old in enumerate(order)}
    coi = np.asarray([remap[int(c)] for c in coi], dtype=np.int64)
    companies = [c["company"] for c in cometa]
    print(f"  {len(companies)} companies: {', '.join(companies)}")

    print("Computing distance metrics ...")
    dist_own = np.linalg.norm(X - C[ci], axis=1).astype(np.float32)
    dist_own_max = float(np.percentile(dist_own, 98))
    dist_from_others = _dist_from_competitor_consensus(X, coi, Cco)
    dist_from_others_max = float(np.percentile(dist_from_others, 98))
    # 50 pre-computed percentile cutoffs so the outliers-only slider (1..50)
    # can just look up the threshold rather than re-sort in JS.
    outlier_cutoffs = [
        float(np.percentile(dist_from_others, 100.0 - p))
        for p in range(1, 51)
    ]

    print("Fitting 2-component GMM on top-20 PCA subspace ...")
    cluster = gmm_cluster_labels(X)

    print(f"Precomputing top-{TOP_NEIGHBORS} nearest neighbors (may take a bit) ...")
    nbrs = top_neighbors(X, TOP_NEIGHBORS)

    print("Curating salient exhibits ...")
    exhibits = curate_exhibits(df, X, dd, prose_axis, ci, C,
                               coi, Cco, companies, cluster)
    print(f"  {len(exhibits)} exhibits selected")

    print("Assembling JSON payload ...")
    co_lookup = {v: i for i, v in enumerate(companies)}
    company_of = df["company"].map(co_lookup).to_numpy(dtype=np.int32).tolist()

    def _snip(t: str) -> str:
        return (t or "")[:1200]

    chunks_json = [
        {
            "text": _snip(df.iloc[i]["text"]),
            "company": str(df.iloc[i]["company"]),
            "doc": str(df.iloc[i]["doc"]),
            "chunk_id": int(df.iloc[i]["chunk_id"]),
        }
        for i in range(len(df))
    ]

    payload = {
        "chunks": chunks_json,
        "companies": companies,
        "company_of": company_of,
        "coords_2d": np.round(Z2, 4).tolist(),
        "coords_3d": np.round(Z3, 4).tolist(),
        "explained_variance": [round(x, 4) for x in ev],
        "prose_tabular_score": np.round(pt_score, 4).tolist(),
        "pt_absmax": round(pt_absmax, 4),
        "digit_density": np.round(dd, 4).tolist(),
        "dist_own_centroid": np.round(dist_own, 4).tolist(),
        "dist_own_max": round(dist_own_max, 4),
        "dist_from_others": np.round(dist_from_others, 4).tolist(),
        "dist_from_others_max": round(dist_from_others_max, 4),
        "outlier_cutoffs": [round(x, 4) for x in outlier_cutoffs],
        "cluster_of": cluster.tolist(),
        "neighbors": nbrs.tolist(),
        "exhibits": exhibits,
        "palette": _PALETTE,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    print("Rendering main HTML ...")
    html = (_HTML
            .replace("__DATA_JSON__", data_json)
            .replace("__REPO_URL__", REPO_URL))
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_HTML}  ({OUT_HTML.stat().st_size / 1024:.1f} KB)")

    print("Rendering companion examples page ...")
    twin_pairs = getattr(curate_exhibits, "_twin_pairs", [])
    examples_html = _render_examples_page(df, exhibits, twin_pairs, companies)
    OUT_EXAMPLES_HTML.write_text(examples_html, encoding="utf-8")
    print(f"wrote {OUT_EXAMPLES_HTML}  ({OUT_EXAMPLES_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
