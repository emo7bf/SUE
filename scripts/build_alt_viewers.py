"""
scripts/build_alt_viewers.py
----------------------------
Emit the interactive t-SNE / UMAP viewers: one self-contained HTML page
per corpus, with a projection picker (t-SNE at four perplexities, UMAP
at four neighbourhood sizes, each in 2D and 3D) and company coloring.

Why these pages exist: the flagship viewer projects with PCA, which
preserves directions of greatest variance but not neighborhoods - true
nearest neighbors may land visually far apart. t-SNE and UMAP make the
opposite trade: they seat true neighbors together at the cost of
distorting long-range distances. Showing all three lets the reader see
which structure survives every lens (that invariance is the finding).

Projections are cached per corpus, keyed by an embedding hash, so
rebuilds are fast (see assets/**/projections_cache.npz).

Usage:
    python scripts/build_alt_viewers.py             # all corpora
    python scripts/build_alt_viewers.py --corpus aerospace_defense

Outputs:
    docs/tsne_umap_viewer.html                      (semiconductor corpus)
    docs/industries/<slug>_tsne_umap.html           (each ingested industry)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

TSNE_PERPLEXITIES = [5, 30, 100, 300]
UMAP_N_NEIGHBORS = [5, 15, 50, 200]
UMAP_MIN_DIST = 0.1

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    "#393b79", "#e6550d", "#31a354", "#756bb1", "#636363",
    "#ad494a", "#8ca252",
]


def _hash(X: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(X.shape).encode())
    step = max(1, X.shape[0] // 256)
    h.update(X[::step].tobytes())
    return h.hexdigest()[:16]


def compute_projections(X: np.ndarray, cache_path: Path) -> dict:
    """All 16 layouts, cached. Mirrors the flagship script's conventions."""
    keys = ([f"tsne_p{p}_d{d}" for p in TSNE_PERPLEXITIES for d in (3, 2)]
            + [f"umap_n{n}_d{d}" for n in UMAP_N_NEIGHBORS for d in (3, 2)])
    want = _hash(X)
    if cache_path.exists():
        try:
            c = np.load(cache_path, allow_pickle=False)
            if str(c.get("_hash", "")) == want and all(k in c for k in keys):
                print(f"  cache hit: {cache_path.name}")
                return {k: c[k].astype(np.float32) for k in keys}
        except Exception as e:
            print(f"  cache read failed ({e!r})")
    out = {}
    from sklearn.manifold import TSNE
    import inspect
    iter_kw = ("max_iter" if "max_iter"
               in inspect.signature(TSNE.__init__).parameters else "n_iter")
    for p in TSNE_PERPLEXITIES:
        for d in (3, 2):
            print(f"  t-SNE perplexity={p} d={d} ...")
            kw = dict(n_components=d, perplexity=float(p), init="pca",
                      random_state=0, learning_rate="auto")
            kw[iter_kw] = 750 if p >= 100 else 1000
            out[f"tsne_p{p}_d{d}"] = TSNE(**kw).fit_transform(X).astype(np.float32)
    import umap
    for n in UMAP_N_NEIGHBORS:
        for d in (3, 2):
            print(f"  UMAP n_neighbors={n} d={d} ...")
            out[f"umap_n{n}_d{d}"] = umap.UMAP(
                n_neighbors=int(n), min_dist=UMAP_MIN_DIST, n_components=d,
                random_state=0, verbose=False).fit_transform(X).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, _hash=np.array(want), **out)
    print(f"  cached -> {cache_path}")
    return out


def _norm(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.mean(axis=0)
    s = float(np.percentile(np.abs(Z), 99)) or 1.0
    return (Z / s).astype(np.float32)


def hover_lines(df: pd.DataFrame) -> list:
    out = []
    for r in df.itertuples():
        doc = getattr(r, "doc", "")
        bits = [f"<b>{r.company}</b>", str(doc)]
        year = str(getattr(r, "report_year", "") or "")
        if year:
            bits[-1] += f" \u00b7 {year}"
        page = getattr(r, "page_start", getattr(r, "page", None))
        if page is not None and not pd.isna(page):
            bits[-1] += f" \u00b7 p.{int(page)}"
        snip = re.sub(r"\s+", " ", str(r.text))[:170]
        bits.append(snip + "\u2026")
        out.append("<br>".join(bits))
    return out


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>t-SNE / UMAP viewer \u2014 SUE \u2014 __CORPUS__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root { --bg:#faf7f2; --panel:#fffdf9; --border:#e4ddd2; --text:#2b2b33;
          --muted:#6b6b76; --accent:#a45a1e; --accent2:#4a6a8a; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font-family:Georgia,'Times New Roman',serif; }
  nav { display:flex; flex-wrap:wrap; gap:14px; align-items:center;
        padding:10px 18px; font-size:13px; border-bottom:1px solid var(--border);
        background:var(--panel); }
  nav a { color:var(--accent2); text-decoration:none; }
  nav a:hover { color:var(--accent); }
  header { padding:14px 18px 0; }
  h1 { font-size:20px; margin:0; color:var(--accent); }
  .sub { color:var(--muted); font-size:13px; margin:4px 0 0; }
  .controls { display:flex; flex-wrap:wrap; gap:18px; align-items:center;
              padding:10px 18px; font-size:13px; }
  .controls label { margin-right:4px; }
  select, input[type=radio] { font-size:13px; }
  #plot { height:78vh; margin:0 10px 14px; background:var(--panel);
          border:1px solid var(--border); border-radius:10px; }
  .note { padding:0 18px 18px; font-size:12.5px; color:var(--muted);
          max-width:900px; line-height:1.5; }
</style>
</head>
<body>
<nav>__NAV__</nav>
<header>
  <h1>t-SNE / UMAP viewer \u2014 __CORPUS__</h1>
  <p class="sub">__N_CHUNKS__ passages \u00b7 __N_COMPANIES__ companies \u00b7
  same MiniLM embeddings as the PCA viewer, different lens.</p>
</header>
<div class="controls">
  <span><label><input type="radio" name="method" value="tsne" checked> t-SNE</label>
        <label><input type="radio" name="method" value="umap"> UMAP</label></span>
  <span id="param-tsne">perplexity
    <select id="tsne-p"><option>5</option><option selected>30</option>
    <option>100</option><option>300</option></select></span>
  <span id="param-umap" style="display:none">n_neighbors
    <select id="umap-n"><option>5</option><option selected>15</option>
    <option>50</option><option>200</option></select></span>
  <span><label><input type="radio" name="dim" value="3" checked> 3D</label>
        <label><input type="radio" name="dim" value="2"> 2D</label></span>
</div>
<div id="plot"></div>
<p class="note"><b>Reading guide.</b> PCA (the flagship viewer) preserves the
directions of greatest overall variance; t-SNE and UMAP instead work to seat
each passage next to its true nearest neighbors, at the price of distorting
long-range distances. The knobs matter: low perplexity / n_neighbors favors
fine local structure, high values favor the global picture. Structure that
survives <i>every</i> setting \u2014 like this corpus's prose\u2013tabular split \u2014
is a property of the data, not of the lens.</p>
<script>
const COORDS = __COORDS__;
const COMPANY_OF = __COMPANY_OF__;
const COMPANIES = __COMPANIES__;
const HOVER = __HOVER__;
const COLORS = __COLORS__;

function currentKey() {
  const method = document.querySelector('input[name=method]:checked').value;
  const dim = document.querySelector('input[name=dim]:checked').value;
  const p = document.getElementById('tsne-p').value;
  const n = document.getElementById('umap-n').value;
  return method === 'tsne' ? `tsne_p${p}_d${dim}` : `umap_n${n}_d${dim}`;
}
function buildTraces(key) {
  const Z = COORDS[key];
  const is3d = key.endsWith('d3');
  const traces = [];
  for (let c = 0; c < COMPANIES.length; c++) {
    const idx = [];
    for (let i = 0; i < COMPANY_OF.length; i++) if (COMPANY_OF[i] === c) idx.push(i);
    traces.push({
      type: is3d ? 'scatter3d' : 'scattergl',
      x: idx.map(i => Z[i][0]), y: idx.map(i => Z[i][1]),
      ...(is3d ? {z: idx.map(i => Z[i][2])} : {}),
      mode: 'markers',
      marker: {size: is3d ? 2.6 : 5.5, color: COLORS[c], opacity: 0.65,
               line: {width: 0}},
      name: COMPANIES[c],
      text: idx.map(i => HOVER[i]),
      hovertemplate: '%{text}<extra></extra>',
    });
  }
  return traces;
}
function layoutFor(key) {
  const is3d = key.endsWith('d3');
  const base = {autosize: true, margin: {l: 0, r: 0, t: 10, b: 0},
                hovermode: 'closest', uirevision: key,
                paper_bgcolor: '#fffdf9', plot_bgcolor: '#fffdf9',
                legend: {itemsizing: 'constant', font: {size: 11}}};
  if (is3d) return {...base, scene: {xaxis: {visible: false},
    yaxis: {visible: false}, zaxis: {visible: false}, aspectmode: 'cube'}};
  return {...base, xaxis: {visible: false}, yaxis: {visible: false,
    scaleanchor: 'x', scaleratio: 1}};
}
function render() {
  const key = currentKey();
  Plotly.react('plot', buildTraces(key), layoutFor(key),
               {displaylogo: false, responsive: true});
}
document.querySelectorAll('input[name=method]').forEach(el =>
  el.addEventListener('change', () => {
    const tsne = document.querySelector('input[name=method]:checked').value === 'tsne';
    document.getElementById('param-tsne').style.display = tsne ? '' : 'none';
    document.getElementById('param-umap').style.display = tsne ? 'none' : '';
    render();
  }));
document.querySelectorAll('input[name=dim]').forEach(el =>
  el.addEventListener('change', render));
document.getElementById('tsne-p').addEventListener('change', render);
document.getElementById('umap-n').addEventListener('change', render);
render();
</script>
</body>
</html>
"""


def build_page(df: pd.DataFrame, proj: dict, corpus_name: str,
               out_path: Path, nav_prefix: str) -> None:
    companies = sorted(df["company"].unique())
    lookup = {c: i for i, c in enumerate(companies)}
    company_of = df["company"].map(lookup).astype(int).tolist()
    coords = {k: np.round(_norm(v), 4).tolist() for k, v in proj.items()}
    nav = (f'<a href="{nav_prefix}semantic_universe_explorer.html">Interactive viewer</a> \u00b7 '
           f'<a href="{nav_prefix}sue_walkthru.html">Walk-through</a> \u00b7 '
           f'<a href="{nav_prefix}math_and_statistics.html">Math &amp; statistics</a>')
    html = (PAGE
            .replace("__CORPUS__", corpus_name)
            .replace("__NAV__", nav)
            .replace("__N_CHUNKS__", f"{len(df):,}")
            .replace("__N_COMPANIES__", str(len(companies)))
            .replace("__COORDS__", json.dumps(coords, separators=(",", ":")))
            .replace("__COMPANY_OF__", json.dumps(company_of, separators=(",", ":")))
            .replace("__COMPANIES__", json.dumps(companies))
            .replace("__HOVER__", json.dumps(hover_lines(df)))
            .replace("__COLORS__", json.dumps(
                [PALETTE[i % len(PALETTE)] for i in range(len(companies))])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None,
                    help="'semiconductors', an industry slug, or omit for all")
    args = ap.parse_args()

    jobs = []
    if args.corpus in (None, "semiconductors"):
        jobs.append(("Semiconductor Equipment",
                     ROOT / "assets" / "chunks.parquet",
                     ROOT / "assets" / "embeddings.npy",
                     ROOT / "assets" / "projections_cache.npz",
                     DOCS / "tsne_umap_viewer.html", ""))
    ind_root = ROOT / "assets" / "industries"
    if ind_root.exists():
        for d in sorted(ind_root.iterdir()):
            if not (d / "chunks.parquet").exists():
                continue
            if args.corpus not in (None, d.name):
                continue
            src = ROOT / "data" / "industries" / f"{d.name}_sources.json"
            name = (json.loads(src.read_text(encoding="utf-8"))["industry"]
                    if src.exists() else d.name.replace("_", " ").title())
            jobs.append((name, d / "chunks.parquet", d / "embeddings.npy",
                         d / "projections_cache.npz",
                         DOCS / "industries" / f"{d.name}_tsne_umap.html", "../"))

    for name, chunks_p, emb_p, cache_p, out_p, nav_prefix in jobs:
        print(f"== {name} ==")
        df = pd.read_parquet(chunks_p).reset_index(drop=True)
        X = np.load(emb_p).astype(np.float32)
        proj = compute_projections(X, cache_p)
        build_page(df, proj, name, out_p, nav_prefix)


if __name__ == "__main__":
    main()
