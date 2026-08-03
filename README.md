# SUE - Semantic Universe Explorer

An interactive corpus observatory and short technical paper about a
specific comparative question:
**five direct competitors in semiconductor-manufacturing equipment**
- Applied Materials, ASML, KLA, Lam Research, and TEL (Tokyo Electron)
- all publish Global Impact / ESG reports on the same topics every year.
*How comparable are they? Which passages sit apart from what the sector
has converged to? Where does the encoder collapse the five voices into
one?*

**Two deliverables live in this repo:**

1. **The interactive viewer** - a single self-contained HTML page. Every
   dot is one ~900-character chunk of one report, embedded with
   `all-MiniLM-L6-v2`, projected with PCA. Filter, color, click, and
   read.
2. **The paper** - `paper/sue.pdf`, plus a Markdown summary with figures
   in [`PAPER.md`](PAPER.md).

> **The one-line takeaway.** Look at your corpus in embedding space
> *before* you build retrieval on top of it. Corpus-level pathologies
> - prose vs. tabular mixing, cross-firm consensus, per-firm outliers -
> show up in the geometry long before they show up as a "our RAG is
> inconsistent" bug report.

## The viewer

Once GitHub Pages is enabled on this repo (see below), the viewer is
live at:

> **https://emo7bf.github.io/SUE/**

What it does:

- **Color modes** - company; prose <-> tabular projection; raw digit
  density; cross-corpus outlierness (distance from the mean of the
  *other four* firms' embeddings); in-doc typicality; and a 2-component
  unsupervised GMM cluster label.
- **Filters** - content type (all / prose / ambiguous / tabular), a
  digit-density range slider, per-company checkboxes, and an
  "outliers only" toggle backed by a percentile slider.
- **Selection with undo** - click any point to spotlight it and its top
  50 nearest neighbors. Buttons at the top of the sidebar let you undo
  back through your click history or clear the selection entirely. The
  "Show N nearest neighbors" slider appears inside the inspector once a
  point is selected.
- **Curated examples panel** - click to jump to five per-company
  off-consensus passages and the sector's "average sentence." The rest
  (per-company typicals, cross-competitor near-twins, prose/tabular
  extremes, GMM cluster centers) live on the companion page.
- **Companion page** -
  [`docs/explorer_examples.html`](docs/explorer_examples.html) is a
  static walk-through with plain-English explanations of each color
  mode and side-by-side "two firms, one sentence" cards for the three
  strongest cross-competitor near-twins.

The 3D scene rotates slowly on load and stops the moment you interact
with it.

## What runs at page-load, and what doesn't

Everything the encoder ever needs to see is computed **once, at build
time**, and cached to disk. The viewer itself is **static HTML** - no
model, no Python, no server-side compute.

At **build time**:

1. `scripts/build_visuals.py` walks `data/sample_data/`, extracts PDF
   text, chunks it (~900 chars, 120 overlap), embeds with MiniLM-L6-v2,
   and caches to `assets/chunks.parquet` and `assets/embeddings.npy`.
2. `scripts/build_semantic_universe_explorer.py` loads the cache, runs
   PCA (2D + 3D), computes per-chunk metrics (digit density, prose <->
   tabular projection, cross-corpus outlierness, in-doc typicality, GMM
   cluster, top-50 nearest neighbors), curates 18 example points, and
   bundles the whole thing as a single ~5 MB HTML in `docs/`.

At **page-load time**, the browser only:

1. Parses the inlined JSON payload.
2. Draws a Plotly scene from precomputed 2D / 3D coordinates.
3. Runs pure-JS filter and highlight logic against precomputed arrays.

**Nothing on the page needs a Python runtime, PyTorch, or the encoder.**

## Repo layout

```
SUE/
|-- README.md
|-- PAPER.md                    <-- Markdown summary of the paper
|-- requirements.txt
|-- .gitignore
|-- docs/                       <-- GitHub Pages source (published site)
|   |-- index.html
|   |-- semantic_universe_explorer.html
|   +-- explorer_examples.html
|-- paper/
|   |-- sue.pdf                 <-- canonical paper PDF
|   +-- figures/                <-- 15 PNG figures referenced by PAPER.md
|-- scripts/
|   |-- build_visuals.py        <-- parse + chunk + embed pipeline
|   |-- build_semantic_universe_explorer.py   <-- cache -> viewer HTMLs
|   |-- fetch_reports.py        <-- downloads source PDFs from manifest URLs
|   +-- paper/                  <-- paper-figure and PDF-build scripts
|-- assets/                     <-- build cache (committed for reproducibility)
|   |-- chunks.parquet
|   +-- embeddings.npy
+-- data/
    |-- manifest.csv            <-- company / industry / source-URL metadata
    +-- sample_data/            <-- populated by fetch_reports.py; gitignored
```

## Quick start

```bash
pip install -r requirements.txt

# Rebuild the viewer from the cached corpus (fast, seconds):
python scripts/build_semantic_universe_explorer.py

# Rebuild the corpus from source PDFs (slow, needs the PDFs first):
python scripts/fetch_reports.py                  # optional, populates data/sample_data/
python scripts/build_visuals.py                  # produces assets/chunks.parquet + embeddings.npy

# Serve the viewer locally:
python -m http.server 8000 --directory docs
# then open http://127.0.0.1:8000/
```

The build reuses cached embeddings from `assets/embeddings.npy` if they
exist. Delete the cache to force a full re-embed.

Runs fully **offline** with `sentence-transformers`. No API keys
required.

## The paper

The canonical version is [`paper/sue.pdf`](paper/sue.pdf). A Markdown
summary with the main figures inline is in
[`PAPER.md`](PAPER.md). The paper argues that centroid geometry in
embedding space is a compact, auditable diagnostic for the internal
structure of a specialized text corpus - a case study of the
sustainability-report genre for five competing semiconductor-equipment
firms.

## Further reading

An opinionated tour of the papers behind SUE - sentence-embedding
geometry, dimensionality reduction, RAG, retrieval evaluation - is
compiled into the paper's bibliography.

## License

Source code: MIT. Paper text and figures: CC BY 4.0. Source PDFs remain
the property of their respective issuers and are excluded from the
repo; use `scripts/fetch_reports.py` to re-download from the URLs in
`data/manifest.csv`.
