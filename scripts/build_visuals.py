"""
scripts/build_visuals.py
------------------------
End-to-end regeneration of every visual referenced by the SUE README.

Pipeline:
  1. Walk data/sample_data/<industry>/<company>/*.pdf
     (also tolerates the older flat data/sample_data/<company>/*.pdf layout)
  2. Extract text (text layer only, no OCR) with pypdf
  3. Chunk each PDF into ~fixed-size character windows
  4. Embed chunks with sentence-transformers/all-MiniLM-L6-v2
  5. PCA -> 2D and 3D
  6. Emit interactive Plotly HTMLs and static matplotlib PNGs to assets/

Points are colored by INDUSTRY (not by company). Company still appears in the
hover text, but the primary color axis is industry so the plot stays readable
as the corpus grows to dozens of companies.

Usage:
    python scripts/build_visuals.py                 # full run
    python scripts/build_visuals.py --max-chars 80000  # faster smoke run

Outputs (all in ./assets/):
    embedding_space_2d.html      # Plotly 2D interactive, chunks + centroids
    embedding_space_3d.html      # Plotly 3D interactive, chunks + centroids
    embedding_space_2d.png       # static screenshot of chunks colored by industry
    embedding_space_3d.png       # static 3D screenshot
    doc_centroids_2d.png         # per-document centroids only, labelled
    chunks.parquet               # cached chunk text + metadata (optional)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pypdf import PdfReader
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "sample_data"
MANIFEST = ROOT / "data" / "manifest.csv"
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(name: str) -> str:
    s = re.sub(r"[^\w\s\-\.&]+", "", (name or "").strip(), flags=re.UNICODE)
    return re.sub(r"\s+", "_", s) or "unknown"


# A colorblind-friendly qualitative palette that scales to ~15 categories.
# Sourced from Tableau 10 + ColorBrewer overflow.
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    "#393b79", "#e6550d", "#31a354", "#756bb1", "#636363",
    "#ad494a", "#8ca252",
]


def load_manifest() -> Dict[str, str]:
    """Return {company_slug: industry_display_name} from data/manifest.csv."""
    mapping: Dict[str, str] = {}
    if not MANIFEST.exists():
        return mapping
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            company = (row.get("company") or "").strip()
            industry = (row.get("industry") or "").strip()
            if company and industry:
                mapping[_slugify(company)] = industry
    return mapping


def assign_industry_colors(industries: List[str]) -> Dict[str, str]:
    ordered = sorted(set(industries))
    return {ind: _PALETTE[i % len(_PALETTE)] for i, ind in enumerate(ordered)}


# ----------------------------- text extraction -----------------------------

def extract_text(pdf_path: Path, max_chars: int) -> str:
    """Return concatenated text-layer content from a PDF (no OCR)."""
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        print(f"  ! pypdf failed on {pdf_path.name}: {e}")
        return ""
    parts: List[str] = []
    total = 0
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if not t:
            continue
        parts.append(t)
        total += len(t)
        if total >= max_chars:
            break
    text = "\n".join(parts)
    # normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_chars]


def chunk_text(text: str, size: int = 900, overlap: int = 120) -> List[str]:
    """Fixed-size character chunks with small overlap; drops very short tails."""
    if not text:
        return []
    step = size - overlap
    out = []
    for i in range(0, len(text), step):
        c = text[i : i + size].strip()
        if len(c) >= 200:
            out.append(c)
    return out


# ----------------------------- corpus build -----------------------------

def _discover_pdfs() -> List[Tuple[Path, str, str]]:
    """
    Yield (pdf_path, industry_folder, company_folder) tuples.

    Preferred layout:   data/sample_data/<industry>/<company>/*.pdf   (depth 2)
    Legacy tolerated:   data/sample_data/<company>/*.pdf              (depth 1, industry = "Unknown")
    """
    out: List[Tuple[Path, str, str]] = []
    if not DATA_DIR.exists():
        return out
    # Depth-2: industry/company/*.pdf
    for pdf in sorted(DATA_DIR.glob("*/*/*.pdf")):
        industry_folder = pdf.parent.parent.name
        company_folder = pdf.parent.name
        out.append((pdf, industry_folder, company_folder))
    # Depth-1 fallback: company/*.pdf (only if that company folder is NOT already
    # covered as an industry folder above)
    covered_top = {p.parent.parent for p, _, _ in out}
    for pdf in sorted(DATA_DIR.glob("*/*.pdf")):
        if pdf.parent in covered_top:
            continue  # this "company" folder is actually an industry that already yielded PDFs
        out.append((pdf, "Unknown", pdf.parent.name))
    return out


def build_corpus(max_chars_per_doc: int) -> pd.DataFrame:
    manifest = load_manifest()  # company_slug -> industry display name

    def resolve_industry(industry_folder: str, company_folder: str) -> str:
        # Manifest wins if it knows the company; else use the folder name.
        m = manifest.get(_slugify(company_folder))
        if m:
            return m
        if industry_folder and industry_folder != "Unknown":
            # de-slugify enough for display
            return industry_folder.replace("_", " ")
        return "Unknown"

    def prettify_company(company_folder: str) -> str:
        return company_folder.replace("_", " ")

    rows = []
    pdfs = _discover_pdfs()
    print(f"Found {len(pdfs)} PDFs under {DATA_DIR}")
    for pdf, ind_folder, comp_folder in pdfs:
        industry = resolve_industry(ind_folder, comp_folder)
        company = prettify_company(comp_folder)
        print(f"  reading [{industry} / {company}] {pdf.name}")
        text = extract_text(pdf, max_chars=max_chars_per_doc)
        chunks = chunk_text(text)
        print(f"    -> {len(chunks)} chunks ({len(text):,} chars)")
        for j, c in enumerate(chunks):
            rows.append({
                "industry": industry,
                "company": company,
                "doc": pdf.name,
                "chunk_id": j,
                "text": c,
            })
    return pd.DataFrame(rows)


# ----------------------------- embed -----------------------------

def embed(df: pd.DataFrame, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Embedding {len(df):,} chunks ...")
    X = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return X


def doc_centroids(df: pd.DataFrame, X: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    tmp = df.copy()
    tmp["_row"] = np.arange(len(tmp))
    centroids = []
    meta = []
    for (industry, company, doc), g in tmp.groupby(["industry", "company", "doc"], sort=True):
        idx = g["_row"].to_numpy()
        centroids.append(X[idx].mean(axis=0))
        meta.append({"industry": industry, "company": company, "doc": doc, "n_chunks": len(idx)})
    return np.array(centroids), pd.DataFrame(meta)


# ----------------------------- plots -----------------------------

def plot_static_2d(Z: np.ndarray, df: pd.DataFrame, colors: Dict[str, str], path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=140)
    for industry, color in colors.items():
        m = df["industry"].values == industry
        if not m.any():
            continue
        ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.55, color=color, label=industry, edgecolors="none")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True, fontsize=8, ncols=2 if len(colors) > 8 else 1)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


def plot_static_3d(Z: np.ndarray, df: pd.DataFrame, colors: Dict[str, str], path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 8), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    for industry, color in colors.items():
        m = df["industry"].values == industry
        if not m.any():
            continue
        ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], s=8, alpha=0.55, color=color, label=industry, edgecolors="none")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True, fontsize=8, ncols=2 if len(colors) > 8 else 1)
    ax.view_init(elev=22, azim=35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


def plot_centroids_2d(Zc: np.ndarray, cmeta: pd.DataFrame, colors: Dict[str, str], path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8), dpi=140)
    for industry, color in colors.items():
        m = cmeta["industry"].values == industry
        if not m.any():
            continue
        ax.scatter(Zc[m, 0], Zc[m, 1], s=180, marker="o", color=color,
                   edgecolors="black", linewidths=0.8, label=industry, alpha=0.85)
    # Only label centroids if the corpus is small enough for readability
    if len(cmeta) <= 20:
        for i, row in cmeta.iterrows():
            label = row["company"]
            label = (label[:22] + "…") if len(label) > 24 else label
            ax.annotate(label, (Zc[i, 0], Zc[i, 1]), xytext=(6, 4),
                        textcoords="offset points", fontsize=7, alpha=0.85)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True, fontsize=8, ncols=2 if len(colors) > 8 else 1)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


def _hover_chunk(r) -> str:
    snippet = r.text[:180].replace(chr(10), " ")
    return (
        f"<b>{r.industry}</b> — {r.company}<br>"
        f"{r.doc} · chunk {r.chunk_id}<br>{snippet}…"
    )


def _hover_centroid(r) -> str:
    return (
        f"<b>{r.industry}</b> — {r.company}<br>"
        f"{r.doc}<br>{r.n_chunks} chunks"
    )


def plot_interactive_2d(Z, df, Zc, cmeta, colors: Dict[str, str], path: Path, title: str):
    import plotly.graph_objects as go
    traces = []
    for industry, color in colors.items():
        m = df["industry"].values == industry
        if not m.any():
            continue
        sub = df[m]
        hover = [_hover_chunk(r) for r in sub.itertuples()]
        traces.append(go.Scattergl(
            x=Z[m, 0], y=Z[m, 1], mode="markers", name=f"{industry} — chunks",
            marker=dict(size=5, color=color, opacity=0.55),
            hovertemplate="%{text}<extra></extra>", text=hover,
        ))
    for industry, color in colors.items():
        m = cmeta["industry"].values == industry
        if not m.any():
            continue
        sub = cmeta[m]
        traces.append(go.Scattergl(
            x=Zc[m, 0], y=Zc[m, 1], mode="markers", name=f"{industry} — centroids",
            marker=dict(size=14, color=color, symbol="x", line=dict(width=1.5, color="black")),
            text=[_hover_centroid(r) for r in sub.itertuples()],
            hovertemplate="%{text}<extra></extra>",
        ))
    fig = go.Figure(traces)
    fig.update_layout(title=title, xaxis_title="PC 1", yaxis_title="PC 2",
                      height=780, legend=dict(itemsizing="constant"))
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    print("  wrote", path)


def plot_interactive_3d(Z, df, Zc, cmeta, colors: Dict[str, str], path: Path, title: str):
    import plotly.graph_objects as go
    traces = []
    for industry, color in colors.items():
        m = df["industry"].values == industry
        if not m.any():
            continue
        sub = df[m]
        hover = [_hover_chunk(r) for r in sub.itertuples()]
        traces.append(go.Scatter3d(
            x=Z[m, 0], y=Z[m, 1], z=Z[m, 2], mode="markers",
            name=f"{industry} — chunks",
            marker=dict(size=2.5, color=color, opacity=0.55),
            hovertemplate="%{text}<extra></extra>", text=hover,
        ))
    for industry, color in colors.items():
        m = cmeta["industry"].values == industry
        if not m.any():
            continue
        sub = cmeta[m]
        traces.append(go.Scatter3d(
            x=Zc[m, 0], y=Zc[m, 1], z=Zc[m, 2], mode="markers",
            name=f"{industry} — centroids",
            marker=dict(size=6, color=color, symbol="x", line=dict(width=1.5, color="black")),
            text=[_hover_centroid(r) for r in sub.itertuples()],
            hovertemplate="%{text}<extra></extra>",
        ))
    fig = go.Figure(traces)
    fig.update_layout(title=title, height=820,
                      scene=dict(xaxis_title="PC 1", yaxis_title="PC 2", zaxis_title="PC 3"),
                      legend=dict(itemsizing="constant"))
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    print("  wrote", path)


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=250_000,
                    help="Cap of characters read per PDF (keeps runs bounded).")
    ap.add_argument("--cache", action="store_true",
                    help="Cache chunks.parquet + embeddings.npy for later reuse.")
    args = ap.parse_args()

    print("== 1) Build corpus ==")
    df = build_corpus(max_chars_per_doc=args.max_chars)
    if df.empty:
        raise SystemExit(
            "No chunks produced. Drop PDFs into "
            "data/sample_data/<industry>/<company>/ and try again."
        )
    n_industries = df["industry"].nunique()
    n_companies = df["company"].nunique()
    print(f"Total chunks: {len(df):,} across {n_companies} companies in {n_industries} industries")

    colors = assign_industry_colors(df["industry"].tolist())

    print("== 2) Embed ==")
    X = embed(df)

    print("== 3) PCA (2D + 3D) ==")
    p2 = PCA(n_components=2, random_state=0).fit(X)
    p3 = PCA(n_components=3, random_state=0).fit(X)
    Z2 = p2.transform(X)
    Z3 = p3.transform(X)
    print(f"  2D explained variance: {p2.explained_variance_ratio_.sum():.1%}")
    print(f"  3D explained variance: {p3.explained_variance_ratio_.sum():.1%}")

    print("== 4) Doc centroids ==")
    C, cmeta = doc_centroids(df, X)
    Zc2 = p2.transform(C)
    Zc3 = p3.transform(C)

    print("== 5) Plots ==")
    plot_static_2d(Z2, df, colors, ASSETS_DIR / "embedding_space_2d.png",
                   "SUE — chunk embedding space, colored by industry (PCA 2D)")
    plot_static_3d(Z3, df, colors, ASSETS_DIR / "embedding_space_3d.png",
                   "SUE — chunk embedding space, colored by industry (PCA 3D)")
    plot_centroids_2d(Zc2, cmeta, colors, ASSETS_DIR / "doc_centroids_2d.png",
                      "SUE — one point per document (centroid of its chunks), colored by industry")
    plot_interactive_2d(Z2, df, Zc2, cmeta, colors,
                        ASSETS_DIR / "embedding_space_2d.html",
                        "SUE — interactive 2D embedding space (hover for company + snippet)")
    plot_interactive_3d(Z3, df, Zc3, cmeta, colors,
                        ASSETS_DIR / "embedding_space_3d.html",
                        "SUE — interactive 3D embedding space (rotate + hover)")

    if args.cache:
        df.to_parquet(ASSETS_DIR / "chunks.parquet")
        np.save(ASSETS_DIR / "embeddings.npy", X)
        print("  cached chunks + embeddings under assets/")

    print("Done.")


if __name__ == "__main__":
    main()
