"""
scripts/plot_vertical.py
------------------------
Emit interactive first-look viewers for one vertical's cached corpus:

  docs/verticals/<vertical>_3d.html   PCA-3D, chunks + document centroids
  docs/verticals/<vertical>_2d.html   PCA-2D companion

Both pages color by company, and hovers carry the full chunk provenance:
document, category, report year, page range, register, and a snippet.
This is the quick-look plot for a newly ingested vertical; the full SUE
explorer treatment (color modes, filters, exhibits) comes once the
vertical is folded into build_semantic_universe_explorer.py.

Usage:
    python scripts/plot_vertical.py                # aerospace_defense
    python scripts/plot_vertical.py --vertical <name>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets" / "verticals"
DOCS_DIR = ROOT / "docs" / "verticals"

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    "#393b79", "#e6550d", "#31a354", "#756bb1", "#636363",
    "#ad494a", "#8ca252",
]


def _hover(r) -> str:
    snippet = r.text[:180].replace(chr(10), " ")
    year = f" · {r.report_year}" if r.report_year else ""
    pages = (f"p.{r.page_start}" if r.page_start == r.page_end
             else f"pp.{r.page_start}–{r.page_end}")
    return (f"<b>{r.company}</b> ({r.ticker}, {r.tier}){year}<br>"
            f"{r.doc} · {r.doc_category} · {pages} · {r.register}<br>"
            f"{snippet}…")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vertical", default="aerospace_defense")
    args = ap.parse_args()

    cache = ASSETS_DIR / args.vertical
    df = pd.read_parquet(cache / "chunks.parquet")
    X = np.load(cache / "embeddings.npy")
    vertical_name = df["vertical"].iloc[0]
    companies = sorted(df["company"].unique())
    colors = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(companies)}
    print(f"{len(df):,} chunks | {len(companies)} companies | {vertical_name}")

    Z3 = PCA(n_components=3, random_state=0).fit_transform(X)

    # one centroid per document
    df = df.reset_index(drop=True)
    cent_rows, cent_pts = [], []
    for (company, doc), g in df.groupby(["company", "doc"], sort=True):
        cent_pts.append(Z3[g.index].mean(axis=0))
        cent_rows.append({"company": company, "doc": doc, "n": len(g)})
    Zc = np.array(cent_pts)
    cmeta = pd.DataFrame(cent_rows)

    import plotly.graph_objects as go

    def traces(dim: int):
        out = []
        for company in companies:
            m = (df["company"] == company).to_numpy()
            hover = [_hover(r) for r in df[m].itertuples()]
            kw = dict(x=Z3[m, 0], y=Z3[m, 1], mode="markers", name=company,
                      text=hover, hovertemplate="%{text}<extra></extra>")
            if dim == 3:
                out.append(go.Scatter3d(
                    z=Z3[m, 2], marker=dict(size=2.4, color=colors[company],
                                            opacity=0.6), **kw))
            else:
                out.append(go.Scattergl(
                    marker=dict(size=5, color=colors[company], opacity=0.6), **kw))
        for company in companies:
            m = (cmeta["company"] == company).to_numpy()
            if not m.any():
                continue
            hov = [f"<b>{r.company}</b><br>{r.doc}<br>{r.n} chunks"
                   for r in cmeta[m].itertuples()]
            kw = dict(x=Zc[m, 0], y=Zc[m, 1], mode="markers",
                      name=f"{company} — docs", showlegend=False, text=hov,
                      hovertemplate="%{text}<extra></extra>")
            if dim == 3:
                out.append(go.Scatter3d(
                    z=Zc[m, 2], marker=dict(size=5.5, color=colors[company],
                                            symbol="x"), **kw))
            else:
                out.append(go.Scattergl(
                    marker=dict(size=13, color=colors[company], symbol="x",
                                line=dict(width=1.4, color="black")), **kw))
        return out

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    title = (f"SUE — {vertical_name}: {len(df):,} chunks from "
             f"{df['doc'].nunique()} reports by {len(companies)} companies "
             f"(PCA of MiniLM embeddings)")

    fig3 = go.Figure(traces(3))
    fig3.update_layout(title=title, height=860,
                       scene=dict(xaxis_title="PC 1", yaxis_title="PC 2",
                                  zaxis_title="PC 3"),
                       legend=dict(itemsizing="constant"))
    out3 = DOCS_DIR / f"{args.vertical}_3d.html"
    fig3.write_html(out3, include_plotlyjs="cdn", full_html=True)
    print("wrote", out3)

    fig2 = go.Figure(traces(2))
    fig2.update_layout(title=title, xaxis_title="PC 1", yaxis_title="PC 2",
                       height=820, legend=dict(itemsizing="constant"))
    out2 = DOCS_DIR / f"{args.vertical}_2d.html"
    fig2.write_html(out2, include_plotlyjs="cdn", full_html=True)
    print("wrote", out2)


if __name__ == "__main__":
    main()
