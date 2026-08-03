"""
scripts/paper/make_figures.py
-----------------------------
Regenerate every figure used by the SUE publication PDF from cached
embeddings and chunk metadata. Writes PNG (for PDF embedding) plus a
PDF-vector copy of each figure to paper/figures/.

Prereqs: `python scripts/build_visuals.py --cache` has been run at least
once, producing assets/chunks.parquet and assets/embeddings.npy.

Usage:
    python -m scripts.paper.make_figures
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

from scripts.paper.style import apply_style, PALETTE, CATEGORICAL


ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKS_PATH = ROOT / "assets" / "chunks.parquet"
EMB_PATH = ROOT / "assets" / "embeddings.npy"
OUT_DIR = ROOT / "paper" / "figures"
STATS_OUT = ROOT / "paper" / "stats.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Data loading and shared derived arrays
# ============================================================

def load_data():
    df = pd.read_parquet(CHUNKS_PATH)
    X = np.load(EMB_PATH)
    assert len(df) == len(X), f"chunks={len(df)} != embeddings={len(X)}"
    return df, X


def prose_table_label(text: str) -> str:
    """Regex-derived chunk-flavor label. Returns 'table' or 'prose'."""
    n_chars = max(len(text), 1)
    n_digits = sum(c.isdigit() for c in text)
    digit_density = n_digits / n_chars
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return "prose"
    short_numeric_lines = sum(
        1 for ln in lines
        if len(ln.strip()) < 60 and sum(c.isdigit() for c in ln) >= 2
    )
    short_line_ratio = short_numeric_lines / max(len(lines), 1)
    # Table-ish rules: numeric-dense or many short numeric lines
    if digit_density > 0.18 or short_line_ratio > 0.35:
        return "table"
    return "prose"


def save(fig, name: str) -> Path:
    """Save a figure as both high-DPI PNG (for PDF embedding) and vector PDF."""
    out_png = OUT_DIR / f"{name}.png"
    out_pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  wrote {out_png.name}")
    return out_png


# ============================================================
# Figure 1 -- Corpus overview
# ============================================================

def fig_corpus_overview(df: pd.DataFrame) -> None:
    counts = (df.groupby(["company"])
                .size().sort_values(ascending=True))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colors = [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(counts))]
    ax.barh(counts.index, counts.values, color=colors, edgecolor="white",
            linewidth=0.6)
    for i, (company, n) in enumerate(counts.items()):
        ax.text(n + 8, i, f"{n}", va="center", fontsize=8, color=PALETTE["ink"])
    ax.set_xlabel("Number of chunks (~900 chars each)")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_color(PALETTE["neutral_light"])
    save(fig, "fig01_corpus_overview")


# ============================================================
# Figure 2 -- PCA-2D scatter colored by company
# ============================================================

def fig_pca2d_by_company(df: pd.DataFrame, Z2: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for i, company in enumerate(sorted(df["company"].unique())):
        m = (df["company"].values == company)
        ax.scatter(Z2[m, 0], Z2[m, 1], s=6, alpha=0.55,
                   color=CATEGORICAL[i % len(CATEGORICAL)],
                   label=company, edgecolors="none")
    ax.set_xlabel(r"PC 1  $\;\;\longleftrightarrow\;\;$  (no interpretation yet)",
                  fontsize=9)
    ax.set_ylabel(r"PC 2  $\;\;\longleftrightarrow\;\;$  (no interpretation yet)",
                  fontsize=9)
    leg = ax.legend(loc="upper right", markerscale=2.0, title="Company")
    leg.get_frame().set_linewidth(0.5)
    save(fig, "fig02_pca_2d_by_company")


# ============================================================
# Figure 3 -- Bimodality: 2-cluster GMM overlay on PCA-2D
# ============================================================

def _draw_ellipse(ax, mean2d, cov2d, color, alpha=0.15, n_std=2.0):
    vals, vecs = np.linalg.eigh(cov2d)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.abs(vals))
    e = Ellipse(xy=mean2d, width=width, height=height, angle=angle,
                facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.2)
    ax.add_patch(e)


def fig_bimodality(df: pd.DataFrame, X: np.ndarray, Z2: np.ndarray) -> dict:
    # Fit GMMs in the top-K-PC subspace: in raw 384-D, full-covariance
    # models add ~74k parameters per component and BIC becomes dominated
    # by that penalty rather than by fit quality. In a 20-PC subspace
    # capturing >50% of the variance, parameter counts are tractable
    # (20 + 20*21/2 = 230 per component) and BIC is well-behaved.
    K = 20
    pca_k = PCA(n_components=K, random_state=0).fit(X)
    Xk = pca_k.transform(X)

    gm2 = GaussianMixture(n_components=2, covariance_type="full",
                          random_state=0, n_init=5).fit(Xk)
    gm1 = GaussianMixture(n_components=1, covariance_type="full",
                          random_state=0).fit(Xk)
    labels = gm2.predict(Xk)
    dbic = gm2.bic(Xk) - gm1.bic(Xk)   # negative => 2-cluster preferred
    pca_k_var = float(pca_k.explained_variance_ratio_.sum())

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    colors = [PALETTE["positive"], PALETTE["negative"]]
    for k in [0, 1]:
        m = (labels == k)
        ax.scatter(Z2[m, 0], Z2[m, 1], s=6, alpha=0.55,
                   color=colors[k], label=f"Cluster {chr(65+k)}",
                   edgecolors="none")

    # For visualization only, we also want to draw the 2D shadows of the
    # clusters. We recompute a 2D-only GMM restricted to the projected
    # data so the ellipses match the picture.
    gm2_vis = GaussianMixture(n_components=2, covariance_type="full",
                              random_state=0, n_init=3).fit(Z2)
    # Match GMM_vis cluster indexing to gm2 by comparing centroid means
    means_hi = np.array([Z2[labels == k].mean(axis=0) for k in [0, 1]])
    for k_vis in [0, 1]:
        distances = np.linalg.norm(gm2_vis.means_[k_vis] - means_hi, axis=1)
        k_hi = int(np.argmin(distances))
        _draw_ellipse(ax, gm2_vis.means_[k_vis], gm2_vis.covariances_[k_vis],
                      colors[k_hi], alpha=0.15)
        ax.plot(*gm2_vis.means_[k_vis], marker="X", color=colors[k_hi],
                markersize=10, markeredgecolor="black", markeredgewidth=0.8)

    ax.set_xlabel("PC 1", fontsize=9)
    ax.set_ylabel("PC 2", fontsize=9)
    txt = (r"$\Delta\mathrm{BIC} = \mathrm{BIC}_2 - \mathrm{BIC}_1 = "
           f"{dbic:,.0f}$"
           f"\n(fit in top-{K} PC subspace, capturing {pca_k_var:.0%} of variance)")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, bbox=dict(facecolor="white", edgecolor="none",
                                    alpha=0.85, pad=3))
    leg = ax.legend(loc="upper right", markerscale=2.0)
    leg.get_frame().set_linewidth(0.5)
    save(fig, "fig03_bimodality_gmm")
    return {"delta_bic": float(dbic),
            "gmm_subspace_dim": K,
            "gmm_subspace_var": pca_k_var,
            "cluster_labels": labels.tolist()}


# ============================================================
# Figure 4 -- Cosine similarity distributions by pair type
# ============================================================

def fig_cosine_distributions(df: pd.DataFrame, X: np.ndarray) -> dict:
    rng = np.random.default_rng(0)
    N = len(X)

    def sample_pairs(mask_fn, k):
        pairs = []
        attempts = 0
        while len(pairs) < k and attempts < 50 * k:
            i, j = rng.integers(0, N, size=2)
            if i == j:
                attempts += 1
                continue
            if mask_fn(i, j):
                pairs.append((i, j))
            attempts += 1
        return np.array(pairs) if pairs else np.zeros((0, 2), dtype=int)

    doc_arr = df["doc"].values
    comp_arr = df["company"].values

    same_doc = sample_pairs(lambda i, j: doc_arr[i] == doc_arr[j], 20000)
    same_comp_diff_doc = sample_pairs(
        lambda i, j: comp_arr[i] == comp_arr[j] and doc_arr[i] != doc_arr[j],
        20000)
    diff_comp = sample_pairs(lambda i, j: comp_arr[i] != comp_arr[j], 20000)
    random_pairs = sample_pairs(lambda i, j: True, 20000)

    def cos(pairs):
        a = X[pairs[:, 0]]; b = X[pairs[:, 1]]
        # embeddings already normalized by build_visuals
        return np.sum(a * b, axis=1)

    groups = [
        ("random pairs",                  cos(random_pairs),      PALETTE["neutral"]),
        ("different companies",           cos(diff_comp),         PALETTE["negative"]),
        ("same company, different doc",   cos(same_comp_diff_doc), PALETTE["accent"]),
        ("same document",                 cos(same_doc),          PALETTE["positive"]),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bins = np.linspace(-0.05, 1.0, 80)
    for label, values, color in groups:
        ax.hist(values, bins=bins, alpha=0.55, density=True, color=color,
                label=f"{label}  ($\\bar x = {values.mean():.2f}$)",
                edgecolor="white", linewidth=0.3)
    ax.axvline(0.0, color=PALETTE["ink"], linewidth=0.6, linestyle=":")
    ax.set_xlabel("cosine similarity  (unrelated $\\leftarrow$   $\\rightarrow$ identical)")
    ax.set_ylabel("density")
    leg = ax.legend(loc="upper left", fontsize=8)
    leg.get_frame().set_linewidth(0.5)
    save(fig, "fig04_cosine_distributions")
    return {
        "cos_mean_random":            float(cos(random_pairs).mean()),
        "cos_mean_diff_company":      float(cos(diff_comp).mean()),
        "cos_mean_same_company":      float(cos(same_comp_diff_doc).mean()),
        "cos_mean_same_doc":          float(cos(same_doc).mean()),
    }


# ============================================================
# Figure 5 -- Anisotropy: random cosine vs isotropic reference
# ============================================================

def fig_anisotropy(X: np.ndarray) -> dict:
    rng = np.random.default_rng(1)
    N, D = X.shape
    K = 50000
    ii = rng.integers(0, N, size=K)
    jj = rng.integers(0, N, size=K)
    valid = ii != jj
    ii, jj = ii[valid], jj[valid]
    cos_real = np.sum(X[ii] * X[jj], axis=1)

    # Isotropic reference: i.i.d. N(0, I_D), normalized.
    Y = rng.standard_normal((2 * K, D))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
    cos_iso = np.sum(Y[:K] * Y[K:2*K], axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bins = np.linspace(-0.4, 1.0, 100)
    ax.hist(cos_iso, bins=bins, alpha=0.55, density=True,
            color=PALETTE["neutral_light"],
            label=fr"isotropic reference  ($\bar x = {cos_iso.mean():.2f}$)",
            edgecolor="white", linewidth=0.3)
    ax.hist(cos_real, bins=bins, alpha=0.65, density=True,
            color=PALETTE["primary"],
            label=fr"SUE corpus  ($\bar x = {cos_real.mean():.2f}$)",
            edgecolor="white", linewidth=0.3)
    ax.axvline(0.0, color=PALETTE["ink"], linewidth=0.6, linestyle=":")
    ax.annotate(
        "", xy=(cos_real.mean(), 0.6 * ax.get_ylim()[1]),
        xytext=(0, 0.6 * ax.get_ylim()[1]),
        arrowprops=dict(arrowstyle="->", color=PALETTE["accent"], linewidth=1.2))
    ax.text(cos_real.mean() / 2, 0.66 * ax.get_ylim()[1],
            "anisotropy gap", ha="center", fontsize=9,
            color=PALETTE["accent"])
    ax.set_xlabel("cosine similarity of random pairs")
    ax.set_ylabel("density")
    leg = ax.legend(loc="upper right", fontsize=8)
    leg.get_frame().set_linewidth(0.5)
    save(fig, "fig05_anisotropy")
    return {"anisotropy_mean": float(cos_real.mean()),
            "isotropic_mean": float(cos_iso.mean())}


# ============================================================
# Figure 6 -- PCA explained-variance ratio
# ============================================================

def fig_explained_variance(X: np.ndarray) -> dict:
    p = PCA(n_components=30, random_state=0).fit(X)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    idx = np.arange(1, 31)
    pc1 = p.explained_variance_ratio_[0]
    top10 = p.explained_variance_ratio_[:10].sum()
    ax.bar(idx, p.explained_variance_ratio_,
           color=PALETTE["primary"],
           edgecolor="white", linewidth=0.5)
    ax.set_xlabel("principal component index")
    ax.set_ylabel("fraction of variance explained")
    ax.text(0.98, 0.95,
            f"PC1 alone: {pc1:.1%}\n"
            f"top 10 combined: {top10:.1%}\n"
            "no single component dominates",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=PALETTE["ink"],
            bbox=dict(facecolor="white", edgecolor=PALETTE["neutral_light"],
                      linewidth=0.5, pad=4))
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    save(fig, "fig06_explained_variance")
    return {"pc1_variance_ratio": float(pc1),
            "cum_top10_variance": float(top10)}


# ============================================================
# Figure 7 -- Whitening before / after
# ============================================================

def fig_whitening(X: np.ndarray) -> None:
    mu = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-8
    Xw = (X - mu) / sig

    Z_raw = PCA(n_components=2, random_state=0).fit_transform(X)
    Z_wht = PCA(n_components=2, random_state=0).fit_transform(Xw)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4))
    for ax, Z, title, xl, yl in [
        (axes[0], Z_raw, "raw MiniLM embeddings", "PC 1", "PC 2"),
        (axes[1], Z_wht, "after per-dim mean/variance whitening",
         "PC 1 (whitened)", "PC 2 (whitened)"),
    ]:
        ax.scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.4,
                   color=PALETTE["primary"], edgecolors="none")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(xl, fontsize=8.5)
        ax.set_ylabel(yl, fontsize=8.5)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "fig07_whitening")


# ============================================================
# Figure 8 -- t-SNE perplexity grid
# ============================================================

def fig_tsne_grid(df: pd.DataFrame, X: np.ndarray, n_sample: int = 1200) -> None:
    rng = np.random.default_rng(2)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    Xs = X[idx]
    companies = df["company"].values[idx]
    uniq = sorted(np.unique(companies))
    cmap = {c: CATEGORICAL[i % len(CATEGORICAL)] for i, c in enumerate(uniq)}
    colors = np.array([cmap[c] for c in companies])

    perplexities = [5, 30, 100, 300]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.4))
    for ax, perp in zip(axes.flat, perplexities):
        Zt = TSNE(n_components=2, perplexity=perp, init="pca",
                  random_state=0, learning_rate="auto").fit_transform(Xs)
        ax.scatter(Zt[:, 0], Zt[:, 1], s=5, alpha=0.6, c=colors,
                   edgecolors="none")
        ax.set_title(f"perplexity = {perp}", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
    # Shared legend below
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=cmap[c], markeredgecolor="none",
                          markersize=6, label=c) for c in uniq]
    fig.legend(handles=handles, loc="lower center", ncols=len(uniq),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig08_tsne_grid")


# ============================================================
# Figure 9 -- UMAP parameter grid (falls back gracefully if unavailable)
# ============================================================

def fig_umap_grid(df: pd.DataFrame, X: np.ndarray, n_sample: int = 1500) -> None:
    try:
        import umap
    except ImportError:
        print("  [skip] umap-learn not installed; skipping fig09")
        return
    rng = np.random.default_rng(3)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    Xs = X[idx]
    companies = df["company"].values[idx]
    uniq = sorted(np.unique(companies))
    cmap = {c: CATEGORICAL[i % len(CATEGORICAL)] for i, c in enumerate(uniq)}
    colors = np.array([cmap[c] for c in companies])

    grid = [(5, 0.1), (15, 0.1), (50, 0.1), (200, 0.1)]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.4))
    for ax, (nn, md) in zip(axes.flat, grid):
        Zu = umap.UMAP(n_neighbors=nn, min_dist=md, n_components=2,
                       random_state=0).fit_transform(Xs)
        ax.scatter(Zu[:, 0], Zu[:, 1], s=5, alpha=0.6, c=colors,
                   edgecolors="none")
        ax.set_title(f"n_neighbors = {nn},  min_dist = {md}", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=cmap[c], markeredgecolor="none",
                          markersize=6, label=c) for c in uniq]
    fig.legend(handles=handles, loc="lower center", ncols=len(uniq),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig09_umap_grid")


# ============================================================
# Figure 10 -- Shepard diagram
# ============================================================

def fig_shepard(X: np.ndarray, Z2: np.ndarray, n_sample: int = 800) -> None:
    rng = np.random.default_rng(4)
    idx = rng.choice(len(X), size=n_sample, replace=False)
    A = X[idx]; B = Z2[idx]

    # Pairwise Euclidean distances (unit-normalized X ⇒ cos-equivalent)
    def pdist_sub(M):
        n = len(M)
        i, j = np.triu_indices(n, k=1)
        return np.linalg.norm(M[i] - M[j], axis=1), i, j

    d_hi, ii, jj = pdist_sub(A)
    d_lo = np.linalg.norm(B[ii] - B[jj], axis=1)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    hb = ax.hexbin(d_hi, d_lo, gridsize=45, cmap="Blues", mincnt=1)
    # Perfect-projection diagonal (scaled by ratio of means)
    scale = d_lo.mean() / max(d_hi.mean(), 1e-9)
    xs = np.linspace(d_hi.min(), d_hi.max(), 100)
    ax.plot(xs, scale * xs, color=PALETTE["accent"], linewidth=1.2,
            linestyle="--", label="distance-preserving projection")
    ax.set_xlabel("original 384-D Euclidean distance")
    ax.set_ylabel("2-D PCA Euclidean distance")
    cb = fig.colorbar(hb, ax=ax, shrink=0.7)
    cb.set_label("pair count", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    leg = ax.legend(loc="upper left", fontsize=8)
    leg.get_frame().set_linewidth(0.5)
    save(fig, "fig10_shepard")


# ============================================================
# Figure 11 -- LDA prose-vs-table direction, 1D density
# ============================================================

def fig_lda_prose_table(df: pd.DataFrame, X: np.ndarray) -> dict:
    labels = df["text"].map(prose_table_label).values
    n_prose = int((labels == "prose").sum())
    n_table = int((labels == "table").sum())

    if n_prose < 20 or n_table < 20:
        print("  [warn] not enough samples for LDA prose/table; skipping fig11")
        return {}

    lda = LinearDiscriminantAnalysis(n_components=1)
    scores = lda.fit_transform(X, labels).ravel()

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.linspace(scores.min(), scores.max(), 80)
    for lbl, color in [("prose", PALETTE["primary"]),
                       ("table", PALETTE["accent"])]:
        m = (labels == lbl)
        ax.hist(scores[m], bins=bins, alpha=0.55, density=True,
                color=color, label=f"{lbl}  ($n = {m.sum():,}$)",
                edgecolor="white", linewidth=0.3)
    ax.set_xlabel(
        r"projection onto LDA prose$\leftrightarrow$table axis   "
        r"(narrative $\leftarrow$   $\rightarrow$ tabular)")
    ax.set_ylabel("density")
    leg = ax.legend(loc="upper right", fontsize=8)
    leg.get_frame().set_linewidth(0.5)
    save(fig, "fig11_lda_prose_table")
    return {"n_prose": n_prose, "n_table": n_table,
            "lda_scores": scores.tolist()}


# ============================================================
# Figure 12 -- Bimodality decomposes on the prose/table axis
# ============================================================

def fig_bimodality_on_prosetable(df: pd.DataFrame, X: np.ndarray,
                                 gmm_labels: list, lda_scores: list) -> None:
    if not lda_scores:
        return
    scores = np.array(lda_scores)
    gmm = np.array(gmm_labels)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bins = np.linspace(scores.min(), scores.max(), 80)
    ax.hist(scores[gmm == 0], bins=bins, alpha=0.55, density=True,
            color=PALETTE["positive"], label="GMM cluster A",
            edgecolor="white", linewidth=0.3)
    ax.hist(scores[gmm == 1], bins=bins, alpha=0.55, density=True,
            color=PALETTE["negative"], label="GMM cluster B",
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel(
        r"projection onto LDA prose$\leftrightarrow$table axis   "
        r"(narrative $\leftarrow$   $\rightarrow$ tabular)")
    ax.set_ylabel("density")
    leg = ax.legend(loc="upper right", fontsize=8,
                    title="unsupervised cluster")
    leg.get_frame().set_linewidth(0.5)
    save(fig, "fig12_bimodality_on_prosetable")


# ============================================================
# Figure 13 -- Doc centroid dendrogram
# ============================================================

def fig_dendrogram(df: pd.DataFrame, X: np.ndarray) -> None:
    grp_keys, centroids, labels = [], [], []
    for (company, doc), g in df.groupby(["company", "doc"], sort=True):
        idx = g.index.to_numpy()
        centroids.append(X[idx].mean(axis=0))
        grp_keys.append((company, doc))
        labels.append(f"{company}\n{re.sub(r'\\.pdf$', '', doc)[:38]}")
    C = np.array(centroids)
    # Ward linkage on Euclidean between centroids
    Z = linkage(C, method="ward")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ddata = dendrogram(Z, labels=labels, orientation="right",
                       color_threshold=0.75 * Z[:, 2].max(),
                       above_threshold_color=PALETTE["neutral"],
                       ax=ax)
    ax.set_xlabel("Ward linkage distance between document centroids")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig13_dendrogram")


# ============================================================
# Exemplar snippets — pulled from real corpus, saved to json
# ============================================================

def save_exemplars(df: pd.DataFrame, X: np.ndarray,
                   lda_scores: list, gmm_labels: list) -> None:
    """Select real chunks that will appear in the paper body as pull-quotes."""
    if not lda_scores:
        return
    scores = np.array(lda_scores)

    def top_k_by(score, k=3, mask=None):
        if mask is None:
            mask = np.ones(len(df), dtype=bool)
        ordered = np.argsort(-score)
        picked = [i for i in ordered if mask[i]][:k]
        return [{"industry": df.iloc[i]["industry"],
                 "company":  df.iloc[i]["company"],
                 "doc":      df.iloc[i]["doc"],
                 "chunk_id": int(df.iloc[i]["chunk_id"]),
                 "text": df.iloc[i]["text"][:280].replace("\n", " ").strip()}
                for i in picked]

    ex = {
        "narrative_extremes": top_k_by(-scores, k=3),   # most narrative
        "tabular_extremes":   top_k_by(scores, k=3),    # most tabular
        "middle_of_road":     top_k_by(-np.abs(scores - scores.mean()), k=2),
    }
    (ROOT / "paper" / "exemplars.json").write_text(
        json.dumps(ex, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  wrote exemplars.json")


# ============================================================
# Orchestrator
# ============================================================

def main() -> None:
    apply_style()
    print(f"Loading cached data from {CHUNKS_PATH}")
    df, X = load_data()
    print(f"  {len(df):,} chunks, dim = {X.shape[1]}")

    print("== PCA (2D) once, reused ==")
    p2 = PCA(n_components=2, random_state=0).fit(X)
    Z2 = p2.transform(X)
    stats = {"n_chunks": len(df),
             "n_docs": int(df["doc"].nunique()),
             "n_companies": int(df["company"].nunique()),
             "n_industries": int(df["industry"].nunique()),
             "companies": sorted(df["company"].unique().tolist()),
             "pca2_explained": float(p2.explained_variance_ratio_.sum())}

    print("== fig01 corpus overview ==");     fig_corpus_overview(df)
    print("== fig02 PCA by company ==");      fig_pca2d_by_company(df, Z2)
    print("== fig03 bimodality (GMM) ==")
    bi = fig_bimodality(df, X, Z2)
    stats.update({"delta_bic": bi["delta_bic"],
                  "gmm_subspace_dim": bi["gmm_subspace_dim"],
                  "gmm_subspace_var": bi["gmm_subspace_var"]})
    print("== fig04 cosine distributions ==")
    stats.update(fig_cosine_distributions(df, X))
    print("== fig05 anisotropy ==")
    stats.update(fig_anisotropy(X))
    print("== fig06 explained variance ==")
    stats.update(fig_explained_variance(X))
    print("== fig07 whitening ==");           fig_whitening(X)
    print("== fig08 t-SNE grid ==");          fig_tsne_grid(df, X)
    print("== fig09 UMAP grid ==");           fig_umap_grid(df, X)
    print("== fig10 Shepard diagram ==");     fig_shepard(X, Z2)
    print("== fig11 LDA prose/table ==")
    lda = fig_lda_prose_table(df, X)
    stats.update({"n_prose": lda.get("n_prose"),
                  "n_table": lda.get("n_table")})
    print("== fig12 bimodality on prose/table axis ==")
    fig_bimodality_on_prosetable(df, X, bi["cluster_labels"],
                                 lda.get("lda_scores", []))
    print("== fig13 dendrogram ==");          fig_dendrogram(df, X)

    print("== exemplars ==")
    save_exemplars(df, X, lda.get("lda_scores", []), bi["cluster_labels"])

    STATS_OUT.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {STATS_OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
