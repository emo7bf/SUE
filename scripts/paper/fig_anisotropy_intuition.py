"""
scripts/paper/fig_anisotropy_intuition.py
-----------------------------------------
Emits paper/figures/fig_anisotropy_intuition.{png,pdf}.

Two-panel schematic explaining what an anisotropic embedding space is,
and why every cosine-similarity histogram in the paper sits well above
zero.  The left panel shows an *isotropic* space: dots fill the sphere
uniformly, so the mean cosine between random pairs is 0.  The right
panel shows an *anisotropic* space: dots are crammed into a narrow
cone, so even "unrelated" pairs sit near cosine ~ 0.3.

This is the visual the reader needs before either the cosine-
distribution histogram (Figure 6 in the paper) or the anisotropy-gap
plot (Figure 7) becomes legible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _fibonacci_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    """Even-ish samples on the unit sphere via the golden spiral."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    theta = 2.0 * np.pi * i / phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    P = np.stack([x, y, z], axis=1)
    # add tiny jitter so it doesn't look mechanical
    P = P + 0.02 * rng.standard_normal(P.shape)
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    return P


def _cone_samples(n: int, cone_axis: np.ndarray,
                  half_angle_deg: float,
                  rng: np.random.Generator) -> np.ndarray:
    """
    n unit vectors uniformly inside a cone around cone_axis with the
    given half-angle.  Uniform on the spherical cap.
    """
    cos_max = np.cos(np.radians(half_angle_deg))
    # sample cos(theta) uniformly in [cos_max, 1]
    u = rng.uniform(cos_max, 1.0, size=n)
    sin_t = np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    # local frame: z along cone_axis, x/y arbitrary orthonormal
    z = cone_axis / np.linalg.norm(cone_axis)
    # pick an arbitrary vector not parallel to z
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(z, tmp); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    P = (u[:, None] * z[None, :]
         + (sin_t * np.cos(phi))[:, None] * x[None, :]
         + (sin_t * np.sin(phi))[:, None] * y[None, :])
    return P


def _plot_panel(ax, P: np.ndarray, title: str, subtitle: str,
                mean_cos_text: str, colour: str,
                cone_axis: np.ndarray | None = None,
                cone_half_deg: float | None = None):
    # a light wireframe unit sphere for reference
    u, v = np.mgrid[0:2 * np.pi:26j, 0:np.pi:16j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color="#c8c8c8", linewidth=0.5, alpha=0.7)

    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=8,
               color=colour, alpha=0.78, edgecolors="none")

    if cone_axis is not None and cone_half_deg is not None:
        za = cone_axis / np.linalg.norm(cone_axis)
        # draw a longer axis so the label doesn't collide with the dots
        arrow_end = 1.55 * za
        ax.plot([0, arrow_end[0]], [0, arrow_end[1]], [0, arrow_end[2]],
                color="#333333", linewidth=1.8)
        ax.text(arrow_end[0] * 1.10, arrow_end[1] * 1.10,
                arrow_end[2] * 1.10 + 0.05,
                "mean direction\n(anisotropy axis)",
                color="#333", fontsize=8.4, ha="left", va="bottom")

        # dashed cone rim on the unit sphere
        theta = np.radians(cone_half_deg)
        rim_r = np.sin(theta)
        rim_h = np.cos(theta)
        tmp = np.array([1.0, 0.0, 0.0]) if abs(za[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_ax = np.cross(za, tmp); x_ax /= np.linalg.norm(x_ax)
        y_ax = np.cross(za, x_ax)
        phis = np.linspace(0, 2 * np.pi, 80)
        rim = (rim_h * za[None, :]
               + rim_r * np.cos(phis)[:, None] * x_ax[None, :]
               + rim_r * np.sin(phis)[:, None] * y_ax[None, :])
        ax.plot(rim[:, 0], rim[:, 1], rim[:, 2],
                color=colour, linewidth=1.5, linestyle="--")

    ax.set_title(title, fontsize=12, weight="bold", pad=6, color="#222")
    ax.text2D(0.5, -0.04, subtitle, transform=ax.transAxes,
              fontsize=9.2, ha="center", va="top", color="#333",
              style="italic")
    ax.text2D(0.5, -0.13, mean_cos_text, transform=ax.transAxes,
              fontsize=10.2, ha="center", va="top",
              color=colour, weight="bold")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=18, azim=28)


def build() -> None:
    rng = np.random.default_rng(42)

    P_iso = _fibonacci_sphere(500, rng)

    cone_axis = np.array([0.6, 0.55, 0.55])
    cone_axis = cone_axis / np.linalg.norm(cone_axis)
    # Half-angle chosen so the cone is visually striking; the real SUE
    # anisotropy is more subtle (mean cos ~0.30 vs isotropic ~0), so we
    # exaggerate for legibility and quote the real numbers in the caption.
    P_aniso = _cone_samples(500, cone_axis, half_angle_deg=45.0, rng=rng)

    fig = plt.figure(figsize=(11, 5.6), dpi=180)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    _plot_panel(
        ax1, P_iso,
        title="Isotropic space (well-behaved)",
        subtitle=("Every direction is used equally; two random points "
                  "are typically far apart on the sphere."),
        mean_cos_text="mean cos(random pair) \u2248 0.00",
        colour="#4a6a8a",
    )

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_panel(
        ax2, P_aniso,
        title="Anisotropic space (what MiniLM does)",
        subtitle=("All embeddings crowd into a cone; even "
                  "\u2018unrelated\u2019 pairs are close together."),
        mean_cos_text="mean cos(random pair) \u2248 +0.30 (real SUE value)",
        colour="#a45a1e",
        cone_axis=cone_axis, cone_half_deg=45.0,
    )

    fig.suptitle(
        "Why every cosine in the SUE corpus sits above zero: "
        "anisotropy is a first-moment property of the encoder",
        fontsize=12.5, y=1.00, color="#222",
    )

    # A single-line caption strip below both panels
    fig.text(
        0.5, 0.005,
        ("Read the cosine-similarity histograms as measurements against "
         "the right-hand cone, not the left-hand sphere. The \u201Coffset\u201D "
         "in every distribution is the encoder\u2019s cone half-angle "
         "made numeric."),
        ha="center", va="bottom", fontsize=9.2, color="#333", style="italic",
    )

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig_anisotropy_intuition.{ext}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    build()
