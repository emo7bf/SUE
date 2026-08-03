"""
scripts/paper/style.py
----------------------
Matplotlib rc-params and color palette matched to the reference-image style
(muted academic, serif, thin gridlines, no top/right spines). Import once at
the top of every figure-producing script:

    from scripts.paper.style import apply_style, PALETTE
    apply_style()
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


# Muted academic palette. Every color is desaturated so a scatter with N
# categories reads as data, not as decoration.
PALETTE = {
    "primary":       "#3B6E8F",  # muted blue
    "accent":        "#B8860B",  # dark goldenrod (highlight / query star)
    "positive":      "#6DA88F",  # sage green
    "negative":      "#C97064",  # muted red
    "neutral":       "#8C8C8C",  # medium gray
    "neutral_light": "#C8C8C8",  # light gray
    "ink":           "#1F1F1F",  # near-black for text
    "paper":         "#FFFFFF",
    "grid":          "#D9D9D9",
}

# Ordered categorical palette for up to 10 groups. Deliberately not the
# matplotlib default; picked for print legibility.
CATEGORICAL = [
    "#3B6E8F",  # blue
    "#B8860B",  # gold
    "#6DA88F",  # sage green
    "#C97064",  # muted red
    "#8C6BB1",  # muted purple
    "#5F8C6A",  # forest
    "#B26A5F",  # terracotta
    "#4E7A9B",  # slate blue
    "#A98D3D",  # olive gold
    "#7A7A7A",  # gray
]


def apply_style() -> None:
    """Apply publication-style rc-params. Call once per script."""
    mpl.rcParams.update({
        # ----- typography (serif to match a LaTeX document) -----
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "STIX Two Text", "Liberation Serif",
            "DejaVu Serif", "serif",
        ],
        "mathtext.fontset": "stix",   # serif-compatible math
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        # ----- axes -----
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": PALETTE["ink"],
        "axes.linewidth": 0.7,
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # ----- grid -----
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        # ----- legend -----
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": PALETTE["neutral_light"],
        "legend.fancybox": False,
        # ----- figure & save -----
        "figure.facecolor": PALETTE["paper"],
        "axes.facecolor":  PALETTE["paper"],
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # ----- lines -----
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "patch.linewidth": 0.6,
    })
