"""
scripts/paper/fig_report_anatomy.py
-----------------------------------
Emits paper/figures/fig00_report_anatomy.{png,pdf}.

A single-panel schematic of what a typical Global Impact Report contains:
CEO letter, materiality, Scope 1/2/3 emissions, water, waste, workforce,
supplier due-diligence, governance, forward-looking targets. Each content
type is drawn as a rounded box with a large emoji glyph, a short label,
and one line of tabular vs. narrative flavour, so a reader can see at a
glance which content types live on the tabular half of the class-centroid
axis vs. the narrative half.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# Prefer an emoji-capable font when available (Windows: Segoe UI Emoji,
# macOS: Apple Color Emoji, Linux: Noto Color Emoji), otherwise fall back
# to DejaVu Sans which has enough BMP glyphs for the symbols below.
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]


ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# (glyph, title, one-line description, flavour) where flavour is:
#   "N" = narrative-heavy (prose)
#   "T" = tabular-heavy (numbers)
#   "M" = mixed
# The glyphs below are BMP characters that ship with DejaVu Sans so the
# figure renders identically on every platform.
_BLOCKS = [
    ("\u2709",  "CEO Letter",              "Framing, priorities, tone for the year.",                    "N"),
    ("\u2696",  "Materiality Assessment",  "Which topics matter to business and stakeholders.",          "M"),
    ("\u25CF",  "Scope 1 Emissions",       "Direct emissions from owned facilities.",                    "T"),
    ("\u26A1",  "Scope 2 Emissions",       "Emissions from purchased electricity and heat.",             "T"),
    ("\u21C4",  "Scope 3 Emissions",       "Value-chain emissions (suppliers, use, disposal).",          "T"),
    ("\u2248",  "Water",                   "Withdrawal + consumption, often by basin stress.",           "T"),
    ("\u267B",  "Waste & Circularity",     "Recycled fractions, hazardous streams, packaging.",          "T"),
    ("\u263B",  "Workforce",               "Composition, safety, turnover, training hours.",             "T"),
    ("\u2699",  "Supplier Due-Diligence",  "Audits, human-rights procedures, code coverage.",            "N"),
    ("\u2691",  "Governance",              "Board independence, committee structure, ethics.",           "N"),
    ("\u2794",  "Forward-Looking Targets", "\u201C50% reduction from 2019 baseline by 2030.\u201D",      "M"),
    ("\u25A6",  "Assurance / Data Tables", "Third-party assurance + machine-readable annexes.",          "T"),
]


_FLAVOUR_COLOUR = {
    "N": ("#e8f0f8", "#4a6a8a", "narrative"),   # blue-ish
    "T": ("#fbeee2", "#a45a1e", "tabular"),     # orange-ish
    "M": ("#efeaf7", "#6a4a8e", "mixed"),       # purple-ish
}


def _wrap(text: str, width: int = 42) -> str:
    """Simple whitespace word-wrap onto <=2 lines within `width` chars."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines[:2])


def build() -> None:
    n = len(_BLOCKS)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols

    box_w, box_h = 3.6, 1.55
    x_gap, y_gap = 0.28, 0.32

    total_w = n_cols * box_w + (n_cols - 1) * x_gap
    total_h = n_rows * box_h + (n_rows - 1) * y_gap

    fig, ax = plt.subplots(
        figsize=(total_w + 0.6, total_h + 1.6), dpi=180,
    )

    ax.set_xlim(-0.3, total_w + 0.3)
    ax.set_ylim(-1.1, total_h + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    for i, (glyph, title, sub, flavour) in enumerate(_BLOCKS):
        col = i % n_cols
        row = i // n_cols
        x = col * (box_w + x_gap)
        y = total_h - (row + 1) * box_h - row * y_gap

        fill, edge, _label = _FLAVOUR_COLOUR[flavour]
        box = FancyBboxPatch(
            (x + 0.02, y + 0.02), box_w - 0.04, box_h - 0.04,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.5, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(box)

        # Glyph column on the left, inside its own coloured circle
        cx, cy, cr = x + 0.42, y + box_h / 2 + 0.05, 0.32
        glyph_bg = plt.Circle(
            (cx, cy), cr,
            facecolor="white", edgecolor=edge, linewidth=1.4, zorder=2,
        )
        ax.add_patch(glyph_bg)
        ax.text(cx, cy, glyph,
                fontsize=20, ha="center", va="center",
                color=edge, zorder=3)

        # Title top-right of glyph (wrap long titles onto two lines)
        text_x = x + 0.9
        wrapped_title = _wrap(title, width=20)
        title_is_two_lines = "\n" in wrapped_title
        title_y = y + box_h - (0.36 if title_is_two_lines else 0.34)
        ax.text(text_x, title_y, wrapped_title,
                fontsize=10.5, weight="bold",
                ha="left", va="top", color=edge,
                linespacing=1.05)

        # Wrapped description below title
        desc_y = y + box_h - (1.02 if title_is_two_lines else 0.82)
        ax.text(text_x, desc_y, _wrap(sub, width=34),
                fontsize=8.4, style="italic",
                ha="left", va="top", color="#333")

        # Small flavour tag in the corner
        tag = _FLAVOUR_COLOUR[flavour][2]
        ax.text(x + box_w - 0.15, y + 0.18, tag,
                fontsize=7.5, ha="right", va="center",
                color=edge, weight="bold",
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", edgecolor=edge, linewidth=0.6))

    # Legend strip below
    legend_y = -0.75
    lx = 0.4
    for key in ["N", "M", "T"]:
        fill, edge, _ = _FLAVOUR_COLOUR[key]
        chip = FancyBboxPatch(
            (lx, legend_y), 0.45, 0.32,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(chip)
        label = {"N": "narrative-heavy (prose)",
                 "M": "mixed",
                 "T": "tabular-heavy (numbers)"}[key]
        ax.text(lx + 0.6, legend_y + 0.16, label,
                fontsize=9.5, ha="left", va="center",
                color=edge, weight="bold")
        lx += 4.0

    plt.title(
        "Anatomy of a Global Impact Report \u2014 twelve recurring "
        "content types, coloured by rhetorical register",
        fontsize=12.5, pad=8, color="#222",
    )

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig00_report_anatomy.{ext}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    build()
