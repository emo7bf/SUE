"""
scripts/build_report_cards.py
-----------------------------
Scaffold for the SUE category report cards: one PDF page per disclosure
category, comparing every company in one industry.

Each page carries:
  1. A picture of the corpus (PCA-2D) with the category's chunks spotlit
     and everything else dimmed.
  2. The KEY PASSAGES: the top chunks by anchor score across companies -
     the literal exhibits the verdict leans on, quoted without commentary.
  3. A VERDICT (2-3 sentences): rule-driven comparison of the companies'
     coverage of the category, with room for editorial notes such as
     over-delivery (a firm publishing more than the rubric demands, e.g.
     independent assurance or a standalone data supplement).

Method (all precomputed, no LLM):
  * Each category is defined by a handful of seed passages written in
    the register of real report prose. Their MiniLM centroid is the
    category ANCHOR a_c.
  * Every chunk x gets the score s_c(x) = x . a_c (cosine, since both
    are unit-norm). Chunks above ANCHOR_THRESHOLD count as "about" the
    category.
  * Per-company statistics over those chunks (share, mean score, digit
    density as a specificity proxy, assurance-document presence) drive
    the verdict templates.

Usage:
    python scripts/build_report_cards.py                       # A&D
    python scripts/build_report_cards.py --industry aerospace_defense

Outputs:
    assets/report_cards/<industry>/<category-slug>.pdf
    assets/report_cards/<industry>/report_card_stats.json
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

ANCHOR_THRESHOLD = 0.42   # cosine score above which a chunk "belongs" to a category
TOP_PASSAGES = 4          # exhibits quoted on the page
MIN_CHUNKS = 3            # a company needs this many on-topic chunks to be ranked

# --------------------------------------------------------------- categories
# Category -> (subtitle, [seed passages]). Seeds are written in the
# register of real Global Impact Report prose; the anchor is their
# embedding centroid. Sub-categories get their own card by being their
# own entry. Draft taxonomy is SASB-informed; refine freely.
CATEGORIES: dict = {
    "GHG emissions (Scope 1 & 2)": (
        "Direct operational and purchased-energy emissions",
        ["Our Scope 1 and Scope 2 greenhouse gas emissions decreased compared "
         "to the prior year, driven by renewable electricity procurement and "
         "energy-efficiency projects across our manufacturing sites.",
         "We report market-based and location-based Scope 2 emissions in "
         "metric tons of CO2 equivalent, with third-party verification.",
         "Total direct greenhouse gas emissions from fuel combustion at our "
         "facilities, including natural gas and fleet vehicles."],
    ),
    "GHG emissions (Scope 3)": (
        "Value-chain emissions: suppliers, product use, logistics",
        ["Our Scope 3 inventory covers purchased goods and services, upstream "
         "transportation, business travel, and the use of sold products.",
         "Emissions from the use phase of our products represent the largest "
         "share of our value-chain footprint, and we engage suppliers to set "
         "science-based reduction targets.",
         "We estimate category 11 use-of-sold-products emissions using "
         "product energy consumption and expected service life."],
    ),
    "Hazardous waste & materials": (
        "Waste generation, recycling, chemical management",
        ["We reduced hazardous waste generation per unit of production and "
         "increased the share of waste diverted from landfill through "
         "recycling and reuse programs.",
         "Chemical management procedures govern the storage, handling, and "
         "disposal of solvents and process chemicals at our facilities.",
         "Total hazardous waste generated, recycled, and disposed, reported "
         "in metric tons with year-over-year comparison."],
    ),
    "Water stewardship": (
        "Withdrawal, consumption, discharge, water risk",
        ["We track water withdrawal and consumption at all manufacturing "
         "sites and prioritize reduction projects in water-stressed basins.",
         "Wastewater from our operations is treated on site before discharge "
         "in compliance with local permits and standards.",
         "Total water withdrawn by source, including municipal supply and "
         "groundwater, reported in thousands of cubic meters."],
    ),
    "Supply chain responsibility": (
        "Supplier audits, conflict minerals, sourcing standards",
        ["Our supplier code of conduct requires compliance with labor, "
         "environmental, and ethics standards, verified through audits and "
         "corrective action plans.",
         "We conduct due diligence on conflict minerals in accordance with "
         "the OECD guidance and require smelter-level reporting from "
         "suppliers.",
         "Supplier assessments completed this year, including on-site audits "
         "and remediation of identified nonconformances."],
    ),
    "Labor & human rights": (
        "Human rights policy, modern slavery, freedom of association",
        ["Our human rights policy prohibits forced labor, child labor, and "
         "discrimination, and applies to all operations and suppliers.",
         "We published our modern slavery statement describing the steps "
         "taken to identify and mitigate forced labor risks in our supply "
         "chain.",
         "Employees have the right to freedom of association and collective "
         "bargaining consistent with applicable law."],
    ),
    "Worker safety": (
        "Injury rates, safety programs, contractor safety",
        ["Our total recordable incident rate improved year over year through "
         "behavior-based safety programs and near-miss reporting.",
         "We investigate all serious incidents, share lessons learned across "
         "sites, and empower employees to stop work when conditions are "
         "unsafe.",
         "Occupational health and safety management systems are certified to "
         "ISO 45001 at our major manufacturing locations."],
    ),
    "Workforce & inclusion": (
        "Demographics, development, pay equity, engagement",
        ["We report workforce demographics by gender and ethnicity and set "
         "goals to increase representation in engineering and leadership "
         "roles.",
         "Annual engagement surveys inform actions on career development, "
         "wellbeing, and inclusion across the company.",
         "We completed a pay-equity analysis and remediated identified gaps."],
    ),
    "Governance & ethics": (
        "Board oversight, anti-corruption, lobbying, compliance",
        ["The board's committee oversees sustainability strategy, enterprise "
         "risk, and progress against public commitments.",
         "Our anti-corruption program includes training, third-party due "
         "diligence, and a confidential ethics helpline with "
         "non-retaliation protections.",
         "Political contributions and lobbying activities are disclosed and "
         "governed by board-approved policy."],
    ),
    "Product safety & consequences": (
        "Product quality, safe use, export controls, end-use governance",
        ["Product safety reviews and quality management systems govern design, "
         "test, and field monitoring of every product line we ship.",
         "We comply with export control regulations and screen transactions "
         "to ensure our products are sold only to authorized end users.",
         "Our review process evaluates the responsible use of our "
         "technologies, including human rights considerations in product "
         "deployment."],
    ),
    "Cybersecurity & data protection": (
        "Security governance, incident response, customer data",
        ["Our cybersecurity program is aligned to the NIST framework, with "
         "board-level oversight and regular penetration testing.",
         "We operate a security operations center, run incident response "
         "exercises, and require security training for all employees.",
         "Customer and employee data is protected through encryption, access "
         "controls, and privacy-by-design practices."],
    ),
    "Climate risk & assurance": (
        "TCFD scenario analysis, targets, third-party assurance",
        ["We assessed physical and transition climate risks under multiple "
         "warming scenarios consistent with the TCFD recommendations.",
         "Our near-term emissions reduction targets are validated by the "
         "Science Based Targets initiative.",
         "An independent third party provided limited assurance over our "
         "reported Scope 1 and Scope 2 emissions data."],
    ),
}

# doc categories that signal over-delivery (publishing beyond the rubric)
OVERDELIVERY_DOCS = {"assurance": "independent assurance",
                     "data_supplement": "a standalone data supplement",
                     "climate_report": "a dedicated climate/TCFD report",
                     "human_rights": "a standalone human-rights disclosure",
                     "reporting_index": "a full GRI/SASB reporting index"}


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\s\-&]+", "", name.lower())
    return re.sub(r"[\s&]+", "_", s).strip("_")


def load_industry(industry: str):
    cache = ASSETS / "industries" / industry
    df = pd.read_parquet(cache / "chunks.parquet")
    X = np.load(cache / "embeddings.npy").astype(np.float32)
    return df.reset_index(drop=True), X


def embed_anchors():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    names = list(CATEGORIES.keys())
    anchors = {}
    for name in names:
        seeds = CATEGORIES[name][1]
        V = model.encode(seeds, normalize_embeddings=True, convert_to_numpy=True)
        a = V.mean(axis=0)
        anchors[name] = a / np.linalg.norm(a)
    return anchors


def company_stats(df: pd.DataFrame, scores: np.ndarray):
    """Per-company coverage statistics for one category."""
    on_topic = scores >= ANCHOR_THRESHOLD
    rows = []
    for company, g in df.groupby("company", sort=True):
        idx = g.index.to_numpy()
        n_on = int(on_topic[idx].sum())
        top10 = np.sort(scores[idx])[-10:]
        rows.append({
            "company": company,
            "n_chunks": len(idx),
            "n_on_topic": n_on,
            "share_on_topic": n_on / len(idx),
            "depth": float(top10.mean()),               # strength of best coverage
            "specificity": float(df.loc[idx, "digit_density"][on_topic[idx]].mean())
                           if n_on else 0.0,             # numbers = concreteness
            "overdelivery": sorted({
                OVERDELIVERY_DOCS[c] for c in g.loc[on_topic[idx], "doc_category"]
                if c in OVERDELIVERY_DOCS}),
        })
    return pd.DataFrame(rows)


def verdict_for(cat: str, stats: pd.DataFrame) -> str:
    """2-3 sentence verdict. Template chosen by the statistics; the key
    passages on the page are the evidence it leans on."""
    ranked = stats[stats.n_on_topic >= MIN_CHUNKS].sort_values(
        "depth", ascending=False)
    if len(ranked) < 2:
        return (f"Too few companies disclose enough about {cat.lower()} for a "
                "peer comparison; the passages above are the strongest "
                "available material. Treat any ranking in this category as "
                "provisional until more reports are ingested.")
    lead, tail = ranked.iloc[0], ranked.iloc[-1]
    spread = lead.depth - tail.depth
    lines = []
    if spread < 0.03:
        lines.append(
            f"On {cat.lower()}, the peer group is essentially level: "
            f"coverage depth is tightly clustered "
            f"({tail.depth:.2f}-{lead.depth:.2f}), suggesting a shared, "
            "rubric-shaped template rather than differentiated effort.")
    else:
        lines.append(
            f"{lead.company} leads the peer group on {cat.lower()} "
            f"(depth {lead.depth:.2f} vs. group tail {tail.depth:.2f}), and "
            "its passages above are noticeably more concrete than the "
            "group norm."
            if lead.specificity >= stats.specificity.mean() else
            f"{lead.company} devotes the most attention to {cat.lower()} "
            f"(depth {lead.depth:.2f}), though its coverage runs more "
            "narrative than numeric; the gap to the group tail "
            f"({tail.company}, {tail.depth:.2f}) is real but rhetorical "
            "depth should not be mistaken for performance.")
    over = [(r.company, r.overdelivery) for r in ranked.itertuples()
            if r.overdelivery]
    if over:
        c, kinds = over[0]
        lines.append(
            f"Credit where due: {c} goes beyond the standard report here, "
            f"publishing {kinds[0]}"
            + (f" and {kinds[1]}" if len(kinds) > 1 else "")
            + " - over-delivery of this kind is a stronger signal than "
              "polished prose.")
    weak = ranked[ranked.share_on_topic < 0.01]
    if len(weak) and len(lines) < 3:
        lines.append(
            f"{', '.join(weak.company.tolist()[:3])} barely touch the topic "
            "at all; silence in a category the industry considers material "
            "is itself a disclosure choice worth noticing.")
    return " ".join(lines[:3])


def render_card(pdf_path: Path, industry_name: str, cat: str, subtitle: str,
                df: pd.DataFrame, Z2: np.ndarray, scores: np.ndarray,
                stats: pd.DataFrame, verdict: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    on = scores >= ANCHOR_THRESHOLD
    top_idx = np.argsort(-scores)[:TOP_PASSAGES]

    fig = plt.figure(figsize=(8.5, 11), dpi=150)
    gs = GridSpec(3, 1, height_ratios=[3.2, 4.4, 1.6], hspace=0.32,
                  left=0.07, right=0.95, top=0.93, bottom=0.05)

    fig.suptitle(f"{cat}", fontsize=15, fontweight="bold", x=0.07, ha="left")
    fig.text(0.07, 0.945, f"SUE report card - {industry_name} | {subtitle}",
             fontsize=8.5, color="#555")

    # --- panel 1: the cluster picture
    ax = fig.add_subplot(gs[0])
    ax.scatter(Z2[~on, 0], Z2[~on, 1], s=4, c="#d8d8dc", alpha=0.35,
               linewidths=0, label="other passages")
    sc = ax.scatter(Z2[on, 0], Z2[on, 1], s=10, c=scores[on], cmap="viridis",
                    alpha=0.9, linewidths=0, label="on-topic passages")
    ax.scatter(Z2[top_idx, 0], Z2[top_idx, 1], s=90, facecolors="none",
               edgecolors="#a45a1e", linewidths=1.6, label="key passages")
    for r, i in enumerate(top_idx, start=1):
        ax.annotate(str(r), (Z2[i, 0], Z2[i, 1]), xytext=(5, 4),
                    textcoords="offset points", fontsize=8, color="#a45a1e",
                    fontweight="bold")
    fig.colorbar(sc, ax=ax, shrink=0.75, label="anchor score")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{int(on.sum())} of {len(df)} passages are about this "
                 "category (PCA projection)", fontsize=9)
    ax.legend(loc="lower right", fontsize=7, frameon=True)

    # --- panel 2: key passages, verbatim
    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    y = 1.0
    ax2.text(0, y, "KEY PASSAGES (the evidence the verdict leans on)",
             fontsize=9.5, fontweight="bold", va="top")
    y -= 0.075
    for r, i in enumerate(top_idx, start=1):
        row = df.iloc[i]
        pages = (f"p.{row.page_start}" if row.page_start == row.page_end
                 else f"pp.{row.page_start}-{row.page_end}")
        head = (f"[{r}] {row.company} - {row.doc} ({pages}) - "
                f"score {scores[i]:.2f}")
        body = textwrap.fill(re.sub(r"\s+", " ", row.text)[:420] + "...",
                             width=110)
        ax2.text(0, y, head, fontsize=7.8, fontweight="bold",
                 color="#a45a1e", va="top")
        y -= 0.045
        ax2.text(0, y, body, fontsize=7.4, va="top", family="serif",
                 color="#222")
        y -= 0.045 * (body.count("\n") + 1) + 0.035

    # --- panel 3: verdict
    ax3 = fig.add_subplot(gs[2]); ax3.axis("off")
    ax3.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax3.transAxes,
                                facecolor="#f4f1ea", edgecolor="#a45a1e",
                                linewidth=1.2))
    ax3.text(0.02, 0.86, "VERDICT", fontsize=10, fontweight="bold",
             color="#a45a1e", va="top", transform=ax3.transAxes)
    ax3.text(0.02, 0.62, textwrap.fill(verdict, width=118), fontsize=8.2,
             va="top", transform=ax3.transAxes, family="serif")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    plt.close(fig)
    print("  wrote", pdf_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--industry", default="aerospace_defense")
    args = ap.parse_args()

    df, X = load_industry(args.industry)
    industry_name = df["industry"].iloc[0]
    Z2 = PCA(n_components=2, random_state=0).fit_transform(X)
    print(f"{industry_name}: {len(df):,} chunks, "
          f"{df['company'].nunique()} companies")

    anchors = embed_anchors()
    out_dir = ASSETS / "report_cards" / args.industry
    all_stats = {}
    for cat, (subtitle, _seeds) in CATEGORIES.items():
        scores = X @ anchors[cat]
        stats = company_stats(df, scores)
        verdict = verdict_for(cat, stats)
        render_card(out_dir / f"{slugify(cat)}.pdf", industry_name, cat,
                    subtitle, df, Z2, scores, stats, verdict)
        all_stats[cat] = {
            "verdict": verdict,
            "companies": stats.to_dict(orient="records"),
        }
    (out_dir / "report_card_stats.json").write_text(
        json.dumps(all_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_dir / 'report_card_stats.json'}")
    print(f"{len(CATEGORIES)} category cards in {out_dir}")


if __name__ == "__main__":
    main()
