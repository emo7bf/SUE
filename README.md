# Semantic Universe Explorer — SUE

**Reachable pages**

- **[Interactive viewer](https://emo7bf.github.io/SUE/semantic_universe_explorer.html)** — the 3-D / 2-D plot with
  every chunk clickable
- **[Walk-through](https://emo7bf.github.io/SUE/sue_walkthru.html)** — what the colour modes mean
  and where each firm sits along the prose–tabular axis
- **[Math & statistics](https://emo7bf.github.io/SUE/math_and_statistics.html)** — LDA, GMM, t-SNE, UMAP with
  every variable defined

> If you’re reading this on GitHub, the **[more readable view of the
> walk-through](https://emo7bf.github.io/SUE/sue_walkthru.html)** has hover thumbnails and rendered math
> that the Markdown version doesn’t.

---

## What is a Global Impact Report?

A **Global Impact Report** (sometimes an ESG or Sustainability Report)
documents a corporation’s environmental, social, and governance
performance — emissions, water, workforce, safety, board oversight,
and supply-chain ethics.

Firms publish them for regulators, capital markets, and the growing pool
of investors who want their portfolios guided by corporate responsibility
as well as by returns.

This demo of the Semantic Universe Explorer (SUE) compares the Global
Impact Reports of five direct competitors in semiconductor-manufacturing
equipment: [Applied Materials](https://www.appliedmaterials.com/content/dam/site/company/csr/doc/2025_impact_report.pdf.coredownload.inline.pdf),
[ASML](https://ourbrand.asml.com/m/71076aaad607de4d/original/asml-2025-annual-report-based-on-us-gaap.pdf),
[KLA](https://www.kla.com/wp-content/uploads/2024-KLA-Global-Impact-Report.pdf),
[Lam Research](https://www.lamresearch.com/wp-content/uploads/2026/05/Lam-Research-2025-Global-Impact-Report.pdf),
and [Tokyo Electron](https://www.tel.com/ir/library/ar/pjsoh100000000rc-att/ir2025_all_en.pdf).

Every dot in the interactive viewer represents one passage —
approximately 900 characters — from one report. The passages are
positioned by *semantic similarity* rather than by which firm wrote them.

**Filter, colour, click, and read.**

## Anatomy of a Global Impact Report

Under the hood, a Global Impact Report is built from the same dozen
recurring content types every year. Some read like corporate prose
(a CEO letter, a governance chapter, a description of supplier
audits); others are almost pure spreadsheet (Scope 1 / 2 / 3 emissions
tables, water withdrawal by basin, workforce demographics). The
diagram below names those twelve blocks and colours them by rhetorical
register — blue for narrative-heavy, orange for tabular-heavy,
purple for mixed. The viewer’s *Prose→Tabular* axis is
empirically the same axis that separates the orange half from the blue
half.

![Anatomy of a Global Impact Report](sue_release/paper/figures/fig00_report_anatomy.png)

## What each color mode means

- **Company.** Every dot coloured by which of the five firms authored the
  underlying passage. Useful for eyeballing whether any single firm
  produces a visually distinct region of the space.
- **Prose→Tabular score.** A number for each chunk from "very
  narrative" (blue) through zero (mixed) to "very tabular" (red).
  Projection onto the axis that connects the average narrative-chunk
  location to the average tabular-chunk location.
- **Digit density of text.** The unlearned control: what percentage of
  the chunk’s characters are numerals. Whenever this and the
  prose↔tabular score paint the same picture, that’s your sign
  that the encoder’s "semantics" on this axis is really just
  numerals.
- **Cross-corpus outlierness.** For every chunk, how far is it from the
  average of the OTHER four competitors’ embeddings? Red = far.
- **In-doc typicality.** For every chunk, how far is it from the centre
  of its own document?
- **Unsupervised cluster (GMM).** A 2-component Gaussian mixture fit on
  the top-20 principal components, without ever being told about prose
  or tables. If it recovers the same partition anyway, that’s
  evidence the split is real.

## Further statistical analyses

The interactive viewer is the friendly front door. Underneath it sit a
handful of standard statistical tools that all point at the same finding
from different angles — the corpus is bimodal along a
prose–tabular axis, and that split survives every dimensionality
reduction we throw at it. See **[Math & statistics](https://emo7bf.github.io/SUE/math_and_statistics.html)** for
plain-English explanations with every variable defined.

- [LDA: the single direction that splits prose from tables](https://emo7bf.github.io/SUE/math_and_statistics.html#fig11_lda_prose_table)
- [GMM: is the corpus one blob, or really two?](https://emo7bf.github.io/SUE/math_and_statistics.html#fig03_bimodality_gmm)
- [t-SNE at four perplexities](https://emo7bf.github.io/SUE/math_and_statistics.html#fig08_tsne_grid)
- [UMAP at four neighbourhood sizes](https://emo7bf.github.io/SUE/math_and_statistics.html#fig09_umap_grid)

## The five companies and their reports

Every dot in the viewer is one ~900-character chunk of one of these
firms’ Global Impact Reports. If a passage catches your eye and you
want to see it in the original document, the reports themselves are one
click away.

| Company | 2025 | 2024 | 2023 | 2022 |
|---|---|---|---|---|
| **Applied Materials** | [Impact Report](https://www.appliedmaterials.com/content/dam/site/company/csr/doc/2025_impact_report.pdf.coredownload.inline.pdf) | [Impact Report](https://www.appliedmaterials.com/content/dam/site/company/csr/doc/2024_impact_report.pdf.coredownload.inline.pdf) | [Sustainability Highlights](https://www.appliedmaterials.com/content/dam/site/company/csr/doc/2023_Sustainability_Highlights.pdf.coredownload.inline.pdf)[^1] | [Sustainability Report](https://www.appliedmaterials.com/content/dam/site/company/csr/doc/2022_Sustainability_F.pdf.coredownload.inline.pdf) |
| **ASML** | [Annual Report](https://www.asml.com/en/investors/annual-report/2025) | [Annual Report](https://www.asml.com/en/investors/annual-report/2024) | [Annual Report](https://www.asml.com/en/investors/annual-report/2023) | [Annual Report](https://www.asml.com/en/investors/annual-report/2022) |
| **KLA** | —[^2] | [Global Impact Report](https://www.kla.com/wp-content/uploads/2024-KLA-Global-Impact-Report.pdf) | [Global Impact Report](https://www.kla.com/wp-content/uploads/KLA-Global-Impact-Report-3.pdf) | [Global Impact Report (archived copy)](https://www.responsibilityreports.com/HostedData/ResponsibilityReportArchive/k/NASDAQ_KLAC_2022.pdf)[^3] |
| **Lam Research** | [Global Impact Report](https://www.lamresearch.com/wp-content/uploads/2026/05/Lam-Research-2025-Global-Impact-Report.pdf) | [Global Impact Report](https://www.lamresearch.com/wp-content/uploads/2025/07/Lam-Research-2024-Global-Impact-Report.pdf) | [ESG Report](https://www.lamresearch.com/wp-content/uploads/2024/06/Lam-Research-2023-ESG-Report.pdf) | [ESG Report](https://www.lamresearch.com/wp-content/uploads/2024/06/Lam-Research-2022-ESG-Report.pdf) |
| **Tokyo Electron** | [Integrated Report](https://www.tel.com/ir/library/ar/pjsoh100000000rc-att/ir2025_all_en.pdf) | [Integrated Report](https://www.tel.com/ir/library/ar/egp82m00000000h7-att/ir2024_all_en.pdf) | [Integrated Report](https://www.tel.com/ir/library/ar/f3gfkt000000003v-att/ir2023_all_en_r.pdf) | [Sustainability Report](https://www.tel.com/sustainability/report/index.html)[^4] |

[^1]: Shorter 'Sustainability Highlights' brochure rather than a full report.
[^2]: No 2025 report published yet at time of writing.
[^3]: The 2022 KLA report was pulled from KLA's own site; linked here via the third-party archive at responsibilityreports.com.
[^4]: TEL's 2022 sustainability content was published as a web page rather than a single downloadable PDF.

---

**[Open the walk-through in a more readable view](https://emo7bf.github.io/SUE/sue_walkthru.html)** if
anything above was hard to follow on GitHub — the HTML version has
rendered math, hover thumbnails for each concept, and full-resolution
figures.
