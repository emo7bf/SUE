"""
scripts/paper/build_pdf.py
--------------------------
Assemble paper/sue.pdf from the figures in paper/figures/ and the prose
below. Uses ReportLab for typography and layout; math is rendered as small
matplotlib-mathtext PNGs and inlined.

Usage:
    python -m scripts.paper.build_pdf
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from PIL import Image as PILImage

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    Image, KeepTogether, PageBreak, Table, TableStyle, Flowable,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parent.parent.parent
FIGDIR = ROOT / "paper" / "figures"
OUT_PDF = ROOT / "paper" / "sue.pdf"
STATS = json.loads((ROOT / "paper" / "stats.json").read_text())
EXEMPLARS_PATH = ROOT / "paper" / "exemplars.json"
EXEMPLARS = json.loads(EXEMPLARS_PATH.read_text(encoding="utf-8")) \
    if EXEMPLARS_PATH.exists() else {}


# ============================================================
# Fonts. Prefer Times-family; fall back to built-in.
# ============================================================
FONT_BODY = "Times-Roman"
FONT_BOLD = "Times-Bold"
FONT_ITAL = "Times-Italic"
FONT_BI   = "Times-BoldItalic"

# Try to upgrade to a real serif if available on this Windows install
for candidate, target in [
    ("C:/Windows/Fonts/times.ttf",   "SerifBody"),
    ("C:/Windows/Fonts/timesbd.ttf", "SerifBold"),
    ("C:/Windows/Fonts/timesi.ttf",  "SerifItalic"),
    ("C:/Windows/Fonts/timesbi.ttf", "SerifBI"),
]:
    p = Path(candidate)
    if p.exists():
        try:
            pdfmetrics.registerFont(TTFont(target, str(p)))
        except Exception:
            pass

if all(name in pdfmetrics.getRegisteredFontNames()
       for name in ("SerifBody", "SerifBold", "SerifItalic", "SerifBI")):
    FONT_BODY, FONT_BOLD, FONT_ITAL, FONT_BI = (
        "SerifBody", "SerifBold", "SerifItalic", "SerifBI")

# --- Math-symbol fallback font (Times lacks glyphs like ∈, ℝ, ⊂, ‖). ---
# DejaVu Serif ships with matplotlib and has good Unicode math coverage.
FONT_MATH = FONT_BODY  # sensible fallback if we can't find DejaVu
try:
    import matplotlib as _mpl
    _dj_path = Path(_mpl.get_data_path()) / "fonts" / "ttf" / "DejaVuSerif.ttf"
    _dj_ital = Path(_mpl.get_data_path()) / "fonts" / "ttf" / "DejaVuSerif-Italic.ttf"
    if _dj_path.exists():
        pdfmetrics.registerFont(TTFont("MathSerif", str(_dj_path)))
        FONT_MATH = "MathSerif"
    if _dj_ital.exists():
        pdfmetrics.registerFont(TTFont("MathSerifItalic", str(_dj_ital)))
except Exception:
    pass

def MS(fragment: str) -> str:
    """Wrap a fragment (containing Unicode math symbols) in the math-fallback
    font so glyphs missing from Times render correctly."""
    return f'<font name="{FONT_MATH}">{fragment}</font>'


INK = HexColor("#1F1F1F")
INK_LIGHT = HexColor("#555555")
RULE = HexColor("#B0B0B0")
ACCENT = HexColor("#B8860B")


# ============================================================
# Styles
# ============================================================

def make_styles():
    body = ParagraphStyle(
        "body", fontName=FONT_BODY, fontSize=10.5, leading=13.5,
        alignment=TA_JUSTIFY, spaceAfter=6, textColor=INK,
        firstLineIndent=0)
    body_indent = ParagraphStyle(
        "body_indent", parent=body, firstLineIndent=14)
    title = ParagraphStyle(
        "title", fontName=FONT_BOLD, fontSize=19, leading=24,
        alignment=TA_LEFT, textColor=INK, spaceAfter=6)
    subtitle = ParagraphStyle(
        "subtitle", fontName=FONT_ITAL, fontSize=12, leading=15,
        alignment=TA_LEFT, textColor=INK_LIGHT, spaceAfter=18)
    author = ParagraphStyle(
        "author", fontName=FONT_BODY, fontSize=11, leading=14,
        alignment=TA_LEFT, textColor=INK, spaceAfter=4)
    section = ParagraphStyle(
        "section", fontName=FONT_BOLD, fontSize=13, leading=16,
        alignment=TA_LEFT, textColor=INK,
        spaceBefore=14, spaceAfter=6)
    subsection = ParagraphStyle(
        "subsection", fontName=FONT_BOLD, fontSize=11, leading=14,
        alignment=TA_LEFT, textColor=INK,
        spaceBefore=10, spaceAfter=4)
    abstract_head = ParagraphStyle(
        "abstract_head", fontName=FONT_BOLD, fontSize=10, leading=13,
        alignment=TA_LEFT, textColor=INK, spaceAfter=2)
    abstract = ParagraphStyle(
        "abstract", parent=body, fontSize=10, leading=13,
        leftIndent=18, rightIndent=18, spaceAfter=6)
    caption = ParagraphStyle(
        "caption", fontName=FONT_ITAL, fontSize=9, leading=11.5,
        alignment=TA_CENTER, textColor=INK, spaceBefore=4, spaceAfter=14,
        leftIndent=24, rightIndent=24)
    figref = ParagraphStyle(
        "figref", parent=caption, fontName=FONT_BI)
    pullquote = ParagraphStyle(
        "pullquote", fontName=FONT_ITAL, fontSize=9.5, leading=12.5,
        alignment=TA_LEFT, textColor=INK,
        leftIndent=28, rightIndent=18, spaceBefore=4, spaceAfter=6,
        borderPadding=0)
    pullquote_meta = ParagraphStyle(
        "pullquote_meta", fontName=FONT_BODY, fontSize=8, leading=10,
        alignment=TA_LEFT, textColor=INK_LIGHT,
        leftIndent=28, rightIndent=18, spaceAfter=10)
    ref = ParagraphStyle(
        "ref", fontName=FONT_BODY, fontSize=9, leading=11.5,
        alignment=TA_LEFT, textColor=INK,
        leftIndent=16, firstLineIndent=-16, spaceAfter=3)
    return dict(body=body, body_indent=body_indent, title=title,
                subtitle=subtitle, author=author,
                section=section, subsection=subsection,
                abstract_head=abstract_head, abstract=abstract,
                caption=caption, figref=figref,
                pullquote=pullquote, pullquote_meta=pullquote_meta,
                ref=ref)


# ============================================================
# Math rendering via matplotlib mathtext -> PNG in-memory
# ============================================================

def math_png(tex: str, fontsize: int = 12, dpi: int = 300) -> Image:
    """Render a LaTeX-ish math string as a ReportLab Image."""
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0.0)
    t = fig.text(0, 0, f"${tex}$", fontsize=fontsize,
                 color="#1F1F1F", va="baseline")
    fig.canvas.draw()
    bbox = t.get_window_extent()
    w_in = bbox.width  / dpi
    h_in = bbox.height / dpi
    fig.set_size_inches(w_in + 0.02, h_in + 0.02)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True,
                bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    buf.seek(0)
    pil = PILImage.open(buf)
    w_pt = pil.width  * 72 / dpi
    h_pt = pil.height * 72 / dpi
    img = Image(buf, width=w_pt, height=h_pt)
    return img


class NumberedEquation(Flowable):
    """A centered display equation with a right-aligned (n) tag."""
    def __init__(self, tex: str, number: int, page_width: float,
                 fontsize: int = 13):
        Flowable.__init__(self)
        self.img = math_png(tex, fontsize=fontsize)
        self.number = number
        self.page_width = page_width
        self.height = self.img.drawHeight + 8
        self.width = page_width

    def draw(self):
        c = self.canv
        img_w = self.img.drawWidth
        img_h = self.img.drawHeight
        # Centered image
        x = (self.width - img_w) / 2.0
        y = 4
        self.img.drawOn(c, x, y)
        # Right-aligned number (skip for definitional equations)
        if self.number:
            c.setFont(FONT_BODY, 10)
            c.setFillColor(INK)
            c.drawRightString(self.width - 4, y + img_h / 2 - 3,
                              f"({self.number})")

    def wrap(self, aw, ah):
        return self.width, self.height


# ============================================================
# Page template: running header + footer + rule
# ============================================================

TITLE_SHORT = "SUE: Looking Before You Build"
AUTHOR = "Elizabeth Orrico"

def draw_page_frame(canvas, doc):
    canvas.saveState()
    # Header text
    canvas.setFont(FONT_ITAL, 9)
    canvas.setFillColor(INK_LIGHT)
    canvas.drawString(1.0 * inch, LETTER[1] - 0.6 * inch, TITLE_SHORT)
    canvas.drawRightString(LETTER[0] - 1.0 * inch,
                           LETTER[1] - 0.6 * inch, AUTHOR)
    # Header rule
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(1.0 * inch, LETTER[1] - 0.72 * inch,
                LETTER[0] - 1.0 * inch, LETTER[1] - 0.72 * inch)
    # Footer page number
    canvas.setFont(FONT_BODY, 9)
    canvas.setFillColor(INK_LIGHT)
    canvas.drawCentredString(LETTER[0] / 2.0, 0.55 * inch,
                             f"{doc.page}")
    canvas.restoreState()


def draw_title_page_frame(canvas, doc):
    # Title page: no header, just page number
    canvas.saveState()
    canvas.setFont(FONT_BODY, 9)
    canvas.setFillColor(INK_LIGHT)
    canvas.drawCentredString(LETTER[0] / 2.0, 0.55 * inch,
                             f"{doc.page}")
    canvas.restoreState()


# ============================================================
# Figure helper
# ============================================================

def figure_block(name: str, number: int, caption_text: str,
                 styles, target_width: float = 6.2 * inch) -> List:
    """Return a KeepTogether flowable: image + caption."""
    path = FIGDIR / f"{name}.png"
    pil = PILImage.open(path)
    ratio = pil.height / pil.width
    w = target_width
    h = w * ratio
    img = Image(str(path), width=w, height=h)
    cap = Paragraph(
        f"<font name='{FONT_BI}'>Figure {number}: </font>"
        f"<i>{caption_text}</i>",
        styles["caption"])
    return [KeepTogether([Spacer(1, 4), img, cap])]


def schematic_block(name: str, label: str, caption_text: str,
                    styles, target_width: float = 6.2 * inch) -> List:
    """Schematic diagram with a lettered label (e.g. "Diagram A")."""
    path = FIGDIR / f"{name}.png"
    pil = PILImage.open(path)
    ratio = pil.height / pil.width
    w = target_width
    h = w * ratio
    img = Image(str(path), width=w, height=h)
    cap = Paragraph(
        f"<font name='{FONT_BI}'>Diagram {label}: </font>"
        f"<i>{caption_text}</i>",
        styles["caption"])
    return [KeepTogether([Spacer(1, 4), img, cap])]


def pullquote(text: str, meta: str, styles) -> List:
    return [
        Paragraph(f"&#8220;{text}&#8221;", styles["pullquote"]),
        Paragraph(meta, styles["pullquote_meta"]),
    ]


def hbar(width: float, color=RULE, thickness: float = 0.4) -> Flowable:
    class _Hr(Flowable):
        def __init__(self):
            Flowable.__init__(self)
            self.width = width
            self.height = 2
        def draw(self):
            self.canv.setStrokeColor(color)
            self.canv.setLineWidth(thickness)
            self.canv.line(0, 1, self.width, 1)
        def wrap(self, aw, ah):
            return self.width, self.height
    return _Hr()


# ============================================================
# Prose blocks — the paper's actual text
# ============================================================

def build_story(styles) -> List:
    s = styles
    body = s["body"]
    body_i = s["body_indent"]
    section = s["section"]
    subsection = s["subsection"]

    N = STATS["n_chunks"]
    NDOC = STATS["n_docs"]
    NCOMP = STATS["n_companies"]
    companies = ", ".join(STATS["companies"])
    ANISO = STATS.get("anisotropy_mean", 0.0)
    ISO = STATS.get("isotropic_mean", 0.0)
    PC1 = STATS.get("pc1_variance_ratio", 0.0)
    TOP10 = STATS.get("cum_top10_variance", 0.0)
    DBIC = STATS.get("delta_bic", 0.0)
    COS_RAND = STATS.get("cos_mean_random", 0.0)
    COS_SAMEDOC = STATS.get("cos_mean_same_doc", 0.0)
    COS_SAMECO = STATS.get("cos_mean_same_company", 0.0)
    COS_DIFFCO = STATS.get("cos_mean_diff_company", 0.0)
    NPROSE = STATS.get("n_prose") or 0
    NTABLE = STATS.get("n_table") or 0

    story: List = []

    # ---------------- Title block ----------------
    story += [
        Paragraph("SUE: Looking Before You Build", s["title"]),
        Paragraph("A visual diagnostic study of corporate sustainability "
                  "report embeddings", s["subtitle"]),
        Paragraph(AUTHOR, s["author"]),
        Paragraph("Corpus preview edition &#8212; Semiconductors subset",
                  ParagraphStyle("date", parent=s["author"],
                                 fontName=FONT_ITAL,
                                 textColor=INK_LIGHT)),
        Spacer(1, 18),
        hbar(6.5 * inch),
        Spacer(1, 10),
    ]

    # ---------------- Abstract ----------------
    story += [
        Paragraph("Abstract", s["abstract_head"]),
        Paragraph(
            f"We embed {N:,} chunks drawn from {NDOC} publicly-available "
            f"sustainability, ESG, and integrated reports "
            f"({NCOMP} companies in the semiconductor equipment sector) "
            "into the 384-dimensional MiniLM sentence-embedding space "
            "and ask a diagnostic rather than a modeling question: "
            "<i>what shape does this corpus have, and does that shape "
            "line up with properties an author would name for it?</i> "
            "Our central observation is elementary and generalisable: "
            "when the content of a document class shifts, the mean "
            "embedding of that class shifts with it, and the "
            "displacement between class centroids is itself an "
            "interpretable signal. In the present corpus, the "
            "financial-disclosure passages of an integrated report and "
            "its narrative ESG passages sit at measurably distinct "
            "centroids, and a single linear direction separates them "
            f"cleanly ({NPROSE:,} vs. {NTABLE:,} chunks). The same "
            "direction accounts for a bimodal Gaussian-mixture fit "
            f"(&Delta;BIC = {DBIC:,.0f} in the top-20 principal-"
            "component subspace) that is invisible to any per-company "
            "or per-year partition. We argue that this <i>centroid-"
            "shift-as-signal</i> pattern is a general-purpose "
            "diagnostic for text collections whose author-declared "
            "structure ought to imprint on their embedding-space "
            "geometry, and we lay out its mathematical scaffolding "
            "(cosine, anisotropy, PCA, perplexity, BIC, Fisher&#8217;s "
            "ratio) before showing the corpus itself.",
            s["abstract"]),
        Spacer(1, 6),
        hbar(6.5 * inch),
        Spacer(1, 12),
    ]

    # ---------------- 1. Introduction ----------------
    story += [Paragraph("1 &#160; Introduction", section)]

    story += [Paragraph("1.1 &#160; What a Global Impact Report is",
                        subsection)]
    story += [Paragraph(
        "A <i>Global Impact Report</i> &#8212; variously titled "
        "&#8220;Sustainability Report,&#8221; "
        "&#8220;ESG Report,&#8221; "
        "&#8220;Corporate Responsibility Report,&#8221; or, when "
        "consolidated with the annual financial disclosure, "
        "&#8220;Integrated Report&#8221; &#8212; is an annual, "
        "predominantly non-financial disclosure document that a "
        "publicly-traded firm publishes for its investors, "
        "regulators, employees, customers, and civil society. Until "
        "roughly the middle of the last decade its publication was "
        "almost entirely voluntary; today its content is governed "
        "de facto by a growing patchwork of frameworks (GRI, SASB, "
        "TCFD, and the ISSB standards that have begun consolidating "
        "them) and, in some jurisdictions, de jure by regulation "
        "&#8212; most prominently the European Union&#8217;s "
        "Corporate Sustainability Reporting Directive (CSRD), which "
        "began phasing in mandatory disclosure obligations for "
        "large firms in 2024.",
        body)]
    story += [Paragraph(
        "The document is a genre. Any given report will typically "
        "combine, in one PDF running 60 to 200 pages, several "
        "recurring content types: an opening letter from the chief "
        "executive framing the year&#8217;s priorities; a "
        "materiality assessment naming the topics the firm considers "
        "most consequential to its business and to its stakeholders; "
        "quantitative disclosure of greenhouse-gas emissions "
        "partitioned into Scope 1 (direct), Scope 2 (purchased "
        "energy), and Scope 3 (value-chain) categories; water "
        "withdrawal and consumption; waste and circularity metrics; "
        "workforce composition and safety statistics; supplier "
        "auditing and human-rights due-diligence procedures; "
        "governance disclosures covering board independence and "
        "committee structure; and a set of forward-looking targets, "
        "typically framed against a base year and a horizon (for "
        "example &#8220;a 50% absolute reduction in Scope 1 and 2 "
        "emissions from a 2019 baseline by 2030&#8221;).",
        body_i)]
    story += schematic_block(
        "fig00_report_anatomy", "A",
        "The twelve content types that recur across essentially "
        "every Global Impact Report in the corpus, coloured by "
        "rhetorical register. Narrative-heavy blocks (blue) read "
        "like corporate prose; tabular-heavy blocks (orange) are "
        "flattened spreadsheets of quantitative disclosure; mixed "
        "blocks (purple) combine both. The class-centroid axis of "
        "\u00a72 is, empirically, the axis that separates the "
        "orange half from the blue half.", s)
    story += [Paragraph(
        "For a study of embedding-space geometry this genre is "
        "uniquely useful. Two rhetorical registers are placed inside "
        "the same document, on the same topics, at roughly the same "
        "length: <i>narrative</i> passages, which read like corporate "
        "prose, and <i>tabular</i> passages, which read like a "
        "spreadsheet flattened into text during PDF extraction. Both "
        "registers describe the same underlying commitments and the "
        "same underlying measurements. A sentence encoder that "
        "represents meaning ought to treat them as close; a sentence "
        "encoder that represents surface form ought to separate them. "
        "The corpus therefore admits a diagnostic question that most "
        "text collections do not: <i>which of these two things is our "
        "encoder actually doing?</i>",
        body_i)]

    # --- 1.2 What is at stake in reading these documents well ---
    story += [Paragraph("1.2 &#160; What is at stake in reading these "
                        "documents well", subsection)]
    story += [Paragraph(
        "Two questions animate the rest of the paper. The first is a "
        "<i>disclosure</i> question: for any given report, how many "
        "of its chunks land on the numeric side of the "
        "narrative\u2194tabular axis versus the narrative side, and "
        "what is each side actually saying? Are targets framed "
        "against comparable baselines, or is one firm quietly using "
        "a 2019 base year while another uses 2015? The class-centroid "
        "decomposition of \u00a72 gives a first-cut, quantitative "
        "answer, and converts a stack of 3,000-page reports into an "
        "auditable quantity.", body)]
    story += [Paragraph(
        "The second is a <i>representation</i> question: when an "
        "off-the-shelf sentence encoder is handed text of this genre "
        "&#8212; two rhetorical registers describing the same "
        "commitments on the same topics inside the same document "
        "&#8212; does it treat them as close (representing meaning) "
        "or does it separate them (representing surface form)? We "
        "will show empirically that it separates them: a single, "
        "learnable direction in embedding space cleanly divides "
        "prose from tabular chunks. That direction is a discovered "
        "representational axis at the corpus level, of the same "
        "family as the axes mechanistic interpretability recovers at "
        "the token or circuit level.", body_i)]
    story += [Paragraph(
        "Both questions turn out to be the same question in "
        "different vocabularies: <i>what is inside these documents, "
        "and where is the boundary between what a system claims to "
        "represent and what it actually represents?</i>", body_i)]

    # --- 1.3 (was 1.2) ---
    story += [Paragraph("1.3 &#160; Why look at the vectors before "
                        "building on top of them", subsection)]
    story += [Paragraph(
        "Modern retrieval-augmented systems treat a corpus as a bag of "
        "vectors: each chunk is embedded, indexed, and retrieved by "
        "cosine similarity to a query embedding. The design of such a "
        "system rarely returns to look at the vectors themselves. This "
        "paper argues, through a small, replicable case study, that "
        "<i>the vectors are worth looking at</i>: naive assumptions "
        "about corpus homogeneity fail on data that would seem, on "
        "paper, to be as homogeneous as it gets.",
        body)]
    story += [Paragraph(
        "Our corpus is a set of Global Impact, Environmental Social and "
        "Governance, and Integrated Reports published by the "
        f"five largest semiconductor-equipment firms: {companies}. "
        "Every document is written for a comparable audience &#8212; "
        "investors, regulators, employees, journalists &#8212; on a "
        "comparable set of topics &#8212; emissions, water, supply "
        "chain, workforce, governance. The naive prediction is that "
        "the sentence embeddings of chunks drawn from these documents "
        "form a single blob: a lot of overlap, little separation. "
        "That prediction is wrong, and its being wrong is instructive.",
        body_i)]
    story += [Paragraph(
        "The paper is organised around a single claim and its "
        "consequences. \u00a72 states the claim in its most compact "
        "form: for a text collection whose authors mix identifiable "
        "content classes into one document, the class centroids in "
        "embedding space are displaced from one another, and that "
        "displacement is a rich, interpretable signal. "
        "\u00a73 is a mathematical warm-up &#8212; a short primer on "
        "the objects (cosine similarity, variance, anisotropy, "
        "perplexity, BIC, Fisher&#8217;s ratio) that the rest of the "
        "paper is written in terms of. "
        "\u00a74 introduces the corpus itself. "
        "\u00a75 exhibits the first-order surprise the abstract "
        "advertised: the point cloud is bimodal, and the split is not "
        "by company or year. "
        "\u00a76 quantifies anisotropy and clarifies what PCA does and "
        "does not see. "
        "\u00a77 replays the corpus under nonlinear projections "
        "(t-SNE, UMAP). "
        "\u00a78 discusses what changes downstream once the centroid "
        "structure is admitted, and \u00a79 sketches the research "
        "directions this study opens &#8212; including where else the "
        "centroid-shift diagnostic is likely to be productive.",
        body_i)]

    # --- 1.4 How to read this paper (NEW) ---
    story += [Paragraph("1.4 &#160; How to read this paper", subsection)]
    story += [Paragraph(
        "This paper is written with a reader in mind who is comfortable "
        "reading, but not necessarily fluent in, the mathematics of "
        "high-dimensional statistics. Every equation is introduced by "
        "an intuition first, unpacked symbol by symbol second, and "
        "used in a figure third. If a piece of notation feels new "
        "&#8212; the covariance matrix, Bayesian information "
        "criterion, perplexity, Fisher&#8217;s discriminant ratio "
        "&#8212; it is defined in \u00a73 before it is used later. "
        "Readers already fluent in this material can skim or skip "
        "\u00a73 without missing the argument.", body)]
    story += [Paragraph(
        "Every empirical claim in the paper is accompanied by either "
        "a figure or a quotation of a real chunk from the corpus, so "
        "that a reader can audit the claim rather than take it on "
        "trust. The corpus itself, the embedding matrix, and the code "
        "that produces every figure and number in this document are "
        "included in the accompanying repository.", body_i)]

    # ================================================================
    # 2. Centroids as a data-story engine (NEW &#8212; central thesis)
    # ================================================================
    story += [Paragraph("2 &#160; Centroids as a data-story engine",
                        section)]
    story += [Paragraph(
        "The technical measurements in the remainder of this paper "
        "all serve a single, elementary claim. State it first so the "
        "figures can be read as evidence for it, rather than as a "
        "sequence of unmotivated diagnostics.", body)]
    story += [Paragraph(
        "<b>Claim.</b> Given a text corpus <i>C</i> and a partition "
        "of its chunks into content-defined classes "
        r"<i>C</i> = <i>C</i><sub>1</sub> &#8746; &#8230; &#8746; "
        "<i>C</i><sub><i>k</i></sub>, and any well-behaved sentence "
        "encoder <i>E</i>, the class centroids",
        body_i)]
    story += [NumberedEquation(
        r"\mu_j \ \ =\ \ \frac{1}{|C_j|}\sum_{x \in C_j} E(x) "
        r"\ \ \in\ \ \mathbb{R}^{384}",
        number=1, page_width=6.5 * inch, fontsize=13)]
    story += [Paragraph(
        "are displaced from one another whenever the classes are "
        "genuinely different in content, and the displacements "
        r"<i>&#956;<sub>i</sub></i> &#8722; <i>&#956;<sub>j</sub></i> "
        "carry interpretable information about how the classes differ. "
        "Shifted content \u2192 shifted embeddings \u2192 shifted "
        "centroid. The equation is a bookkeeping identity; the "
        "empirical claim is that in real corpora the displacement is "
        "measurable, stable, and human-readable.", body)]
    story += [Paragraph(
        "This is not a subtle statement. Its power is that the "
        "converse chain runs in the useful direction: <i>a measurable "
        "centroid displacement in embedding space is evidence of a "
        "content distinction that a partition may or may not already "
        "know about</i>. If the classes are known in advance, the "
        "displacement quantifies their separability. If the classes "
        "are not known, unsupervised clustering that recovers the "
        "same displacement retrieves the latent content axis. The "
        "rest of this paper does both, on one worked corpus.", body_i)]

    story += [Paragraph("2.1 &#160; The minimal example: financial "
                        "disclosure vs. narrative ESG", subsection)]
    story += [Paragraph(
        "The corpus that motivates the paper is a set of integrated, "
        "sustainability, and ESG reports (\u00a74) that mixes, within "
        "the same document, two content classes an author would "
        "readily name: <i>financial-disclosure passages</i> "
        "&#8212; income statements, balance-sheet fragments, "
        "line-item tables of costs, revenues, and emissions figures "
        "&#8212; and <i>narrative ESG passages</i> &#8212; prose "
        "about strategy, materiality, and progress against targets. "
        "Both classes describe overlapping subject matter, in the "
        "same document, at comparable length. Only the register "
        "differs. We assign each chunk to one class by a light "
        "regex-derived label (digit density, line-length statistics), "
        f"which yields {NPROSE:,} narrative chunks and {NTABLE:,} "
        "tabular chunks.", body)]
    story += [Paragraph(
        "Linear discriminant analysis on this two-class label "
        "recovers a single direction "
        f"<i>w</i> {MS('&#8712;')} {MS('&#8477;')}<super>384</super> "
        "along which the two class centroids are maximally separated "
        "relative to their internal spread. Figure 1 projects the "
        "corpus onto that one-dimensional axis; the two class "
        "distributions are cleanly separated, and their means "
        "&#8212; the projections of "
        r"<i>&#956;</i><sub>narrative</sub> and "
        r"<i>&#956;</i><sub>tabular</sub> &#8212; sit far apart on a "
        "scale where within-class spread is small. Content shifts; "
        "centroids shift; the shift is visible in one dimension.",
        body_i)]
    story += figure_block(
        "fig11_lda_prose_table", 1,
        "Projection of every corpus chunk onto the single "
        "direction that maximises the ratio of between-class to "
        "within-class scatter (Fisher&#8217;s ratio, defined in "
        "\u00a73.7). The two class centroids &#8212; narrative and "
        "tabular &#8212; are displaced along this axis by many "
        "within-class standard deviations. This is the paper&#8217;s "
        "central image.", s)

    story += [Paragraph("2.2 &#160; Why the centroid displacement is "
                        "the whole story", subsection)]
    story += [Paragraph(
        "Two independent lines of evidence, developed in later "
        "sections, converge on the same axis. First, an unsupervised "
        "Gaussian-mixture fit to the raw embeddings (\u00a75) "
        "discovers two components without being told about the "
        "regex labels. Second, when the corpus is projected onto the "
        "supervised centroid axis of Figure 1 and coloured by the "
        "unsupervised cluster assignment (Figure 2 below), the two "
        "clusterings coincide almost exactly. The label the corpus "
        "would give itself is the label the geometry has already "
        "found.", body)]
    story += figure_block(
        "fig12_bimodality_on_prosetable", 2,
        "Chunks re-projected onto the supervised centroid axis "
        "of Figure 1, coloured by the <i>unsupervised</i> two-"
        "component Gaussian mixture fit of \u00a75. The two "
        "assignment schemes agree: the bimodality of the raw "
        "embedding cloud lives on the class-centroid displacement.",
        s)

    story += [Paragraph("2.3 &#160; Why this generalises", subsection)]
    story += [Paragraph(
        "Nothing in the claim is specific to sustainability "
        "disclosure. The same construction applies to any corpus "
        "whose author-declared structure mixes identifiable content "
        "classes into a single stream of text: scientific papers "
        "(methods vs. discussion), medical records (structured "
        "history vs. free-text note), legal filings (statutory "
        "citation vs. argument), transcripts (speaker A vs. speaker "
        "B), or the technical vs. rhetorical passages of a research "
        "grant. In each case the centroid displacement "
        r"<i>&#956;<sub>i</sub></i> &#8722; "
        "<i>&#956;<sub>j</sub></i> is a one-line diagnostic that can "
        "be computed, plotted, and audited before any retrieval or "
        "downstream model is built on top of the embeddings. Where a "
        "domain lacks an author-supplied label, an unsupervised "
        "mixture fit followed by centroid inspection recovers the "
        "same information; we execute both routes in \u00a75. We "
        "return to further application domains in \u00a79.", body_i)]

    story += [Paragraph(
        "The rest of the paper is scaffolding for this claim: "
        "\u00a73 defines the mathematical objects, \u00a74 introduces "
        "the corpus, \u00a75\u2013\u00a77 measure the geometry that "
        "makes the claim rigorous, \u00a78 discusses downstream "
        "implications, and \u00a79 catalogues where the diagnostic "
        "should be pushed next.", body)]

    # ================================================================
    # 3. Mathematical warm-up (was 2)
    # ================================================================
    story += [Paragraph("3 &#160; Mathematical warm-up", section)]
    story += [Paragraph(
        "This section defines every object the rest of the paper uses, "
        "in plain language and with a picture in mind before any "
        "algebra. A reader already fluent in embeddings, PCA, entropy, "
        "and BIC can skim; a reader for whom any of the following "
        "objects is new should read this section carefully &#8212; "
        "every subsequent figure is stated in these terms.", body)]

    # 2.1 Embeddings
    story += [Paragraph("3.1 &#160; What an embedding actually is",
                        subsection)]
    story += [Paragraph(
        "An <i>embedding</i> is a function that maps a piece of text "
        "&#8212; a word, a sentence, or in our case a "
        "~900-character chunk &#8212; to a point in a "
        "high-dimensional space. For us that space is "
        f"{MS('&#8477;')}<super>384</super>: every chunk becomes a list of 384 "
        "real numbers. Formally, if <i>T</i> is the set of possible "
        "text inputs, an embedding is a map",
        body)]
    story += [NumberedEquation(
        r"E : T \ \ \longrightarrow\ \ \mathbb{R}^{384},",
        number=None, page_width=6.5 * inch, fontsize=13)]
    # We use a hidden numbering scheme for math-primer equations,
    # renumbered later so the reader-facing counter is contiguous.
    story += [Paragraph(
        "and the numbers themselves are meaningless in isolation; what "
        "matters is <i>relative</i> geometry. If two chunks land at "
        "nearby points, the encoder is claiming they mean similar "
        "things. If they land far apart, the encoder is claiming they "
        "mean different things. The goal of training an encoder is "
        "precisely to arrange the mapping so that this geometric "
        "claim is reliable.", body_i)]
    story += [Paragraph(
        "The encoder we use, <i>all-MiniLM-L6-v2</i>, additionally "
        "<i>unit-normalises</i> its outputs: after producing "
        f"<i>x</i> {MS('&#8712;')} {MS('&#8477;')}<super>384</super>, it divides by "
        f"the length {MS('&#8214;')}<i>x</i>{MS('&#8214;')} so that every embedded "
        "chunk lies on the surface of the unit sphere "
        f"<i>S</i><super>383</super> "
        f"{MS('&#8834;')} {MS('&#8477;')}<super>384</super>. This means every "
        "geometric statement we will make can be made either as a "
        "statement about angles or as a statement about distances "
        "&#8212; the two are the same object on the sphere.", body)]

    # 2.2 Cosine
    story += [Paragraph("3.2 &#160; Cosine similarity, three ways",
                        subsection)]
    story += [Paragraph(
        "The <i>cosine similarity</i> of two vectors is the cosine of "
        "the angle between them, viewed as arrows from the origin. "
        "Three equivalent expressions are worth having in mind:", body)]
    story += [NumberedEquation(
        r"\cos(a, b)\ \ =\ \ \frac{\langle a, b \rangle}{\|a\|\,\|b\|}"
        r"\ \ =\ \ \langle \hat a,\hat b \rangle"
        r"\ \ =\ \ 1 - \frac{1}{2}\|\hat a - \hat b\|^2,",
        number=None, page_width=6.5 * inch, fontsize=13)]
    story += [Paragraph(
        f"where <i>{MS('&#10216;')}a, b{MS('&#10217;')}</i> is the dot product "
        f"<i>{MS('&#8721;')} a<sub>i</sub> b<sub>i</sub></i>, and "
        f"<i>{MS('â')}</i>, <i>b{MS('&#770;')}</i> are the unit-normalised "
        "versions of <i>a</i>, <i>b</i>. The three forms say the "
        "same thing: cosine similarity is the dot product of the "
        "normalised vectors, and on the unit sphere, cosine and "
        "Euclidean distance are locked together. Its range is "
        "[&#8722;1, 1]: <b>1</b> when the two vectors point the same "
        "way, <b>0</b> when they are perpendicular, and <b>&#8722;1</b> "
        "when they point opposite. For random unit vectors in a "
        "well-behaved (isotropic) space, the expected value is "
        "<b>0</b>. We will use that fact as a diagnostic in \u00a76.1.",
        body_i)]

    # 2.3 Variance, covariance, PCA
    story += [Paragraph("3.3 &#160; Variance, covariance, "
                        "and principal components", subsection)]
    story += [Paragraph(
        "The <i>variance</i> of a random variable measures how spread "
        "out its values are, on average, around its mean. In one "
        "dimension:", body)]
    story += [NumberedEquation(
        r"\mathrm{Var}(X)\ \ =\ \ \mathbb{E}\left[(X - \mu)^2\right].",
        number=None, page_width=6.5 * inch, fontsize=13)]
    story += [Paragraph(
        "Zero variance means every sample is exactly the mean; large "
        "variance means samples are far-flung. The two-variable "
        "generalisation is <i>covariance</i>:", body_i)]
    story += [NumberedEquation(
        r"\mathrm{Cov}(X, Y)\ \ =\ \ \mathbb{E}\left[(X - \mu_X)(Y - \mu_Y)\right],",
        number=None, page_width=6.5 * inch, fontsize=13)]
    story += [Paragraph(
        "which is positive when <i>X</i> and <i>Y</i> tend to rise "
        "and fall together, negative when they tend to move in "
        "opposite directions, and zero when they are linearly "
        "unrelated. Stacking these covariances into a matrix for a "
        "384-dimensional vector gives the "
        f"<i>covariance matrix</i> &#931; {MS('&#8712;')} "
        f"{MS('&#8477;')}<super>384&#215;384</super>, whose entry "
        "&#931;<sub><i>ij</i></sub> is Cov(<i>X<sub>i</sub></i>, "
        "<i>X<sub>j</sub></i>).", body)]
    story += [Paragraph(
        "<i>Principal component analysis</i> (PCA) diagonalises &#931;. "
        "Its eigenvectors are called <i>principal components</i>; each "
        "eigenvalue tells us how much of the total variance lies along "
        "its eigenvector. PC1 is the direction along which the "
        "corpus is most spread out, PC2 is the next direction "
        "orthogonal to it along which it is next-most spread out, "
        "and so on. Projecting a 384-dimensional cloud onto its "
        "first two principal components gives the flat picture that "
        "every subsequent scatter plot in this paper is drawn on. "
        "One caveat is worth flagging now, and returning to in "
        "\u00a76.2: <b>PCA subtracts the mean before it looks for "
        "variance</b>. Any structure that lives in the mean direction "
        "of the corpus is, by construction, invisible to PCA.", body_i)]

    # 2.4 Anisotropy
    story += [Paragraph("3.4 &#160; Anisotropy: when the cloud has a "
                        "preferred direction", subsection)]
    story += [Paragraph(
        "A cloud of points is <i>isotropic</i> if it has no preferred "
        "direction &#8212; a spherical swarm of gnats. It is "
        "<i>anisotropic</i> if it does &#8212; a school of fish "
        "swimming north. Formally, a corpus of unit vectors is "
        "isotropic when",
        body)]
    story += [NumberedEquation(
        r"\mathbb{E}_{i \ne j}\left[\cos(x_i, x_j)\right]\ \ \approx\ \ 0,",
        number=None, page_width=6.5 * inch, fontsize=13)]
    story += [Paragraph(
        "and anisotropic when this expectation is strictly positive. "
        "Sentence encoders trained with self-supervised objectives "
        "are known to be strongly anisotropic (Ethayarajh, 2019): "
        "their outputs occupy a narrow cone rather than filling the "
        "sphere. Practically, this means the &#8220;baseline&#8221; "
        "cosine similarity between two unrelated chunks is not 0 but "
        "something distinctly positive, and every retrieval decision "
        "that a downstream system makes is being made against that "
        "offset. We measure the offset directly in \u00a76.1.", body_i)]

    # 2.5 Entropy, perplexity, KL
    story += [Paragraph("3.5 &#160; Entropy, perplexity, and "
                        "effective neighbourhood size", subsection)]
    story += [Paragraph(
        "Perplexity is a knob in t-SNE (\u00a77) whose meaning is "
        "not obvious from its name. It comes from information theory, "
        "and it is easiest to explain by first defining "
        "<i>entropy</i>.", body)]
    story += [Paragraph(
        "The Shannon entropy of a probability distribution "
        "<i>P</i> = (<i>p</i><sub>1</sub>, &#8230;, "
        "<i>p<sub>k</sub></i>) over <i>k</i> outcomes is",
        body_i)]
    story += [NumberedEquation(
        r"H(P)\ \ =\ \ -\sum_{i=1}^{k} p_i \log_2 p_i.",
        number=None, page_width=6.5 * inch, fontsize=13)]
    story += [Paragraph(
        "It is measured in bits, and the intuition is: <i>on average, "
        "how many yes/no questions do I need to ask to identify a "
        "sample from this distribution?</i> A uniform distribution "
        "over 8 outcomes has entropy log&#8322; 8 = 3 bits (three "
        "yes/no questions is exactly what a binary search needs). A "
        "distribution concentrated on a single outcome has entropy 0. "
        "Entropy is a measure of how uncertain, or how spread out, "
        "a distribution is.", body)]
    story += [Paragraph(
        "<i>Perplexity</i> is simply the exponential of entropy:",
        body_i)]
    story += [NumberedEquation(
        r"\mathrm{Perp}(P)\ \ =\ \ 2^{H(P)}.",
        number=None, page_width=6.5 * inch, fontsize=13)]
    story += [Paragraph(
        "It converts &#8220;bits of uncertainty&#8221; into an "
        "<i>effective number of choices</i>. A fair coin has entropy "
        "1 bit and perplexity 2 (two equally-plausible outcomes). A "
        "99/1 biased coin has entropy near 0 and perplexity near 1 "
        "(effectively one outcome). A uniform distribution over 30 "
        "outcomes has perplexity 30. Perplexity is entropy in a "
        "unit the eye is trained to read.", body)]
    story += [Paragraph(
        "In t-SNE, we do the following: for each point <i>x<sub>i</sub></i>, "
        "we place a Gaussian &#8220;attention&#8221; distribution "
        "over all other points, with the Gaussian&#8217;s bandwidth "
        "<i>&#963;<sub>i</sub></i> tuned so that the resulting "
        "distribution has some chosen perplexity. Because perplexity "
        "is &#8220;effective number of neighbours,&#8221; setting "
        "perplexity = 30 is telling the algorithm <i>each point "
        "should care about roughly 30 nearest neighbours when "
        "deciding where to land in the 2-D map</i>. Small perplexity "
        f"{MS('&#8594;')} hyper-local emphasis, many small clumps. Large "
        f"perplexity {MS('&#8594;')} global emphasis, one big shape. "
        "\u00a77 renders the same corpus at four perplexity choices "
        "side by side.", body_i)]
    story += [Paragraph(
        "One more information-theoretic object is worth introducing "
        "briefly, because t-SNE and its cousins optimise it: the "
        "<i>Kullback&#8211;Leibler divergence</i> from a "
        "distribution <i>Q</i> to a distribution <i>P</i>,",
        body_i)]
    story += [NumberedEquation(
        r"D_{\mathrm{KL}}(P\,\|\,Q)\ \ =\ \ "
        r"\sum_{i} p_i \log \frac{p_i}{q_i},",
        number=None, page_width=6.5 * inch, fontsize=13)]
    story += [Paragraph(
        "measures how surprised you would be, on average, if you "
        "believed the world was <i>Q</i> and it turned out to be "
        "<i>P</i>. It is zero when <i>P</i> = <i>Q</i> and grows "
        "otherwise. t-SNE builds a neighbourhood distribution "
        "<i>P</i> in the original 384-dimensional space and a "
        "neighbourhood distribution <i>Q</i> in the 2-D map, and "
        "arranges the 2-D coordinates so that <i>D</i><sub>KL</sub>"
        f"(<i>P</i> {MS('&#8214;')} <i>Q</i>) is as small as possible.", body)]

    # 2.6 BIC
    story += [Paragraph("3.6 &#160; The Bayesian Information Criterion",
                        subsection)]
    story += [Paragraph(
        "How do we decide whether a two-cluster model of the corpus "
        "is better than a one-cluster model? Any two-cluster model "
        "has strictly more free parameters, so it will fit the data "
        "strictly better in a raw likelihood sense. The Bayesian "
        "Information Criterion (BIC) trades fit against complexity:",
        body)]
    story += [NumberedEquation(
        r"\mathrm{BIC}(k)\ \ =\ \ k \ln N\ \ -\ \ 2 \ln \hat L_k,",
        number=None, page_width=6.5 * inch, fontsize=13)]
    story += [Paragraph(
        "where <i>k</i> is the number of free parameters, <i>N</i> "
        f"the number of datapoints, and <i>L{MS('&#770;')}<sub>k</sub></i> "
        "the maximised likelihood of the model. Lower BIC is better. "
        "Adding a parameter helps only if it improves the log-"
        "likelihood by at least ln <i>N</i> &#247; 2 &#8776; 4.1 "
        "for our <i>N</i> = 3,685. This is a soft Ockham&#8217;s "
        "razor with a specific numerical threshold. In \u00a75 we use "
        "&#916;BIC = BIC<sub>2</sub> &#8722; BIC<sub>1</sub> to ask "
        "whether the corpus is better described by two Gaussian "
        "clusters than by one.", body_i)]

    # 2.7 Fisher's ratio
    story += [Paragraph("3.7 &#160; Fisher&#8217;s discriminant ratio",
                        subsection)]
    story += [Paragraph(
        "Given points labelled with one of two classes &#8212; in "
        "our case, chunks tagged as &#8220;prose&#8221; or "
        "&#8220;table&#8221; by a regex &#8212; linear discriminant "
        "analysis (LDA) finds the direction <i>w</i> along which the "
        "two classes are maximally separated relative to their own "
        "internal spread. It maximises",
        body)]
    story += [NumberedEquation(
        r"J(w)\ \ =\ \ \frac{w^{\top} S_B\, w}{w^{\top} S_W\, w},",
        number=None, page_width=6.5 * inch, fontsize=13)]
    story += [Paragraph(
        "where <i>S<sub>B</sub></i> is the between-class scatter "
        "(how far the class means are from the overall mean, "
        "weighted by class size) and <i>S<sub>W</sub></i> is the "
        "within-class scatter (how spread out each class is around "
        "its own mean). The ratio is a signal-to-noise ratio: it "
        "grows when class means separate and shrinks when either "
        "class becomes more diffuse. Fisher\u2019s ratio is the object "
        "underlying Figure 1 of \u00a72: the direction it selects is "
        "what makes the centroid displacement legible in one "
        "dimension.", body_i)]

    # 2.8 Sampling
    story += [Paragraph("3.8 &#160; A note on empirical sampling",
                        subsection)]
    story += [Paragraph(
        "Several of the distributions in the rest of the paper "
        "&#8212; the cosine-similarity histograms and the anisotropy "
        "reference of \u00a76 &#8212; are constructed by <i>sampling "
        "pairs</i> "
        "rather than enumerating all of them. With <i>N</i> = 3,685 "
        "chunks the number of pairs is "
        r"<i>N</i>(<i>N</i>&#8722;1)/2 &#8776; 6.8 &#215; 10<super>6</super>, "
        "which is computationally awkward and, more importantly, "
        "unnecessary: a random sample of 20,000 to 50,000 pairs "
        "gives a Monte-Carlo estimate of any distributional summary "
        f"whose standard error is O(1/{MS('&#8730;')}<i>K</i>), i.e. below the "
        "third decimal place. Every empirical distribution reported "
        "below is such a sample; every seed is fixed for "
        "reproducibility.", body)]

    # ---------------- 4. The corpus (was 3) ----------------
    story += [Paragraph("4 &#160; The corpus", section)]
    story += [Paragraph(
        f"After light PDF text extraction, cleaning, and fixed-size "
        f"chunking (approximately 900 characters, 120-character "
        f"overlap), the corpus contains {N:,} chunks across {NDOC} "
        f"reports; chunk counts per company (Figure 3) vary from "
        "about 250 to about 950 per firm, an imbalance inherited "
        "from publishing conventions rather than curation. Each chunk "
        "is passed through "
        "<font name='" + FONT_ITAL + "'>all-MiniLM-L6-v2</font>, "
        "producing a 384-dimensional unit-normalised embedding. On "
        "unit vectors, cosine similarity and Euclidean distance are "
        "monotonically related through "
        r"&#8214;<i>a</i> &#8722; <i>b</i>&#8214;<super>2</super> "
        r"= 2(1 &#8722; cos(<i>a</i>, <i>b</i>)), "
        "so every statement about distance below can be read "
        "equivalently as a statement about cosine similarity.", body)]
    story += figure_block(
        "fig01_corpus_overview", 3,
        "Chunks per company. Every chunk is a contiguous "
        "~900-character window of extracted PDF text.", s)

    # ---------------- 5. First observation (was 4) ----------------
    story += [Paragraph("5 &#160; Bimodality: the corpus splits, "
                        "but not by author", section)]
    story += [Paragraph(
        "Figure 4 shows a principal-component projection of the whole "
        "corpus to two dimensions, coloured by publishing company. "
        "Two features are immediate. First, the colours mix: the five "
        "companies do not occupy distinguishable regions. Any "
        "per-firm retrieval design that assumed firm-level clustering "
        "would be building on sand. Second, the cloud is not one "
        "lump; there are two visible modes.", body)]
    story += figure_block(
        "fig02_pca_2d_by_company", 4,
        "Two-dimensional principal-component projection of "
        f"the {N:,} corpus chunks, coloured by publishing company. "
        "The colours mix; the shape does not. Company identity is "
        "not the axis along which this space is structured.", s)
    story += [Paragraph(
        "To promote the visual impression to a measured claim we fit "
        "two Gaussian mixtures &#8212; with one and two full-"
        "covariance components respectively &#8212; and compare their "
        "BIC. A methodological subtlety is worth flagging: in the "
        "raw 384-dimensional space each additional full-covariance "
        "component introduces roughly "
        "384 + <i>C</i>(384, 2) &#8776; 74,000 free parameters, and "
        "the BIC penalty <i>k</i> ln <i>N</i> swamps any log-"
        "likelihood improvement even when a two-cluster fit is "
        "visually obvious. We therefore fit both mixtures in the "
        "top-20 principal-component subspace, which retains "
        f"{STATS.get('gmm_subspace_var', 0.5):.0%} of corpus variance "
        "and reduces the per-component parameter count to 230. In "
        f"that subspace &Delta;BIC = {DBIC:,.0f}, a decisive "
        "preference for two clusters (Figure 5).", body_i)]
    story += figure_block(
        "fig03_bimodality_gmm", 5,
        "A two-component Gaussian mixture fitted in the top-"
        "20 principal-component subspace, projected here onto the "
        "same principal-component plane as Figure 4. Ellipses are "
        "two standard deviations of each cluster&#8217;s projected "
        "covariance. The bimodality is a property of the embedded "
        "corpus, not of the two-dimensional view.", s)
    if EXEMPLARS:
        story += [Paragraph(
            "A representative chunk from one cluster reads:", body)]
        na = EXEMPLARS.get("narrative_extremes", [])
        ta = EXEMPLARS.get("tabular_extremes", [])
        if na:
            ex = na[0]
            story += pullquote(
                ex["text"],
                f"&#8212; {ex['company']}, {ex['doc']}, chunk "
                f"{ex['chunk_id']} (one cluster)", s)
        story += [Paragraph(
            "and a representative chunk from the other reads:", body)]
        if ta:
            ex = ta[0]
            story += pullquote(
                ex["text"],
                f"&#8212; {ex['company']}, {ex['doc']}, chunk "
                f"{ex['chunk_id']} (the other cluster)", s)
        story += [Paragraph(
            "The first is narrative ESG prose. The second is what "
            "PDF-to-text extraction produces when it walks over a "
            "table of financial line items: numbers, parenthesised "
            "signs, whitespace where columns used to be, no verbs. "
            "These are the two class centroids of \u00a72 made "
            "concrete: the unsupervised mixture has discovered the "
            "same partition that a regex-based content labeller "
            "produces, and the class means displaced along Fisher&#8217;"
            "s direction are what the eye is seeing in Figures 4-5.",
            body)]

    # ---------------- 6. Cosine, anisotropy, PC1 (was 5) ----------------
    story += [Paragraph("6 &#160; Reading cosine similarity honestly",
                        section)]
    story += [Paragraph(
        "Cosine similarity is the atom of every retrieval decision "
        "that follows: to ask &#8220;which chunk is most like this "
        "query?&#8221; is to sort the corpus by it. It is therefore "
        "worth knowing what its distribution looks like across pair "
        "types we would expect to differ. Figure 6 partitions the "
        "corpus into four pair-type populations. Same-document pairs "
        f"average <i>x{MS('&#772;')}</i> &#8776; {COS_SAMEDOC:.2f}; "
        f"same-company pairs average {COS_SAMECO:.2f}; different-"
        f"company pairs average {COS_DIFFCO:.2f}. Two observations. "
        "First, the same-document and same-company distributions are "
        "essentially coincident &#8212; multiple annual reports from "
        "the same firm on the same topics are as textually related "
        "as chunks within a single report. Second, and more "
        "importantly, none of these distributions is centred near "
        f"zero. Even different-company pairs cluster near {COS_DIFFCO:.2f}. "
        "That offset is a property of the encoder, not of our "
        "corpus.", body)]
    story += figure_block(
        "fig04_cosine_distributions", 6,
        "Pairwise cosine-similarity distributions across four "
        "pair populations. Within-document and within-company "
        "distributions coincide almost exactly. All four sit well "
        "above zero &#8212; the subject of \u00a76.1.", s)
    story += [Paragraph(
        "<b>How to read Figure 6.</b> Each coloured histogram is the "
        "answer to one question: <i>if I pick two chunks from a "
        "particular relationship &#8212; same document, same company, "
        "different companies, or random pairs &#8212; how similar do "
        "they typically look to MiniLM?</i> The x-axis is cosine "
        "similarity: 0 means &#8220;unrelated,&#8221; 1 means "
        "&#8220;identical.&#8221; The story the plot tells is a "
        "double surprise. First, the four histograms nearly coincide: "
        "MiniLM barely distinguishes two random reports from two "
        "chunks of the <i>same</i> report. Second, none of them is "
        "centred at zero. Even random-pair similarity averages "
        f"{COS_DIFFCO:.2f}, which is where the next subsection "
        "starts.", body)]

    story += [Paragraph("6.1 &#160; Anisotropy", subsection)]
    story += [Paragraph(
        "An isotropic embedding space would satisfy "
        r"E<sub><i>i</i>&#8800;<i>j</i></sub>[cos(<i>x<sub>i</sub></i>, "
        r"<i>x<sub>j</sub></i>)] &#8776; 0 for random chunk pairs. "
        "In practice pretrained contextual encoders exhibit strong "
        "<i>anisotropy</i> (Ethayarajh, 2019): outputs occupy a "
        "narrow cone, and the left-hand side lands well above zero. "
        f"On our corpus <i>c{MS('&#772;')}</i><sub>SUE</sub> = "
        f"{ANISO:.3f} against an isotropic reference "
        f"<i>c{MS('&#772;')}</i><sub>iso</sub> = {ISO:.3f} (Figure 7). "
        "The gap between the means is the quantity Ethayarajh names, "
        "and every cosine we report should be read against this "
        "offset.", body)]
    story += schematic_block(
        "fig_anisotropy_intuition", "B",
        "Anisotropy, without algebra. Left: an isotropic embedding "
        "space; every direction is used equally and the mean cosine "
        "between random pairs sits at 0. Right: an anisotropic "
        "space, of the kind MiniLM produces; all embeddings crowd "
        "into a cone around a preferred axis, so even "
        "\u201Cunrelated\u201D pairs land at cosine \u2248 +0.30. "
        "The cone half-angle is exaggerated for legibility; the "
        "quoted +0.30 is the measured SUE value. Every histogram in "
        "Figure 6 is a slice through the right-hand cone.", s)
    story += figure_block(
        "fig05_anisotropy", 7,
        "Cosine similarity between random pairs in our corpus "
        "(blue) against an isotropic reference (grey). The gap "
        "between the means is a property of the encoder, not of the "
        "corpus.", s)
    story += [Paragraph(
        "<b>How to read Figure 7.</b> The grey histogram is what "
        "cosine similarity <i>would</i> look like if MiniLM lived on "
        "the full sphere of Diagram B (mean 0). The blue histogram "
        "is what it actually looks like on our corpus (mean "
        f"{ANISO:.2f}). The horizontal gap between the two means is "
        "the anisotropy of the encoder made numeric: it is the "
        "cosine of the cone\u2019s half-angle, expressed as a "
        "similarity. Once we subtract that offset, the "
        "&#8220;signal&#8221; that same-document pairs carry above "
        "random pairs is small &#8212; roughly "
        f"{max(0.0, COS_SAMEDOC - COS_DIFFCO):.2f} in Figure 6 "
        "&#8212; and every retrieval decision the encoder makes is "
        "resolving that thin margin.", body_i)]

    story += [Paragraph("6.2 &#160; What PCA does and does not see",
                        subsection)]
    story += [Paragraph(
        "A common intuition &#8212; worth explicitly dismantling "
        "&#8212; is that PC1 of an anisotropic corpus is dominated "
        "by the anisotropy axis. It is not. Anisotropy is a "
        "<i>first-moment</i> property: the mean direction "
        f"<i>x{MS('&#772;')}</i> is far from zero. PCA operates on "
        "the <i>centred</i> data matrix: it subtracts the mean "
        "before diagonalising the covariance. The direction along "
        "which anisotropy concentrates the mean is therefore "
        "invisible to PCA. What Figure 8 shows is the residual, "
        "second-moment structure of the corpus after that offset "
        "has been removed.", body)]
    story += figure_block(
        "fig06_explained_variance", 8,
        "Fraction of variance explained by each of the first "
        "thirty principal components. PC1 accounts for "
        f"{PC1:.1%} of the total variance, the first ten combined "
        f"for {TOP10:.1%}. No single component dominates.", s)
    story += [Paragraph(
        f"On our corpus PC1 accounts for {PC1:.1%} of variance and "
        f"the first ten combined for {TOP10:.1%}: a relatively flat "
        "spectrum, meaning the two-dimensional principal-component "
        "projections of Figures 4-5 are picking up genuine, if "
        "partial, second-moment structure &#8212; they are not "
        "renderings of the anisotropy vector. Whitening still has a "
        "purpose. A linear approximation to Li et al. (2020) is "
        "per-dimension mean/variance normalisation "
        r"<i>x&#771;</i> = diag(&#963;)<super>&#8722;1</super>"
        r"(<i>x</i> &#8722; <i>&#956;</i>), "
        "after which the covariance is approximately the identity "
        "and the mean is at the origin by construction. Figure 9 "
        "compares the corpus before and after this transformation, "
        "each projected to its own top-two principal components. "
        "The two-lobe shape persists &#8212; the important "
        "observation. It was not an artefact of anisotropy or of "
        "the mean offset.", body_i)]
    story += figure_block(
        "fig07_whitening", 9,
        "Left: raw MiniLM embeddings under two-dimensional "
        "principal-component projection. Right: after per-dimension "
        "mean/variance whitening, projected to its own top-two "
        "principal components. The tilt flattens; the two-lobe "
        "structure persists.", s)

    # ---------------- 7. Nonlinear projections (was 6) ----------------
    story += [Paragraph("7 &#160; Nonlinear projections: t-SNE and UMAP",
                        section)]
    story += [Paragraph(
        "Nothing forces us to use principal components. t-SNE "
        "(van der Maaten and Hinton, 2008) and UMAP (McInnes et al., "
        "2018) are the standard nonlinear alternatives; both have "
        "visible hyperparameters that materially change the picture. "
        "Figures 10 and 11 replay the corpus under four settings "
        "each. The purpose is not to nominate a winner; it is to "
        "make the reader aware that every low-dimensional plot of a "
        "corpus is a rendering through a chosen lens. Across all "
        "settings, the two-lobe structure of \u00a72 and \u00a75 is "
        "preserved &#8212; the class-centroid displacement is "
        "robust to the choice of projection, which is the "
        "essential invariance for our claim.", body)]
    story += figure_block(
        "fig08_tsne_grid", 10,
        "t-SNE projections of a 1200-chunk stratified sample "
        "at four perplexity settings. Same data, same seed, four "
        "different neighbourhood-scale choices; the two-lobe "
        "structure survives every setting.", s)
    if (FIGDIR / "fig09_umap_grid.png").exists():
        story += figure_block(
            "fig09_umap_grid", 11,
            "UMAP projections of the same corpus sample at four "
            "values of <i>n_neighbors</i>, with <i>min_dist</i> "
            "held fixed. Local vs. global emphasis is a user-facing "
            "choice; the bimodality is not.", s)

    # ---------------- 8. Discussion (was 9) ----------------
    story += [Paragraph("8 &#160; Discussion", section)]
    story += [Paragraph(
        "The finding is not that these five companies write "
        "differently from one another. They write very similarly; "
        "at the chunk level our two-dimensional projections do not "
        "resolve company identity. The finding is that any single "
        "report is itself already a bimodal object: it contains a "
        "narrative track and a tabular track, and the geometry of "
        "the embedding space picks up on that structure &#8212; "
        "as a displacement of class centroids &#8212; much more "
        "cleanly than it picks up on authorship. A retrieval system "
        "that treats every chunk as a member of a single population, "
        "and every top-<i>k</i> neighbourhood as a semantically "
        "coherent context, is likely to fail differently on the two "
        "tracks. The prose track will support "
        "&#8220;what does this company plan to do about X?&#8221; "
        "questions; the tabular track will pollute them with "
        "vocabulary-overlapping but intent-mismatched context.", body)]
    story += [Paragraph(
        "The corresponding design implication is that if the two "
        "tracks are learnable from the raw embeddings &#8212; "
        "\u00a72 shows they are &#8212; then a downstream system "
        "can route on them. One natural route is separate indices "
        "with separate top-<i>k</i> budgets; a lighter one is a "
        "single index re-ranked by the query&#8217;s projection "
        "onto the class-centroid axis. Which is worth doing depends "
        "on the query mix a system is served, and on how often the "
        "two tracks are answering the same question in incompatible "
        "languages.", body_i)]
    story += [Paragraph(
        "The present study is limited in three ways worth naming. "
        "First, only one industry is represented, so all industry-"
        "level claims remain hypotheses. Second, only one embedding "
        "model is used; the anisotropy figure would look different "
        "for a model trained with a stronger uniformity objective, "
        "and the class-centroid separation would potentially sharpen "
        "or blur. Third, the content-class label is regex-based and "
        "coarse; a trained classifier might refine it. None of these "
        "limitations undermines the diagnostic claim, which is "
        "elementary: <i>look at the class centroids of your corpus "
        "in embedding space before you build on top of it.</i>", body)]

    # ---------------- 9. Future directions (was 10) ----------------
    story += [Paragraph("9 &#160; Future directions", section)]
    story += [Paragraph(
        "The measurements above are deliberately minimal &#8212; "
        "one sector, one encoder, one chunk size &#8212; and even "
        "at that minimum, class-centroid displacement is doing "
        "something interpretable and actionable. Five directions "
        "follow naturally.", body)]

    story += [Paragraph("9.1 &#160; Chunk-window ablation", subsection)]
    story += [Paragraph(
        "Every result is computed on ~900-character chunks with a "
        "120-character overlap. A systematic ablation across, say, "
        "{300, 900, 2000} characters would let us ask whether the "
        "class-centroid axis sharpens as chunks lengthen (more "
        "context per point) or blurs (mixing registers inside one "
        "window). Intuition points in opposite directions for the "
        "two ends: prose should benefit from more context, tables "
        "should become harder to identify as prose leaks in.", body)]

    story += [Paragraph("9.2 &#160; Cross-encoder comparison",
                        subsection)]
    story += [Paragraph(
        "MiniLM was chosen for compactness, not for state-of-the-art "
        "geometry. Encoders trained with explicit uniformity "
        "objectives (SimCSE) or with a contrastive-decoder "
        "architecture (Karpukhin et al., 2020) would occupy the "
        "sphere differently and would potentially reduce the "
        "anisotropy gap of Figure 7. The question is whether the "
        "class-centroid displacement survives the encoder swap; a "
        "grid of encoders on the same corpus turns the current "
        "single-encoder diagnostic into a comparative statement "
        "about representation quality.", body)]

    story += [Paragraph("9.3 &#160; Cross-industry expansion",
                        subsection)]
    story += [Paragraph(
        "The corpus here is one sector. Adding sectors whose "
        "reporting is likely to look qualitatively different "
        "&#8212; oil and gas, pharmaceuticals, consumer banking, "
        "apparel &#8212; would let us ask whether the "
        "financial\u2194narrative class-centroid axis is universal "
        "(a property of the disclosure genre) or sector-specific "
        "(a property of what each sector must quantify).", body)]

    story += [Paragraph("9.4 &#160; Interpretability of the "
                        "class-centroid direction", subsection)]
    story += [Paragraph(
        "The direction "
        r"<i>&#956;</i><sub>tabular</sub> &#8722; "
        r"<i>&#956;</i><sub>narrative</sub> is itself a 384-"
        "dimensional vector, and its projection onto known token-"
        "level features (digit density, punctuation frequency, mean "
        "line length, presence of currency symbols) would decompose "
        "the axis into human-readable components. This is the same "
        "species of question mechanistic interpretability asks about "
        "individual attention heads, exported outward to the corpus "
        "level.", body)]

    story += [Paragraph("9.5 &#160; Centroid-shift analyses in other "
                        "document classes", subsection)]
    story += [Paragraph(
        "The claim of \u00a72 is general. Any collection of text "
        "whose author-declared structure mixes distinguishable "
        "content classes into a single stream admits the same "
        "diagnostic: compute the class centroids, plot their "
        "displacement, and read the axis. Several domains present "
        "themselves immediately.", body)]
    story += [Paragraph(
        "In <b>scientific writing</b>, the methods and discussion "
        "sections of a paper mix hypothesis and procedure; the "
        "centroid displacement across sections of the same paper is "
        "a candidate signal for measuring how tightly a "
        "field&#8217;s reporting norms hold. In <b>clinical "
        "records</b>, structured history and free-text notes "
        "describe the same encounter in incompatible registers; the "
        "class-centroid axis would give a per-record score of how "
        "narratively vs. procedurally the encounter is being "
        "documented, with direct implications for downstream "
        "extraction. In <b>legal filings</b>, statutory-citation "
        "passages and argumentative passages have distinct rhetorical "
        "structure; their centroid displacement is a candidate axis "
        "for a document-triage tool for practitioners handling large "
        "brief volumes. In <b>parliamentary or committee "
        "transcripts</b>, speaker-A vs. speaker-B centroids trace "
        "who occupies which rhetorical register on which topic. In "
        "<b>research grants</b>, the technical vs. rhetorical "
        "passages of a proposal have measurable centroid separation, "
        "and the ratio of chunks on each side is a proxy for how "
        "specifically the proposal has been costed.", body_i)]
    story += [Paragraph(
        "In each case the diagnostic is one line of code once the "
        "corpus is embedded: compute "
        r"<i>&#956;<sub>i</sub></i> &#8722; "
        "<i>&#956;<sub>j</sub></i>, project every chunk onto that "
        "difference vector, and inspect. The value of the framing is "
        "not in any single instance but in the fact that a corpus "
        "reader now has an inexpensive, general-purpose first look "
        "at whether the structure the authors of a corpus claim to "
        "have imposed on it has left a geometric footprint the "
        "encoder can see.", body)]

    story += [Paragraph("9.6 &#160; Policy tooling", subsection)]
    story += [Paragraph(
        "For the policy audience of \u00a71.2, the most immediate "
        "artefact is a per-report score along the class-centroid "
        "axis of \u00a72. Given a target set of disclosure topics "
        "(Scope 3 emissions, water withdrawal, supplier auditing "
        "coverage), one can ask how many chunks a report devotes to "
        "each topic and how many land on the numeric versus the "
        "narrative side of the axis. That statistic is a first-order "
        "proxy for disclosure quality &#8212; not a substitute for "
        "expert audit, but a useful triage tool over the 200-plus "
        "reports published each year by the S&amp;P 500 alone.",
        body)]

    # ---------------- 11. References ----------------
    story += [PageBreak()]
    story += [Paragraph("References", section)]
    refs = [
        ("Reimers, N. and Gurevych, I. (2019).",
         "Sentence-BERT: Sentence Embeddings using Siamese "
         "BERT-Networks.",
         "EMNLP. arXiv:1908.10084."),
        ("Ethayarajh, K. (2019).",
         "How Contextual are Contextualized Word Representations? "
         "Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings.",
         "EMNLP."),
        ("Li, B., Zhou, H., He, J., Wang, M., Yang, Y., and Li, L. (2020).",
         "On the Sentence Embeddings from Pre-trained Language Models.",
         "EMNLP. arXiv:2011.05864."),
        ("Gao, T., Yao, X., and Chen, D. (2021).",
         "SimCSE: Simple Contrastive Learning of Sentence Embeddings.",
         "EMNLP. arXiv:2104.08821."),
        ("van der Maaten, L. and Hinton, G. (2008).",
         "Visualizing Data using t-SNE.",
         "Journal of Machine Learning Research 9, 2579\u20132605."),
        ("McInnes, L., Healy, J., and Melville, J. (2018).",
         "UMAP: Uniform Manifold Approximation and Projection for "
         "Dimension Reduction.",
         "arXiv:1802.03426."),
        ("Coenen, A., Reif, E., Yuan, A., Kim, B., Pearce, A., "
         "Vi\u00e9gas, F., and Wattenberg, M. (2019).",
         "Visualizing and Measuring the Geometry of BERT.",
         "NeurIPS."),
        ("Karpukhin, V., O\u011fuz, B., Min, S., Lewis, P., Wu, L., "
         "Edunov, S., Chen, D., and Yih, W.-t. (2020).",
         "Dense Passage Retrieval for Open-Domain Question Answering.",
         "EMNLP."),
        ("Lewis, P. et al. (2020).",
         "Retrieval-Augmented Generation for Knowledge-Intensive NLP "
         "Tasks.",
         "NeurIPS."),
        ("Thakur, N., Reimers, N., R\u00fcckl\u00e9, A., Srivastava, A., "
         "and Gurevych, I. (2021).",
         "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of "
         "Information Retrieval Models.",
         "NeurIPS Datasets and Benchmarks."),
        ("Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. (2023).",
         "MTEB: Massive Text Embedding Benchmark.",
         "EACL."),
        ("Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., "
         "and Manning, C. D. (2024).",
         "RAPTOR: Recursive Abstractive Processing for Tree-Organized "
         "Retrieval.",
         "ICLR."),
    ]
    for authors, title_txt, venue in refs:
        story += [Paragraph(
            f"{authors} <i>{title_txt}</i> {venue}",
            s["ref"])]

    return story


# ============================================================
# Document assembly
# ============================================================

def build() -> None:
    doc = BaseDocTemplate(
        str(OUT_PDF), pagesize=LETTER,
        leftMargin=1.0 * inch, rightMargin=1.0 * inch,
        topMargin=0.95 * inch, bottomMargin=0.85 * inch,
        title="SUE: Looking Before You Build",
        author=AUTHOR)

    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height, id="normal",
        leftPadding=0, rightPadding=0,
        topPadding=0, bottomPadding=0)

    # Title page has no running header; subsequent pages do
    title_tpl = PageTemplate(id="title", frames=[frame],
                             onPage=draw_title_page_frame)
    body_tpl = PageTemplate(id="body", frames=[frame],
                            onPage=draw_page_frame)
    doc.addPageTemplates([title_tpl, body_tpl])

    styles = make_styles()
    story = build_story(styles)
    # Switch to body template after the abstract, using PageBreak +
    # NextPageTemplate. ReportLab honors the initial template until we
    # explicitly change. For simplicity, we accept the running header
    # from page 2 onward by placing a NextPageTemplate directive early.
    from reportlab.platypus import NextPageTemplate
    story = [NextPageTemplate("body")] + story

    doc.build(story)
    print(f"wrote {OUT_PDF}  ({OUT_PDF.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build()
