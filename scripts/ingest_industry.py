"""
scripts/ingest_industry.py
--------------------------
Parse, chunk, embed, and tag every downloaded PDF of one industry into a
cached corpus under assets/industries/<industry>/.

Pipeline (kept deliberately identical to the semiconductor corpus so the
two industries stay comparable):
  1. Walk data/sample_data/<industry>/<Company>/*.pdf
  2. Extract the text layer per page with pypdf (no OCR), normalizing
     whitespace per page so page->character offsets stay valid
  3. Concatenate pages (250k char cap per doc), chunk into ~900-char
     windows with 120-char overlap, dropping tails under 200 chars
  4. Record page_start / page_end for every chunk (provenance for the
     future page-thumbnail feature)
  5. Tag each chunk with company, ticker, universe tier, document
     category, report year, source URL (from
     data/industries/<industry>_downloads.json), digit density, and the
     three-way prose/ambiguous/tabular register used by the viewer
  6. Embed with sentence-transformers/all-MiniLM-L6-v2 (unit-norm)

Outputs:
  assets/industries/<industry>/chunks.parquet
  assets/industries/<industry>/embeddings.npy

Usage:
    python scripts/ingest_industry.py                  # aerospace_defense
    python scripts/ingest_industry.py --max-chars 80000   # smoke run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent.parent
INDUSTRIES_DIR = ROOT / "data" / "industries"
DATA_DIR = ROOT / "data" / "sample_data"
ASSETS_DIR = ROOT / "assets" / "industries"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
MIN_CHUNK_CHARS = 200
TABLE_DIGIT_THRESHOLD = 0.18   # keep in sync with build_semantic_universe_explorer.py
_DIGIT_RE = re.compile(r"\d")


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\s\-\.&]+", "", (name or "").strip(), flags=re.UNICODE)
    return re.sub(r"\s+", "_", s) or "unknown"


def digit_density(text: str) -> float:
    return len(_DIGIT_RE.findall(text)) / max(1, len(text)) if text else 0.0


def register_of(dd: float) -> str:
    if dd >= TABLE_DIGIT_THRESHOLD:
        return "tabular"
    if dd <= 0.05:
        return "prose"
    return "ambiguous"


def extract_pages(pdf_path: Path, max_chars: int) -> List[str]:
    """One normalized text string per page, stopping once max_chars is hit."""
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        print(f"  ! pypdf failed on {pdf_path.name}: {e}")
        return []
    pages: List[str] = []
    total = 0
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        pages.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return pages


def chunk_pages(pages: List[str], max_chars: int) -> List[Tuple[str, int, int]]:
    """Return (chunk_text, page_start, page_end) tuples (pages 1-indexed)."""
    # Character offset at which each page begins in the concatenated doc.
    text_parts, starts, pos = [], [], 0
    for p in pages:
        starts.append(pos)
        text_parts.append(p)
        pos += len(p) + 1                      # +1 for the joining newline
    text = "\n".join(text_parts)[:max_chars]
    starts_arr = np.array(starts)

    def page_at(offset: int) -> int:
        return int(np.searchsorted(starts_arr, offset, side="right"))

    out: List[Tuple[str, int, int]] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(text), step):
        raw = text[i: i + CHUNK_SIZE]
        c = raw.strip()
        if len(c) < MIN_CHUNK_CHARS:
            continue
        lead = len(raw) - len(raw.lstrip())
        start_off = i + lead
        end_off = start_off + len(c) - 1
        out.append((c, page_at(start_off), page_at(end_off)))
    return out


def build_corpus(industry: str, max_chars: int) -> pd.DataFrame:
    downloads_file = INDUSTRIES_DIR / f"{industry}_downloads.json"
    meta = {}
    if downloads_file.exists():
        meta = json.loads(downloads_file.read_text(encoding="utf-8"))
    sources_file = INDUSTRIES_DIR / f"{industry}_sources.json"
    industry_name = json.loads(
        sources_file.read_text(encoding="utf-8"))["industry"]

    ind_dir = DATA_DIR / slugify(industry_name)
    pdfs = sorted(ind_dir.glob("*/*.pdf"))
    print(f"Found {len(pdfs)} PDFs under {ind_dir}")

    rows = []
    for pdf in pdfs:
        m = meta.get(pdf.name, {})
        company = m.get("company_name") or pdf.parent.name.replace("_", " ")
        print(f"  reading [{company}] {pdf.name}")
        pages = extract_pages(pdf, max_chars=max_chars)
        chunks = chunk_pages(pages, max_chars=max_chars)
        print(f"    -> {len(chunks)} chunks over {len(pages)} pages")
        for j, (c, p_start, p_end) in enumerate(chunks):
            dd = digit_density(c)
            rows.append({
                "industry": industry_name,
                "company": company,
                "ticker": m.get("ticker", ""),
                "tier": m.get("universe_tier", ""),
                "doc": pdf.name,
                "doc_category": m.get("category", ""),
                "report_year": m.get("report_year", ""),
                "source_url": m.get("url", ""),
                "chunk_id": j,
                "text": c,
                "page_start": p_start,
                "page_end": p_end,
                "digit_density": dd,
                "register": register_of(dd),
            })
    return pd.DataFrame(rows)


def embed(texts: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(f"Embedding {len(texts):,} chunks ...")
    return model.encode(texts, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--industry", default="aerospace_defense")
    ap.add_argument("--max-chars", type=int, default=250_000,
                    help="Character cap per PDF (matches the core corpus).")
    args = ap.parse_args()

    df = build_corpus(args.industry, max_chars=args.max_chars)
    if df.empty:
        raise SystemExit("No chunks produced. Run scripts/fetch_industry.py first.")
    print(f"\nTotal: {len(df):,} chunks | {df['company'].nunique()} companies "
          f"| {df['doc'].nunique()} documents")
    print(df["register"].value_counts().to_string())

    X = embed(df["text"].tolist())

    out_dir = ASSETS_DIR / args.industry
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "chunks.parquet", index=False)
    np.save(out_dir / "embeddings.npy", X.astype(np.float32))
    print(f"\nwrote {out_dir / 'chunks.parquet'}")
    print(f"wrote {out_dir / 'embeddings.npy'}  {X.shape}")


if __name__ == "__main__":
    main()
