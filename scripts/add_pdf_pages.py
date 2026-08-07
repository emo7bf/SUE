"""
Add a `page` column to assets/chunks.parquet by locating each cached chunk
inside its original PDF (per-page text search). Also creates a template
`data/pdf_urls.json` mapping every PDF filename to an empty URL, ready for
the user to fill in with the companies' investor-relations URLs.

Neither operation re-embeds anything. The MiniLM embeddings.npy cache is
left untouched. This script exists so PDF-page linking can be added to
the viewer without a full corpus rebuild.

Usage:
    python scripts/add_pdf_pages.py

Reads:
    assets/chunks.parquet
    data/sample_data/<company>/*.pdf   (the actual source PDFs)

Writes:
    assets/chunks.parquet              (with new `page` int column)
    data/pdf_urls.json                 (template only; existing keys preserved)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent.parent          # .../sue_release
REPO_ROOT = ROOT.parent                                # .../SUE
CHUNKS_PARQUET = ROOT / "assets" / "chunks.parquet"
PDF_URLS_JSON = ROOT / "data" / "pdf_urls.json"

# PDFs may live in either sue_release/data/sample_data or the original
# repo-root data/sample_data. Prefer whichever actually has PDFs.
_release_data = ROOT / "data" / "sample_data"
_repo_data = REPO_ROOT / "data" / "sample_data"
if _release_data.exists() and any(_release_data.rglob("*.pdf")):
    DATA_DIR = _release_data
else:
    DATA_DIR = _repo_data


def extract_pages(pdf_path: Path) -> List[str]:
    """Return one text string per PDF page (1-indexed via list order)."""
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        print(f"  ! pypdf failed on {pdf_path.name}: {e}")
        return []
    pages: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages.append(t)
    return pages


def _normalize(s: str) -> str:
    """Whitespace-collapse a string so matching survives line breaks."""
    return re.sub(r"\s+", " ", s).strip()


def locate_page(chunk_text: str, pages_norm: List[str]) -> int:
    """Return 1-indexed page number containing the chunk's opening text.

    We match against the first 80 non-whitespace characters of the chunk
    (its "fingerprint"). If a chunk spans two pages, the fingerprint is
    almost always on the first of them, which is the natural target for
    a "jump to page" link.
    """
    fingerprint = _normalize(chunk_text[:120])[:80]
    if not fingerprint:
        return 1
    for i, page_norm in enumerate(pages_norm):
        if fingerprint in page_norm:
            return i + 1
    # Fall back: try progressively shorter fingerprints
    for prefix in (60, 40, 24):
        fp = _normalize(chunk_text[:120])[:prefix]
        if not fp:
            continue
        for i, page_norm in enumerate(pages_norm):
            if fp in page_norm:
                return i + 1
    return 1


def build_doc_index() -> Dict[str, Path]:
    """Return {pdf_filename: absolute_path} across all sample_data subfolders."""
    idx: Dict[str, Path] = {}
    if not DATA_DIR.exists():
        return idx
    for pdf in DATA_DIR.rglob("*.pdf"):
        idx[pdf.name] = pdf
    return idx


def main() -> None:
    if not CHUNKS_PARQUET.exists():
        raise SystemExit(f"missing {CHUNKS_PARQUET}. Run build_visuals.py first.")

    df = pd.read_parquet(CHUNKS_PARQUET)
    print(f"Loaded {len(df):,} chunks across {df['doc'].nunique()} PDFs.")

    if "page" in df.columns:
        print("  (existing `page` column found; will be overwritten)")

    doc_index = build_doc_index()
    print(f"Found {len(doc_index)} source PDFs under {DATA_DIR}")

    missing = sorted(set(df["doc"].unique()) - set(doc_index.keys()))
    if missing:
        print("  ! WARNING: no PDF file found for these docs; they will get page=1:")
        for m in missing:
            print(f"      - {m}")

    # Per-doc page cache so we only extract once per PDF, not once per chunk
    pages_cache: Dict[str, List[str]] = {}

    pages_col: List[int] = []
    for doc_name, chunk_text in zip(df["doc"], df["text"]):
        if doc_name not in pages_cache:
            pdf_path = doc_index.get(doc_name)
            if pdf_path is None:
                pages_cache[doc_name] = []
                print(f"  reading (skipped, missing): {doc_name}")
            else:
                raw_pages = extract_pages(pdf_path)
                pages_cache[doc_name] = [_normalize(p) for p in raw_pages]
                print(f"  read {len(raw_pages):>4} pages : {doc_name}")
        pages_norm = pages_cache[doc_name]
        if not pages_norm:
            pages_col.append(1)
        else:
            pages_col.append(locate_page(chunk_text, pages_norm))

    df["page"] = pages_col
    df.to_parquet(CHUNKS_PARQUET, index=False)
    print(f"\nWrote page column: min={df['page'].min()}  max={df['page'].max()}  "
          f"median={int(df['page'].median())}")
    print(f"  -> {CHUNKS_PARQUET}")

    # --- pdf_urls.json template
    existing: Dict[str, str] = {}
    if PDF_URLS_JSON.exists():
        try:
            existing = json.loads(PDF_URLS_JSON.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    all_docs = sorted(df["doc"].unique().tolist())
    template: Dict[str, str] = {}
    for d in all_docs:
        template[d] = existing.get(d, "")   # preserve any URLs the user already added
    PDF_URLS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PDF_URLS_JSON.write_text(
        json.dumps(template, indent=2, ensure_ascii=False),
        encoding="utf-8")
    filled = sum(1 for v in template.values() if v)
    print(f"\nWrote {PDF_URLS_JSON}  ({filled}/{len(template)} URLs filled)")
    print("  Open that file and paste the canonical PDF URL next to each filename.")


if __name__ == "__main__":
    main()
