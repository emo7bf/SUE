"""
scripts/fetch_industry.py
-------------------------
Download every direct-PDF report listed in a industry's source file
(data/industries/<industry>_sources.json) into
data/sample_data/<industry>/<Company>/<year> <title>.pdf.

Design goals (mirrors scripts/fetch_reports.py):
- Never overwrite: rows whose target file already exists are skipped.
- Only url_kind == "direct_pdf" rows are fetched; hubs and archives are
  crawl seeds for a future discovery pass, not downloads.
- SHA-256 hash every downloaded PDF and drop exact duplicates (some
  companies serve the same file from multiple URLs).
- Corporate CDNs commonly refuse non-browser user agents, so we present
  a plain browser UA string.
- Write data/industries/<industry>_downloads.json mapping each saved
  filename to its full source record (ticker, tier, category, year,
  url), which scripts/ingest_industry.py uses to tag chunks.

Usage:
    python scripts/fetch_industry.py                       # P0 + P1
    python scripts/fetch_industry.py --priority P0         # primary reports only
    python scripts/fetch_industry.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent.parent
INDUSTRIES_DIR = ROOT / "data" / "industries"
DATA_DIR = ROOT / "data" / "sample_data"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/128.0 Safari/537.36"),
    "Accept": "application/pdf,*/*",
}
TIMEOUT_SECONDS = 60
MAX_RETRIES = 2
RETRY_SLEEP_SECONDS = 3


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\s\-\.&]+", "", (name or "").strip(), flags=re.UNICODE)
    return re.sub(r"\s+", "_", s) or "unknown"


def target_path(industry: str, rec: dict) -> Path:
    year = (rec.get("report_year") or "").strip()
    title = slugify(rec.get("document_title") or rec.get("category") or "report")
    fname = f"{year} {title}.pdf".strip() if year else f"{title}.pdf"
    return DATA_DIR / slugify(industry) / slugify(rec["company_name"]) / fname


def download(url: str, dest: Path) -> str | None:
    """Download url -> dest. Returns None on success, error string on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            with requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS,
                              stream=True, allow_redirects=True) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "")
                blocks = resp.iter_content(chunk_size=64 * 1024)
                first = next(blocks, b"")
                if b"%PDF" not in first[:1024] and "pdf" not in ctype.lower():
                    return f"not a PDF (content-type {ctype!r})"
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    f.write(first)
                    for block in blocks:
                        f.write(block)
                tmp.replace(dest)
                return None
        except requests.RequestException as e:
            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)
                continue
            return f"{type(e).__name__}: {e}"
    return "unknown error"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--industry", default="aerospace_defense",
                    help="Basename of data/industries/<industry>_sources.json")
    ap.add_argument("--priority", nargs="+", default=["P0", "P1"],
                    help="crawl_priority tiers to fetch (default: P0 P1)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_file = INDUSTRIES_DIR / f"{args.industry}_sources.json"
    if not src_file.exists():
        raise SystemExit(f"missing {src_file}")
    payload = json.loads(src_file.read_text(encoding="utf-8"))
    industry = payload["industry"]
    records = payload["sources"]

    todo = [r for r in records
            if r.get("url_kind") == "direct_pdf"
            and r.get("crawl_priority") in args.priority
            and (r.get("url") or "").strip()]
    print(f"{len(records)} source records; {len(todo)} direct PDFs at "
          f"priority {'/'.join(args.priority)}")

    downloads_file = INDUSTRIES_DIR / f"{args.industry}_downloads.json"
    downloads: dict = {}
    if downloads_file.exists():
        downloads = json.loads(downloads_file.read_text(encoding="utf-8"))

    seen_hashes: dict[str, str] = {
        v["sha256"]: k for k, v in downloads.items() if v.get("sha256")}
    fetched, skipped, deduped, failed = 0, 0, 0, []

    for rec in todo:
        dest = target_path(industry, rec)
        rel = dest.relative_to(ROOT)
        if dest.exists():
            print(f"  [skip] {rel} (exists)")
            skipped += 1
            continue
        if args.dry_run:
            print(f"  [dry ] {rel}\n         from {rec['url']}")
            continue
        print(f"  [get ] {rel}")
        err = download(rec["url"], dest)
        if err:
            print(f"         FAIL: {err}")
            failed.append((rec["company_name"], rec["url"], err))
            continue
        digest = sha256_of(dest)
        if digest in seen_hashes:
            print(f"         duplicate of {seen_hashes[digest]}; removed")
            dest.unlink()
            deduped += 1
            continue
        seen_hashes[digest] = dest.name
        downloads[dest.name] = {**rec, "sha256": digest,
                                "path": str(rel).replace("\\", "/")}
        fetched += 1

    if not args.dry_run:
        downloads_file.write_text(
            json.dumps(downloads, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"\nwrote {downloads_file} ({len(downloads)} records)")

    print("\n---- summary ----")
    print(f"  downloaded: {fetched}   skipped(existing): {skipped}   "
          f"deduped: {deduped}   failed: {len(failed)}")
    for company, url, err in failed:
        print(f"    - {company}: {err}\n      {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
