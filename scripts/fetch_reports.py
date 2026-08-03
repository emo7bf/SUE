"""
scripts/fetch_reports.py
------------------------
Download Global Impact / Sustainability / ESG reports listed in data/manifest.csv
into data/sample_data/<industry>/<company>/<year>.pdf.

Design goals:
- Never overwrite. Skip rows whose target file already exists.
- Skip rows whose report_url column is empty (manifest is safe to ship
  with blank URLs; users fill them in as they curate).
- Be a well-behaved HTTP client: identify ourselves, honor a modest timeout,
  and retry a couple of times on transient failure.
- Print a clear summary at the end so it's obvious what to fix next.

Usage:
    # Fetch every row whose report_url is populated:
    python scripts/fetch_reports.py

    # Restrict to a single industry:
    python scripts/fetch_reports.py --industry "Pharmaceuticals"

    # Restrict to a single company:
    python scripts/fetch_reports.py --company Pfizer

    # Dry-run (list what would be downloaded, do nothing):
    python scripts/fetch_reports.py --dry-run

The manifest schema (data/manifest.csv):
    industry,company,ticker,report_year,report_url,notes

The `report_url` column is intentionally blank in the shipped manifest.
Populate it with a direct link to a publicly available PDF (the company's
own investor-relations or sustainability site is the right source of truth).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.csv"
DATA_DIR = ROOT / "data" / "sample_data"

USER_AGENT = (
    "SUE-Corpus-Fetcher/1.0 (research; contact via repository) "
    "python-urllib"
)
TIMEOUT_SECONDS = 30
MAX_RETRIES = 2
RETRY_SLEEP_SECONDS = 3


def slugify(name: str) -> str:
    """Turn a display name into a filesystem-safe folder name."""
    s = name.strip()
    s = re.sub(r"[^\w\s\-\.&]+", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    return s or "unknown"


def target_path(industry: str, company: str, year: str) -> Path:
    return DATA_DIR / slugify(industry) / slugify(company) / f"{year}.pdf"


def download(url: str, dest: Path) -> Optional[str]:
    """Download `url` to `dest`. Returns None on success, error string on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                ctype = resp.headers.get("Content-Type", "")
                if "pdf" not in ctype.lower() and not url.lower().endswith(".pdf"):
                    return f"content-type not PDF: {ctype!r}"
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(64 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                tmp.replace(dest)
                return None
        except (HTTPError, URLError, TimeoutError) as e:
            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)
                continue
            return f"{type(e).__name__}: {e}"
        except Exception as e:
            return f"{type(e).__name__}: {e}"
    return "unknown error"


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--industry", type=str, default=None,
                    help="Only fetch rows for this industry (exact match).")
    ap.add_argument("--company", type=str, default=None,
                    help="Only fetch rows for this company (exact match).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be downloaded, then exit.")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"ERROR: manifest not found at {MANIFEST}", file=sys.stderr)
        return 2
    rows = load_manifest(MANIFEST)
    total = len(rows)

    if args.industry:
        rows = [r for r in rows if r.get("industry", "").strip() == args.industry]
    if args.company:
        rows = [r for r in rows if r.get("company", "").strip() == args.company]

    with_url = [r for r in rows if (r.get("report_url") or "").strip()]
    without_url = [r for r in rows if not (r.get("report_url") or "").strip()]

    print(f"Manifest: {total} rows total ({len(rows)} match filters)")
    print(f"  {len(with_url)} rows have a report_url populated")
    print(f"  {len(without_url)} rows are missing report_url (skipped)\n")

    downloaded, skipped_existing, failed = 0, 0, []
    for r in with_url:
        industry = r["industry"].strip()
        company = r["company"].strip()
        year = (r.get("report_year") or "unknown").strip()
        url = r["report_url"].strip()
        dest = target_path(industry, company, year)

        if dest.exists():
            print(f"  [skip] {industry}/{company}/{year}.pdf already exists")
            skipped_existing += 1
            continue

        rel = dest.relative_to(ROOT)
        if args.dry_run:
            print(f"  [dry-run] would fetch -> {rel}\n            from {url}")
            continue

        print(f"  [get ] {industry}/{company}/{year}.pdf")
        print(f"         from {url}")
        err = download(url, dest)
        if err is None:
            print(f"         ok  -> {rel}")
            downloaded += 1
        else:
            print(f"         FAIL: {err}")
            failed.append((company, err))

    print("\n---- summary ----")
    print(f"  downloaded:      {downloaded}")
    print(f"  skipped (had):   {skipped_existing}")
    print(f"  skipped (no URL): {len(without_url)}")
    print(f"  failed:          {len(failed)}")
    if failed:
        print("  failures:")
        for company, err in failed:
            print(f"    - {company}: {err}")
    if without_url and not args.dry_run:
        print("\n  Next step: populate the report_url column in data/manifest.csv")
        print("  for the rows above (each company's own IR / sustainability page")
        print("  is the right source of truth).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
