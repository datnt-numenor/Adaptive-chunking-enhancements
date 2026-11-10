"""Download candidate PDFs listed in a TSV (name, category, lang, url) and record sha256 / pages / status.

Usage (cmd):  python -X utf8 research\\data_expansion\\download.py research\\data_expansion\\candidates_v0.tsv
Writes data/clair_ext/pdfs/<category>_<name>.pdf and research/data_expansion/download_log.csv (appends).
"""
from __future__ import annotations

import csv
import hashlib
import sys
import time
from pathlib import Path

import requests

OUT = Path("data/clair_ext/pdfs")
LOG = Path("research/data_expansion/download_log.csv")
UA = {"User-Agent": "Mozilla/5.0 (research corpus builder)"}


def main(tsv: str, sleep: float = 0.0) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"))
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["name", "category", "lang", "url", "status", "bytes", "pages", "sha256", "time"])
        for r in rows:
            path = OUT / f"{r['category']}_{r['name']}.pdf"
            status, size, pages, digest = "exists", 0, "", ""
            try:
                if not path.exists():
                    resp = requests.get(r["url"], headers=UA, timeout=60, allow_redirects=True)
                    data = resp.content
                    if resp.status_code != 200 or not data.startswith(b"%PDF"):
                        status = f"fail_{resp.status_code}_{resp.headers.get('content-type', '')[:30]}"
                        data = None
                    else:
                        path.write_bytes(data)
                        status = "ok"
                    time.sleep(sleep)
                if path.exists():
                    data = path.read_bytes()
                    size, digest = len(data), hashlib.sha256(data).hexdigest()
                    try:
                        import fitz  # PyMuPDF, if available
                        pages = fitz.open(path).page_count
                    except Exception:
                        pages = data.count(b"/Type /Page") - data.count(b"/Type /Pages") or ""
            except Exception as e:  # network errors
                status = f"error_{type(e).__name__}"
            w.writerow([r["name"], r["category"], r["lang"], r["url"], status, size, pages, digest,
                        time.strftime("%Y-%m-%d %H:%M:%S")])
            print(f"{status:30} {pages!s:>5} {r['name']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
