"""Apply the pre-registered inclusion rule and build the test-set data folder.

Keeps parsed documents with 2,000-80,000 tokens (o200k_base), copies them to <out>/adi_parsed/ (the layout the
paper pipeline expects), and writes <out>/doc_ids.json (harness rule: "doc-" + sha256(name \\0 sha256(full_text))[:16])
and <out>/set_manifest.json (sha256 per document, excluded documents with reason).

Usage (cmd):  python -X utf8 research\\data_expansion\\build_set.py --parsed data\\clair_ext\\parsed --out data\\clair_ext_set
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("o200k_base")


def h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=80000)
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} exists - choose a new folder")
    (args.out / "adi_parsed").mkdir(parents=True)
    kept, excluded, ids = {}, {}, {}
    for p in sorted(args.parsed.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        n = len(ENC.encode(d["full_text"], disallowed_special=()))
        if not (args.min_tokens <= n <= args.max_tokens):
            excluded[p.stem] = f"{n} tokens"
            continue
        if not d["split_points"] or len(d["pages"]) < 2:
            excluded[p.stem] = "no split points / single page"
            continue
        shutil.copy2(p, args.out / "adi_parsed" / p.name)
        ids[p.stem] = "doc-" + h(f"{p.stem}\0{h(d['full_text'])}")[:16]
        kept[p.stem] = {"tokens": n, "pages": len(d["pages"]), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (args.out / "doc_ids.json").write_text(json.dumps(ids, indent=1, ensure_ascii=False), encoding="utf-8")
    (args.out / "set_manifest.json").write_text(json.dumps(
        {"rule": f"{args.min_tokens} <= tokens <= {args.max_tokens}", "kept": kept, "excluded": excluded},
        indent=1, ensure_ascii=False), encoding="utf-8")
    by_cat = {}
    for k in kept:
        by_cat[k.split("_", 1)[0]] = by_cat.get(k.split("_", 1)[0], 0) + 1
    print(f"kept {len(kept)} {by_cat}, excluded {len(excluded)}: {excluded}")
    print(f"total tokens {sum(v['tokens'] for v in kept.values()):,}")


if __name__ == "__main__":
    main()
