"""Buoc 6: tao tap con vai tai lieu de do thoi gian tinh metrics.

Cach dung: python -X utf8 runs\make_subset.py [so_tai_lieu=2] [so_token_muc_tieu=10000]
Chon cac tai lieu co do dai gan muc tieu nhat, chep JSON + mentions sang runs/subset/.
"""
import json
import shutil
import sys
from pathlib import Path

import tiktoken

SRC = Path("data/clair")
DST = Path("runs/subset")
count = int(sys.argv[1]) if len(sys.argv) > 1 else 2
target = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

enc = tiktoken.get_encoding("o200k_base")
docs = []
for path in (SRC / "adi_parsed").glob("*.json"):
    tokens = len(enc.encode(json.loads(path.read_text(encoding="utf-8"))["full_text"]))
    docs.append((abs(tokens - target), tokens, path))
docs.sort()

if DST.exists():
    shutil.rmtree(DST)
(DST / "adi_parsed").mkdir(parents=True)
(DST / "mentions").mkdir()

corpus_tokens = sum(tokens for _, tokens, _ in docs)
subset_tokens = 0
for _, tokens, path in docs[:count]:
    shutil.copy2(path, DST / "adi_parsed" / path.name)
    mentions = [m for m in (SRC / "mentions").iterdir() if m.stem == path.stem]
    for mention in mentions:
        shutil.copy2(mention, DST / "mentions" / mention.name)
    subset_tokens += tokens
    print(f"{tokens:>8,} token  {len(mentions)} file mentions  {path.stem}")

print(f"Tap con: {subset_tokens:,} / {corpus_tokens:,} token = {subset_tokens / corpus_tokens:.2%} corpus")
print(f"He so nhan de uoc luong ca corpus: x{corpus_tokens / subset_tokens:.1f}")