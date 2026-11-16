"""Buoc 5b: tong ket chi phi OpenAI va tim tai lieu ma LLM-regex khong tach duoc."""
import json
from pathlib import Path

import pandas as pd

lines = Path("runs/openai_cache/usage.jsonl").read_text(encoding="utf-8").splitlines()
usage = [json.loads(line) for line in lines]
print(f"So request da tra tien: {len(usage)}")
print(f"Model thuc te: {sorted({u['model'] for u in usage})}")
print(f"Token dau vao: {sum(u['input_tokens'] for u in usage):,}  dau ra: {sum(u['output_tokens'] for u in usage):,}")

raw = pd.read_parquet("runs/full/chunks/raw/chunks.parquet")
per_doc = raw[raw["method"] == "llm_regex"].groupby("doc_name").size()
print(f"\nSo tai lieu co chunk llm_regex: {len(per_doc)}")
failed = per_doc[per_doc == 1]
print(f"Tai lieu ma regex khong tach duoc (ca tai lieu = 1 chunk raw): {len(failed)}")
for name in failed.index:
    print("   ", name)