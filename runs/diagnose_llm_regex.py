"""Buoc 5b audit: vi sao LLM-regex khong tach duoc mot so tai lieu?

Dung lai dung prompt ma repo goc gui cho OpenAI, tim phan hoi tuong ung trong
runs/openai_cache/, roi kiem tra regex: thieu the <regex>, regex loi, hay regex
hop le nhung khong khop cho nao trong tai lieu. Khong goi API.
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from adaptive_chunking.paper.replicate import _build_few_shot_prompt, count_tokens_func
from adaptive_chunking.paper.splitters import extract_llm_regex
from adaptive_chunking.splitters import combine_blocks

DATA = Path("data/clair/adi_parsed")
CACHE = Path("runs/openai_cache")
MODEL = "gpt-4o"  # ten model ma replicate.py truyen vao API

raw = pd.read_parquet("runs/full/chunks/raw/chunks.parquet")
per_doc = raw[raw["method"] == "llm_regex"].groupby("doc_name").size()
failed = sorted(per_doc[per_doc == 1].index)
base_prompt = _build_few_shot_prompt()

for doc_name in failed:
    text = json.loads((DATA / f"{doc_name}.json").read_text(encoding="utf-8"))["full_text"]
    context = combine_blocks(text.splitlines(keepends=True), max_tokens=8000, count_tokens_func=count_tokens_func)
    request = {"model": MODEL, "messages": [{"role": "user", "content": base_prompt + "<Input>" + context + "</Input>"}],
               "temperature": 0}
    key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    path = CACHE / f"{key}.json"

    print("=" * 100)
    print(doc_name)
    if not path.exists():
        print("  Khong tim thay phan hoi trong cache (prompt khac voi luc chay?)")
        continue
    answer = json.loads(path.read_text(encoding="utf-8"))["choices"][0]["message"]["content"]
    print(f"  Phan hoi cua LLM: {answer.strip()[:300]!r}")
    pattern = extract_llm_regex(answer)
    if pattern is None:
        print("  => Nguyen nhan: khong co the <regex> hoac regex khong hop le -> ca tai lieu thanh 1 chunk")
        continue
    matches = [m.start() for m in re.finditer(pattern, text) if m.start() > 0]
    in_sample = sum(1 for pos in matches if pos < len(context))
    print(f"  Regex: {pattern!r}")
    print(f"  So cho khop trong ca tai lieu: {len(matches)}  (trong doan 8.000 token LLM duoc xem: {in_sample})")
    if not matches:
        print("  => Nguyen nhan: regex hop le nhung khong khop cho nao -> ca tai lieu thanh 1 chunk")
    else:
        print("  => Regex co khop; can xem them regex_splitter (min_len, attach_to)")