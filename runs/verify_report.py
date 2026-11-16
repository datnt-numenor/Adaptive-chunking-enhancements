"""In lai cac con so chinh tu artifact de doi chieu voi AUDIT_REPORT.md.

Cach dung (tu thu muc goc repo): python -X utf8 runs\verify_report.py
"""
import json
from pathlib import Path

import pandas as pd

RUN = Path("runs/full_kaggle")


def section(title):
    print(f"\n=== {title} ===")


section("Moi truong (runs/pip_freeze.txt) -> muc 2")
freeze = Path("runs/pip_freeze.txt")
if freeze.exists():
    wanted = ("torch", "pandas", "numpy", "stanza", "transformers", "sentence-transformers", "tiktoken", "openai")
    for line in freeze.read_text(encoding="utf-16" if freeze.read_bytes()[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8").splitlines():
        name = line.split("==")[0].split(" @ ")[0].strip().lower()
        if name in wanted:
            print(" ", line.strip())
else:
    print("  thieu runs/pip_freeze.txt")

section("OpenAI (runs/openai_cache/usage.jsonl) -> muc 2")
usage = [json.loads(x) for x in Path("runs/openai_cache/usage.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
print(f"  request: {len(usage)}  model: {sorted({u['model'] for u in usage})}")
print(f"  token vao: {sum(u['input_tokens'] for u in usage):,}  token ra: {sum(u['output_tokens'] for u in usage):,}")

section("Thoi gian shard Kaggle -> muc 3")
for status in sorted(Path("runs/kaggle").rglob("run_status.json")):
    manifest = json.loads((status.parent / "shard_manifest.json").read_text(encoding="utf-8"))
    jobs = json.loads(status.read_text(encoding="utf-8"))["jobs"]
    print(f"  {status.parent.parent.name:12} shard {manifest['shard']} ({len(manifest['docs'])} tai lieu): "
          + ", ".join(f"{k} rc={v['returncode']} {v['minutes']} phut" for k, v in jobs.items()))

section("Metric da gop -> muc 3")
for sub in ("results", "results_raw"):
    df = pd.read_parquet(RUN / sub / "chunking_metrics.parquet")
    print(f"  {sub}: {len(df):,} dong, {df['doc_name'].nunique()} tai lieu, {df['chunking_method'].nunique()} phuong phap, "
          f"trung lap {int(df.duplicated(['doc_name', 'chunking_method', 'metric_name']).sum())}")

section("Table 3 (audit/table3_local_vs_paper.csv) -> muc 4.3")
t3 = pd.read_csv(RUN / "audit" / "table3_local_vs_paper.csv")
print(t3[["method", "RC", "ICC", "DCC", "BI", "SC", "mean", "paper_mean", "delta"]].to_string(index=False))

section("Table 4 (audit/adaptive_selection.csv) -> muc 4.4")
print(pd.read_csv(RUN / "audit" / "adaptive_selection.csv", index_col=0).iloc[:, 0].value_counts().to_string())

section("Audit BI (audit/bi_audit_selections.csv) -> F6")
sel = pd.read_csv(RUN / "audit" / "bi_audit_selections.csv", index_col=0)
base = sel.iloc[:, 0]
for col in sel.columns:
    counts = sel[col].value_counts().to_dict()
    print(f"  {col:22} {counts}  doi so voi A: {int((sel[col] != base).sum())}")

section("LLM-regex that bai (runs/llm_regex_diagnosis.txt) -> F3")
diag = Path("runs/llm_regex_diagnosis.txt").read_text(encoding="utf-8", errors="replace")
failed = [line.strip() for line in diag.splitlines() if line.strip() and not line.startswith((" ", "="))]
print(f"  {len(failed)} tai lieu: {failed}")
chosen = pd.read_csv(RUN / "audit" / "adaptive_selection.csv", index_col=0).iloc[:, 0]
print(f"  Adaptive chon llm_regex o tai lieu that bai: {[d for d, m in chosen.items() if m == 'llm_regex' and d in failed]}")
