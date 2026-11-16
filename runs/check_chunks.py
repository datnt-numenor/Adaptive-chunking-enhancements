"""Buoc 5: kiem tra ket qua chunking va doi chieu voi Table 2 cua paper."""
import sys
from pathlib import Path

import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/nosem")
# method: (stage dung de so sanh, paper mean, max, min, # chunks) -- paper Table 2
PAPER_TABLE2 = {
    "llm_regex":             ("small_merged", 518, 1146, 69, 2279),
    "our_recurs_1100":       ("small_merged", 878, 1141, 104, 1345),
    "our_recurs_600":        ("small_merged", 496, 691, 102, 2381),
    "page":                  ("small_merged", 663, 1146, 72, 1780),
    "langch_recurs_1100":    ("raw", 706, 1101, 2, 1675),
    "langch_recurs_default": ("raw", 773, 1364, 57, 1557),
    "page (raw)":            ("raw", 669, 1765, 0, 1765),
    "semantic":              ("raw", 693, 17146, 1, 1706),
    "sentence":              ("raw", 146, 1025, 4, 8125),
}

files = sorted(OUT.rglob("*.parquet"))
print("Cac file parquet tim thay:")
for f in files:
    print("  ", f)

stages = {f.parent.name: pd.read_parquet(f) for f in files if f.name == "chunks.parquet"}
for stage, df in stages.items():
    print(f"\n[{stage}] {len(df):,} chunks, {df['doc_name'].nunique()} tai lieu")
    print(df.groupby("method")["doc_name"].nunique().to_string())

print(f"\n{'method':24}{'stage':>14}{'mean':>7}{'max':>7}{'min':>6}{'#chunks':>9}   | paper mean/max/min/#chunks")
for name, (stage, *paper) in PAPER_TABLE2.items():
    method = name.replace(" (raw)", "")
    df = stages.get(stage)
    if df is None or method not in set(df["method"]):
        print(f"{name:24}{stage:>14}   (khong co - da bo qua?)")
        continue
    lens = df.loc[df["method"] == method, "chunk_len"]
    print(f"{name:24}{stage:>14}{round(lens.mean()):>7}{lens.max():>7}{lens.min():>6}{len(lens):>9}   | {paper}")