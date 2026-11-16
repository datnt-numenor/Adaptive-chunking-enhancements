"""Buoc 7: tu tinh Table 3 + hang Adaptive (4 ung vien nhu paper) + Table 4, doi chieu voi paper.

Cach dung: python -X utf8 runs\table3_adaptive.py [thu_muc_ket_qua=runs/full_kaggle]
Doc results/ (chunk da hau xu ly) va results_raw/ (chunk tho). Ghi CSV vao <thu muc>/audit/.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/full_kaggle")
FIVE = {"references_completeness": "RC", "intrachunk_cohesion": "ICC",
        "document_contextual_coherence": "DCC", "block_integrity": "BI", "size_compliance": "SC"}
CANDIDATES = ["llm_regex", "our_recurs_1100", "our_recurs_600", "page"]   # cac hang * trong paper
# (ten hien thi, ten method, nguon) -> gia tri paper Table 3: RC, ICC, DCC, BI, SC, mean
ROWS = [
    ("* LLM regex",            "llm_regex",             "proc", (98.0, 70.9, 82.4, 98.1, 99.6, 89.80)),
    ("* our recursive 1100",   "our_recurs_1100",       "proc", (99.0, 66.6, 89.7, 98.1, 100.0, 90.68)),
    ("* our recursive 600",    "our_recurs_600",        "proc", (97.2, 69.6, 84.7, 94.8, 100.0, 89.24)),
    ("* page (post-processed)", "page",                 "proc", (97.2, 69.2, 86.4, 99.9, 99.9, 90.52)),
    ("† LC recursive 1100",    "langch_recurs_1100",    "raw",  (98.4, 64.7, 86.8, 98.6, 93.3, 88.37)),
    ("† LC recursive default", "langch_recurs_default", "raw",  (96.1, 65.6, 88.8, 95.0, 97.7, 88.62)),
    ("† page (raw)",           "page",                  "raw",  (97.1, 69.3, 86.1, 100.0, 92.7, 89.03)),
    ("† semantic",             "semantic",              "raw",  (97.5, 69.3, 76.3, 91.3, 48.1, 76.49)),
    ("† sentence",             "sentence",              "raw",  (86.3, 78.4, 72.5, 61.9, 67.2, 73.26)),
]
PAPER_ADAPTIVE = (99.0, 68.2, 88.8, 99.4, 99.9, 91.07)
PAPER_TABLE4 = {"page": 48, "our_recurs_1100": 42, "llm_regex": 6, "our_recurs_600": 3}


def load(sub):
    df = pd.read_parquet(RUN / sub / "chunking_metrics.parquet")
    missing = set(FIVE) - set(df["metric_name"])
    if missing:
        sys.exit(f"Khong thay metric {sorted(missing)}. Cac metric co trong file: {sorted(df['metric_name'].unique())}")
    df = df[df["metric_name"].isin(FIVE)]
    return df.pivot_table(index=["chunking_method", "doc_name"], columns="metric_name", values="score")[list(FIVE)] * 100


tables = {"proc": load("results"), "raw": load("results_raw")}
out_dir = RUN / "audit"
out_dir.mkdir(exist_ok=True)

# Adaptive: moi tai lieu chon ung vien co trung binh 5 metric cao nhat (bo qua metric NaN, nhu code goc)
cand = tables["proc"].loc[CANDIDATES]
doc_mean = cand.mean(axis=1).rename("mean").reset_index()
best_idx = doc_mean.groupby("doc_name")["mean"].idxmax()
selection = doc_mean.loc[best_idx, ["doc_name", "chunking_method"]].set_index("doc_name")["chunking_method"]
adaptive = pd.DataFrame([cand.loc[(m, d)] for d, m in selection.items()], index=selection.index)
selection.to_csv(out_dir / "adaptive_selection.csv")


def summarize(df):
    means, stds = df.mean(), df.std()
    return means, stds, means.mean()


lines, per_doc = [], {}
header = f"{'Method':26}" + "".join(f"{v:>13}" for v in FIVE.values()) + f"{'Mean':>8}{'Paper':>8}{'Delta':>7}"
print(header)
print("-" * len(header))
for label, method, src, paper in ROWS + [("Adaptive (4 ung vien)", None, None, PAPER_ADAPTIVE)]:
    if method is None:
        df = adaptive
    elif method in tables[src].index.get_level_values(0):
        df = tables[src].loc[method]
    else:
        print(f"{label:26}  (khong co - da bo qua khi chay)")
        continue
    per_doc[label] = df.mean(axis=1)
    means, stds, mean = summarize(df)
    cells = "".join(f"{means[k]:6.1f}±{stds[k]:<5.1f}" for k in FIVE)
    print(f"{label:26}{cells}{mean:8.2f}{paper[5]:8.2f}{mean - paper[5]:+7.2f}")
    lines.append({"method": label, **{FIVE[k]: round(means[k], 2) for k in FIVE},
                  "mean": round(mean, 2), "paper_mean": paper[5], "delta": round(mean - paper[5], 2),
                  **{f"paper_{FIVE[k]}": p for k, p in zip(FIVE, paper)}})
pd.DataFrame(lines).to_csv(out_dir / "table3_local_vs_paper.csv", index=False)

print("\nTable 4 - ty le chon cua Adaptive (paper: % tren 33 tai lieu)")
counts = selection.value_counts()
for method in CANDIDATES:
    n = int(counts.get(method, 0))
    print(f"  {method:18} {n:2d}/33 = {n / len(selection) * 100:5.1f}%   (paper {PAPER_TABLE4[method]}%)")

print("\nWilcoxon tung cap theo tai lieu: Adaptive so voi tung phuong phap (diem trung binh 5 metric)")
print("  Luu y: Adaptive la max cua 4 ung vien, nen so voi chinh ung vien thi luon >= theo cach xay dung.")
ada = per_doc["Adaptive (4 ung vien)"]
for label, scores in per_doc.items():
    if label.startswith("Adaptive"):
        continue
    pair = pd.concat([ada, scores], axis=1, join="inner").dropna()
    diff = pair.iloc[:, 0] - pair.iloc[:, 1]
    nonzero = int((diff != 0).sum())
    p = wilcoxon(pair.iloc[:, 0], pair.iloc[:, 1]).pvalue if nonzero else float("nan")
    print(f"  {label:26} n={len(pair):2d}  khac 0: {nonzero:2d}  chenh TB {diff.mean():+6.2f}  p={p:.2g}")
print(f"\nDa luu: {out_dir / 'table3_local_vs_paper.csv'} va {out_dir / 'adaptive_selection.csv'}")