"""Compare SDP runs with the paper's candidates and the equal-weight Adaptive selector.

Baseline numbers come from the second reproduction (runs/full_kaggle/results, post-processed `*` rows).
For an SDP run, full 5-metric scores are read from <run>/metrics/chunking_metrics.parquet when present,
otherwise only the CPU metrics (SC, BI, RC) from <run>/cheap_metrics.parquet are shown.

Usage:
    python -X utf8 research/structure_chunker/compare.py runs/sdp/A_default [runs/sdp/other ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE = Path("runs/full_kaggle/results/chunking_metrics.parquet")
FIVE = {"references_completeness": "RC", "intrachunk_cohesion": "ICC",
        "document_contextual_coherence": "DCC", "block_integrity": "BI", "size_compliance": "SC"}
CANDIDATES = ["llm_regex", "our_recurs_1100", "our_recurs_600", "page"]


def pivot(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["metric_name"].isin(FIVE)]
    p = df.pivot_table(index=["chunking_method", "doc_name"], columns="metric_name", values="score", dropna=False) * 100
    return p.reindex(columns=list(FIVE))


def adaptive(table: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    cand = table.loc[methods]
    mean = cand.mean(axis=1).rename("m").reset_index()
    idx = mean.groupby("doc_name")["m"].idxmax()
    sel = mean.loc[idx].set_index("doc_name")["chunking_method"]
    return pd.DataFrame([cand.loc[(m, d)] for d, m in sel.items()], index=sel.index), sel


def boot_ci(diff: np.ndarray, n: int = 10000, seed: int = 2026) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_table(run: Path) -> pd.DataFrame:
    full = run / "metrics" / "chunking_metrics.parquet"
    return pivot(pd.read_parquet(full if full.exists() else run / "cheap_metrics.parquet"))


def main(runs: list[Path]) -> None:
    base = pivot(pd.read_parquet(BASE))
    docs = set(base.index.get_level_values(1))
    for run in runs:
        docs &= set(run_table(run).index.get_level_values(1))
    base = base[base.index.get_level_values(1).isin(docs)]
    print(f"Documents compared: {len(docs)}")
    ada, _ = adaptive(base, CANDIDATES)
    rows = {m: base.loc[m] for m in CANDIDATES}
    rows["Adaptive (4 cand.)"] = ada
    full_runs = []
    for run in runs:
        full = run / "metrics" / "chunking_metrics.parquet"
        t = run_table(run)
        t = t[t.index.get_level_values(1).isin(docs)]
        for method in t.index.get_level_values(0).unique():
            label = f"{run.name}:{method}"
            rows[label] = t.loc[method]
            if full.exists():
                full_runs.append((label, t.loc[method]))
                both = pd.concat([base.loc[CANDIDATES], t.loc[[method]]])
                rows[f"Adaptive + {label}"], sel = adaptive(both, CANDIDATES + [method])
                print(f"[{label}] chosen by Adaptive+5th candidate in {int((sel == method).sum())}/{len(sel)} docs")

    print(f"\n{'Method':34}" + "".join(f"{v:>8}" for v in FIVE.values()) + f"{'Mean5':>8}{'SC+BI+RC':>10}{'ICC+DCC':>9}")
    for label, df in rows.items():
        m = df.mean()
        mean5 = m.mean() if m.notna().all() else float("nan")
        cells = "".join(f"{m[k]:8.2f}" for k in FIVE)
        print(f"{label:34}{cells}{mean5:8.2f}{m[['size_compliance', 'block_integrity', 'references_completeness']].sum():10.2f}"
              f"{m[['intrachunk_cohesion', 'document_contextual_coherence']].sum():9.2f}")

    if full_runs:
        print("\nPer-document paired comparison of the 5-metric mean (document = unit)")
        ada_doc = ada.mean(axis=1)
        refs = {"Adaptive (4 cand.)": ada_doc, "our_recurs_1100": base.loc["our_recurs_1100"].mean(axis=1)}
        for label, df in full_runs:
            mine = df.mean(axis=1)
            for ref_name, ref in refs.items():
                pair = pd.concat([mine, ref], axis=1, join="inner").dropna()
                diff = (pair.iloc[:, 0] - pair.iloc[:, 1]).to_numpy()
                lo, hi = boot_ci(diff)
                p = wilcoxon(diff, zero_method="pratt").pvalue if np.any(diff) else float("nan")
                print(f"  {label} - {ref_name:20} mean {diff.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  "
                      f"wins {int((diff > 0).sum())}/{len(diff)}  Wilcoxon p={p:.3g}")


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]])
