"""Choose an SDP configuration by document-level 5-fold CV (CLAUDE.md rule 5) and report held-out results.

For each fold k (runs/table5_v2/prepared/folds.jsonl, 5 folds, seed 2026):
  * candidate configs = the given run dirs that have full 5-metric scores (<run>/metrics/chunking_metrics.parquet);
  * pick the config with the highest mean Table-3 score (mean of the 5 metrics, per doc, averaged) on the
    training folds only;
  * record that config's per-document scores on the test fold.
The 33 held-out scores form "SDP (CV-selected)", compared per document with Adaptive (4 candidates),
our_recurs_1100 and page: mean difference, 10,000-sample document bootstrap CI, Wilcoxon (Pratt), Holm.

Usage:
    python -X utf8 research/structure_chunker/tune.py --out runs/sdp/tune_<name> runs/sdp/A_target600 runs/sdp/A_target800 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import BASE, CANDIDATES, FIVE, adaptive, boot_ci, pivot  # noqa: E402

FOLDS = Path("runs/table5_v2/prepared/folds.jsonl")


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj, running = [0.0] * len(pvals), 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[idx]))
        adj[idx] = running
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    folds = pd.read_json(FOLDS, lines=True).set_index("doc_name")["fold"]
    per_doc = {}
    for run in args.runs:
        f = run / "metrics" / "chunking_metrics.parquet"
        if not f.exists():
            sys.exit(f"{run}: no full metrics yet ({f})")
        t = pivot(pd.read_parquet(f))
        method = t.index.get_level_values(0).unique()
        assert len(method) == 1, run
        doc_scores = t.loc[method[0]].mean(axis=1)  # NaN metrics skipped, as in the paper's selector
        missing = set(folds.index) - set(doc_scores.index)
        if missing:
            sys.exit(f"{run}: missing {len(missing)} docs")
        per_doc[run.name] = doc_scores
    scores = pd.DataFrame(per_doc).loc[folds.index]

    chosen, held_out = {}, pd.Series(index=folds.index, dtype=float)
    for k in sorted(folds.unique()):
        train = folds.index[folds != k]
        test = folds.index[folds == k]
        best = scores.loc[train].mean().idxmax()
        chosen[int(k)] = {"config": best, "train_means": scores.loc[train].mean().round(3).to_dict(),
                          "test_mean": round(float(scores.loc[test, best].mean()), 3)}
        held_out.loc[test] = scores.loc[test, best]
        print(f"fold {k}: train-best = {best:18} test mean = {scores.loc[test, best].mean():.2f}")

    base = pivot(pd.read_parquet(BASE))
    ada, _ = adaptive(base, CANDIDATES)
    refs = {"Adaptive (4 cand.)": ada.mean(axis=1), "our_recurs_1100": base.loc["our_recurs_1100"].mean(axis=1),
            "page": base.loc["page"].mean(axis=1)}
    rows = []
    for name, ref in refs.items():
        diff = (held_out - ref.loc[held_out.index]).to_numpy()
        lo, hi = boot_ci(diff)
        p = wilcoxon(diff, zero_method="pratt").pvalue if np.any(diff) else 1.0
        rows.append({"reference": name, "sdp_cv_mean": held_out.mean(), "ref_mean": ref.loc[held_out.index].mean(),
                     "diff": diff.mean(), "ci_low": lo, "ci_high": hi, "wins": int((diff > 0).sum()),
                     "losses": int((diff < 0).sum()), "p": p})
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res["p"].tolist())
    print(f"\nSDP (CV-selected), held-out mean over 33 docs: {held_out.mean():.2f}")
    print(res.round(4).to_string(index=False))

    args.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out / "comparisons.csv", index=False)
    held_out.rename("sdp_cv").to_csv(args.out / "held_out_scores.csv")
    scores.to_csv(args.out / "per_config_doc_scores.csv")
    (args.out / "manifest.json").write_text(json.dumps({
        "runs": [str(r) for r in args.runs], "folds": str(FOLDS), "criterion": "mean of 5 Table-3 metrics on train folds",
        "chosen_per_fold": chosen, "baseline": str(BASE), "bootstrap": {"n": 10000, "seed": 2026},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
