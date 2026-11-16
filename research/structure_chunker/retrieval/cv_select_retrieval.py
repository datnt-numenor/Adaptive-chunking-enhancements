"""Choose a chunking configuration by RETRIEVAL with document-level 5-fold CV, report held-out scores.

For each family (a list of candidate system ids), on each fold the system with the best mean metric on the
TRAINING documents (document-level mean of per-query scores) is chosen; its scores on the held-out documents
are kept. The 33 held-out document scores are then compared with reference systems (bootstrap CI over
documents, Wilcoxon Pratt, Holm across all comparisons of the run).

Input: one or more per_query_metrics.jsonl (harness or proxy_retrieval.py layout).

Usage (cmd):
  python -X utf8 research\\structure_chunker\\retrieval\\cv_select_retrieval.py ^
     --per-query runs\\proxy\\r1_repr runs\\proxy\\r2_fixed ^
     --family "sdp_cv@doc=sdp_a@doc,sdp_b@doc,sdp_bq@doc,sdp_a_def@doc" ^
     --references adaptive@doc,adaptive@plain --out runs\\proxy\\cv_r12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SEED, N_BOOT = 2026, 10_000


def holm(p):
    order, out, run = np.argsort(p), [0.0] * len(p), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        out[i] = run
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", nargs="+", type=Path, required=True, help="run dirs or jsonl files")
    ap.add_argument("--folds", type=Path, default=Path("runs/table5_v2/prepared/folds.jsonl"))
    ap.add_argument("--family", action="append", required=True, help='"name=sys1,sys2,..."')
    ap.add_argument("--references", required=True)
    ap.add_argument("--metric", default="recall@10")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists - choose a new folder")

    rows = []
    for p in args.per_query:
        f = p / "evaluation-full" / "per_query_metrics.jsonl" if p.is_dir() else p
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    pq = pd.DataFrame(rows).drop_duplicates(["system_id", "qa_id"], keep="last")
    folds = pd.read_json(args.folds, lines=True).set_index("doc_id")["fold"]
    metrics = ["recall@10", "ndcg@10", "recall@5", "hit@10", "mrr@10"]
    doc = pq.groupby(["system_id", "doc_id"])[metrics].mean()

    held = {}  # name -> DataFrame(doc_id -> metrics)
    choices = {}
    for fam in args.family:
        name, members = fam.split("=", 1)
        members = [m for m in members.split(",") if m]
        missing = [m for m in members if m not in doc.index.get_level_values(0)]
        if missing:
            raise SystemExit(f"{name}: unknown systems {missing}")
        parts, choices[name] = [], {}
        for k in sorted(folds.unique()):
            test = folds[folds == k].index
            train = folds[folds != k].index
            score = {m: doc.loc[m].loc[train, args.metric].mean() for m in members}
            best = max(members, key=lambda m: (score[m], -members.index(m)))
            choices[name][int(k)] = {"chosen": best, "train_scores": {m: round(v, 4) for m, v in score.items()}}
            parts.append(doc.loc[best].loc[test])
        held[name] = pd.concat(parts)
    for r in args.references.split(","):
        held[r] = doc.loc[r]

    table = pd.DataFrame({n: d[metrics].mean() for n, d in held.items()}).T
    print(table.round(4).to_string())
    for name, c in choices.items():
        print(f"{name}: " + ", ".join(f"fold{k}->{v['chosen']}" for k, v in c.items()))

    rng = np.random.default_rng(SEED)
    comp = []
    fams = [f.split("=", 1)[0] for f in args.family]
    refs = args.references.split(",")
    for m in ["recall@10", "ndcg@10"]:
        block = []
        for c in fams:
            for r in refs:
                d = (held[c][m] - held[r][m].reindex(held[c].index)).to_numpy()
                idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
                lo, hi = np.quantile(d[idx].mean(axis=1), [0.025, 0.975])
                p = wilcoxon(d, zero_method="pratt").pvalue if np.any(d) else 1.0
                block.append({"metric": m, "candidate": c, "reference": r, "diff": d.mean(), "ci_low": lo,
                              "ci_high": hi, "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p": p})
        comp += block
    for b, ph in zip(comp, holm([b["p"] for b in comp])):
        b["p_holm"] = ph
    comp = pd.DataFrame(comp)
    print("\n" + comp.round(4).to_string(index=False))

    args.out.mkdir(parents=True)
    table.to_csv(args.out / "held_out_summary.csv")
    comp.to_csv(args.out / "comparisons.csv", index=False)
    pd.concat({n: d[metrics] for n, d in held.items()}, names=["system", "doc_id"]).to_csv(args.out / "held_out_documents.csv")
    (args.out / "manifest.json").write_text(json.dumps({
        "metric_for_selection": args.metric, "families": args.family, "references": refs,
        "inputs": [str(p) for p in args.per_query], "folds": str(args.folds), "choices": choices,
        "bootstrap": {"n": N_BOOT, "seed": SEED}, "holm_family": "all comparisons in this run"}, indent=2),
        encoding="utf-8")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
