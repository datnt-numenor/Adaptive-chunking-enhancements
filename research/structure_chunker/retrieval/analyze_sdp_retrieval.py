"""Document-level statistics for the SDP retrieval run. Primary metric: recall@10 (user decision, 2026-11-09).

recall@k in the harness = fraction of evidence characters covered by the top-k chunks, so systems with larger
chunks get more text into the top 10. Two controls are reported next to recall@10:
  * top10_tokens   : tokens (o200k_base) actually retrieved in the top 10 (context cost);
  * recall@budget  : evidence coverage when the ranked list is cut at the same token budget for every system
                     (default: median top-10 tokens of the reference system). Computed from the saved ranked
                     chunks with the harness's own interval logic.

Unit = document (mean of its questions). 10,000-sample document bootstrap CI (seed 2026), Wilcoxon (Pratt),
Holm across the comparisons of each metric.

Usage (cmd):
  python -X utf8 research\\structure_chunker\\retrieval\\analyze_sdp_retrieval.py ^
     --run runs\\table5_sdp_results\\table5_sdp --prepared runs\\table5_sdp_v1\\prepared ^
     --candidates sdp_cv,sdp_b --references adaptive,raw__langch_recurs_default,raw__page ^
     --out runs\\table5_sdp_analysis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
from scipy.stats import wilcoxon

SEED, N_BOOT = 2026, 10_000
ENC = tiktoken.get_encoding("o200k_base")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def union_len(iv):
    total, cur = 0, None
    for s, e in sorted(iv):
        if cur is None or s > cur[1]:
            if cur:
                total += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    return total + (cur[1] - cur[0] if cur else 0)


def coverage(evidence, doc_id, chunks) -> float:
    spans = [(int(e["source_start"]), int(e["source_end"])) for e in evidence]
    total = sum(e - s for s, e in spans)
    cov = 0
    for es, ee in spans:
        ov = [(max(es, int(c["source_start"])), min(ee, int(c["source_end"]))) for c in chunks if c.get("doc_id") == doc_id]
        cov += union_len([(s, e) for s, e in ov if e > s])
    return cov / total


def holm(p):
    order, out, run = np.argsort(p), [0.0] * len(p), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        out[i] = run
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="folder with evaluation-full/ and retrieval-full/")
    ap.add_argument("--qa", type=Path, default=Path("runs/table5_v2/qa-full/qa_frozen.jsonl"))
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--references", default="adaptive,raw__langch_recurs_default,raw__page")
    ap.add_argument("--budget", type=int, default=0, help="token budget; 0 = median top-10 tokens of first reference")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists - choose a new folder")
    cands = [s for s in args.candidates.split(",") if s]
    refs = [s for s in args.references.split(",") if s]
    systems = refs + cands
    qas = {q["qa_id"]: q for q in read_jsonl(args.qa)}

    pq = pd.DataFrame(read_jsonl(args.run / "evaluation-full" / "per_query_metrics.jsonl"))
    pq = pq[pq.system_id.isin(systems)]
    extra = []
    ranked_by = {}
    for s in systems:
        for row in read_jsonl(args.run / "retrieval-full" / f"{s}.jsonl"):
            chunks = [r["meta"] | {"tokens": len(ENC.encode(r["content"]))} for r in row["results"]]
            ranked_by[(s, row["qa_id"])] = chunks
            extra.append({"system_id": s, "qa_id": row["qa_id"], "top10_tokens": sum(c["tokens"] for c in chunks[:10])})
    pq = pq.merge(pd.DataFrame(extra), on=["system_id", "qa_id"])
    budget = args.budget or int(pq[pq.system_id == refs[0]].top10_tokens.median())

    def budget_recall(row):
        qa, used, kept = qas[row.qa_id], 0, []
        for c in ranked_by[(row.system_id, row.qa_id)]:
            if used + c["tokens"] > budget:
                break
            kept.append(c)
            used += c["tokens"]
        return coverage(qa["evidence"], qa["doc_id"], kept)

    pq["recall@budget"] = pq.apply(budget_recall, axis=1)
    metrics = ["recall@10", "recall@budget", "recall@5", "ndcg@10", "mrr@10", "hit@10", "top10_tokens"]
    doc = pq.groupby(["system_id", "doc_id"])[metrics].mean()

    print(f"Token budget for recall@budget: {budget}")
    summary = pq.groupby("system_id")[metrics].mean().loc[systems]
    print(summary.round(4).to_string())

    rng = np.random.default_rng(SEED)
    rows = []
    for m in metrics[:-1]:
        block = []
        for c in cands:
            for r in refs:
                d = (doc.loc[c][m] - doc.loc[r][m]).dropna().to_numpy()
                idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
                lo, hi = np.quantile(d[idx].mean(axis=1), [0.025, 0.975])
                p = wilcoxon(d, zero_method="pratt").pvalue if np.any(d) else 1.0
                block.append({"metric": m, "candidate": c, "reference": r, "diff": d.mean(), "ci_low": lo, "ci_high": hi,
                              "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()), "p": p})
        for b, ph in zip(block, holm([b["p"] for b in block])):
            b["p_holm"] = ph
        rows += block
    comp = pd.DataFrame(rows)
    print("\n" + comp[comp.metric.isin(["recall@10", "recall@budget"])].round(4).to_string(index=False))

    args.out.mkdir(parents=True)
    summary.to_csv(args.out / "system_summary.csv")
    comp.to_csv(args.out / "comparisons.csv", index=False)
    doc.reset_index().to_csv(args.out / "document_metrics.csv", index=False)
    (args.out / "manifest.json").write_text(json.dumps({
        "primary": "recall@10", "budget_tokens": budget, "candidates": cands, "references": refs,
        "run": str(args.run), "qa": str(args.qa), "bootstrap": {"n": N_BOOT, "seed": SEED}, "unit": "document"},
        indent=2), encoding="utf-8")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
