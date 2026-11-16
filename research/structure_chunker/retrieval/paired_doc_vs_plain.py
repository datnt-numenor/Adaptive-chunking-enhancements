"""Paired comparison <system>__doc vs <system> (doc-title prefix effect), document-level bootstrap + Wilcoxon (Pratt),
Holm across systems per metric. Usage: python -X utf8 paired_doc_vs_plain.py <run_dir> <out_dir>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

run, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists():
    sys.exit(f"{out} exists")
pq = pd.read_json(run / "evaluation-full" / "per_query_metrics.jsonl", lines=True)
pq["system_id"] = pq.system_id.str.replace("@plain", "", regex=False).str.replace("@doc", "__doc", regex=False)  # proxy naming
bases = sorted(s for s in pq.system_id.unique() if f"{s}__doc" in set(pq.system_id))
rows = []
for m in ["recall@10", "ndcg@10", "recall@5", "mrr@10"]:
    doc = pq.groupby(["system_id", "doc_id"])[m].mean()
    rng, block = np.random.default_rng(2026), []
    for s in bases:
        d = (doc.loc[f"{s}__doc"] - doc.loc[s]).to_numpy()
        idx = rng.integers(0, len(d), size=(10_000, len(d)))
        lo, hi = np.quantile(d[idx].mean(axis=1), [0.025, 0.975])
        block.append({"metric": m, "system": s, "plain": doc.loc[s].mean(), "doc": doc.loc[f"{s}__doc"].mean(),
                      "diff": d.mean(), "ci_low": lo, "ci_high": hi, "wins": int((d > 0).sum()),
                      "losses": int((d < 0).sum()), "p": wilcoxon(d, zero_method="pratt").pvalue if np.any(d) else 1.0})
    order, run_max = np.argsort([b["p"] for b in block]), 0.0
    for r, i in enumerate(order):
        run_max = max(run_max, min(1.0, (len(block) - r) * block[i]["p"]))
        block[i]["p_holm"] = run_max
    rows += block
res = pd.DataFrame(rows)
out.mkdir(parents=True)
res.to_csv(out / "doc_vs_plain.csv", index=False)
(out / "manifest.json").write_text(json.dumps({"run": str(run), "unit": "document (33)", "bootstrap": 10000,
                                               "seed": 2026, "holm": "across systems, per metric"}, indent=2))
print(res.round(4).to_string(index=False))
