"""Check the pod's 4B proxy against the official A5000 harness run on CLAIR (same 99 questions).
  python compare_with_a5000.py <pod_clair_run> <a5000_per_query_metrics.jsonl>"""
import sys
import pandas as pd
pod = pd.read_json(f"{sys.argv[1]}/evaluation-full/per_query_metrics.jsonl", lines=True)
ref = pd.read_json(sys.argv[2], lines=True)
pod["system_id"] = pod.system_id.str.replace("@plain", "", regex=False).str.replace("@doc", "__doc", regex=False)
m = pod.merge(ref, on=["system_id", "qa_id"], suffixes=("_pod", "_a5000"))
for s, g in m.groupby("system_id"):
    for k in ("recall@10", "ndcg@10"):
        a, b = g[f"{k}_pod"], g[f"{k}_a5000"]
        print(f"{s:28} {k:10} pod={a.mean():.4f} a5000={b.mean():.4f} identical={(abs(a - b) < 1e-9).mean():.0%} r={a.corr(b):.3f} n={len(g)}")
