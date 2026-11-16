from pathlib import Path
import pandas as pd
root = Path("runs/gpu_metrics_ext_v2_results/extracted_verified/metrics_out/clair")
for run in ("sw0_rc1", "keepcont_rc1", "noart_rc1"):
    out = Path("runs/sdp_pod_clair_v1") / run / "metrics"
    out.mkdir(parents=True, exist_ok=False)
    d = pd.concat([pd.read_parquet(p) for p in sorted((root / run).glob("gpu*/chunking_metrics.parquet"))], ignore_index=True)
    d.to_parquet(out / "chunking_metrics.parquet")
    print(run, d.doc_name.nunique(), "docs ->", out)
# cross-machine detail for sw0_rc1
F = ["references_completeness", "intrachunk_cohesion", "document_contextual_coherence", "block_integrity", "size_compliance"]
lap = pd.read_parquet("runs/sdp_sweep_v2/sw0_rc1/metrics/chunking_metrics.parquet"); pod = pd.read_parquet("runs/sdp_pod_clair_v1/sw0_rc1/metrics/chunking_metrics.parquet")
m = lap[lap.metric_name.isin(F)].merge(pod[pod.metric_name.isin(F)], on=["doc_name", "metric_name"], suffixes=("_laptop", "_pod"))
m["abs"] = (m.score_laptop.astype(float) - m.score_pod.astype(float)).abs()
print("\nsw0_rc1 laptop (RTX 4060) vs pod (H100), 0-1 scale:")
print(m.groupby("metric_name")["abs"].agg(["max", "mean"]).round(5).to_string())
print(m.sort_values("abs", ascending=False).head(3)[["doc_name", "metric_name", "score_laptop", "score_pod", "abs"]].to_string(index=False))
