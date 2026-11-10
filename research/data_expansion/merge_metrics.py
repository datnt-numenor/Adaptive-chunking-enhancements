"""Merge per-GPU pod metrics with the documents already scored on the laptop into one chunking_metrics.parquet
per run. Unit = document (all methods of a document come from one machine). Laptop wins only for documents the pod
skipped (skip.json); every other document comes from the pod. Overlaps are compared (cross-machine check).
  python -X utf8 research/data_expansion/merge_metrics.py --pod runs/gpu_metrics_ext_v1_results/extracted_verified/metrics_out \
      --laptop ext_v1=runs/ext_v1/metrics --laptop sdp_a=runs/sdp_ext/A_target600/metrics \
      --laptop sdp_b=runs/sdp_ext/B_default/metrics --out runs/ext_metrics_merged_v1"""
import argparse, json, sys
from pathlib import Path
import pandas as pd

FIVE = ["size_compliance", "block_integrity", "intrachunk_cohesion", "document_contextual_coherence", "references_completeness"]
ap = argparse.ArgumentParser()
ap.add_argument("--pod", type=Path, required=True)
ap.add_argument("--laptop", action="append", default=[])
ap.add_argument("--chunks", action="append", default=["ext_v1=runs/ext_v1", "sdp_a=runs/sdp_ext/A_target600", "sdp_b=runs/sdp_ext/B_default"])
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
if a.out.exists():
    sys.exit(f"{a.out} exists")
laptop = dict(x.split("=", 1) for x in a.laptop)
chunks = dict(x.split("=", 1) for x in a.chunks)
report = {}
for run, rdir in chunks.items():
    pod = pd.concat([pd.read_parquet(p) for p in sorted((a.pod / run).glob("gpu*/chunking_metrics.parquet"))])
    lap = pd.read_parquet(Path(laptop[run]) / "chunking_metrics.parquet") if run in laptop and (Path(laptop[run]) / "chunking_metrics.parquet").exists() else pod.iloc[:0]
    lap_only = sorted(set(lap.doc_name) - set(pod.doc_name))
    both = sorted(set(lap.doc_name) & set(pod.doc_name))
    merged = pd.concat([pod, lap[lap.doc_name.isin(lap_only)]], ignore_index=True)
    c = pd.read_parquet(Path(rdir) / "chunks" / "small_merged" / "chunks.parquet")
    expected = set(map(tuple, c[["doc_name", "method"]].drop_duplicates().to_numpy()))
    got = set(map(tuple, merged[merged.metric_name == "size_compliance"][["doc_name", "chunking_method"]].to_numpy()))
    dup = merged.duplicated(["doc_name", "chunking_method", "metric_name"]).sum()
    diff = None
    if both:
        k = ["doc_name", "chunking_method", "metric_name"]
        m = pod[pod.metric_name.isin(FIVE)].merge(lap[lap.metric_name.isin(FIVE)], on=k, suffixes=("_pod", "_lap"))
        diff = float((m.score_pod.astype(float) - m.score_lap.astype(float)).abs().max())
    report[run] = {"docs_pod": int(pod.doc_name.nunique()), "docs_laptop_only": lap_only, "docs_overlap": len(both),
                   "max_abs_diff_overlap_pod_vs_laptop": diff, "missing_doc_method": len(expected - got),
                   "extra_doc_method": len(got - expected), "duplicates": int(dup)}
    print(run, json.dumps({k: (v if k != "docs_laptop_only" else len(v)) for k, v in report[run].items()}))
    if expected - got or dup:
        sys.exit(f"{run}: incomplete or duplicated")
    (a.out / run).mkdir(parents=True)
    merged.to_parquet(a.out / run / "chunking_metrics.parquet")
(a.out / "merge_manifest.json").write_text(json.dumps({"pod": str(a.pod), "laptop": laptop, "runs": report}, indent=1, ensure_ascii=False), encoding="utf-8")
print("Saved ->", a.out)
