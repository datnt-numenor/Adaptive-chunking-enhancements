"""Zip everything the pod needs to score the clair_ext test set (run_metrics_remote.sh). Documents already scored
on the laptop (runs/ext_v1/metrics) go to skip.json so the pod does not repeat them.
  python -X utf8 research/data_expansion/make_metrics_bundle.py --out runs/gpu_metrics_ext_v1/metrics_bundle.zip"""
import argparse, hashlib, json, zipfile
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
if a.out.exists():
    raise SystemExit(f"{a.out} exists")
a.out.parent.mkdir(parents=True, exist_ok=True)
skip = {"ext_v1": sorted(pd.read_parquet("runs/ext_v1/metrics/chunking_metrics.parquet").doc_name.unique().tolist())}
files = [Path(p) for p in ["pyproject.toml", "README.md", "LICENSE", "research/structure_chunker/score_full.py",
                           "research/data_expansion/score_shard.py",
                           "runs/full_kaggle/chunks/small_merged/chunks.parquet",
                           "runs/full_kaggle/results/chunking_metrics.parquet",
                           "runs/ext_v1/chunks/small_merged/chunks.parquet",
                           "runs/sdp_ext/A_target600/chunks/small_merged/chunks.parquet",
                           "runs/sdp_ext/B_default/chunks/small_merged/chunks.parquet"]]
for d in ["src/adaptive_chunking", "data/clair_ext_set/adi_parsed", "data/clair_ext_set/mentions",
          "data/clair/adi_parsed", "data/clair/mentions"]:
    files += [p for p in Path(d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.as_posix())
    sh = Path("research/data_expansion/run_metrics_remote.sh").read_bytes().replace(b"\r\n", b"\n")
    z.writestr("run_metrics_remote.sh", sh)
    z.writestr("skip.json", json.dumps(skip, ensure_ascii=False, indent=1))
sha = hashlib.sha256(a.out.read_bytes()).hexdigest()
a.out.with_suffix(".zip.sha256").write_text(f"{sha}  {a.out.name}\n")
print(f"{len(files)} files, skip {len(skip['ext_v1'])} ext_v1 docs already scored -> {a.out} ({a.out.stat().st_size/1e6:.1f} MB) sha256 {sha}")
