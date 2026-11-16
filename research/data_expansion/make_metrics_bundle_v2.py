"""Zip what the pod needs for the second look at clair_ext (run_metrics_remote_v2.sh) and freeze the chunk files.

Writes next to the zip: chunks_sha256.json (sha256 of every chunk file that will be scored; quoted by the
pre-registration) and <zip>.sha256.
  python -X utf8 research/data_expansion/make_metrics_bundle_v2.py --out runs/gpu_metrics_ext_v2/metrics_bundle_v2.zip"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

CHUNKS = {
    "ext/sw0_rc1": "runs/sdp_sweep_ext_v4/sw0_rc1/chunks/small_merged/chunks.parquet",
    "ext/rc1": "runs/sdp_sweep_ext_v4/rc1/chunks/small_merged/chunks.parquet",
    "ext/keepcont_rc1": "runs/sdp_sweep_ext_v4/keepcont_rc1/chunks/small_merged/chunks.parquet",
    "ext/noart_rc1": "runs/sdp_sweep_ext_v4/noart_rc1/chunks/small_merged/chunks.parquet",
    "clair/sw0_rc1": "runs/sdp_sweep_v2/sw0_rc1/chunks/small_merged/chunks.parquet",
    "clair/keepcont_rc1": "runs/sdp_sweep_v4/keepcont_rc1/chunks/small_merged/chunks.parquet",
    "clair/noart_rc1": "runs/sdp_sweep_v4/noart_rc1/chunks/small_merged/chunks.parquet",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
if a.out.exists():
    raise SystemExit(f"{a.out} exists")
a.out.parent.mkdir(parents=True, exist_ok=True)
frozen = {k: {"path": v, "sha256": sha(Path(v))} for k, v in CHUNKS.items()}
files = [Path(p) for p in ["pyproject.toml", "README.md", "LICENSE", "research/structure_chunker/score_full.py",
                           "research/data_expansion/score_shard.py",
                           "runs/full_kaggle/chunks/small_merged/chunks.parquet",      # for `validate`
                           "runs/full_kaggle/results/chunking_metrics.parquet"]] + [Path(v) for v in CHUNKS.values()]
for d in ["src/adaptive_chunking", "data/clair_ext_set/adi_parsed", "data/clair_ext_set/mentions",
          "data/clair/adi_parsed", "data/clair/mentions"]:
    files += [p for p in Path(d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.as_posix())
    sh = Path("research/data_expansion/run_metrics_remote_v2.sh").read_bytes().replace(b"\r\n", b"\n")
    z.writestr("run_metrics_remote_v2.sh", sh)
    z.writestr("chunks_sha256.json", json.dumps(frozen, indent=1))
(a.out.parent / "chunks_sha256.json").write_text(json.dumps(frozen, indent=1), encoding="utf-8")
digest = sha(a.out)
a.out.with_suffix(".zip.sha256").write_text(f"{digest}  {a.out.name}\n")
print(f"{len(files)} files -> {a.out} ({a.out.stat().st_size / 1e6:.1f} MB) sha256 {digest}")
for k, v in frozen.items():
    print(f"  {k:20} {v['sha256'][:16]}  {v['path']}")
