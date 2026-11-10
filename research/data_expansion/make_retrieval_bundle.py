"""Zip what run_retrieval_ext_remote.sh needs (python extraction on the pod, never `unzip`).
  python -X utf8 research/data_expansion/make_retrieval_bundle.py --out runs/gpu_retrieval_ext_4b/retrieval_bundle.zip"""
import argparse, hashlib, json, zipfile
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
if a.out.exists():
    raise SystemExit(f"{a.out} exists")
a.out.parent.mkdir(parents=True, exist_ok=True)
HARN = Path("D:/Project/Adaptive-chunking-enhancements/research/week3/phase_g_runner.py")
files = {p: p.as_posix() for p in map(Path, [
    "pyproject.toml", "README.md", "LICENSE", "research/structure_chunker/retrieval/proxy_retrieval.py",
    "runs/qa_ext_v1/qa_frozen.jsonl", "runs/table5_v2/qa-full/qa_frozen.jsonl", "data/clair_ext_set/doc_ids.json"])}
for d in ["src/adaptive_chunking", "data/clair_ext_set/adi_parsed", "data/clair/adi_parsed"]:
    files.update({p: p.as_posix() for p in Path(d).rglob("*") if p.is_file() and "__pycache__" not in p.parts})
for prep, systems in [("runs/ext_prepared_v1/prepared", ["adaptive", "raw__page", "sdp_b", "raw__langch_recurs_default", "sdp_a_t600"]),
                      ("runs/table5_repr_v1/prepared", ["adaptive", "sdp_a_t600"])]:
    for f in ["prepare_manifest.json", "documents.jsonl"]:
        if (Path(prep) / f).exists():
            files[Path(prep) / f] = f"{prep}/{f}"
    for s in systems:
        files[Path(prep) / "systems" / f"{s}.jsonl"] = f"{prep}/systems/{s}.jsonl"
files[HARN] = "harness/research/week3/phase_g_runner.py"
files[Path("research/data_expansion/compare_with_a5000.py")] = "compare_with_a5000.py"
files[Path("runs/table5_repr_results/table5_sdp/evaluation-full/per_query_metrics.jsonl")] = "a5000_per_query_metrics.jsonl"
with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
    for src, arc in files.items():
        z.write(src, arc)
    z.writestr("run_retrieval_ext_remote.sh", Path("research/data_expansion/run_retrieval_ext_remote.sh").read_bytes().replace(b"\r\n", b"\n"))
sha = hashlib.sha256(a.out.read_bytes()).hexdigest()
a.out.with_suffix(".zip.sha256").write_text(f"{sha}  {a.out.name}\n")
print(f"{len(files) + 1} files -> {a.out} ({a.out.stat().st_size / 1e6:.1f} MB) sha256 {sha}")
