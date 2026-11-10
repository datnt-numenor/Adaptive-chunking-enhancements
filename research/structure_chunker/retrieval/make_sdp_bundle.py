"""Zip the group harness + a prepared SDP retrieval folder + launcher/remote script for the GPU machine.

Usage (cmd):
  python -X utf8 research\\structure_chunker\\retrieval\\make_sdp_bundle.py ^
     --repo D:\\Project\\Adaptive-chunking-enhancements --input runs\\table5_sdp_v1 ^
     --systems adaptive,raw__langch_recurs_default,raw__page,sdp_cv,sdp_b --out-dir runs\\gpu_sdp_v1
Output: table5_sdp_bundle.zip + .sha256. Inside, the input folder is renamed table5_sdp/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_FILES = ["pyproject.toml", "README.md", "LICENSE", "NOTICE",
              "research/week3/phase_g_runner.py", "research/week3/requirements-gpu.txt"]
PARTS = ["prepared", "qa-full", "qa-smoke"]
SKIP = {"__pycache__", ".git", ".venv", ".venv-judge", ".pytest_cache"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and not SKIP.intersection(p.relative_to(root).parts) and p.suffix != ".pyc":
            yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True, help="folder made by prepare_sdp_retrieval.py")
    ap.add_argument("--systems", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    out_zip = args.out_dir / "table5_sdp_bundle.zip"
    if out_zip.exists():
        sys.exit(f"{out_zip} already exists")
    manifest_in = json.loads((args.input / "prepared" / "prepare_manifest.json").read_text(encoding="utf-8"))
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems) - set(manifest_in["systems"])
    if unknown:
        sys.exit(f"systems not in prepared manifest: {sorted(unknown)}")
    sdp = [s for s in systems if s in manifest_in.get("sdp_extension", {}).get("new_systems", [])]
    for part in ("qa-full", "qa-smoke"):
        m = json.loads((args.input / part / "qa_frozen_manifest.json").read_text(encoding="utf-8"))
        if sha(args.input / part / "qa_frozen.jsonl") != m["qa_sha256"]:
            sys.exit(f"{part}: checksum mismatch")

    commit = subprocess.run(["git", "-C", str(args.repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(args.repo), "status", "--porcelain", "--", "src", "research/week3/phase_g_runner.py"],
                           capture_output=True, text=True).stdout.strip()

    entries = [(args.repo / f, f) for f in REPO_FILES]
    entries += [(p, "src/" + p.relative_to(args.repo / "src").as_posix()) for p in walk(args.repo / "src")]
    for part in PARTS:
        entries += [(p, f"table5_sdp/{part}/" + p.relative_to(args.input / part).as_posix()) for p in walk(args.input / part)]
    entries += [(HERE / "sdp_launcher.py", "sdp_launcher.py"),
                (HERE / "run_sdp_retrieval_remote.sh", "run_sdp_retrieval_remote.sh")]

    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "harness_repo_commit": commit,
                "harness_uncommitted_changes": dirty, "input": str(args.input), "systems": systems,
                "sdp_systems": sdp, "files": {arc: sha(p) for p, arc in entries}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, arc in entries:
            data = path.read_bytes()
            zf.writestr(arc, data.replace(b"\r\n", b"\n") if arc.endswith(".sh") else data)
        zf.writestr("table5_sdp/systems.txt", ",".join(systems) + "\n")
        zf.writestr("table5_sdp/sdp_systems.txt", ",".join(sdp) + "\n")
        zf.writestr("bundle_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    digest = sha(out_zip)
    (args.out_dir / "table5_sdp_bundle.zip.sha256").write_bytes(f"{digest}  table5_sdp_bundle.zip\n".encode())
    print(f"{out_zip} ({out_zip.stat().st_size / 1e6:.1f} MB, {len(entries)} files), sha256 {digest}")
    print(f"systems: {systems}; SDP: {sdp}; harness commit {commit}" + (f"  WARNING dirty: {dirty}" if dirty else ""))


if __name__ == "__main__":
    main()
