"""Dong goi code harness (repo nhom) + thu muc table5_v2 thanh 1 file zip de dua len may GPU.

Chay tren may cua ban (Windows), dat file nay va run_table5_v2_remote.sh trong CUNG mot thu muc:
  python -X utf8 make_table5_bundle.py --repo D:\\Project\\Adaptive-chunking-enhancements
      --table5-dir D:\\Project\\adaptive-chunking\\runs\\table5_v2 --out-dir D:\\Project\\adaptive-chunking\\runs
Tao ra: table5_v2_bundle.zip + table5_v2_bundle.zip.sha256 (dinh dang cua lenh sha256sum -c).
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

REPO_FILES = ["pyproject.toml", "README.md", "LICENSE", "NOTICE",
              "research/week3/phase_g_runner.py", "research/week3/requirements-gpu.txt"]
TABLE5_PARTS = ["prepared", "qa-full", "qa-smoke"]
SKIP = {"__pycache__", ".git", ".venv", ".pytest_cache"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and not SKIP.intersection(path.relative_to(root).parts) and path.suffix != ".pyc":
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True, help="ban clone repo nhom (co research/week3)")
    parser.add_argument("--table5-dir", type=Path, required=True, help="thu muc do prepare_table5_new_qa.py tao ra")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    remote_script = Path(__file__).resolve().parent / "run_table5_v2_remote.sh"
    out_zip = args.out_dir / "table5_v2_bundle.zip"
    if out_zip.exists():
        sys.exit(f"{out_zip} da ton tai - xoa hoac doi --out-dir")
    missing = [f for f in REPO_FILES if not (args.repo / f).is_file()]
    missing += [p for p in TABLE5_PARTS if not (args.table5_dir / p).is_dir()]
    if not (args.repo / "src" / "adaptive_chunking").is_dir():
        missing.append("src/adaptive_chunking")
    if not remote_script.is_file():
        missing.append(str(remote_script))
    if missing:
        sys.exit("Thieu: " + ", ".join(missing))
    for part, expect in [("qa-full", "full"), ("qa-smoke", "smoke")]:
        m = json.loads((args.table5_dir / part / "qa_frozen_manifest.json").read_text(encoding="utf-8"))
        if m.get("mode") != expect or sha(args.table5_dir / part / "qa_frozen.jsonl") != m.get("qa_sha256"):
            sys.exit(f"{part}: manifest khong khop - tao lai bang prepare_table5_new_qa.py")

    try:
        commit = subprocess.run(["git", "-C", str(args.repo), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(args.repo), "status", "--porcelain", "--", "src", "research/week3"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", "git not available"

    entries: list[tuple[Path, str]] = [(args.repo / f, f) for f in REPO_FILES]
    entries += [(p, "src/" + p.relative_to(args.repo / "src").as_posix()) for p in walk(args.repo / "src")]
    for part in TABLE5_PARTS:
        entries += [(p, f"table5_v2/{part}/" + p.relative_to(args.table5_dir / part).as_posix())
                    for p in walk(args.table5_dir / part)]
    entries.append((remote_script, "run_table5_v2_remote.sh"))

    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "harness_repo_commit": commit,
                "harness_uncommitted_changes": dirty, "table5_dir": str(args.table5_dir),
                "files": {arc: sha(p) for p, arc in entries}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, arc in entries:
            if arc.endswith(".sh"):                      # bo CRLF cua Windows de bash chay duoc
                zf.writestr(arc, path.read_bytes().replace(b"\r\n", b"\n"))
            else:
                zf.write(path, arc)
        zf.writestr("bundle_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    digest = sha(out_zip)
    (args.out_dir / "table5_v2_bundle.zip.sha256").write_bytes(f"{digest}  table5_v2_bundle.zip\n".encode())
    print(f"Da tao {out_zip} ({out_zip.stat().st_size / 1e6:.1f} MB, {len(entries)} file)")
    print(f"sha256 {digest}\ncommit harness {commit}" + (f"  (CANH BAO thay doi chua commit: {dirty})" if dirty else ""))


if __name__ == "__main__":
    main()
