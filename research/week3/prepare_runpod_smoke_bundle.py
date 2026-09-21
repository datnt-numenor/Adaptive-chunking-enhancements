"""Create a secret-free, checksum-validated Phase G GPU smoke bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "research" / "week3" / "artifacts"
OUTPUT = ARTIFACTS / "runpod_phase_g_smoke.zip"
CHECKSUM = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")
MANIFEST = OUTPUT.with_suffix(OUTPUT.suffix + ".manifest.json")

SINGLE_FILES = (
    ROOT / "pyproject.toml",
    ROOT / "research" / "week3" / "phase_g_runner.py",
    ROOT / "research" / "week3" / "requirements-local.txt",
    ROOT / "research" / "week3" / "requirements-gpu.txt",
    ARTIFACTS / "qa-smoke" / "qa_frozen.jsonl",
    ARTIFACTS / "qa-smoke" / "qa_frozen_manifest.json",
)
TREE_ROOTS = (
    ROOT / "src",
    ARTIFACTS / "prepared",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files() -> list[Path]:
    files = list(SINGLE_FILES)
    for tree_root in TREE_ROOTS:
        files.extend(path for path in tree_root.rglob("*") if path.is_file())
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing bundle inputs: {missing}")
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    files = selected_files()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, relative)
            records.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )

    archive_sha256 = sha256(OUTPUT)
    CHECKSUM.write_text(f"{archive_sha256}  {OUTPUT.name}\n", encoding="ascii")
    MANIFEST.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "phase-g-two-document-gpu-smoke",
                "archive": OUTPUT.name,
                "archive_bytes": OUTPUT.stat().st_size,
                "archive_sha256": archive_sha256,
                "files": records,
                "excludes": [".env", "qa API caches", "QA candidates", "API keys"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"archive": str(OUTPUT), "sha256": archive_sha256, "files": len(files)}))


if __name__ == "__main__":
    main()
