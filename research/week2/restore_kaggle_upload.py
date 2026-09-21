"""Restore original document filenames from a Kaggle-safe upload bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Sequence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_child(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe relative path in mapping: {relative!r}")
    root_resolved = root.resolve()
    child = (root / relative_path).resolve()
    if child != root_resolved and root_resolved not in child.parents:
        raise ValueError(f"Mapped path escapes its root: {relative!r}")
    return child


def restore_kaggle_names(
    input_dir: Path,
    mapping_path: Path,
    output_dir: Path,
) -> dict:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    records = mapping.get("records", [])
    if not records:
        raise ValueError("Filename mapping contains no records")

    copied = 0
    skipped = 0
    for record in records:
        source = _safe_child(input_dir, record["safe_relative"])
        target = _safe_child(output_dir, record["original_relative"])
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = sha256(source)
        if source_hash != record["sha256"]:
            raise ValueError(f"SHA256 mismatch for safe input: {source}")
        if target.is_file():
            if sha256(target) != source_hash:
                raise ValueError(f"Existing restored file has different content: {target}")
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1

    restored_parsed = list(output_dir.glob("shard-*/adi_parsed/*.json"))
    expected_documents = int(mapping.get("document_count", 0))
    if len(restored_parsed) != expected_documents:
        raise ValueError(
            f"Expected {expected_documents} restored documents, found {len(restored_parsed)}"
        )
    return {
        "status": "ok",
        "output_dir": str(output_dir),
        "copied_files": copied,
        "already_valid_files": skipped,
        "restored_documents": len(restored_parsed),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = restore_kaggle_names(args.input_dir, args.mapping, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
