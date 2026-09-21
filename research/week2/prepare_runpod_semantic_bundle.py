"""Build and verify the secret-free input bundle for the RunPod semantic run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile


PINNED_COMMIT = "ea87ce8e1a97888f3f179e7f1359ff7f43fb179d"
FIXED_ZIP_TIMESTAMP = (2026, 9, 12, 0, 0, 0)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_bytes(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content)


def build_bundle(repository_root: Path, output_zip: Path) -> dict:
    parsed_dir = repository_root / "data" / "clair" / "adi_parsed"
    runner = repository_root / "research" / "week2" / "reproduction_runner.py"
    guide = repository_root / "research" / "week2" / "KAGGLE_REPRODUCTION_GUIDE.md"
    documents = sorted(parsed_dir.glob("*.json"), key=lambda path: path.name)
    if len(documents) != 33:
        raise ValueError(f"Expected 33 parsed documents, found {len(documents)}")
    for required in (runner, guide):
        if not required.is_file():
            raise FileNotFoundError(required)

    sources = [runner, guide, *documents]
    records = []
    payloads: list[tuple[str, bytes]] = []
    for source in sources:
        relative = source.relative_to(repository_root).as_posix()
        content = source.read_bytes()
        records.append(
            {"path": relative, "bytes": len(content), "sha256": sha256_bytes(content)}
        )
        payloads.append((relative, content))

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": PINNED_COMMIT,
        "purpose": "exact semantic chunker benchmark and resumable full run",
        "document_count": len(documents),
        "files": records,
    }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_zip.with_suffix(output_zip.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w") as archive:
        for relative, content in payloads:
            _write_bytes(archive, relative, content)
        _write_bytes(archive, "runpod_bundle_manifest.json", manifest_bytes)
    temporary.replace(output_zip)

    validate_bundle(output_zip)
    digest = sha256_file(output_zip)
    sidecar = output_zip.with_suffix(output_zip.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {output_zip.name}\n", encoding="ascii")
    return {
        "output_zip": str(output_zip),
        "sha256": digest,
        "sidecar": str(sidecar),
        "document_count": len(documents),
        "file_count": len(records) + 1,
    }


def validate_bundle(bundle: Path) -> dict:
    with zipfile.ZipFile(bundle) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"Corrupt ZIP member: {corrupt}")
        manifest = json.loads(archive.read("runpod_bundle_manifest.json"))
        expected = {record["path"]: record for record in manifest["files"]}
        names = set(archive.namelist()) - {"runpod_bundle_manifest.json"}
        if names != set(expected):
            raise ValueError("Bundle file list does not match its manifest")
        for name, record in expected.items():
            content = archive.read(name)
            if len(content) != record["bytes"]:
                raise ValueError(f"Size mismatch for {name}")
            if sha256_bytes(content) != record["sha256"]:
                raise ValueError(f"SHA256 mismatch for {name}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()
    result = build_bundle(args.repository_root.resolve(), args.output_zip.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
