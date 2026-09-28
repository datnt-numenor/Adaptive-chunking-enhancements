"""Package only the inputs needed for the Phase J single-system GPU run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
WEEK3 = ROOT / "research" / "week3"
ARTIFACTS = WEEK3 / "artifacts"
SYSTEM = "adaptive_bi_neutral"
OUTPUT = ARTIFACTS / "phase-j-bi-neutral" / "runpod_phase_j.zip"
QA_SHA256 = "a33a07c0f52441a946f505cb34be199fcf1abbf9f01178aa8faa635407d7546a"
SOURCE_COMMIT = "ea87ce8e1a97888f3f179e7f1359ff7f43fb179d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files() -> dict[str, Path]:
    """Use a fixed allowlist; never recursively include generated artifacts."""
    prepared = ARTIFACTS / "phase-j-bi-neutral" / "prepared"
    files = {
        path.relative_to(ROOT).as_posix(): path
        for path in (
            ROOT / "pyproject.toml",
            ROOT / "README.md",
            ROOT / "LICENSE",
            WEEK3 / "phase_g_runner.py",
            WEEK3 / "requirements-gpu.txt",
            WEEK3 / "run_phase_j_remote.sh",
            ARTIFACTS / "qa-full" / "qa_frozen.jsonl",
            ARTIFACTS / "qa-full" / "qa_frozen_manifest.json",
            ARTIFACTS / "prepared" / "prepare_manifest.json",
            prepared / "prepare_manifest.json",
            prepared / f"systems/{SYSTEM}.jsonl",
            prepared / f"{SYSTEM}_selections.json",
            prepared / "folds.jsonl",
        )
    }
    files.update(
        {
            path.relative_to(ROOT).as_posix(): path
            for path in (ROOT / "src").rglob("*.py")
            if path.is_file()
        }
    )
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase J bundle inputs: {missing}")
    return dict(sorted(files.items()))


def validate_inputs(files: dict[str, Path]) -> None:
    qa = ARTIFACTS / "qa-full" / "qa_frozen.jsonl"
    qa_manifest = json.loads((ARTIFACTS / "qa-full" / "qa_frozen_manifest.json").read_text(encoding="utf-8"))
    if sha256(qa) != QA_SHA256 or qa_manifest.get("qa_sha256") != QA_SHA256:
        raise ValueError("Frozen 99-QA checksum differs from the locked Phase G set")
    if qa_manifest.get("qa_count") != 99 or qa_manifest.get("documents") != 33:
        raise ValueError("Frozen QA coverage must be 99 questions across 33 documents")
    qa_rows = [json.loads(line) for line in qa.read_text(encoding="utf-8").splitlines()]
    qa_doc_ids = {row["doc_id"] for row in qa_rows}
    if len(qa_rows) != 99 or len(qa_doc_ids) != 33:
        raise ValueError("Frozen QA content does not match 99 questions and 33 documents")
    prepared = ARTIFACTS / "phase-j-bi-neutral" / "prepared"
    manifest = json.loads((prepared / "prepare_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_commit") != SOURCE_COMMIT or manifest.get("fold_seed") != 2026:
        raise ValueError("Phase J prepared manifest differs from locked source commit or fold seed")
    system_record = manifest.get("systems", {}).get(SYSTEM)
    if manifest.get("status") != "complete" or not system_record:
        raise ValueError("Phase J prepared manifest is incomplete")
    system_path = prepared / system_record["path"]
    if system_path != prepared / f"systems/{SYSTEM}.jsonl":
        raise ValueError("Unexpected Phase J system path")
    if sha256(system_path) != system_record.get("sha256"):
        raise ValueError("Phase J system checksum mismatch")
    if system_record.get("documents") != 33:
        raise ValueError("Phase J system must cover 33 documents")
    system_doc_ids = {
        json.loads(line)["doc_id"]
        for line in system_path.read_text(encoding="utf-8").splitlines()
    }
    if system_doc_ids != qa_doc_ids:
        raise ValueError("Phase J system and frozen QA document sets differ")
    fold_doc_ids = {
        json.loads(line)["doc_id"]
        for line in (prepared / "folds.jsonl").read_text(encoding="utf-8").splitlines()
    }
    if fold_doc_ids != qa_doc_ids:
        raise ValueError("Phase J folds and frozen QA document sets differ")
    if sha256(prepared / "folds.jsonl") != manifest.get("folds_sha256"):
        raise ValueError("Phase J folds checksum mismatch")
    if sha256(prepared / f"{SYSTEM}_selections.json") != manifest.get("selections_sha256"):
        raise ValueError("Phase J selections checksum mismatch")
    if len(files) != len(set(files)):
        raise ValueError("Duplicate archive paths")


def main() -> None:
    files = selected_files()
    validate_inputs(files)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in files.items()
    ]
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
    archive_hash = sha256(OUTPUT)
    OUTPUT.with_suffix(".zip.sha256").write_bytes(
        f"{archive_hash}  {OUTPUT.name}\n".encode("ascii")
    )
    OUTPUT.with_suffix(".zip.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "phase-j-bi-neutral-single-system-gpu-run",
                "archive": OUTPUT.name,
                "archive_bytes": OUTPUT.stat().st_size,
                "archive_sha256": archive_hash,
                "frozen_qa_sha256": QA_SHA256,
                "files": records,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"archive": str(OUTPUT), "sha256": archive_hash, "files": len(files)}))


if __name__ == "__main__":
    main()
