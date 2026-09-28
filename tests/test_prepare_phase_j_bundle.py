"""Local checks for the Phase J transport allowlist and checksums."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "research/week3/prepare_phase_j_bundle.py"
spec = importlib.util.spec_from_file_location("prepare_phase_j_bundle", SCRIPT)
assert spec and spec.loader
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def put(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fixture_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    week3 = root / "research/week3"
    artifacts = week3 / "artifacts"
    prepared = artifacts / "phase-j-bi-neutral/prepared"
    monkeypatch.setattr(bundle, "ROOT", root)
    monkeypatch.setattr(bundle, "WEEK3", week3)
    monkeypatch.setattr(bundle, "ARTIFACTS", artifacts)
    monkeypatch.setattr(bundle, "OUTPUT", artifacts / "phase-j-bi-neutral/runpod_phase_j.zip")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        put(root / name, name)
    put(root / "src/adaptive_chunking/__init__.py", "")
    for name in ("phase_g_runner.py", "requirements-gpu.txt", "run_phase_j_remote.sh"):
        put(week3 / name, name)
    qa_rows = [{"qa_id": f"doc-{i}::q{j}", "doc_id": f"doc-{i}"} for i in range(33) for j in range(3)]
    qa_text = "".join(json.dumps(row) + "\n" for row in qa_rows)
    put(artifacts / "qa-full/qa_frozen.jsonl", qa_text)
    qa_hash = hashlib.sha256((artifacts / "qa-full/qa_frozen.jsonl").read_bytes()).hexdigest()
    monkeypatch.setattr(bundle, "QA_SHA256", qa_hash)
    put(artifacts / "qa-full/qa_frozen_manifest.json", json.dumps({
        "qa_sha256": qa_hash, "qa_count": 99, "documents": 33,
    }))
    system_text = "".join(json.dumps({"doc_id": f"doc-{i}"}) + "\n" for i in range(33))
    put(prepared / "systems/adaptive_bi_neutral.jsonl", system_text)
    put(prepared / "folds.jsonl", system_text)
    put(prepared / "adaptive_bi_neutral_selections.json", "{}")
    put(prepared / "prepare_manifest.json", json.dumps({
        "status": "complete",
        "source_commit": bundle.SOURCE_COMMIT,
        "fold_seed": 2026,
        "folds_sha256": hashlib.sha256((prepared / "folds.jsonl").read_bytes()).hexdigest(),
        "selections_sha256": hashlib.sha256((prepared / "adaptive_bi_neutral_selections.json").read_bytes()).hexdigest(),
        "systems": {"adaptive_bi_neutral": {
            "path": "systems/adaptive_bi_neutral.jsonl",
            "sha256": hashlib.sha256((prepared / "systems/adaptive_bi_neutral.jsonl").read_bytes()).hexdigest(),
            "documents": 33,
        }},
    }))
    put(artifacts / "prepared/prepare_manifest.json", "{}")
    put(root / ".env", "DO_NOT_PACKAGE")
    put(artifacts / "index-full/document_store.json", "DO_NOT_PACKAGE")
    return root, prepared


def test_bundle_contains_only_allowlisted_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_tree(tmp_path, monkeypatch)
    bundle.main()
    manifest = json.loads(bundle.OUTPUT.with_suffix(".zip.manifest.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(bundle.OUTPUT) as archive:
        names = set(archive.namelist())
        assert names == {row["path"] for row in manifest["files"]}
        assert all(hashlib.sha256(archive.read(row["path"])).hexdigest() == row["sha256"] for row in manifest["files"])
    assert "research/week3/artifacts/index-full/document_store.json" not in names
    assert ".env" not in names
    assert hashlib.sha256(bundle.OUTPUT.read_bytes()).hexdigest() == manifest["archive_sha256"]
    sidecar = bundle.OUTPUT.with_suffix(".zip.sha256").read_bytes()
    assert sidecar.endswith(b"\n") and b"\r" not in sidecar


def test_bundle_rejects_modified_system(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, prepared = fixture_tree(tmp_path, monkeypatch)
    with (prepared / "systems/adaptive_bi_neutral.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"doc_id":"doc-0"}\n')
    with pytest.raises(ValueError, match="checksum mismatch"):
        bundle.validate_inputs(bundle.selected_files())
