"""Round-trip tests for Kaggle-safe filename packaging."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


packager = _load_module(
    "week2_prepare_kaggle_upload",
    ROOT / "research/week2/prepare_kaggle_upload.py",
)
restorer = _load_module(
    "week2_restore_kaggle_upload",
    ROOT / "research/week2/restore_kaggle_upload.py",
)


def test_kaggle_safe_archive_round_trip_preserves_special_filenames(
    tmp_path: Path,
):
    data_shards = tmp_path / "data_shards"
    shard = data_shards / "shard-00"
    parsed = shard / "adi_parsed"
    mentions = shard / "mentions"
    parsed.mkdir(parents=True)
    mentions.mkdir()
    original_name = "Document & author's résumé"
    payloads = {
        parsed / f"{original_name}.json": b"parsed",
        mentions / f"{original_name}.json": b"mention-json",
        mentions / f"{original_name}.parquet": b"mention-parquet",
    }
    for path, payload in payloads.items():
        path.write_bytes(payload)
    (data_shards / "shards_manifest.json").write_text("{}", encoding="utf-8")
    (shard / "shard_manifest.json").write_text("{}", encoding="utf-8")
    runner = tmp_path / "reproduction_runner.py"
    restore_script = tmp_path / "restore_kaggle_upload.py"
    runner.write_text("# runner", encoding="utf-8")
    restore_script.write_text("# restorer", encoding="utf-8")

    archive_path = tmp_path / "upload.zip"
    result = packager.create_kaggle_safe_zip(
        data_shards, runner, restore_script, archive_path
    )

    assert result["document_count"] == 1
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert all(packager.SAFE_ARCHIVE_NAME.fullmatch(name) for name in names)
        assert all("&" not in name and "'" not in name and "\\" not in name for name in names)
        extracted = tmp_path / "extracted"
        archive.extractall(extracted)

    restored = tmp_path / "restored"
    report = restorer.restore_kaggle_names(
        extracted / "data_shards_safe",
        extracted / "filename_mapping.json",
        restored,
    )
    assert report["restored_documents"] == 1
    for original_path, payload in payloads.items():
        relative = original_path.relative_to(data_shards)
        assert (restored / relative).read_bytes() == payload

    mapping = json.loads(
        (extracted / "filename_mapping.json").read_text(encoding="utf-8")
    )
    assert mapping["document_count"] == 1
