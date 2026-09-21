"""Create a Kaggle-safe ZIP without changing repository document identities.

Kaggle rejects several characters that are legal in the source filenames,
including backslashes, ampersands, and ASCII apostrophes.  This packager stores
document files under stable ASCII aliases and includes a mapping consumed by
``restore_kaggle_upload.py`` inside the notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Sequence


SAFE_ARCHIVE_NAME = re.compile(r"^[A-Za-z0-9._/-]+$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _add_file(
    archive: zipfile.ZipFile,
    source: Path,
    archive_name: str,
) -> None:
    if not SAFE_ARCHIVE_NAME.fullmatch(archive_name):
        raise ValueError(f"Unsafe Kaggle archive entry: {archive_name!r}")
    archive.write(source, arcname=archive_name)


def create_kaggle_safe_zip(
    data_shards_dir: Path,
    runner_path: Path,
    restore_script_path: Path,
    output_zip: Path,
) -> dict:
    if not (data_shards_dir / "shards_manifest.json").is_file():
        raise FileNotFoundError(
            f"Missing data shard manifest: {data_shards_dir / 'shards_manifest.json'}"
        )
    for required in (runner_path, restore_script_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    records: list[dict] = []
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_zip.with_suffix(output_zip.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            top_manifest = data_shards_dir / "shards_manifest.json"
            safe_relative = "shards_manifest.json"
            _add_file(
                archive,
                top_manifest,
                f"data_shards_safe/{safe_relative}",
            )
            records.append(
                {
                    "safe_relative": safe_relative,
                    "original_relative": safe_relative,
                    "sha256": sha256(top_manifest),
                }
            )

            shard_dirs = sorted(
                path
                for path in data_shards_dir.glob("shard-*")
                if path.is_dir()
            )
            global_document_count = 0
            for shard_dir in shard_dirs:
                shard_name = shard_dir.name
                shard_manifest = shard_dir / "shard_manifest.json"
                if not shard_manifest.is_file():
                    raise FileNotFoundError(shard_manifest)
                manifest_relative = f"{shard_name}/shard_manifest.json"
                _add_file(
                    archive,
                    shard_manifest,
                    f"data_shards_safe/{manifest_relative}",
                )
                records.append(
                    {
                        "safe_relative": manifest_relative,
                        "original_relative": manifest_relative,
                        "sha256": sha256(shard_manifest),
                    }
                )

                parsed_dir = shard_dir / "adi_parsed"
                document_paths = sorted(parsed_dir.glob("*.json"))
                if not document_paths:
                    raise ValueError(f"No parsed documents in {parsed_dir}")
                for document_index, document_path in enumerate(document_paths):
                    alias = f"doc-{shard_name.removeprefix('shard-')}-{document_index:02d}"
                    original_stem = document_path.stem
                    related_files = [document_path]
                    mentions_dir = shard_dir / "mentions"
                    related_files.extend(sorted(mentions_dir.glob(f"{original_stem}.*")))
                    if len(related_files) < 2:
                        raise ValueError(
                            f"Missing mention artifacts for {original_stem!r}"
                        )
                    for source in related_files:
                        relative_parent = (
                            "adi_parsed" if source.parent == parsed_dir else "mentions"
                        )
                        safe_relative = (
                            f"{shard_name}/{relative_parent}/{alias}{source.suffix}"
                        )
                        original_relative = (
                            f"{shard_name}/{relative_parent}/{source.name}"
                        )
                        _add_file(
                            archive,
                            source,
                            f"data_shards_safe/{safe_relative}",
                        )
                        records.append(
                            {
                                "safe_relative": safe_relative,
                                "original_relative": original_relative,
                                "sha256": sha256(source),
                            }
                        )
                    global_document_count += 1

            mapping = {
                "schema_version": 1,
                "document_count": global_document_count,
                "file_count": len(records),
                "records": records,
            }
            archive.writestr(
                "filename_mapping.json",
                json.dumps(mapping, ensure_ascii=False, indent=2).encode("utf-8"),
            )
            _add_file(archive, runner_path, "reproduction_runner.py")
            _add_file(
                archive,
                restore_script_path,
                "restore_kaggle_upload.py",
            )

        with zipfile.ZipFile(temporary) as archive:
            names = archive.namelist()
            unsafe = [name for name in names if not SAFE_ARCHIVE_NAME.fullmatch(name)]
            if unsafe:
                raise ValueError(f"Unsafe entries remained in ZIP: {unsafe[:5]}")
            if len(names) != len(set(names)):
                raise ValueError("ZIP contains duplicate entry names")
        temporary.replace(output_zip)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "output_zip": str(output_zip),
        "zip_sha256": sha256(output_zip),
        "document_count": global_document_count,
        "mapped_file_count": len(records),
        "archive_entry_count": len(names),
        "all_entry_names_kaggle_safe": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-shards-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--restore-script", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = create_kaggle_safe_zip(
        args.data_shards_dir,
        args.runner,
        args.restore_script,
        args.output_zip,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
