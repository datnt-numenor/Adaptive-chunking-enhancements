"""Low-cost, resumable helpers for reproducing paper Tables 1--3.

The upstream algorithms live under ``src/adaptive_chunking`` and are not
modified here.  This module only shards independent documents, runs the
upstream semantic configuration one document at a time, merges artifacts, and
validates the resulting data layout.

Run ``python research/week2/reproduction_runner.py --help`` from the repository
root for the available commands.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


STAGES = ("raw", "no_oversizing", "small_merged")
PINNED_COMMIT = "ea87ce8e1a97888f3f179e7f1359ff7f43fb179d"
ALL_METHODS = {
    "page",
    "sentence",
    "langch_recurs_default",
    "langch_recurs_1100",
    "our_recurs_1100",
    "our_recurs_600",
    "semantic",
    "llm_regex",
}
NONSEMANTIC_METHODS = ALL_METHODS - {"semantic"}
RAW_METRIC_METHODS = {
    "page",
    "sentence",
    "semantic",
    "langch_recurs_1100",
    "langch_recurs_default",
}
CORE_METRICS = {
    "size_compliance",
    "block_integrity",
    "intrachunk_cohesion",
    "document_contextual_coherence",
    "references_completeness",
}
BASIC_METRICS = {
    "avg_chunk_tokens",
    "stddev_chunk_tokens",
    "max_chunk_tokens",
    "min_chunk_tokens",
    "num_chunks",
}
ALL_RECORDED_METRICS = CORE_METRICS | BASIC_METRICS

SEMANTIC_MODEL = "Qwen/Qwen3-Embedding-0.6B"
METRIC_EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
METRIC_EMBEDDING_MODEL_REVISION = "ab036b023d30b4d1138c4c3bfa9f0c445ab455d6"
METRIC_EMBEDDING_CODE_REPOSITORY = "jinaai/xlm-roberta-flash-implementation"
METRIC_EMBEDDING_CODE_REVISION = "bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3"
SEMANTIC_CONFIG = {
    "attention_implementation": "flash_attention_2",
    "dtype": "bfloat16",
    "batch_size": 16,
    "breakpoint_threshold_type": "gradient",
}
SEMANTIC_IDENTITY_KEYS = (
    "source_commit",
    "model",
    "model_revision",
    "attention_implementation",
    "dtype",
    "batch_size",
    "breakpoint_threshold_type",
)


class ReproductionError(RuntimeError):
    """Raised when an artifact violates a reproduction invariant."""


@dataclass(frozen=True)
class DocumentInfo:
    name: str
    path: Path
    weight: int


def _json_dump_atomic(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _parquet_write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_snapshot() -> dict:
    snapshot = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        if torch.cuda.is_available():
            snapshot["gpu"] = torch.cuda.get_device_name(0)
            snapshot["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    except ImportError:
        snapshot["torch"] = None
    return snapshot


def _assert_pinned_source_commit() -> None:
    """Refuse a semantic run when the upstream checkout is not the audited commit."""
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    actual = result.stdout.strip()
    if result.returncode != 0 or not actual:
        raise ReproductionError(
            f"Cannot verify the source commit under {repository_root}"
        )
    if actual != PINNED_COMMIT:
        raise ReproductionError(
            f"Semantic reproduction requires commit {PINNED_COMMIT}; found {actual}"
        )


def _resolve_parsed_dir(data_dir: Path) -> Path:
    parsed = data_dir / "adi_parsed"
    if not parsed.is_dir():
        raise ReproductionError(f"Missing parsed document directory: {parsed}")
    if not any(parsed.glob("*.json")):
        raise ReproductionError(f"No parsed JSON documents found in: {parsed}")
    return parsed


def _document_infos(data_dir: Path, weight_kind: str = "tokens") -> list[DocumentInfo]:
    parsed_dir = _resolve_parsed_dir(data_dir)
    count_tokens = None
    if weight_kind == "tokens":
        from adaptive_chunking.chunking_utils import count_tokens as count_tokens_func

        count_tokens = count_tokens_func

    documents: list[DocumentInfo] = []
    for path in sorted(parsed_dir.glob("*.json"), key=lambda value: value.name):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if weight_kind == "tokens":
            weight = int(count_tokens(payload["full_text"], model="gpt-4o"))
        else:
            weight = int(path.stat().st_size)
        documents.append(DocumentInfo(path.stem, path, weight))
    return documents


def balanced_assignments(
    weighted_items: Sequence[tuple[str, int]], shard_count: int
) -> list[list[tuple[str, int]]]:
    """Balance weights with deterministic LPT while keeping shard sizes even."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not weighted_items:
        raise ValueError("weighted_items must not be empty")
    if shard_count > len(weighted_items):
        raise ValueError("shard_count cannot exceed the number of items")
    if len({name for name, _ in weighted_items}) != len(weighted_items):
        raise ValueError("item names must be unique")

    shards: list[list[tuple[str, int]]] = [[] for _ in range(shard_count)]
    totals = [0] * shard_count
    base_size, remainder = divmod(len(weighted_items), shard_count)
    capacities = [
        base_size + (1 if index < remainder else 0)
        for index in range(shard_count)
    ]
    for name, weight in sorted(weighted_items, key=lambda item: (-item[1], item[0])):
        eligible = [
            index
            for index in range(shard_count)
            if len(shards[index]) < capacities[index]
        ]
        target = min(eligible, key=lambda index: (totals[index], index))
        shards[target].append((name, int(weight)))
        totals[target] += int(weight)
    for shard in shards:
        shard.sort(key=lambda item: item[0])
    return shards


def _copy_mentions_for_documents(
    source_mentions: Path, target_mentions: Path, document_names: set[str]
) -> list[str]:
    copied: list[str] = []
    if not source_mentions.is_dir():
        return copied
    target_mentions.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_mentions.iterdir(), key=lambda value: value.name):
        if path.is_file() and path.stem in document_names:
            shutil.copy2(path, target_mentions / path.name)
            copied.append(path.name)
    return copied


def make_data_shards(data_dir: Path, output_dir: Path, shard_count: int) -> dict:
    documents = _document_infos(data_dir, weight_kind="tokens")
    assignments = balanced_assignments(
        [(document.name, document.weight) for document in documents], shard_count
    )
    by_name = {document.name: document for document in documents}
    source_mentions = data_dir / "mentions"

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_records = []
    for shard_index, assigned in enumerate(assignments):
        shard_dir = output_dir / f"shard-{shard_index:02d}"
        parsed_dir = shard_dir / "adi_parsed"
        mentions_dir = shard_dir / "mentions"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        names = {name for name, _ in assigned}
        for name in sorted(names):
            source = by_name[name].path
            shutil.copy2(source, parsed_dir / source.name)
        copied_mentions = _copy_mentions_for_documents(
            source_mentions, mentions_dir, names
        )
        actual_parsed = {path.stem for path in parsed_dir.glob("*.json")}
        if actual_parsed != names:
            raise ReproductionError(
                f"{shard_dir} contains stale or missing parsed documents; "
                "use a new empty output directory"
            )
        copied_mention_stems = {Path(filename).stem for filename in copied_mentions}
        if copied_mention_stems != names:
            raise ReproductionError(
                f"{shard_dir} is missing precomputed mentions for: "
                f"{sorted(names - copied_mention_stems)}"
            )
        record = {
            "shard": shard_index,
            "directory": str(shard_dir),
            "document_count": len(assigned),
            "total_tokens_o200k_base": sum(weight for _, weight in assigned),
            "documents": [
                {
                    "doc_name": name,
                    "tokens_o200k_base": weight,
                    "sha256": _sha256(by_name[name].path),
                }
                for name, weight in assigned
            ],
            "mention_files": copied_mentions,
        }
        _json_dump_atomic(record, shard_dir / "shard_manifest.json")
        shard_records.append(record)

    all_names = [
        item["doc_name"]
        for shard in shard_records
        for item in shard["documents"]
    ]
    if len(all_names) != len(set(all_names)) or set(all_names) != set(by_name):
        raise ReproductionError("Shard assignment lost or duplicated documents")

    manifest = {
        "schema_version": 1,
        "source_commit": PINNED_COMMIT,
        "strategy": (
            "deterministic capacity-constrained greedy LPT by o200k_base "
            "token count"
        ),
        "source_data_dir": str(data_dir),
        "document_count": len(documents),
        "shard_count": shard_count,
        "shards": shard_records,
    }
    _json_dump_atomic(manifest, output_dir / "shards_manifest.json")
    return manifest


def _stage_file_candidates(root: Path, stage: str, filename: str) -> list[Path]:
    direct = [root / "chunks" / stage / filename, root / stage / filename]
    found = [path for path in direct if path.is_file()]
    if not found and root.is_dir():
        found = list(root.rglob(f"chunks/{stage}/{filename}"))
        found.extend(root.rglob(f"{stage}/{filename}"))
    unique: dict[str, Path] = {}
    for path in found:
        unique[str(path.resolve())] = path
    return sorted(unique.values(), key=lambda value: str(value))


def _concat_strict(
    paths: Sequence[Path], key_columns: Sequence[str], artifact_name: str
) -> pd.DataFrame:
    if not paths:
        raise ReproductionError(f"No files found for {artifact_name}")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        missing = set(key_columns) - set(frame.columns)
        if missing:
            raise ReproductionError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.copy()
        frame["__source_artifact"] = str(path)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    duplicate_mask = merged.duplicated(list(key_columns), keep=False)
    if duplicate_mask.any():
        duplicates = merged.loc[duplicate_mask, list(key_columns)].head(10)
        raise ReproductionError(
            f"Duplicate keys while merging {artifact_name}:\n{duplicates.to_string(index=False)}"
        )
    return merged.drop(columns="__source_artifact")


def _validate_chunk_frame(
    frame: pd.DataFrame,
    artifact_name: str,
    allow_empty_page_chunks: bool = False,
) -> None:
    required = {
        "doc_name",
        "method",
        "chunk_index",
        "chunk_text",
        "chunk_len",
        "chunk_pages",
        "titles_context",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ReproductionError(f"{artifact_name} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ReproductionError(f"{artifact_name} contains no rows")
    empty_text = frame["chunk_text"].isna() | frame["chunk_text"].astype(str).eq("")
    if empty_text.any():
        empty_methods = set(frame.loc[empty_text, "method"].unique())
        if not allow_empty_page_chunks or empty_methods != {"page"}:
            raise ReproductionError(
                f"{artifact_name} contains empty chunks for methods: "
                f"{sorted(empty_methods)}"
            )
    if frame.duplicated(["doc_name", "method", "chunk_index"]).any():
        raise ReproductionError(f"{artifact_name} contains duplicate chunk keys")


def merge_chunks(input_roots: Sequence[Path], output_dir: Path) -> dict:
    if not input_roots:
        raise ValueError("At least one input root is required")
    records = []
    for stage in STAGES:
        chunk_paths: list[Path] = []
        performance_paths: list[Path] = []
        for root in input_roots:
            chunk_paths.extend(_stage_file_candidates(root, stage, "chunks.parquet"))
            performance_paths.extend(
                _stage_file_candidates(root, stage, "performances.parquet")
            )
        chunks = _concat_strict(
            chunk_paths,
            ["doc_name", "method", "chunk_index"],
            f"{stage} chunks",
        )
        _validate_chunk_frame(
            chunks,
            f"{stage} chunks",
            allow_empty_page_chunks=stage in {"raw", "no_oversizing"},
        )
        chunks = chunks.sort_values(
            ["doc_name", "method", "chunk_index"], kind="stable"
        ).reset_index(drop=True)
        stage_dir = output_dir / "chunks" / stage
        _parquet_write_atomic(chunks, stage_dir / "chunks.parquet")

        performance_rows = None
        if performance_paths:
            performances = _concat_strict(
                performance_paths,
                ["doc_name", "method"],
                f"{stage} performances",
            )
            performances = performances.sort_values(
                ["doc_name", "method"], kind="stable"
            ).reset_index(drop=True)
            _parquet_write_atomic(performances, stage_dir / "performances.parquet")
            performance_rows = len(performances)

        records.append(
            {
                "stage": stage,
                "chunk_sources": [str(path) for path in chunk_paths],
                "performance_sources": [str(path) for path in performance_paths],
                "rows": len(chunks),
                "performance_rows": performance_rows,
                "document_count": int(chunks["doc_name"].nunique()),
                "methods": sorted(chunks["method"].unique().tolist()),
                "empty_page_chunks": int(
                    (
                        (chunks["method"] == "page")
                        & (
                            chunks["chunk_text"].isna()
                            | chunks["chunk_text"].astype(str).eq("")
                        )
                    ).sum()
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "source_commit": PINNED_COMMIT,
        "stages": records,
    }
    _json_dump_atomic(manifest, output_dir / "chunks_merge_manifest.json")
    return manifest


def _nonsemantic_part_complete(output_dir: Path, doc_name: str) -> bool:
    """Return True only when one document has all seven non-semantic methods."""
    for stage in STAGES:
        path = output_dir / "chunks" / stage / "chunks.parquet"
        if not path.is_file():
            return False
        try:
            frame = pd.read_parquet(path)
            _validate_chunk_frame(
                frame,
                f"{doc_name} {stage} chunks",
                allow_empty_page_chunks=stage in {"raw", "no_oversizing"},
            )
        except Exception:
            return False
        if set(frame["doc_name"].unique()) != {doc_name}:
            return False
        if set(frame["method"].unique()) != NONSEMANTIC_METHODS:
            return False
    return True


def _run_nonsemantic_document(
    *,
    data_dir: Path,
    output_dir: Path,
    device: str,
    log_path: Path,
    cache_dir: Path,
) -> int:
    """Run the upstream CLI for one document through the durable API cache."""
    cache_wrapper = Path(__file__).with_name("cached_openai_replicate.py")
    if not cache_wrapper.is_file():
        raise ReproductionError(f"Missing API cache wrapper: {cache_wrapper}")
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(cache_wrapper),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--steps",
        "chunking",
        "--device",
        device,
        "--skip-semantic",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["ADAPTIVE_OPENAI_CACHE_DIR"] = str(cache_dir)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return result.returncode


def _archive_directory(path: Path, archive_root: Path, label: str) -> Path:
    """Move an obsolete partial directory aside without deleting evidence."""
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    destination = archive_root / f"{label}-{timestamp}"
    counter = 1
    while destination.exists():
        destination = archive_root / f"{label}-{timestamp}-{counter:02d}"
        counter += 1
    archive_root.mkdir(parents=True, exist_ok=True)
    path.replace(destination)
    return destination


def run_nonsemantic_shard(
    data_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    resume: bool = False,
    min_interval_seconds: float = 65.0,
    initial_cooldown_seconds: float = 70.0,
    allow_fresh_paid_run: bool = False,
    *,
    run_document=None,
    sleep_func=time.sleep,
) -> dict:
    """Run LLM-regex one document at a time with durable checkpoints.

    The upstream model, prompt, splitters, and post-processing remain unchanged.
    Only request scheduling is serialized to stay below account-level TPM limits.
    """
    if min_interval_seconds < 0 or initial_cooldown_seconds < 0:
        raise ValueError("Cooldown values must be non-negative")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ReproductionError("OPENAI_API_KEY is required for LLM-regex")

    parsed_dir = _resolve_parsed_dir(data_dir)
    documents = sorted(parsed_dir.glob("*.json"), key=lambda path: path.name)
    if not documents:
        raise ReproductionError(f"No documents found under {parsed_dir}")

    if any(output_dir.glob("chunks/*/chunks.parquet")):
        validation = validate_artifacts(
            output_dir, expected_docs=len(documents), profile="nonsemantic"
        )
        expected_names = {document.stem for document in documents}
        for stage in STAGES:
            frame = pd.read_parquet(output_dir / "chunks" / stage / "chunks.parquet")
            if set(frame["doc_name"]) != expected_names:
                raise ReproductionError("Existing shard output has different documents")
        print("SHARD_ALREADY_COMPLETE: reusing existing output; no API requests", flush=True)
        return {"status": "success", "reused_existing_output": True,
                "validation": validation}

    checkpoint_dir = output_dir.with_name(output_dir.name + "-checkpoints")
    parts_dir = checkpoint_dir / "parts"
    logs_dir = checkpoint_dir / "logs"
    failed_dir = checkpoint_dir / "failed"
    manifest_path = checkpoint_dir / "run_manifest.json"
    if checkpoint_dir.exists() and not resume and manifest_path.is_file():
        raise ReproductionError(
            f"Checkpoint exists at {checkpoint_dir}; pass --resume to continue safely"
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed = []
    for index, document in enumerate(documents):
        part_output = parts_dir / f"doc-{index:02d}"
        if _nonsemantic_part_complete(part_output, document.stem):
            completed.append(document.stem)
        elif part_output.exists() and any(part_output.rglob("chunks.parquet")):
            raise ReproductionError(
                f"Existing output needs recovery for {document.stem}: {part_output}. "
                "Refusing to repeat a potentially paid request. Preserve these "
                "files and inspect the failed stage before continuing."
            )

    print(f"Validated checkpoints: {len(completed)}/{len(documents)}", flush=True)

    cache_dir = checkpoint_dir / "api_cache"
    cached_responses = list(cache_dir.glob("*.json")) if cache_dir.is_dir() else []
    if not completed and not cached_responses and not allow_fresh_paid_run:
        raise ReproductionError(
            "No completed checkpoint or API cache exists. Refusing a fresh paid "
            "run; pass --allow-fresh-paid-run only after confirming the shard ID."
        )

    manifest = {
        "schema_version": 1,
        "source_commit": PINNED_COMMIT,
        "status": "running",
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "device": device,
        "document_count": len(documents),
        "model": "gpt-4o",
        "temperature": 0,
        "request_scheduling": "sequential_per_document",
        "min_interval_seconds": min_interval_seconds,
        "api_cache_dir": str(cache_dir),
        "sdk_automatic_retries": 0,
        "completed_documents": completed,
    }
    _json_dump_atomic(manifest, manifest_path)

    pending = [document for document in documents if document.stem not in completed]
    runner = run_document or _run_nonsemantic_document
    if pending and initial_cooldown_seconds:
        print(
            f"Initial TPM cooldown: waiting {initial_cooldown_seconds:.0f} seconds"
        )
        sleep_func(initial_cooldown_seconds)

    for pending_index, document in enumerate(pending):
        document_index = documents.index(document)
        part_name = f"doc-{document_index:02d}"
        part_data = checkpoint_dir / "data" / part_name
        part_output = parts_dir / part_name
        log_path = logs_dir / f"{part_name}.log"

        if part_output.exists():
            archived = _archive_directory(part_output, failed_dir, part_name)
            print(f"Archived incomplete checkpoint: {archived}")
        if part_data.exists():
            shutil.rmtree(part_data)
        (part_data / "adi_parsed").mkdir(parents=True)
        shutil.copy2(document, part_data / "adi_parsed" / document.name)

        manifest["active_document"] = document.stem
        manifest["status"] = "running"
        _json_dump_atomic(manifest, manifest_path)
        print(
            f"[{len(completed) + 1}/{len(documents)}] Running {document.name}",
            flush=True,
        )
        return_code = runner(
            data_dir=part_data,
            output_dir=part_output,
            device=device,
            log_path=log_path,
            cache_dir=cache_dir,
        )
        if return_code != 0 or not _nonsemantic_part_complete(
            part_output, document.stem
        ):
            manifest["status"] = "failed"
            manifest["failed_document"] = document.stem
            manifest["return_code"] = return_code
            _json_dump_atomic(manifest, manifest_path)
            tail = ""
            if log_path.is_file():
                tail = log_path.read_text(
                    encoding="utf-8", errors="replace"
                )[-6000:]
            raise ReproductionError(
                f"Document failed: {document.name}; log={log_path}\n{tail}"
            )

        completed.append(document.stem)
        manifest["completed_documents"] = completed
        manifest.pop("failed_document", None)
        manifest["return_code"] = 0
        _json_dump_atomic(manifest, manifest_path)
        print(f"Checkpoint saved: {document.name}", flush=True)

        if pending_index < len(pending) - 1 and min_interval_seconds:
            print(
                f"TPM cooldown: waiting {min_interval_seconds:.0f} seconds",
                flush=True,
            )
            sleep_func(min_interval_seconds)

    part_outputs = [parts_dir / f"doc-{index:02d}" for index in range(len(documents))]
    for index, (document, part_output) in enumerate(zip(documents, part_outputs)):
        if not _nonsemantic_part_complete(part_output, document.stem):
            raise ReproductionError(f"Missing completed checkpoint doc-{index:02d}")

    merged_temp = checkpoint_dir / "merged-complete"
    if merged_temp.exists():
        _archive_directory(merged_temp, failed_dir, "stale-merged")
    merge_manifest = merge_chunks(part_outputs, merged_temp)
    validation = validate_artifacts(
        merged_temp, expected_docs=len(documents), profile="nonsemantic"
    )

    archived_previous_output = None
    if output_dir.exists():
        archived_previous_output = _archive_directory(
            output_dir, failed_dir, "previous-shard-output"
        )
    shutil.move(str(merged_temp), str(output_dir))

    master_log = output_dir.with_suffix(".log")
    with master_log.open("w", encoding="utf-8") as destination:
        for index, document in enumerate(documents):
            source_log = logs_dir / f"doc-{index:02d}.log"
            destination.write(f"\n===== {document.name} =====\n")
            if source_log.is_file():
                destination.write(
                    source_log.read_text(encoding="utf-8", errors="replace")
                )

    manifest["status"] = "success"
    manifest["active_document"] = None
    manifest["completed_documents"] = completed
    manifest["archived_previous_output"] = (
        str(archived_previous_output) if archived_previous_output else None
    )
    manifest["merge"] = merge_manifest
    manifest["validation"] = validation
    _json_dump_atomic(manifest, manifest_path)
    return manifest


def _complete_semantic_documents(output_dir: Path) -> set[str]:
    complete_sets = []
    for stage in STAGES:
        path = output_dir / "chunks" / stage / "chunks.parquet"
        if not path.is_file():
            return set()
        frame = pd.read_parquet(path, columns=["doc_name", "method"])
        frame = frame[frame["method"] == "semantic"]
        complete_sets.append(set(frame["doc_name"].unique()))
    return set.intersection(*complete_sets) if complete_sets else set()


def _previous_semantic_timings(output_dir: Path) -> list[dict]:
    path = output_dir / "semantic_run_manifest.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    timings = payload.get("all_timings", payload.get("session_timings", []))
    by_document = {
        timing["doc_name"]: timing
        for timing in timings
        if {"doc_name", "tokens_o200k_base", "seconds"} <= set(timing)
    }
    return list(by_document.values())


def _semantic_artifacts_exist(output_dir: Path) -> bool:
    return any(
        (output_dir / "chunks" / stage / "chunks.parquet").is_file()
        for stage in STAGES
    )


def _semantic_source_records(documents: Sequence[DocumentInfo]) -> list[dict]:
    return [
        {
            "doc_name": document.name,
            "tokens_o200k_base": document.weight,
            "sha256": _sha256(document.path),
        }
        for document in sorted(documents, key=lambda item: item.name)
    ]


def _load_semantic_resume_manifest(
    output_dir: Path,
    *,
    resume: bool,
    configuration: dict,
    source_documents: Sequence[dict],
) -> dict | None:
    """Validate that a resumed run cannot mix configurations or datasets."""
    manifest_path = output_dir / "semantic_run_manifest.json"
    artifacts_exist = _semantic_artifacts_exist(output_dir)
    manifest_exists = manifest_path.is_file()

    if (artifacts_exist or manifest_exists) and not resume:
        raise ReproductionError(
            f"{output_dir} contains semantic run state; use --resume after auditing it"
        )
    if artifacts_exist and not manifest_exists:
        raise ReproductionError(
            "Semantic artifacts exist without semantic_run_manifest.json; refusing "
            "to resume because their configuration cannot be verified"
        )
    if not manifest_exists:
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ReproductionError(
            f"Cannot read semantic resume manifest: {manifest_path}"
        ) from error

    previous_configuration = manifest.get("configuration")
    if not isinstance(previous_configuration, dict):
        raise ReproductionError("Semantic resume manifest has no configuration")
    mismatches = {
        key: (previous_configuration.get(key), configuration.get(key))
        for key in SEMANTIC_IDENTITY_KEYS
        if previous_configuration.get(key) != configuration.get(key)
    }
    if mismatches:
        raise ReproductionError(
            "Semantic resume configuration mismatch; use a separate output "
            f"directory instead of mixing runs: {mismatches}"
        )

    previous_sources = manifest.get("source_documents")
    if previous_sources != list(source_documents):
        raise ReproductionError(
            "Semantic resume dataset identity mismatch; use a separate output "
            "directory instead of mixing runs"
        )
    return manifest


def _estimate_total_semantic_hours(timings: Sequence[dict], total_tokens: int):
    observed_tokens = sum(int(item["tokens_o200k_base"]) for item in timings)
    if not timings or observed_tokens <= 0:
        return None
    observed_seconds = sum(float(item["seconds"]) for item in timings)
    return observed_seconds / observed_tokens * total_tokens / 3600


def _conservative_next_document_seconds(
    timings: Sequence[dict], document: DocumentInfo
) -> float | None:
    if not timings:
        return None
    slowest_rate = max(
        float(item["seconds"]) / max(int(item["tokens_o200k_base"]), 1)
        for item in timings
    )
    longest_observed = max(float(item["seconds"]) for item in timings)
    return max(longest_observed, slowest_rate * document.weight) * 1.25


def _upsert_document_rows(source_path: Path, destination_path: Path, doc_name: str) -> None:
    new_rows = pd.read_parquet(source_path)
    if "doc_name" not in new_rows.columns:
        raise ReproductionError(f"Missing doc_name column in {source_path}")
    if set(new_rows["doc_name"].unique()) != {doc_name}:
        raise ReproductionError(f"Temporary artifact contains unexpected documents: {source_path}")
    if destination_path.is_file():
        current = pd.read_parquet(destination_path)
        current = current[current["doc_name"] != doc_name]
        new_rows = pd.concat([current, new_rows], ignore_index=True)
    _parquet_write_atomic(new_rows, destination_path)


def _representative_documents(
    documents: Sequence[DocumentInfo], count: int
) -> list[DocumentInfo]:
    if count < 1:
        raise ValueError("benchmark document count must be at least 1")
    ordered = sorted(documents, key=lambda document: (document.weight, document.name))
    if count >= len(ordered):
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = {
        round(position * (len(ordered) - 1) / (count - 1))
        for position in range(count)
    }
    return [ordered[index] for index in sorted(indices)]


def _semantic_preflight(device: str, attention_implementation: str, dtype_name: str):
    import torch

    if not device.startswith("cuda"):
        raise ReproductionError("Semantic reproduction requires a CUDA device")
    if not torch.cuda.is_available():
        raise ReproductionError("CUDA is not available")
    if attention_implementation == "flash_attention_2":
        capability = torch.cuda.get_device_capability(torch.device(device))
        if capability < (8, 0):
            raise ReproductionError(
                "flash_attention_2 requires an Ampere/Ada/Hopper GPU (compute capability >= 8.0)"
            )
        if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ReproductionError("Selected GPU does not support bfloat16")
    if dtype_name not in {"bfloat16", "float16"}:
        raise ReproductionError(f"Unsupported semantic dtype: {dtype_name}")
    return torch


def _build_semantic_splitter(
    device: str,
    model_revision: str,
    attention_implementation: str,
    dtype_name: str,
    batch_size: int,
):
    torch = _semantic_preflight(device, attention_implementation, dtype_name)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from adaptive_chunking.paper.splitters import SemanticChunkerWrapper

    torch_dtype = getattr(torch, dtype_name)
    embeddings = HuggingFaceEmbeddings(
        model_name=SEMANTIC_MODEL,
        model_kwargs={
            "device": device,
            "revision": model_revision,
            "model_kwargs": {
                "attn_implementation": attention_implementation,
                "torch_dtype": torch_dtype,
            },
            "tokenizer_kwargs": {"padding_side": "left"},
        },
        encode_kwargs={"batch_size": batch_size},
    )
    return SemanticChunkerWrapper(
        embeddings=embeddings,
        breakpoint_threshold_type=SEMANTIC_CONFIG["breakpoint_threshold_type"],
    )


async def _run_semantic_document(
    document: DocumentInfo,
    splitter,
    temporary_root: Path,
) -> dict[str, dict[str, Path]]:
    from adaptive_chunking.paper.replicate import SEPARATORS, count_tokens_func
    from adaptive_chunking.postprocessing import (
        merge_small_chunks_from_df,
        merge_small_chunks_to_neighbours,
        split_oversized_chunks,
        split_oversized_chunks_from_df,
    )
    from adaptive_chunking.split_documents import split_documents_from_dir
    from adaptive_chunking.splitters import RecursiveSplitter

    parsed_dir = temporary_root / "data" / "adi_parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(document.path, parsed_dir / document.path.name)
    raw_dir = temporary_root / "chunks" / "raw"
    no_oversizing_dir = temporary_root / "chunks" / "no_oversizing"
    small_merged_dir = temporary_root / "chunks" / "small_merged"

    await split_documents_from_dir(
        parsed_docs_dir=parsed_dir,
        sync_splitters={"semantic": splitter},
        async_splitters={},
        output_dir=raw_dir,
        count_tokens_func=count_tokens_func,
        skip_non_english=False,
        replace_all_results=True,
    )

    oversized_splitter = RecursiveSplitter(
        separators=SEPARATORS,
        chunk_size=1100,
        chunk_overlap=0,
        is_separator_regex=True,
        attach_separator_to="start",
        length_function=count_tokens_func,
        merging="to_chunk_size",
    )

    def split_oversized(chunks):
        return split_oversized_chunks(
            chunks,
            oversized_splitter,
            count_tokens_func,
            max_chunk_tokens=1100,
        )

    split_oversized_chunks_from_df(
        parsed_docs_dir=parsed_dir,
        chunks_path=raw_dir / "chunks.parquet",
        output_dir=no_oversizing_dir,
        methods_to_be_regularized={"semantic"},
        split_oversized_func=split_oversized,
        count_tokens_func=count_tokens_func,
        replace_all_results=True,
    )

    def merge_small(chunks):
        return merge_small_chunks_to_neighbours(
            chunks,
            count_tokens_func,
            min_limit=100,
            max_limit=1150,
            merge_to="next",
        )

    merge_small_chunks_from_df(
        parsed_docs_dir=parsed_dir,
        chunks_path=no_oversizing_dir / "chunks.parquet",
        output_dir=small_merged_dir,
        methods_to_be_regularized={"semantic"},
        merge_small_chunks_func=merge_small,
        count_tokens_func=count_tokens_func,
        replace_all_results=True,
    )

    return {
        stage: {
            "chunks": temporary_root / "chunks" / stage / "chunks.parquet",
            "performances": temporary_root
            / "chunks"
            / stage
            / "performances.parquet",
        }
        for stage in STAGES
    }


def run_semantic_only(
    data_dir: Path,
    output_dir: Path,
    device: str,
    model_revision: str,
    benchmark_documents: int | None,
    resume: bool,
    attention_implementation: str,
    dtype_name: str,
    batch_size: int,
    max_runtime_hours: float | None,
) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise ReproductionError(
            "--model-revision must be an immutable 40-character Hugging Face commit SHA"
        )
    _assert_pinned_source_commit()
    documents = _document_infos(data_dir, weight_kind="tokens")
    total_tokens = sum(document.weight for document in documents)
    config = {
        "source_commit": PINNED_COMMIT,
        "model": SEMANTIC_MODEL,
        "model_revision": model_revision,
        "attention_implementation": attention_implementation,
        "dtype": dtype_name,
        "batch_size": batch_size,
        "breakpoint_threshold_type": SEMANTIC_CONFIG[
            "breakpoint_threshold_type"
        ],
        "device": device,
        "exact_upstream_configuration": (
            attention_implementation == SEMANTIC_CONFIG["attention_implementation"]
            and dtype_name == SEMANTIC_CONFIG["dtype"]
            and batch_size == SEMANTIC_CONFIG["batch_size"]
        ),
        "runtime": _runtime_snapshot(),
    }
    source_documents = _semantic_source_records(documents)
    previous_manifest = _load_semantic_resume_manifest(
        output_dir,
        resume=resume,
        configuration=config,
        source_documents=source_documents,
    )
    existing = _complete_semantic_documents(output_dir)
    selected = (
        _representative_documents(documents, benchmark_documents)
        if benchmark_documents
        else list(documents)
    )
    selected = [document for document in selected if document.name not in existing]

    prior_timings = _previous_semantic_timings(output_dir) if resume else []
    prior_runner_elapsed = (
        float(previous_manifest.get("total_runner_elapsed_seconds", 0.0))
        if previous_manifest
        else 0.0
    )
    initial_manifest = {
        "schema_version": 2,
        "status": "initializing",
        "configuration": config,
        "source_documents": source_documents,
        "total_source_documents": len(documents),
        "selected_documents": [item.name for item in selected],
        "completed_documents": sorted(existing),
        "completed_document_count": len(existing),
        "session_timings": [],
        "all_timings": prior_timings,
        "session_elapsed_seconds": 0.0,
        "total_runner_elapsed_seconds": prior_runner_elapsed,
        "estimated_total_hours_from_session_mean": (
            _estimate_total_semantic_hours(prior_timings, total_tokens)
        ),
        "stopped_by_runtime_limit": False,
    }
    _json_dump_atomic(initial_manifest, output_dir / "semantic_run_manifest.json")

    splitter = _build_semantic_splitter(
        device, model_revision, attention_implementation, dtype_name, batch_size
    )
    started = time.perf_counter()
    timings: list[dict] = []
    stopped_by_runtime_limit = False
    for index, document in enumerate(selected, start=1):
        elapsed_hours = (time.perf_counter() - started) / 3600
        if max_runtime_hours is not None:
            remaining_seconds = max_runtime_hours * 3600 - elapsed_hours * 3600
            conservative_seconds = _conservative_next_document_seconds(
                [*prior_timings, *timings], document
            )
            if remaining_seconds <= 0 or (
                conservative_seconds is not None
                and conservative_seconds > remaining_seconds
            ):
                stopped_by_runtime_limit = True
                break
        print(
            f"\n[{index}/{len(selected)}] semantic: {document.name} "
            f"({document.weight} o200k tokens)"
        )
        document_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="adaptive_semantic_") as temporary:
            artifacts = asyncio.run(
                _run_semantic_document(document, splitter, Path(temporary))
            )
            for stage, paths in artifacts.items():
                destination = output_dir / "chunks" / stage
                _upsert_document_rows(
                    paths["chunks"], destination / "chunks.parquet", document.name
                )
                _upsert_document_rows(
                    paths["performances"],
                    destination / "performances.parquet",
                    document.name,
                )
        duration = time.perf_counter() - document_started
        timings.append(
            {
                "doc_name": document.name,
                "tokens_o200k_base": document.weight,
                "seconds": duration,
            }
        )
        all_timings = [*prior_timings, *timings]
        current_complete = sorted(_complete_semantic_documents(output_dir))
        elapsed = time.perf_counter() - started
        manifest = {
            "schema_version": 2,
            "status": "running",
            "configuration": config,
            "source_documents": source_documents,
            "total_source_documents": len(documents),
            "selected_documents": [item.name for item in selected],
            "completed_documents": current_complete,
            "completed_document_count": len(current_complete),
            "session_timings": timings,
            "all_timings": all_timings,
            "session_elapsed_seconds": elapsed,
            "total_runner_elapsed_seconds": prior_runner_elapsed + elapsed,
            "estimated_total_hours_from_session_mean": (
                _estimate_total_semantic_hours(all_timings, total_tokens)
            ),
            "stopped_by_runtime_limit": False,
        }
        _json_dump_atomic(manifest, output_dir / "semantic_run_manifest.json")

    completed = sorted(_complete_semantic_documents(output_dir))
    elapsed = time.perf_counter() - started
    all_timings = [*prior_timings, *timings]
    if stopped_by_runtime_limit:
        final_status = "stopped_by_runtime_limit"
    elif len(completed) == len(documents):
        final_status = "complete"
    else:
        final_status = "benchmark_complete"
    result = {
        "schema_version": 2,
        "status": final_status,
        "configuration": config,
        "source_documents": source_documents,
        "total_source_documents": len(documents),
        "selected_documents": [item.name for item in selected],
        "completed_documents": completed,
        "completed_document_count": len(completed),
        "session_timings": timings,
        "all_timings": all_timings,
        "session_elapsed_seconds": elapsed,
        "total_runner_elapsed_seconds": prior_runner_elapsed + elapsed,
        "estimated_total_hours_from_session_mean": (
            _estimate_total_semantic_hours(all_timings, total_tokens)
        ),
        "stopped_by_runtime_limit": stopped_by_runtime_limit,
    }
    _json_dump_atomic(result, output_dir / "semantic_run_manifest.json")
    return result


def _resolve_chunks_root(chunks_dir: Path) -> Path:
    if (chunks_dir / "chunks" / "raw" / "chunks.parquet").is_file():
        return chunks_dir / "chunks"
    if (chunks_dir / "raw" / "chunks.parquet").is_file():
        return chunks_dir
    raise ReproductionError(
        f"Expected chunks/raw/chunks.parquet or raw/chunks.parquet under {chunks_dir}"
    )


def make_metric_shards(chunks_dir: Path, output_dir: Path, shard_count: int) -> dict:
    chunks_root = _resolve_chunks_root(chunks_dir)
    stage_frames = {
        stage: pd.read_parquet(chunks_root / stage / "chunks.parquet")
        for stage in STAGES
        if (chunks_root / stage / "chunks.parquet").is_file()
    }
    if "raw" not in stage_frames or "small_merged" not in stage_frames:
        raise ReproductionError("Metric sharding requires raw and small_merged chunks")
    document_names = set(stage_frames["raw"]["doc_name"].unique())
    for stage, frame in stage_frames.items():
        if set(frame["doc_name"].unique()) != document_names:
            raise ReproductionError(
                f"Raw and {stage} stages contain different documents"
            )
    weights = []
    for name in sorted(document_names):
        weight = sum(int((frame["doc_name"] == name).sum()) for frame in stage_frames.values())
        weights.append((name, weight))
    assignments = balanced_assignments(weights, shard_count)

    records = []
    for shard_index, assigned in enumerate(assignments):
        names = {name for name, _ in assigned}
        shard_dir = output_dir / f"shard-{shard_index:02d}"
        stage_records = []
        for stage, frame in stage_frames.items():
            subset = frame[frame["doc_name"].isin(names)].copy()
            _validate_chunk_frame(
                subset,
                f"metric shard {shard_index} {stage}",
                allow_empty_page_chunks=stage in {"raw", "no_oversizing"},
            )
            destination = shard_dir / "chunks" / stage / "chunks.parquet"
            _parquet_write_atomic(subset, destination)
            stage_records.append({"stage": stage, "rows": len(subset)})
        record = {
            "shard": shard_index,
            "document_count": len(names),
            "total_chunk_rows": sum(weight for _, weight in assigned),
            "documents": sorted(names),
            "stages": stage_records,
        }
        _json_dump_atomic(record, shard_dir / "shard_manifest.json")
        records.append(record)
    manifest = {
        "schema_version": 1,
        "source_commit": PINNED_COMMIT,
        "strategy": (
            "deterministic capacity-constrained greedy LPT by total chunk rows"
        ),
        "document_count": len(document_names),
        "shard_count": shard_count,
        "shards": records,
    }
    _json_dump_atomic(manifest, output_dir / "metric_shards_manifest.json")
    return manifest


def run_metrics_shard(
    kind: str,
    chunks_parquet: Path,
    data_dir: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    allow_jina_api: bool,
) -> dict:
    started = time.perf_counter()
    if os.environ.get("JINA_API_KEY") and not allow_jina_api:
        raise ReproductionError(
            "JINA_API_KEY is set. Remove it for full local-model metrics, or explicitly use --allow-jina-api."
        )
    if not chunks_parquet.is_file():
        raise ReproductionError(f"Missing chunks parquet: {chunks_parquet}")
    from adaptive_chunking.compute_metrics import compute_metrics_per_origin

    if os.environ.get("JINA_API_KEY") and allow_jina_api:
        from adaptive_chunking.jina_embedder import JinaEmbedder

        embedder = JinaEmbedder()
        backend = "jina_api"
    else:
        from huggingface_hub import HfApi
        from sentence_transformers import SentenceTransformer

        resolved_code_revision = HfApi().model_info(
            METRIC_EMBEDDING_CODE_REPOSITORY
        ).sha
        if resolved_code_revision != METRIC_EMBEDDING_CODE_REVISION:
            raise ReproductionError(
                "Jina custom-code HEAD changed: "
                f"expected {METRIC_EMBEDDING_CODE_REVISION}, "
                f"got {resolved_code_revision}. Refusing an unreviewed metrics run."
            )
        embedder = SentenceTransformer(
            METRIC_EMBEDDING_MODEL,
            revision=METRIC_EMBEDDING_MODEL_REVISION,
            trust_remote_code=True,
            # Sentence Transformers consumes this value while resolving the
            # main model's custom_st.py. The secondary auto_map repository is
            # guarded immediately above because its loader does not expose a
            # separate nested code_revision argument.
            model_kwargs={"code_revision": METRIC_EMBEDDING_MODEL_REVISION},
        ).to(device)
        backend = "sentence_transformers_local"

    source_frame = pd.read_parquet(chunks_parquet)
    _validate_chunk_frame(
        source_frame,
        str(chunks_parquet),
        allow_empty_page_chunks=kind == "raw",
    )
    if kind == "raw":
        source_frame = source_frame[source_frame["method"].isin(RAW_METRIC_METHODS)]
    elif kind != "processed":
        raise ValueError("kind must be 'processed' or 'raw'")
    if source_frame.empty:
        raise ReproductionError(f"No rows remain for {kind} metrics")

    input_dir = output_dir / "_input"
    _parquet_write_atomic(source_frame, input_dir / "chunks.parquet")
    metrics_dir = output_dir / ("results_raw" if kind == "raw" else "results")
    compute_metrics_per_origin(
        chunks_dir=input_dir,
        mentions_dir=data_dir / "mentions",
        parsed_docs_dir=data_dir / "adi_parsed",
        models={"sentence_embedder": embedder},
        output_dir=metrics_dir,
        batch_size=batch_size,
    )
    manifest = {
        "schema_version": 1,
        "source_commit": PINNED_COMMIT,
        "kind": kind,
        "embedding_backend": backend,
        "embedding_model": METRIC_EMBEDDING_MODEL,
        "embedding_model_revision": (
            METRIC_EMBEDDING_MODEL_REVISION
            if backend == "sentence_transformers_local"
            else None
        ),
        "embedding_code_revision": (
            METRIC_EMBEDDING_CODE_REVISION
            if backend == "sentence_transformers_local"
            else None
        ),
        "embedding_code_repository": (
            METRIC_EMBEDDING_CODE_REPOSITORY
            if backend == "sentence_transformers_local"
            else None
        ),
        "device": device,
        "batch_size": batch_size,
        "source_chunks": str(chunks_parquet),
        "source_sha256": _sha256(chunks_parquet),
        "document_count": int(source_frame["doc_name"].nunique()),
        "methods": sorted(source_frame["method"].unique().tolist()),
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": _runtime_snapshot(),
        "metrics_sha256": _sha256(metrics_dir / "chunking_metrics.parquet"),
    }
    _json_dump_atomic(manifest, output_dir / f"{kind}_metrics_manifest.json")
    return manifest


def _discover_named_files(roots: Sequence[Path], relative_suffix: str) -> list[Path]:
    discovered: dict[str, Path] = {}
    normalized_suffix = relative_suffix.replace("\\", "/")
    for root in roots:
        if root.is_file() and root.name == Path(relative_suffix).name:
            discovered[str(root.resolve())] = root
            continue
        if not root.is_dir():
            continue
        for path in root.rglob(Path(relative_suffix).name):
            if path.as_posix().endswith(normalized_suffix):
                discovered[str(path.resolve())] = path
    return sorted(discovered.values(), key=lambda value: str(value))


def _merge_metric_kind(roots: Sequence[Path], kind: str, destination: Path) -> dict:
    source_subdir = "results_raw" if kind == "raw" else "results"
    metric_paths = _discover_named_files(
        roots, f"{source_subdir}/chunking_metrics.parquet"
    )
    performance_paths = _discover_named_files(
        roots, f"{source_subdir}/metrics_performance.parquet"
    )
    metrics = _concat_strict(
        metric_paths,
        ["doc_name", "chunking_method", "metric_name"],
        f"{kind} metrics",
    )
    metrics = metrics.sort_values(
        ["doc_name", "chunking_method", "metric_name"], kind="stable"
    ).reset_index(drop=True)
    _parquet_write_atomic(metrics, destination / "chunking_metrics.parquet")
    if performance_paths:
        performances = _concat_strict(
            performance_paths,
            ["doc_name", "metric"],
            f"{kind} metric performances",
        )
        performances = performances.sort_values(
            ["doc_name", "metric"], kind="stable"
        ).reset_index(drop=True)
        _parquet_write_atomic(
            performances, destination / "metrics_performance.parquet"
        )
    return {
        "kind": kind,
        "metric_sources": [str(path) for path in metric_paths],
        "performance_sources": [str(path) for path in performance_paths],
        "rows": len(metrics),
        "document_count": int(metrics["doc_name"].nunique()),
        "methods": sorted(metrics["chunking_method"].unique().tolist()),
        "metrics": sorted(metrics["metric_name"].unique().tolist()),
    }


def merge_metrics(
    processed_roots: Sequence[Path],
    raw_roots: Sequence[Path],
    output_dir: Path,
    chunks_dir: Path | None,
) -> dict:
    processed = _merge_metric_kind(processed_roots, "processed", output_dir / "results")
    raw = _merge_metric_kind(raw_roots, "raw", output_dir / "results_raw")
    chunks_manifest = None
    if chunks_dir is not None:
        source_root = _resolve_chunks_root(chunks_dir)
        chunks_manifest = merge_chunks([source_root], output_dir)
    manifest = {
        "schema_version": 1,
        "source_commit": PINNED_COMMIT,
        "processed": processed,
        "raw": raw,
        "chunks": chunks_manifest,
    }
    _json_dump_atomic(manifest, output_dir / "metrics_merge_manifest.json")
    return manifest


def _profile_methods(profile: str) -> set[str] | None:
    return {
        "full": ALL_METHODS,
        "nonsemantic": NONSEMANTIC_METHODS,
        "semantic": {"semantic"},
        "auto": None,
    }[profile]


def validate_artifacts(output_dir: Path, expected_docs: int, profile: str) -> dict:
    expected_methods = _profile_methods(profile)
    report: dict[str, object] = {
        "schema_version": 1,
        "output_dir": str(output_dir),
        "expected_documents": expected_docs,
        "profile": profile,
        "chunks": {},
        "metrics": {},
        "warnings": [],
    }

    chunks_root = None
    try:
        chunks_root = _resolve_chunks_root(output_dir)
    except ReproductionError:
        pass
    if chunks_root:
        for stage in STAGES:
            path = chunks_root / stage / "chunks.parquet"
            if not path.is_file():
                raise ReproductionError(f"Missing chunk artifact: {path}")
            frame = pd.read_parquet(path)
            _validate_chunk_frame(
                frame,
                str(path),
                allow_empty_page_chunks=stage in {"raw", "no_oversizing"},
            )
            methods = set(frame["method"].unique())
            docs = int(frame["doc_name"].nunique())
            if docs != expected_docs:
                raise ReproductionError(
                    f"{stage}: expected {expected_docs} documents, found {docs}"
                )
            if expected_methods is not None and methods != expected_methods:
                raise ReproductionError(
                    f"{stage}: expected methods {sorted(expected_methods)}, found {sorted(methods)}"
                )
            if expected_methods is not None:
                for name, group in frame.groupby("doc_name"):
                    if set(group["method"]) != expected_methods:
                        raise ReproductionError(f"{stage}: incomplete methods for {name}")
            report["chunks"][stage] = {
                "path": str(path),
                "rows": len(frame),
                "documents": docs,
                "methods": sorted(methods),
            }

    metric_specs = [
        ("processed", output_dir / "results" / "chunking_metrics.parquet", ALL_METHODS),
        ("raw", output_dir / "results_raw" / "chunking_metrics.parquet", RAW_METRIC_METHODS),
    ]
    for kind, path, methods_expected in metric_specs:
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        key = ["doc_name", "chunking_method", "metric_name"]
        missing_columns = set(key + ["score"]) - set(frame.columns)
        if missing_columns:
            raise ReproductionError(f"{path} missing columns: {sorted(missing_columns)}")
        if frame.duplicated(key).any():
            raise ReproductionError(f"{path} contains duplicate metric keys")
        docs = int(frame["doc_name"].nunique())
        methods = set(frame["chunking_method"].unique())
        metrics = set(frame["metric_name"].unique())
        if docs != expected_docs:
            raise ReproductionError(
                f"{kind} metrics: expected {expected_docs} documents, found {docs}"
            )
        if profile == "full" and methods != methods_expected:
            raise ReproductionError(
                f"{kind} metrics: expected {sorted(methods_expected)}, found {sorted(methods)}"
            )
        if metrics != ALL_RECORDED_METRICS:
            raise ReproductionError(
                f"{kind} metrics: expected {sorted(ALL_RECORDED_METRICS)}, found {sorted(metrics)}"
            )
        expected_rows = expected_docs * len(methods) * len(ALL_RECORDED_METRICS)
        if len(frame) != expected_rows:
            raise ReproductionError(
                f"{kind} metrics: expected {expected_rows} rows, found {len(frame)}"
            )
        null_rc = int(
            frame.loc[frame["metric_name"] == "references_completeness", "score"]
            .isna()
            .sum()
        )
        if null_rc:
            report["warnings"].append(
                f"{kind} metrics contains {null_rc} null references_completeness scores"
            )
        null_counts = {
            metric_name: int(count)
            for metric_name, count in frame.loc[frame["score"].isna()]
            .groupby("metric_name")
            .size()
            .items()
        }
        report["metrics"][kind] = {
            "path": str(path),
            "rows": len(frame),
            "documents": docs,
            "methods": sorted(methods),
            "metrics": sorted(metrics),
            "null_references_completeness": null_rc,
            "null_scores_by_metric": null_counts,
        }

    if not report["chunks"] and not report["metrics"]:
        raise ReproductionError(f"No recognizable artifacts found under {output_dir}")
    _json_dump_atomic(report, output_dir / "validation_report.json")
    return report


def _paths(values: Iterable[str] | None) -> list[Path]:
    return [Path(value) for value in values or []]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Low-cost sharding, semantic, merge, and validation helpers."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_data = subparsers.add_parser("make-data-shards")
    make_data.add_argument("--data-dir", type=Path, required=True)
    make_data.add_argument("--output-dir", type=Path, required=True)
    make_data.add_argument("--shards", type=int, default=3)

    semantic = subparsers.add_parser("run-semantic-only")
    semantic.add_argument("--data-dir", type=Path, required=True)
    semantic.add_argument("--output-dir", type=Path, required=True)
    semantic.add_argument("--device", default="cuda:0")
    semantic.add_argument(
        "--model-revision",
        required=True,
        help="Immutable Hugging Face commit SHA for the semantic embedding model",
    )
    semantic.add_argument("--benchmark-documents", type=int)
    semantic.add_argument("--resume", action="store_true")
    semantic.add_argument(
        "--attention-implementation",
        choices=["flash_attention_2", "sdpa"],
        default=SEMANTIC_CONFIG["attention_implementation"],
    )
    semantic.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default=SEMANTIC_CONFIG["dtype"],
    )
    semantic.add_argument(
        "--batch-size", type=int, default=SEMANTIC_CONFIG["batch_size"]
    )
    semantic.add_argument("--max-runtime-hours", type=float)

    nonsemantic = subparsers.add_parser("run-nonsemantic-shard")
    nonsemantic.add_argument("--data-dir", type=Path, required=True)
    nonsemantic.add_argument("--output-dir", type=Path, required=True)
    nonsemantic.add_argument("--device", default="cpu")
    nonsemantic.add_argument("--resume", action="store_true")
    nonsemantic.add_argument("--min-interval-seconds", type=float, default=65.0)
    nonsemantic.add_argument("--initial-cooldown-seconds", type=float, default=70.0)
    nonsemantic.add_argument("--allow-fresh-paid-run", action="store_true")

    merge_chunk_parser = subparsers.add_parser("merge-chunks")
    merge_chunk_parser.add_argument("--inputs", nargs="+", required=True)
    merge_chunk_parser.add_argument("--output-dir", type=Path, required=True)

    metric_shards = subparsers.add_parser("make-metric-shards")
    metric_shards.add_argument("--chunks-dir", type=Path, required=True)
    metric_shards.add_argument("--output-dir", type=Path, required=True)
    metric_shards.add_argument("--shards", type=int, default=6)

    run_metrics = subparsers.add_parser("run-metrics-shard")
    run_metrics.add_argument("--kind", choices=["processed", "raw"], required=True)
    run_metrics.add_argument("--chunks-parquet", type=Path, required=True)
    run_metrics.add_argument("--data-dir", type=Path, required=True)
    run_metrics.add_argument("--output-dir", type=Path, required=True)
    run_metrics.add_argument("--device", default="cuda:0")
    run_metrics.add_argument("--batch-size", type=int, default=32)
    run_metrics.add_argument("--allow-jina-api", action="store_true")

    merge_metric_parser = subparsers.add_parser("merge-metrics")
    merge_metric_parser.add_argument("--processed-inputs", nargs="+", required=True)
    merge_metric_parser.add_argument("--raw-inputs", nargs="+", required=True)
    merge_metric_parser.add_argument("--output-dir", type=Path, required=True)
    merge_metric_parser.add_argument("--chunks-dir", type=Path)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--expected-docs", type=int, default=33)
    validate.add_argument(
        "--profile",
        choices=["full", "nonsemantic", "semantic", "auto"],
        default="full",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "make-data-shards":
            result = make_data_shards(args.data_dir, args.output_dir, args.shards)
        elif args.command == "run-semantic-only":
            result = run_semantic_only(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                device=args.device,
                model_revision=args.model_revision,
                benchmark_documents=args.benchmark_documents,
                resume=args.resume,
                attention_implementation=args.attention_implementation,
                dtype_name=args.dtype,
                batch_size=args.batch_size,
                max_runtime_hours=args.max_runtime_hours,
            )
        elif args.command == "run-nonsemantic-shard":
            result = run_nonsemantic_shard(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                device=args.device,
                resume=args.resume,
                min_interval_seconds=args.min_interval_seconds,
                initial_cooldown_seconds=args.initial_cooldown_seconds,
                allow_fresh_paid_run=args.allow_fresh_paid_run,
            )
        elif args.command == "merge-chunks":
            result = merge_chunks(_paths(args.inputs), args.output_dir)
        elif args.command == "make-metric-shards":
            result = make_metric_shards(args.chunks_dir, args.output_dir, args.shards)
        elif args.command == "run-metrics-shard":
            result = run_metrics_shard(
                kind=args.kind,
                chunks_parquet=args.chunks_parquet,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
                allow_jina_api=args.allow_jina_api,
            )
        elif args.command == "merge-metrics":
            result = merge_metrics(
                processed_roots=_paths(args.processed_inputs),
                raw_roots=_paths(args.raw_inputs),
                output_dir=args.output_dir,
                chunks_dir=args.chunks_dir,
            )
        elif args.command == "validate":
            result = validate_artifacts(
                args.output_dir, args.expected_docs, args.profile
            )
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    except (ReproductionError, ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
