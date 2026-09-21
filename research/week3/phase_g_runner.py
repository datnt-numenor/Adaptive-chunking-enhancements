"""Protocol-faithful, resumable Phase G retrieval benchmark.

The runner intentionally separates free/local preparation from paid QA
generation and GPU indexing.  Week 2 artifacts are read-only inputs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
SOURCE_COMMIT = "ea87ce8e1a97888f3f179e7f1359ff7f43fb179d"
ADAPTIVE_CANDIDATES = (
    "our_recurs_1100",
    "our_recurs_600",
    "page",
    "llm_regex",
)
METRICS = (
    "size_compliance",
    "block_integrity",
    "intrachunk_cohesion",
    "document_contextual_coherence",
    "references_completeness",
)
WEIGHTS = {metric: 0.2 for metric in METRICS}

FIXED_SYSTEMS: dict[str, tuple[str, str]] = {
    "processed__our_recurs_1100": ("small_merged", "our_recurs_1100"),
    "processed__our_recurs_600": ("small_merged", "our_recurs_600"),
    "processed__page": ("small_merged", "page"),
    "processed__llm_regex": ("small_merged", "llm_regex"),
    "raw__langch_recurs_1100": ("raw", "langch_recurs_1100"),
    "raw__langch_recurs_default": ("raw", "langch_recurs_default"),
    "raw__page": ("raw", "page"),
    "raw__semantic": ("raw", "semantic"),
    "raw__sentence": ("raw", "sentence"),
}
ALL_SYSTEMS = ("adaptive", *FIXED_SYSTEMS)
PAPER_PROTOCOL_SYSTEMS = (
    "adaptive",
    "raw__langch_recurs_default",
    "raw__page",
)

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
RERANKER_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
RERANKER_REVISION = "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"
QA_MODEL = "gpt-4.1-2025-04-14"
QA_PRICE_INPUT_PER_MILLION = 2.0
QA_PRICE_OUTPUT_PER_MILLION = 8.0
QA_PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-4.1"
QA_PRICING_CHECKED_AT = "2026-09-20"
QA_PAIRS_PER_DOCUMENT = 3
QA_CONTEXT_TOKENS = 10_000
QA_MAX_OUTPUT_TOKENS = 3_500
FOLD_SEED = 2026
TOP_K_VALUES = (1, 3, 5, 10)
QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer "
    "the query\nQuery: "
)


class PhaseGError(RuntimeError):
    """Raised when an artifact or safety gate is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _harness_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PhaseGError(f"Missing JSONL artifact: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise PhaseGError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _domain(doc_name: str) -> str:
    return doc_name.split(" ", 1)[0]


def _doc_id(doc_name: str, source_sha256: str) -> str:
    return "doc-" + _sha256_text(f"{doc_name}\0{source_sha256}")[:16]


def _plain_pages(value: Any) -> list[int]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [int(item) for item in value]


def _load_source_documents(data_dir: Path) -> dict[str, dict[str, Any]]:
    parsed_dir = data_dir / "adi_parsed"
    documents: dict[str, dict[str, Any]] = {}
    for path in sorted(parsed_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        doc_name = path.stem
        full_text = str(payload["full_text"])
        source_sha256 = _sha256_text(full_text)
        page_spans: list[dict[str, Any]] = []
        cursor = 0
        for page_number, page_text_value in payload["pages"].items():
            page_text = str(page_text_value)
            start = full_text.find(page_text, cursor)
            if start < 0:
                start = full_text.find(page_text)
            if start < 0:
                raise PhaseGError(
                    f"Could not map page {page_number} into full_text for {doc_name}"
                )
            end = start + len(page_text)
            page_spans.append(
                {
                    "page": int(page_number),
                    "source_start": start,
                    "source_end": end,
                    "text": page_text,
                }
            )
            cursor = end
        documents[doc_name] = {
            "doc_name": doc_name,
            "doc_id": _doc_id(doc_name, source_sha256),
            "domain": _domain(doc_name),
            "source_sha256": source_sha256,
            "full_text": full_text,
            "page_spans": page_spans,
            "source_path": str(path),
        }
    if len(documents) != 33:
        raise PhaseGError(f"Expected 33 parsed documents, found {len(documents)}")
    return documents


def _find_chunk_spans(
    frame: pd.DataFrame, documents: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (doc_name, method), group in frame.groupby(["doc_name", "method"], sort=False):
        document = documents[str(doc_name)]
        full_text = document["full_text"]
        cursor = 0
        for row in group.sort_values("chunk_index").itertuples(index=False):
            raw_text = str(row.chunk_text)
            # The validated Week 2 raw stage contains four known empty page
            # chunks. They cannot be embedded or made relevant to evidence, so
            # omit them explicitly while retaining the raw stage for the
            # baseline itself.
            if not raw_text.strip():
                continue
            candidates = [raw_text]
            stripped = raw_text.strip()
            if stripped != raw_text:
                candidates.append(stripped)
            position = -1
            matched_text = ""
            for candidate in candidates:
                if not candidate:
                    continue
                position = full_text.find(candidate, cursor)
                if position < 0:
                    position = full_text.find(candidate)
                if position >= 0:
                    matched_text = candidate
                    break
            if position < 0:
                raise PhaseGError(
                    f"Could not map chunk {doc_name}/{method}/{row.chunk_index}"
                )
            end = position + len(matched_text)
            cursor = end
            output.append(
                {
                    "chunk_id": (
                        f"{document['doc_id']}::{method}::{int(row.chunk_index):06d}"
                    ),
                    "doc_id": document["doc_id"],
                    "doc_name": doc_name,
                    "method": method,
                    "chunk_index": int(row.chunk_index),
                    "chunk_text": raw_text.strip(),
                    "titles_context": str(getattr(row, "titles_context", "")),
                    "chunk_pages": _plain_pages(getattr(row, "chunk_pages", [])),
                    "source_start": position,
                    "source_end": end,
                }
            )
    return output


def _select_adaptive_methods(metrics: pd.DataFrame) -> dict[str, str]:
    from adaptive_chunking.paper.analysis import find_best_method

    filtered = metrics[metrics["chunking_method"].isin(ADAPTIVE_CANDIDATES)]
    if set(filtered["chunking_method"].unique()) != set(ADAPTIVE_CANDIDATES):
        raise PhaseGError("Processed metrics do not contain all Adaptive candidates")
    selected: dict[str, str] = {}
    for doc_name, group in filtered.groupby("doc_name"):
        pivot = group.pivot(
            index="metric_name", columns="chunking_method", values="score"
        )
        scores = pivot.reindex(METRICS)
        method, _ = find_best_method(scores, WEIGHTS)
        if method not in ADAPTIVE_CANDIDATES:
            raise PhaseGError(f"Selector escaped candidate set for {doc_name}: {method}")
        selected[str(doc_name)] = method
    if len(selected) != 33:
        raise PhaseGError(f"Expected 33 Adaptive selections, found {len(selected)}")
    return selected


def _build_folds(documents: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    from sklearn.model_selection import StratifiedGroupKFold

    docs = sorted(documents.values(), key=lambda item: item["doc_id"])
    y = [item["domain"] for item in docs]
    groups = [item["doc_id"] for item in docs]
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=FOLD_SEED
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold, (_, test_indices) in enumerate(splitter.split(groups, y, groups)):
        for index in test_indices:
            doc_id = docs[int(index)]["doc_id"]
            if doc_id in seen:
                raise PhaseGError(f"Document appears in multiple folds: {doc_id}")
            seen.add(doc_id)
            rows.append(
                {
                    "fold": fold,
                    "doc_id": doc_id,
                    "doc_name": docs[int(index)]["doc_name"],
                    "domain": docs[int(index)]["domain"],
                }
            )
    if seen != set(groups):
        raise PhaseGError("Fold assignment does not cover every document exactly once")
    return sorted(rows, key=lambda item: (item["fold"], item["doc_id"]))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    final_dir = args.final_dir
    output_dir = args.output_dir
    validation_path = final_dir / "validation_report.json"
    merge_path = final_dir / "metrics_merge_manifest.json"
    if not validation_path.is_file() or not merge_path.is_file():
        raise PhaseGError("Week 2 final artifacts are incomplete")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    if merge.get("source_commit") != SOURCE_COMMIT:
        raise PhaseGError("Week 2 source commit does not match the Phase G pin")
    if validation["metrics"]["processed"]["rows"] != 2640:
        raise PhaseGError("Unexpected processed metric row count")
    if validation["metrics"]["raw"]["rows"] != 1650:
        raise PhaseGError("Unexpected raw metric row count")

    documents = _load_source_documents(args.data_dir)
    processed_chunks = pd.read_parquet(
        final_dir / "chunks" / "small_merged" / "chunks.parquet"
    )
    raw_chunks = pd.read_parquet(final_dir / "chunks" / "raw" / "chunks.parquet")
    metrics = pd.read_parquet(final_dir / "results" / "chunking_metrics.parquet")
    adaptive_methods = _select_adaptive_methods(metrics)

    systems_dir = output_dir / "systems"
    system_records: dict[str, dict[str, Any]] = {}
    fixed_frames: dict[str, pd.DataFrame] = {}
    for system_id, (stage, method) in FIXED_SYSTEMS.items():
        source = processed_chunks if stage == "small_merged" else raw_chunks
        frame = source[source["method"] == method].copy()
        if frame["doc_name"].nunique() != 33:
            raise PhaseGError(f"{system_id} does not cover 33 documents")
        fixed_frames[system_id] = frame

    adaptive_parts: list[pd.DataFrame] = []
    for doc_name, method in adaptive_methods.items():
        adaptive_parts.append(
            processed_chunks[
                (processed_chunks["doc_name"] == doc_name)
                & (processed_chunks["method"] == method)
            ]
        )
    system_frames = {"adaptive": pd.concat(adaptive_parts, ignore_index=True), **fixed_frames}

    smoke_candidates = pd.concat(
        [
            frame[["doc_name", "chunk_text"]]
            for frame in system_frames.values()
        ],
        ignore_index=True,
    )
    smoke_candidates["chunk_chars"] = smoke_candidates["chunk_text"].map(
        lambda value: len(str(value))
    )
    maximum_by_document = (
        smoke_candidates.groupby("doc_name")["chunk_chars"].max().sort_values(ascending=False)
    )
    first_smoke_name = str(maximum_by_document.index[0])
    first_smoke_domain = documents[first_smoke_name]["domain"]
    second_smoke_name = next(
        str(doc_name)
        for doc_name in maximum_by_document.index[1:]
        if documents[str(doc_name)]["domain"] != first_smoke_domain
    )
    smoke_documents = [
        {
            "doc_id": documents[doc_name]["doc_id"],
            "doc_name": doc_name,
            "domain": documents[doc_name]["domain"],
            "maximum_chunk_characters": int(maximum_by_document.loc[doc_name]),
        }
        for doc_name in (first_smoke_name, second_smoke_name)
    ]

    for system_id, frame in system_frames.items():
        rows = _find_chunk_spans(frame, documents)
        excluded_empty_chunks = int(frame["chunk_text"].fillna("").str.strip().eq("").sum())
        path = systems_dir / f"{system_id}.jsonl"
        _write_jsonl(path, rows)
        system_records[system_id] = {
            # Keep prepared artifacts relocatable: this directory is copied to the
            # GPU worker after QA review.
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": _sha256_file(path),
            "chunks": len(rows),
            "excluded_empty_chunks": excluded_empty_chunks,
            "documents": len({row["doc_id"] for row in rows}),
            "stage": "mixed_processed_selection" if system_id == "adaptive" else FIXED_SYSTEMS[system_id][0],
            "method": "per_document_equal_weight" if system_id == "adaptive" else FIXED_SYSTEMS[system_id][1],
        }

    document_rows = [
        {
            key: value
            for key, value in document.items()
            if key not in {"full_text", "page_spans"}
        }
        | {
            "page_spans": [
                {key: value for key, value in page.items() if key != "text"}
                for page in document["page_spans"]
            ]
        }
        for document in sorted(documents.values(), key=lambda item: item["doc_id"])
    ]
    _write_jsonl(output_dir / "documents.jsonl", document_rows)
    _json_dump(
        output_dir / "adaptive_selections.json",
        {
            "candidate_methods": list(ADAPTIVE_CANDIDATES),
            "weights": WEIGHTS,
            "selections": adaptive_methods,
        },
    )
    folds = _build_folds(documents)
    _write_jsonl(output_dir / "folds.jsonl", folds)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "harness_sha256": _harness_sha256(),
        "status": "complete",
        "created_at": _utc_now(),
        "source_commit": SOURCE_COMMIT,
        "week2_validation_sha256": _sha256_file(validation_path),
        "week2_merge_sha256": _sha256_file(merge_path),
        "documents": 33,
        "folds": 5,
        "fold_seed": FOLD_SEED,
        "smoke_selection": {
            "rule": "largest_chunk_then_largest_different_domain",
            "documents": smoke_documents,
        },
        "adaptive_candidates": list(ADAPTIVE_CANDIDATES),
        "paper_protocol_systems": list(PAPER_PROTOCOL_SYSTEMS),
        "systems": system_records,
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    _json_dump(output_dir / "prepare_manifest.json", manifest)
    return manifest


def _count_tokens(text: str) -> int:
    from adaptive_chunking.chunking_utils import count_tokens

    return int(count_tokens(text, model="gpt-4o"))


def _build_document_context(document: dict[str, Any]) -> tuple[str, list[int]]:
    parts: list[str] = []
    included_pages: list[int] = []
    for page in document["page_spans"]:
        header = f"\n<page number=\"{page['page']}\">\n"
        footer = "\n</page>\n"
        available_lines: list[str] = []
        for line in page["text"].splitlines(keepends=True):
            candidate = "".join(parts) + header + "".join(available_lines) + line + footer
            if _count_tokens(candidate) > QA_CONTEXT_TOKENS:
                break
            available_lines.append(line)
        if not available_lines:
            break
        parts.extend([header, "".join(available_lines), footer])
        included_pages.append(int(page["page"]))
        if _count_tokens("".join(parts)) >= QA_CONTEXT_TOKENS:
            break
    return "".join(parts).strip(), included_pages


def _qa_prompt(doc_name: str, context: str) -> str:
    return f"""You are creating a frozen evaluation set for a retrieval-augmented generation benchmark.

Document: {doc_name}

Create exactly three diverse question-answer pairs. Each answer must be no more than 120 words and fully supported by the supplied document excerpt. Use exactly one evidence item per pair unless a second item is essential. Every evidence quote must be 15 to 40 consecutive words copied character-for-character from exactly one <page> block. Preserve spelling, capitalization, punctuation, and typos. Never join non-adjacent text, paraphrase, correct, or use ellipses inside a quote. Before returning, verify that every quote is an exact substring of its stated page. Do not use outside knowledge. Vary question type and difficulty.

Document excerpt:
{context}
"""


def _qa_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "qas": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question": {"type": "string", "minLength": 1},
                        "answer": {"type": "string", "minLength": 1},
                        "evidence": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 3,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "page": {"type": "integer", "minimum": 1},
                                    "quote": {"type": "string", "minLength": 1},
                                },
                                "required": ["page", "quote"],
                            },
                        },
                    },
                    "required": ["question", "answer", "evidence"],
                },
            }
        },
        "required": ["qas"],
    }


def _map_evidence(
    document: dict[str, Any], evidence: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    pages = {int(page["page"]): page for page in document["page_spans"]}
    mapped: list[dict[str, Any]] = []
    for item in evidence:
        page_number = int(item["page"])
        quote = str(item["quote"])
        if page_number not in pages:
            raise PhaseGError(f"Evidence refers to absent page {page_number}")
        page = pages[page_number]
        occurrences: list[int] = []
        start = 0
        while True:
            position = page["text"].find(quote, start)
            if position < 0:
                break
            occurrences.append(position)
            start = position + 1
        if len(occurrences) != 1:
            raise PhaseGError(
                f"Evidence quote must occur exactly once on page {page_number}; found {len(occurrences)}"
            )
        source_start = int(page["source_start"]) + occurrences[0]
        mapped.append(
            {
                "page": page_number,
                "quote": quote,
                "source_start": source_start,
                "source_end": source_start + len(quote),
                "quote_sha256": _sha256_text(quote),
            }
        )
    return mapped


def _qa_content_hash(qa: dict[str, Any]) -> str:
    """Return a stable identity for the reviewed QA content and evidence spans."""
    identity = {
        "doc_id": qa["doc_id"],
        "question": qa["question"],
        "reference_answer": qa["reference_answer"],
        "evidence": [
            {
                "page": item["page"],
                "quote": item["quote"],
                "source_start": item["source_start"],
                "source_end": item["source_end"],
            }
            for item in qa["evidence"]
        ],
    }
    return _sha256_text(json.dumps(identity, ensure_ascii=False, sort_keys=True))


def _load_prepared_sources(
    prepared_dir: Path, data_dir: Path
) -> dict[str, dict[str, Any]]:
    manifest = json.loads(
        (prepared_dir / "prepare_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "complete" or manifest.get("source_commit") != SOURCE_COMMIT:
        raise PhaseGError("Prepared Phase G inputs are not valid")
    return _load_source_documents(data_dir)


def _estimate_qa_cost(input_tokens: int, documents: int) -> dict[str, Any]:
    maximum_output_tokens = documents * QA_MAX_OUTPUT_TOKENS
    input_cost = input_tokens / 1_000_000 * QA_PRICE_INPUT_PER_MILLION
    output_cost = maximum_output_tokens / 1_000_000 * QA_PRICE_OUTPUT_PER_MILLION
    return {
        "input_tokens": input_tokens,
        "maximum_output_tokens": maximum_output_tokens,
        "input_cost_usd": input_cost,
        "maximum_output_cost_usd": output_cost,
        "worst_case_cost_usd": input_cost + output_cost,
        "pricing_source": QA_PRICING_SOURCE,
        "pricing_checked_at": QA_PRICING_CHECKED_AT,
    }


def _generate_one_qa(
    *, client: Any, prompt: str, request_hash: str, cache_path: Path
) -> dict[str, Any]:
    response = client.responses.create(
        model=QA_MODEL,
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=QA_MAX_OUTPUT_TOKENS,
        text={
            "format": {
                "type": "json_schema",
                "name": "phase_g_qa_batch",
                "strict": True,
                "schema": _qa_json_schema(),
            }
        },
        store=False,
    )
    usage = getattr(response, "usage", None)
    payload = {
        "request_hash": request_hash,
        "model": QA_MODEL,
        "response_id": response.id,
        "usage": {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        },
        "response_status": getattr(response, "status", None),
        "incomplete_details": str(getattr(response, "incomplete_details", None)),
        "raw_output": response.output_text,
        "created_at": _utc_now(),
    }
    # Persist usage and raw output before parsing so truncated/invalid JSON is
    # still auditable and never triggers an implicit paid retry.
    _json_dump(cache_path, payload)
    parsed = json.loads(response.output_text)
    payload["parsed"] = parsed
    _json_dump(cache_path, payload)
    return payload


def _actual_cost(cache_payloads: Iterable[dict[str, Any]]) -> float:
    input_tokens = sum(item["usage"]["input_tokens"] for item in cache_payloads)
    output_tokens = sum(item["usage"]["output_tokens"] for item in cache_payloads)
    return (
        input_tokens / 1_000_000 * QA_PRICE_INPUT_PER_MILLION
        + output_tokens / 1_000_000 * QA_PRICE_OUTPUT_PER_MILLION
    )


def _known_cache_cost(cache_dir: Path) -> float:
    payloads: list[dict[str, Any]] = []
    if cache_dir.is_dir():
        for path in cache_dir.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "usage" in payload:
                payloads.append(payload)
    return _actual_cost(payloads)


def generate_qa(args: argparse.Namespace) -> dict[str, Any]:
    if args.request_delay_seconds < 0:
        raise PhaseGError("--request-delay-seconds must be non-negative")
    documents = _load_prepared_sources(args.prepared_dir, args.data_dir)
    selected = sorted(documents.values(), key=lambda item: item["doc_id"])
    mode = "full"
    if args.smoke_docs:
        if args.smoke_docs != 2:
            raise PhaseGError("Protocol smoke test requires exactly --smoke-docs 2")
        prepared_manifest = json.loads(
            (args.prepared_dir / "prepare_manifest.json").read_text(encoding="utf-8")
        )
        smoke_ids = {
            item["doc_id"]
            for item in prepared_manifest["smoke_selection"]["documents"]
        }
        selected = [item for item in selected if item["doc_id"] in smoke_ids]
        if len(selected) != 2:
            raise PhaseGError("Prepared smoke selection does not resolve to two documents")
        mode = "smoke"

    requests: list[dict[str, Any]] = []
    for document in selected:
        context, included_pages = _build_document_context(document)
        prompt = _qa_prompt(document["doc_name"], context)
        request_hash = _sha256_text(
            json.dumps(
                {
                    "model": QA_MODEL,
                    "prompt": prompt,
                    "source_sha256": document["source_sha256"],
                    "schema": _qa_json_schema(),
                    "max_output_tokens": QA_MAX_OUTPUT_TOKENS,
                    "store": False,
                },
                sort_keys=True,
            )
        )
        requests.append(
            {
                "document": document,
                "prompt": prompt,
                "prompt_sha256": _sha256_text(prompt),
                "request_hash": request_hash,
                "included_pages": included_pages,
                "input_tokens": _count_tokens(prompt),
                "cache_path": args.output_dir / "cache" / f"{request_hash}.json",
            }
        )
    estimate = _estimate_qa_cost(
        sum(item["input_tokens"] for item in requests), len(requests)
    )
    if estimate["worst_case_cost_usd"] > args.budget_usd:
        raise PhaseGError(
            f"Worst-case QA cost ${estimate['worst_case_cost_usd']:.2f} exceeds ${args.budget_usd:.2f} cap"
        )
    known_prior_cost = _known_cache_cost(args.output_dir / "cache")
    if known_prior_cost + estimate["worst_case_cost_usd"] > args.budget_usd:
        raise PhaseGError(
            "Known cached spend plus this worst-case run exceeds the QA budget cap"
        )
    estimate_path = args.output_dir / f"qa_{mode}_estimate.json"
    _json_dump(
        estimate_path,
        {
            "schema_version": SCHEMA_VERSION,
            "harness_sha256": _harness_sha256(),
            "mode": mode,
            "model": QA_MODEL,
            "documents": len(requests),
            "estimate": estimate,
            "budget_usd": args.budget_usd,
            "known_prior_cache_cost_usd": known_prior_cost,
            "requests": [
                {
                    key: value
                    for key, value in item.items()
                    if key
                    in {
                        "prompt_sha256",
                        "request_hash",
                        "included_pages",
                        "input_tokens",
                    }
                }
                | {"doc_id": item["document"]["doc_id"]}
                for item in requests
            ],
        },
    )
    if not args.allow_paid_run:
        return {
            "status": "estimate_only",
            "mode": mode,
            "estimate_path": str(estimate_path),
            "estimate": estimate,
        }
    if mode == "full":
        if args.smoke_manifest is None or not args.smoke_manifest.is_file():
            raise PhaseGError(
                "Paid full QA generation requires --smoke-manifest from the two-document run"
            )
        smoke_manifest = json.loads(args.smoke_manifest.read_text(encoding="utf-8"))
        if (
            smoke_manifest.get("status") != "complete"
            or smoke_manifest.get("mode") != "smoke"
            or smoke_manifest.get("documents_requested") != 2
            or smoke_manifest.get("qa_candidates") != 6
        ):
            raise PhaseGError("QA smoke manifest is not a complete two-document run")
        if (
            args.smoke_evaluation_manifest is None
            or not args.smoke_evaluation_manifest.is_file()
        ):
            raise PhaseGError(
                "Paid full QA generation requires --smoke-evaluation-manifest from the end-to-end smoke run"
            )
        smoke_evaluation = json.loads(
            args.smoke_evaluation_manifest.read_text(encoding="utf-8")
        )
        if (
            smoke_evaluation.get("status") != "complete"
            or smoke_evaluation.get("mode") != "smoke"
            or smoke_evaluation.get("qa_count") != 6
            or set(smoke_evaluation.get("systems", [])) != set(ALL_SYSTEMS)
        ):
            raise PhaseGError("End-to-end smoke evaluation manifest is not valid")
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parents[2] / ".env", override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        raise PhaseGError("OPENAI_API_KEY is required only for --allow-paid-run")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise PhaseGError(
            "OpenAI SDK is missing; install the project's paper extra before a paid run"
        ) from exc

    cache_payloads: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    last_paid_request_started: float | None = None
    with OpenAI(max_retries=0) as client:
        for request in requests:
            cache_path = request["cache_path"]
            try:
                if cache_path.is_file():
                    payload = json.loads(cache_path.read_text(encoding="utf-8"))
                    if payload.get("request_hash") != request["request_hash"]:
                        raise PhaseGError(f"Cache identity mismatch: {cache_path}")
                else:
                    if last_paid_request_started is not None:
                        elapsed = time.monotonic() - last_paid_request_started
                        remaining = args.request_delay_seconds - elapsed
                        if remaining > 0:
                            time.sleep(remaining)
                    last_paid_request_started = time.monotonic()
                    payload = _generate_one_qa(
                        client=client,
                        prompt=request["prompt"],
                        request_hash=request["request_hash"],
                        cache_path=cache_path,
                    )
                cache_payloads.append(payload)
            except Exception as exc:  # record once; no hidden retry
                failures.append(
                    {"doc_id": request["document"]["doc_id"], "error": repr(exc)}
                )
                break

    candidates: list[dict[str, Any]] = []
    validation_issues: list[dict[str, str]] = []
    request_by_hash = {item["request_hash"]: item for item in requests}
    for payload in cache_payloads:
        request = request_by_hash[payload["request_hash"]]
        document = request["document"]
        qas = payload.get("parsed", {}).get("qas", [])
        if len(qas) != QA_PAIRS_PER_DOCUMENT:
            failures.append(
                {
                    "doc_id": document["doc_id"],
                    "error": f"Expected 3 QA pairs, found {len(qas)}",
                }
            )
            continue
        for index, qa in enumerate(qas, 1):
            validation_status = "valid"
            validation_error = ""
            try:
                evidence = _map_evidence(document, qa["evidence"])
            except Exception as exc:
                evidence = []
                validation_status = "invalid_evidence"
                validation_error = str(exc)
                validation_issues.append(
                    {
                        "qa_id": f"{document['doc_id']}::q{index}",
                        "error": validation_error,
                    }
                )
            candidate = {
                    "schema_version": SCHEMA_VERSION,
                    "qa_id": f"{document['doc_id']}::q{index}",
                    "doc_id": document["doc_id"],
                    "doc_name": document["doc_name"],
                    "domain": document["domain"],
                    "question": str(qa["question"]).strip(),
                    "reference_answer": str(qa["answer"]).strip(),
                    "evidence": evidence,
                    "original_evidence": qa["evidence"],
                    "validation_status": validation_status,
                    "validation_error": validation_error,
                    "generation": {
                        "model": QA_MODEL,
                        "request_hash": payload["request_hash"],
                        "prompt_sha256": request["prompt_sha256"],
                        "source_sha256": document["source_sha256"],
                    },
                }
            candidate["qa_hash"] = _qa_content_hash(candidate)
            candidates.append(candidate)
    candidates.sort(key=lambda item: item["qa_id"])
    _write_jsonl(args.output_dir / "qa_candidates.jsonl", candidates)
    review_path = args.output_dir / "qa_review.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "qa_id",
                "doc_name",
                "question",
                "reference_answer",
                "evidence_json",
                "validation_status",
                "validation_error",
                "status",
                "edited_question",
                "edited_answer",
                "edited_evidence_json",
                "notes",
            ),
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "qa_id": candidate["qa_id"],
                    "doc_name": candidate["doc_name"],
                    "question": candidate["question"],
                    "reference_answer": candidate["reference_answer"],
                    "evidence_json": json.dumps(
                        candidate["evidence"] or candidate["original_evidence"],
                        ensure_ascii=False,
                    ),
                    "validation_status": candidate["validation_status"],
                    "validation_error": candidate["validation_error"],
                    "status": "",
                    "edited_question": "",
                    "edited_answer": "",
                    "edited_evidence_json": "",
                    "notes": "",
                }
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "harness_sha256": _harness_sha256(),
        "status": "complete" if not failures else "incomplete",
        "mode": mode,
        "created_at": _utc_now(),
        "model": QA_MODEL,
        "documents_requested": len(requests),
        "documents_cached": len(cache_payloads),
        "qa_candidates": len(candidates),
        "valid_candidates": sum(
            item["validation_status"] == "valid" for item in candidates
        ),
        "invalid_candidates": sum(
            item["validation_status"] != "valid" for item in candidates
        ),
        "estimate": estimate,
        "actual_cost_usd": _actual_cost(cache_payloads),
        "known_cache_cost_usd": _known_cache_cost(args.output_dir / "cache"),
        "budget_usd": args.budget_usd,
        "failures": failures,
        "validation_issues": validation_issues,
        "automatic_retries": 0,
        "input_prepare_manifest_sha256": _sha256_file(
            args.prepared_dir / "prepare_manifest.json"
        ),
        "request_hashes": [item["request_hash"] for item in requests],
        "prompt_sha256s": [item["prompt_sha256"] for item in requests],
    }
    manifest_path = args.output_dir / f"qa_{mode}_manifest.json"
    _json_dump(manifest_path, manifest)
    if failures:
        raise PhaseGError(f"QA generation incomplete; inspect {manifest_path}")
    expected = len(requests) * QA_PAIRS_PER_DOCUMENT
    if len(candidates) != expected:
        raise PhaseGError(f"Expected {expected} QA candidates, found {len(candidates)}")
    return manifest


def freeze_qa(args: argparse.Namespace) -> dict[str, Any]:
    documents = _load_prepared_sources(args.prepared_dir, args.data_dir)
    documents_by_id = {document["doc_id"]: document for document in documents.values()}
    candidates = {row["qa_id"]: row for row in _read_jsonl(args.qa_dir / "qa_candidates.jsonl")}
    if args.review_decisions is not None:
        decision_payload = json.loads(
            args.review_decisions.read_text(encoding="utf-8")
        )
        reviews = decision_payload["decisions"]
        reviewer = decision_payload.get("reviewer", "unspecified")
        reviewed_at = decision_payload.get("reviewed_at")
    else:
        review_path = args.qa_dir / "qa_review.csv"
        if not review_path.is_file():
            raise PhaseGError(f"Missing review file: {review_path}")
        with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reviews = list(csv.DictReader(handle))
        reviewer = "csv-reviewer"
        reviewed_at = None
    if {row["qa_id"] for row in reviews} != set(candidates):
        raise PhaseGError("Review rows do not match QA candidates exactly")
    frozen: list[dict[str, Any]] = []
    rejected: list[str] = []
    for review in reviews:
        status = review["status"].strip().lower()
        if status not in {"accept", "edit", "reject"}:
            raise PhaseGError(f"Invalid review status for {review['qa_id']}: {status!r}")
        if status == "reject":
            rejected.append(review["qa_id"])
            continue
        candidate = dict(candidates[review["qa_id"]])
        if status == "accept" and candidate.get("validation_status") != "valid":
            raise PhaseGError(
                f"Invalid evidence must be edited, not accepted: {review['qa_id']}"
            )
        if status == "edit":
            question = review.get("edited_question", "").strip()
            answer = review.get("edited_answer", "").strip()
            evidence_json = review.get("edited_evidence_json", "").strip()
            if not question or not answer or not evidence_json:
                raise PhaseGError(f"Edited QA is incomplete: {review['qa_id']}")
            evidence_input = json.loads(evidence_json)
            candidate["question"] = question
            candidate["reference_answer"] = answer
            candidate["evidence"] = _map_evidence(
                documents_by_id[candidate["doc_id"]], evidence_input
            )
            candidate["validation_status"] = "valid_after_edit"
            candidate["validation_error"] = ""
        candidate["review"] = {
            "status": status,
            "notes": review.get("notes", "").strip(),
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
        }
        candidate["qa_hash"] = _qa_content_hash(candidate)
        frozen.append(candidate)
    expected_qa = args.expected_documents * QA_PAIRS_PER_DOCUMENT
    if rejected:
        raise PhaseGError(
            f"Freeze requires exactly {expected_qa} accepted/edited QA; rejected: {rejected}"
        )
    if len(frozen) != expected_qa:
        raise PhaseGError(f"Freeze requires {expected_qa} QA, found {len(frozen)}")
    counts = pd.Series([row["doc_id"] for row in frozen]).value_counts()
    if len(counts) != args.expected_documents or set(counts.tolist()) != {3}:
        raise PhaseGError("Frozen QA must contain exactly three questions per document")
    if len({row["qa_id"] for row in frozen}) != len(frozen):
        raise PhaseGError("Frozen QA IDs must be unique")
    if len({row["qa_hash"] for row in frozen}) != len(frozen):
        raise PhaseGError("Frozen QA content hashes must be unique")
    frozen.sort(key=lambda item: item["qa_id"])
    path = args.qa_dir / "qa_frozen.jsonl"
    _write_jsonl(path, frozen)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "harness_sha256": _harness_sha256(),
        "status": "complete",
        "mode": "full" if args.expected_documents == 33 else "smoke",
        "created_at": _utc_now(),
        "qa_count": expected_qa,
        "documents": args.expected_documents,
        "qa_sha256": _sha256_file(path),
        "generation_model": QA_MODEL,
        "human_review_required": True,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
    }
    _json_dump(args.qa_dir / "qa_frozen_manifest.json", manifest)
    return manifest


def _selected_system_ids(value: str) -> list[str]:
    if value == "all":
        return list(ALL_SYSTEMS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(selected) - set(ALL_SYSTEMS)
    if unknown:
        raise PhaseGError(f"Unknown systems: {sorted(unknown)}")
    return selected


def _resolve_model(repo_id: str, revision: str, cache_dir: Path | None) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise PhaseGError("huggingface_hub is required for GPU stages") from exc
    return snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)


def _gpu_preflight(device: str) -> dict[str, Any]:
    try:
        import flash_attn
        import torch
    except ImportError as exc:
        raise PhaseGError("GPU stages require torch and flash-attn") from exc
    if device != "cuda:0" or not torch.cuda.is_available():
        raise PhaseGError("Phase G GPU stages require CUDA device cuda:0")
    name = torch.cuda.get_device_name(0)
    capability = tuple(int(item) for item in torch.cuda.get_device_capability(0))
    if "A5000" not in name:
        raise PhaseGError(f"Expected an RTX A5000, found {name}")
    if capability < (8, 0):
        raise PhaseGError(f"Compute capability must be >= 8.0, found {capability}")
    if not torch.cuda.is_bf16_supported():
        raise PhaseGError("Selected GPU/runtime does not support bfloat16")
    return {
        "name": name,
        "compute_capability": list(capability),
        "bf16_supported": True,
        "torch_version": torch.__version__,
        "flash_attn_version": getattr(flash_attn, "__version__", None)
        or version("flash-attn"),
    }


def _environment_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PhaseGError(f"Missing pip-freeze environment file: {path}")
    return {"path": str(path), "sha256": _sha256_file(path)}


def _haystack_documents(rows: Sequence[dict[str, Any]]) -> list[Any]:
    try:
        from haystack import Document
    except ImportError as exc:
        raise PhaseGError("haystack-ai is required for GPU stages") from exc
    return [
        Document(
            content=row["chunk_text"],
            meta={
                key: row[key]
                for key in (
                    "chunk_id",
                    "doc_id",
                    "doc_name",
                    "method",
                    "chunk_index",
                    "chunk_pages",
                    "source_start",
                    "source_end",
                )
            },
        )
        for row in rows
    ]


def index_systems(args: argparse.Namespace) -> dict[str, Any]:
    gpu = _gpu_preflight(args.device)
    environment = _environment_record(args.environment_file)
    from adaptive_chunking.paper.rag_utils import index_documents

    prepared_manifest = json.loads(
        (args.prepared_dir / "prepare_manifest.json").read_text(encoding="utf-8")
    )
    frozen_manifest_path = args.qa_dir / "qa_frozen_manifest.json"
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    qa_path = args.qa_dir / "qa_frozen.jsonl"
    if _sha256_file(qa_path) != frozen_manifest.get("qa_sha256"):
        raise PhaseGError("Frozen QA checksum mismatch before indexing")
    qas = _read_jsonl(qa_path)
    qa_doc_ids = {row["doc_id"] for row in qas}
    if len(qa_doc_ids) not in {2, 33}:
        raise PhaseGError(
            "Indexing accepts only the two-document smoke set or full 33-document QA set"
        )
    if len(qa_doc_ids) == 33:
        if (
            args.smoke_evaluation_manifest is None
            or not args.smoke_evaluation_manifest.is_file()
        ):
            raise PhaseGError(
                "Full indexing requires --smoke-evaluation-manifest from the end-to-end two-document run"
            )
        smoke_evaluation = json.loads(
            args.smoke_evaluation_manifest.read_text(encoding="utf-8")
        )
        if (
            smoke_evaluation.get("status") != "complete"
            or smoke_evaluation.get("mode") != "smoke"
            or smoke_evaluation.get("qa_count") != 6
        ):
            raise PhaseGError("Smoke retrieval evaluation manifest is not valid")
    systems = _selected_system_ids(args.systems)
    embedding_path = _resolve_model(
        EMBEDDING_MODEL, EMBEDDING_REVISION, args.model_cache_dir
    )
    existing_manifest_path = args.output_dir / "index_manifest.json"
    if existing_manifest_path.is_file() and not args.force:
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("input_qa_sha256") != frozen_manifest["qa_sha256"]:
            raise PhaseGError(
                "Existing indexes were built for a different QA set; use a separate output directory"
            )
    records: dict[str, Any] = {}
    for system_id in systems:
        source_path = args.prepared_dir / prepared_manifest["systems"][system_id]["path"]
        if _sha256_file(source_path) != prepared_manifest["systems"][system_id]["sha256"]:
            raise PhaseGError(f"Prepared system checksum mismatch: {system_id}")
        rows = [
            row
            for row in _read_jsonl(source_path)
            if row["doc_id"] in qa_doc_ids
        ]
        if {row["doc_id"] for row in rows} != qa_doc_ids:
            raise PhaseGError(f"{system_id} does not cover every frozen-QA document")
        output_dir = args.output_dir / system_id
        store_path = output_dir / "document_store.json"
        if store_path.is_file() and not args.force:
            records[system_id] = {
                "status": "reused",
                "path": str(store_path),
                "sha256": _sha256_file(store_path),
                "chunks": len(rows),
                "source_documents": len(qa_doc_ids),
            }
            continue
        index_documents(
            documents=_haystack_documents(rows),
            output_dir=output_dir,
            embedding_model_name=embedding_path,
            device=args.device,
            batch_size=args.batch_size,
        )
        if not store_path.is_file():
            raise PhaseGError(f"Index was not created: {store_path}")
        records[system_id] = {
            "status": "created",
            "path": str(store_path),
            "sha256": _sha256_file(store_path),
            "chunks": len(rows),
            "source_documents": len(qa_doc_ids),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "harness_sha256": _harness_sha256(),
        "status": "complete",
        "created_at": _utc_now(),
        "source_commit": SOURCE_COMMIT,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_REVISION,
        "device": args.device,
        "dtype": "bfloat16",
        "attention_implementation": "flash_attention_2",
        "batch_size": args.batch_size,
        "mode": "full" if len(qa_doc_ids) == 33 else "smoke",
        "qa_count": len(qas),
        "source_documents": len(qa_doc_ids),
        "input_qa_sha256": frozen_manifest["qa_sha256"],
        "input_prepare_manifest_sha256": _sha256_file(
            args.prepared_dir / "prepare_manifest.json"
        ),
        "document_prompt_sha256": _sha256_text(""),
        "query_prompt_sha256": _sha256_text(QUERY_PROMPT),
        "systems": records,
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "gpu": gpu,
        "environment": environment,
    }
    _json_dump(args.output_dir / "index_manifest.json", manifest)
    return manifest


def retrieve(args: argparse.Namespace) -> dict[str, Any]:
    gpu = _gpu_preflight(args.device)
    environment = _environment_record(args.environment_file)
    from adaptive_chunking.paper.rag_utils import create_retrieval_pipeline
    import torch

    frozen_manifest = json.loads(
        (args.qa_dir / "qa_frozen_manifest.json").read_text(encoding="utf-8")
    )
    qa_path = args.qa_dir / "qa_frozen.jsonl"
    if _sha256_file(qa_path) != frozen_manifest["qa_sha256"]:
        raise PhaseGError("Frozen QA checksum mismatch")
    qas = _read_jsonl(qa_path)
    systems = _selected_system_ids(args.systems)
    index_manifest_path = args.index_dir / "index_manifest.json"
    if not index_manifest_path.is_file():
        raise PhaseGError(f"Missing index manifest: {index_manifest_path}")
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    if (
        index_manifest.get("embedding_revision") != EMBEDDING_REVISION
        or index_manifest.get("source_commit") != SOURCE_COMMIT
    ):
        raise PhaseGError("Index manifest does not match pinned Phase G configuration")
    if index_manifest.get("input_qa_sha256") != frozen_manifest["qa_sha256"]:
        raise PhaseGError("Index and frozen QA sets do not match")
    embedding_path = _resolve_model(
        EMBEDDING_MODEL, EMBEDDING_REVISION, args.model_cache_dir
    )
    reranker_path = _resolve_model(
        RERANKER_MODEL, RERANKER_REVISION, args.model_cache_dir
    )
    outputs: dict[str, Any] = {}
    for system_id in systems:
        output_path = args.output_dir / f"{system_id}.jsonl"
        if output_path.is_file() and not args.force:
            existing_rows = _read_jsonl(output_path)
            if {row["qa_id"] for row in existing_rows} != {
                qa["qa_id"] for qa in qas
            }:
                raise PhaseGError(
                    f"Existing retrieval output belongs to a different QA set: {system_id}"
                )
            outputs[system_id] = {
                "status": "reused",
                "path": str(output_path),
                "sha256": _sha256_file(output_path),
                "queries": len(existing_rows),
            }
            continue
        store_path = args.index_dir / system_id / "document_store.json"
        if not store_path.is_file():
            raise PhaseGError(f"Missing index for {system_id}: {store_path}")
        pipeline = create_retrieval_pipeline(
            document_store_path=store_path,
            embedding_model=embedding_path,
            embedder_config_kwargs={"attn_implementation": "flash_attention_2"},
            embedder_model_kwargs={"torch_dtype": torch.bfloat16},
            embedder_batch_size=args.batch_size,
            reranker_model=reranker_path,
            reranker_batch_size=args.batch_size,
            top_k_semantic_search=50,
            top_k_keyword_search=50,
            top_k_reranker=10,
            device=args.device,
        )
        rows: list[dict[str, Any]] = []
        for qa in qas:
            result = pipeline.run(
                {
                    "text_embedder": {
                        "text": qa["question"],
                        "prompt_name": "query",
                    },
                    "bm25_retriever": {"query": qa["question"]},
                    "ranker": {"query": qa["question"]},
                }
            )
            documents = result.get("ranker", {}).get("documents", [])
            rows.append(
                {
                    "qa_id": qa["qa_id"],
                    "system_id": system_id,
                    "results": [
                        {
                            "rank": rank,
                            "score": float(document.score),
                            "content": document.content,
                            "meta": document.meta,
                        }
                        for rank, document in enumerate(documents, 1)
                    ],
                }
            )
        _write_jsonl(output_path, rows)
        outputs[system_id] = {
            "status": "created",
            "path": str(output_path),
            "sha256": _sha256_file(output_path),
            "queries": len(rows),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "harness_sha256": _harness_sha256(),
        "status": "complete",
        "created_at": _utc_now(),
        "source_commit": SOURCE_COMMIT,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_REVISION,
        "reranker_model": RERANKER_MODEL,
        "reranker_revision": RERANKER_REVISION,
        "retrieval": {
            "bm25_top_k": 50,
            "dense_top_k": 50,
            "reranker_top_k": 10,
            "reranker_strategy": "bi_encoder_cosine",
            "reranker_query_prompt": "model_prompt_name:query",
        },
        "device": args.device,
        "dtype": "bfloat16",
        "attention_implementation": "flash_attention_2",
        "query_prompt_sha256": _sha256_text(QUERY_PROMPT),
        "input_qa_sha256": frozen_manifest["qa_sha256"],
        "input_index_manifest_sha256": _sha256_file(index_manifest_path),
        "systems": outputs,
        "gpu": gpu,
        "environment": environment,
    }
    _json_dump(args.output_dir / "retrieval_manifest.json", manifest)
    return manifest


def _interval_overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _union_length(intervals: Sequence[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def query_metrics(
    evidence_doc_id: str,
    evidence: Sequence[dict[str, Any]],
    ranked_chunks: Sequence[dict[str, Any]],
    corpus_chunks: Sequence[dict[str, Any]],
) -> dict[str, float]:
    evidence_spans = [
        (int(item["source_start"]), int(item["source_end"])) for item in evidence
    ]
    evidence_total = sum(end - start for start, end in evidence_spans)
    if evidence_total <= 0:
        raise PhaseGError("Evidence spans must be non-empty")

    def relevance(chunk: dict[str, Any]) -> float:
        if chunk.get("doc_id") != evidence_doc_id:
            return 0.0
        span = (int(chunk["source_start"]), int(chunk["source_end"]))
        return max(
            (_interval_overlap(span, evidence_span) / (evidence_span[1] - evidence_span[0]))
            for evidence_span in evidence_spans
        )

    ranked_relevance = [relevance(chunk) for chunk in ranked_chunks]
    corpus_relevance = sorted(
        (relevance(chunk) for chunk in corpus_chunks), reverse=True
    )
    result: dict[str, float] = {}
    for k in TOP_K_VALUES:
        selected = ranked_chunks[:k]
        coverage = 0
        for evidence_start, evidence_end in evidence_spans:
            overlaps: list[tuple[int, int]] = []
            for chunk in selected:
                if chunk.get("doc_id") != evidence_doc_id:
                    continue
                start = max(evidence_start, int(chunk["source_start"]))
                end = min(evidence_end, int(chunk["source_end"]))
                if end > start:
                    overlaps.append((start, end))
            coverage += _union_length(overlaps)
        result[f"hit@{k}"] = float(any(value > 0 for value in ranked_relevance[:k]))
        result[f"recall@{k}"] = coverage / evidence_total
    result["mrr@10"] = next(
        (1.0 / rank for rank, value in enumerate(ranked_relevance[:10], 1) if value > 0),
        0.0,
    )
    dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(ranked_relevance[:10], 1))
    idcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(corpus_relevance[:10], 1))
    result["ndcg@10"] = dcg / idcg if idcg else 0.0
    return result


def _mean_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise PhaseGError("Cannot aggregate an empty metric collection")
    keys = [key for key in rows[0] if key not in {"qa_id", "doc_id", "system_id", "fold"}]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def evaluate_retrieval(args: argparse.Namespace) -> dict[str, Any]:
    retrieval_manifest_path = args.retrieval_dir / "retrieval_manifest.json"
    if not retrieval_manifest_path.is_file():
        raise PhaseGError(f"Missing retrieval manifest: {retrieval_manifest_path}")
    retrieval_manifest = json.loads(
        retrieval_manifest_path.read_text(encoding="utf-8")
    )
    if (
        retrieval_manifest.get("embedding_revision") != EMBEDDING_REVISION
        or retrieval_manifest.get("reranker_revision") != RERANKER_REVISION
        or retrieval_manifest.get("source_commit") != SOURCE_COMMIT
    ):
        raise PhaseGError("Retrieval manifest does not match pinned Phase G configuration")
    qas = {row["qa_id"]: row for row in _read_jsonl(args.qa_dir / "qa_frozen.jsonl")}
    folds = {row["doc_id"]: int(row["fold"]) for row in _read_jsonl(args.prepared_dir / "folds.jsonl")}
    systems = _selected_system_ids(args.systems)
    per_query: list[dict[str, Any]] = []
    for system_id in systems:
        corpus = _read_jsonl(args.prepared_dir / "systems" / f"{system_id}.jsonl")
        corpus_by_doc: dict[str, list[dict[str, Any]]] = {}
        for chunk in corpus:
            corpus_by_doc.setdefault(chunk["doc_id"], []).append(chunk)
        retrieval_rows = _read_jsonl(args.retrieval_dir / f"{system_id}.jsonl")
        if {row["qa_id"] for row in retrieval_rows} != set(qas):
            raise PhaseGError(f"Retrieval QA coverage mismatch for {system_id}")
        for row in retrieval_rows:
            qa = qas[row["qa_id"]]
            ranked = [
                result["meta"] | {"score": result["score"]}
                for result in row["results"]
            ]
            metrics = query_metrics(
                qa["doc_id"], qa["evidence"], ranked, corpus_by_doc[qa["doc_id"]]
            )
            per_query.append(
                {
                    "qa_id": qa["qa_id"],
                    "doc_id": qa["doc_id"],
                    "fold": folds[qa["doc_id"]],
                    "system_id": system_id,
                    **metrics,
                }
            )
    _write_jsonl(args.output_dir / "per_query_metrics.jsonl", per_query)

    summary = {
        system_id: _mean_metrics(
            [row for row in per_query if row["system_id"] == system_id]
        )
        for system_id in systems
    }
    fixed_systems = [system for system in systems if system != "adaptive"]
    if not fixed_systems:
        raise PhaseGError("Evaluation requires at least one fixed-method system")
    best_fixed_rows: list[dict[str, Any]] = []
    observed_folds = sorted({row["fold"] for row in per_query})
    for fold in observed_folds:
        training = [row for row in per_query if row["fold"] != fold and row["system_id"] in fixed_systems]
        if not training:
            continue
        by_system = {
            system: np.mean([row["ndcg@10"] for row in training if row["system_id"] == system])
            for system in fixed_systems
        }
        chosen = max(by_system, key=by_system.get)
        best_fixed_rows.extend(
            row
            for row in per_query
            if row["fold"] == fold and row["system_id"] == chosen
        )
    if len(best_fixed_rows) == len(qas):
        summary["best_fixed_cv"] = _mean_metrics(best_fixed_rows)

    oracle_rows: list[dict[str, Any]] = []
    for qa_id in qas:
        options = [
            row
            for row in per_query
            if row["qa_id"] == qa_id and row["system_id"] in fixed_systems
        ]
        oracle_rows.append(max(options, key=lambda row: row["ndcg@10"]))
    summary["oracle"] = _mean_metrics(oracle_rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "harness_sha256": _harness_sha256(),
        "status": "complete",
        "created_at": _utc_now(),
        "source_commit": SOURCE_COMMIT,
        "mode": "full" if len({row["doc_id"] for row in qas.values()}) == 33 else "smoke",
        "qa_count": len(qas),
        "systems": systems,
        "paper_protocol_systems": list(PAPER_PROTOCOL_SYSTEMS),
        "relevance": "source-character interval overlap",
        "embedding_model": retrieval_manifest["embedding_model"],
        "embedding_revision": retrieval_manifest["embedding_revision"],
        "reranker_model": retrieval_manifest["reranker_model"],
        "reranker_revision": retrieval_manifest["reranker_revision"],
        "device": retrieval_manifest["device"],
        "dtype": retrieval_manifest["dtype"],
        "attention_implementation": retrieval_manifest["attention_implementation"],
        "query_prompt_sha256": retrieval_manifest["query_prompt_sha256"],
        "input_qa_sha256": _sha256_file(args.qa_dir / "qa_frozen.jsonl"),
        "input_folds_sha256": _sha256_file(args.prepared_dir / "folds.jsonl"),
        "input_retrieval_manifest_sha256": _sha256_file(retrieval_manifest_path),
        "input_retrieval_sha256": {
            system_id: _sha256_file(args.retrieval_dir / f"{system_id}.jsonl")
            for system_id in systems
        },
        "summary": summary,
    }
    _json_dump(args.output_dir / "retrieval_evaluation.json", manifest)
    table_rows = []
    ordered = [*PAPER_PROTOCOL_SYSTEMS, *[s for s in fixed_systems if s not in PAPER_PROTOCOL_SYSTEMS], "best_fixed_cv", "oracle"]
    for system_id in dict.fromkeys(ordered):
        if system_id in summary:
            table_rows.append({"system_id": system_id, **summary[system_id]})
    pd.DataFrame(table_rows).to_csv(args.output_dir / "retrieval_summary.csv", index=False)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--final-dir", type=Path, required=True)
    prepare_parser.add_argument("--data-dir", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)

    qa = subparsers.add_parser("generate-qa")
    qa.add_argument("--prepared-dir", type=Path, required=True)
    qa.add_argument("--data-dir", type=Path, required=True)
    qa.add_argument("--output-dir", type=Path, required=True)
    qa.add_argument("--smoke-docs", type=int, default=0)
    qa.add_argument("--budget-usd", type=float, default=5.0)
    qa.add_argument("--request-delay-seconds", type=float, default=0.0)
    qa.add_argument("--allow-paid-run", action="store_true")
    qa.add_argument("--smoke-manifest", type=Path)
    qa.add_argument("--smoke-evaluation-manifest", type=Path)

    freeze = subparsers.add_parser("freeze-qa")
    freeze.add_argument("--prepared-dir", type=Path, required=True)
    freeze.add_argument("--data-dir", type=Path, required=True)
    freeze.add_argument("--qa-dir", type=Path, required=True)
    freeze.add_argument("--expected-documents", type=int, default=33)
    freeze.add_argument("--review-decisions", type=Path)

    index = subparsers.add_parser("index")
    index.add_argument("--prepared-dir", type=Path, required=True)
    index.add_argument("--qa-dir", type=Path, required=True)
    index.add_argument("--output-dir", type=Path, required=True)
    index.add_argument("--systems", default="all")
    index.add_argument("--device", default="cuda:0")
    index.add_argument("--batch-size", type=int, default=1)
    index.add_argument("--model-cache-dir", type=Path)
    index.add_argument("--environment-file", type=Path, required=True)
    index.add_argument("--smoke-evaluation-manifest", type=Path)
    index.add_argument("--force", action="store_true")

    retrieve_parser = subparsers.add_parser("retrieve")
    retrieve_parser.add_argument("--prepared-dir", type=Path, required=True)
    retrieve_parser.add_argument("--qa-dir", type=Path, required=True)
    retrieve_parser.add_argument("--index-dir", type=Path, required=True)
    retrieve_parser.add_argument("--output-dir", type=Path, required=True)
    retrieve_parser.add_argument("--systems", default="all")
    retrieve_parser.add_argument("--device", default="cuda:0")
    retrieve_parser.add_argument("--batch-size", type=int, default=1)
    retrieve_parser.add_argument("--model-cache-dir", type=Path)
    retrieve_parser.add_argument("--environment-file", type=Path, required=True)
    retrieve_parser.add_argument("--force", action="store_true")

    evaluate = subparsers.add_parser("evaluate-retrieval")
    evaluate.add_argument("--prepared-dir", type=Path, required=True)
    evaluate.add_argument("--qa-dir", type=Path, required=True)
    evaluate.add_argument("--retrieval-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--systems", default="all")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args)
        elif args.command == "generate-qa":
            result = generate_qa(args)
        elif args.command == "freeze-qa":
            result = freeze_qa(args)
        elif args.command == "index":
            result = index_systems(args)
        elif args.command == "retrieve":
            result = retrieve(args)
        elif args.command == "evaluate-retrieval":
            result = evaluate_retrieval(args)
        else:  # pragma: no cover
            raise PhaseGError(f"Unsupported command: {args.command}")
    except PhaseGError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
