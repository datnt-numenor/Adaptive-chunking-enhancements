"""Low-cost correctness checks for the Phase G retrieval benchmark."""

from __future__ import annotations

import importlib.util
import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd
import pytest


RUNNER_PATH = Path(__file__).parents[1] / "research" / "week3" / "phase_g_runner.py"
SPEC = importlib.util.spec_from_file_location("phase_g_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_protocol_systems_use_four_processed_candidates_and_two_raw_baselines():
    assert runner.ADAPTIVE_CANDIDATES == (
        "our_recurs_1100",
        "our_recurs_600",
        "page",
        "llm_regex",
    )
    assert runner.PAPER_PROTOCOL_SYSTEMS == (
        "adaptive",
        "raw__langch_recurs_default",
        "raw__page",
    )
    assert runner.FIXED_SYSTEMS["raw__langch_recurs_default"] == (
        "raw",
        "langch_recurs_default",
    )
    assert runner.FIXED_SYSTEMS["raw__page"] == ("raw", "page")
    assert len(runner.ALL_SYSTEMS) == 10


def test_output_best_chunks_respects_candidate_methods(tmp_path: Path):
    from adaptive_chunking.paper.analysis import output_best_chunks

    chunks = pd.DataFrame(
        [
            {
                "doc_name": "doc",
                "method": method,
                "chunk_index": 0,
                "chunk_text": method,
                "titles_context": "",
                "chunk_pages": [1],
            }
            for method in ("page", "llm_regex", "semantic")
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "doc_name": "doc",
                "chunking_method": method,
                "metric_name": "score",
                "score": score,
            }
            for method, score in (("page", 0.5), ("llm_regex", 0.6), ("semantic", 1.0))
        ]
    )
    chunks_path = tmp_path / "chunks.parquet"
    metrics_path = tmp_path / "metrics.parquet"
    chunks.to_parquet(chunks_path)
    metrics.to_parquet(metrics_path)

    output_best_chunks(
        chunks_path,
        metrics_path,
        {"score": 1.0},
        tmp_path / "out",
        candidate_methods=["page", "llm_regex"],
    )

    selected = json.loads((tmp_path / "out" / "doc.json").read_text(encoding="utf-8"))
    assert selected["method"] == "llm_regex"

    output_best_chunks(
        chunks_path,
        metrics_path,
        {"score": 1.0},
        tmp_path / "legacy-out",
    )
    legacy = json.loads(
        (tmp_path / "legacy-out" / "doc.json").read_text(encoding="utf-8")
    )
    assert legacy["method"] == "semantic"


def test_query_metrics_ignore_matching_offsets_from_another_document():
    evidence = [{"source_start": 10, "source_end": 20}]
    ranked = [
        {"doc_id": "doc-b", "source_start": 10, "source_end": 20},
        {"doc_id": "doc-a", "source_start": 15, "source_end": 20},
    ]
    corpus = [{"doc_id": "doc-a", "source_start": 10, "source_end": 20}]

    metrics = runner.query_metrics("doc-a", evidence, ranked, corpus)

    assert metrics["hit@1"] == 0.0
    assert metrics["hit@3"] == 1.0
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 0.5
    assert metrics["mrr@10"] == 0.5
    assert metrics["ndcg@10"] == pytest.approx(0.5 / math.log2(3))


def test_query_metrics_use_union_coverage_without_double_counting():
    evidence = [{"source_start": 0, "source_end": 10}]
    ranked = [
        {"doc_id": "doc-a", "source_start": 0, "source_end": 7},
        {"doc_id": "doc-a", "source_start": 5, "source_end": 10},
    ]
    metrics = runner.query_metrics("doc-a", evidence, ranked, ranked)
    assert metrics["recall@1"] == 0.7
    assert metrics["recall@3"] == 1.0


def test_evidence_mapping_requires_exact_unique_quote():
    document = {
        "page_spans": [
            {"page": 1, "source_start": 100, "source_end": 111, "text": "hello world"}
        ]
    }
    mapped = runner._map_evidence(document, [{"page": 1, "quote": "world"}])
    assert mapped[0]["source_start"] == 106
    assert mapped[0]["source_end"] == 111
    with pytest.raises(runner.PhaseGError, match="exactly once"):
        runner._map_evidence(document, [{"page": 1, "quote": "missing"}])


def test_document_grouped_folds_cover_each_document_once():
    documents = {
        f"doc-{index}": {
            "doc_id": f"id-{index}",
            "doc_name": f"doc-{index}",
            "domain": f"domain-{index % 3}",
        }
        for index in range(33)
    }
    folds = runner._build_folds(documents)
    assert len(folds) == 33
    assert len({row["doc_id"] for row in folds}) == 33
    assert {row["fold"] for row in folds} == {0, 1, 2, 3, 4}


def test_qa_hash_changes_when_reviewed_content_changes():
    qa = {
        "doc_id": "doc-a",
        "question": "Question?",
        "reference_answer": "Answer.",
        "evidence": [
            {"page": 1, "quote": "Answer.", "source_start": 5, "source_end": 12}
        ],
    }
    first = runner._qa_content_hash(qa)
    changed = dict(qa, question="Different question?")
    assert len(first) == 64
    assert runner._qa_content_hash(changed) != first


def test_generate_qa_parser_accepts_request_delay():
    args = runner.build_parser().parse_args(
        [
            "generate-qa",
            "--prepared-dir",
            "prepared",
            "--data-dir",
            "data",
            "--output-dir",
            "output",
            "--request-delay-seconds",
            "30",
        ]
    )

    assert args.request_delay_seconds == 30.0


def _bi_neutral_fixture(tmp_path: Path) -> argparse.Namespace:
    final_dir = tmp_path / "final"
    prepared_dir = tmp_path / "base"
    output_dir = tmp_path / "neutral"
    metrics_path = final_dir / "results" / "chunking_metrics.parquet"
    metrics_path.parent.mkdir(parents=True)
    names = [f"doc-{index:02d}" for index in range(33)]
    metrics = []
    systems = {}
    for method in runner.ADAPTIVE_CANDIDATES:
        system_id = f"processed__{method}"
        path = prepared_dir / "systems" / f"{system_id}.jsonl"
        rows = []
        for index, name in enumerate(names):
            rows.append(
                {
                    "chunk_id": f"id-{index}::{method}::000000",
                    "doc_id": f"id-{index}",
                    "doc_name": name,
                    "method": method,
                    "chunk_index": 0,
                    "chunk_text": name,
                    "source_start": 0,
                    "source_end": len(name),
                }
            )
            for metric in (*runner.BI_NEUTRAL_METRICS, "block_integrity"):
                metrics.append(
                    {
                        "doc_name": name,
                        "chunking_method": method,
                        "metric_name": metric,
                        "score": 0.9 if method == "page" and metric == "block_integrity" else (0.8 if method == "our_recurs_1100" else 0.2),
                    }
                )
        runner._write_jsonl(path, rows)
        systems[system_id] = {
            "path": path.relative_to(prepared_dir).as_posix(),
            "sha256": runner._sha256_file(path),
            "stage": "small_merged",
            "method": method,
        }
    # A non-candidate may score higher, but cannot enter the selection.
    metrics.extend(
        {
            "doc_name": name,
            "chunking_method": "semantic",
            "metric_name": metric,
            "score": 1.0,
        }
        for name in names
        for metric in runner.BI_NEUTRAL_METRICS
    )
    pd.DataFrame(metrics).to_parquet(metrics_path)
    runner._write_jsonl(
        prepared_dir / "documents.jsonl",
        [{"doc_name": name, "doc_id": f"id-{index}"} for index, name in enumerate(names)],
    )
    runner._write_jsonl(
        prepared_dir / "folds.jsonl",
        [{"doc_name": name, "doc_id": f"id-{index}", "fold": index % 5} for index, name in enumerate(names)],
    )
    runner._json_dump(
        prepared_dir / "prepare_manifest.json",
        {
            "status": "complete",
            "source_commit": runner.SOURCE_COMMIT,
            "documents": 33,
            "fold_seed": runner.FOLD_SEED,
            "adaptive_candidates": list(runner.ADAPTIVE_CANDIDATES),
            "systems": systems,
        },
    )
    locked_path = tmp_path / "page_neutral_selections.json"
    runner._json_dump(
        locked_path,
        {
            "scoring_metrics": list(runner.BI_NEUTRAL_METRICS),
            "weights": runner.BI_NEUTRAL_WEIGHTS,
            "selections": {name: "our_recurs_1100" for name in names},
        },
    )
    return argparse.Namespace(
        final_dir=final_dir,
        prepared_dir=prepared_dir,
        phase_h_selections=locked_path,
        output_dir=output_dir,
    )


def test_bi_neutral_prepare_matches_locked_selections_and_hashes(tmp_path: Path):
    args = _bi_neutral_fixture(tmp_path)
    manifest = runner.prepare_bi_neutral(args)
    assert runner._selected_system_ids("all") == list(runner.ALL_SYSTEMS)
    assert len(runner.ALL_SYSTEMS) == 10
    assert runner._selected_system_ids("adaptive_bi_neutral") == [runner.BI_NEUTRAL_SYSTEM]
    assert manifest["systems"][runner.BI_NEUTRAL_SYSTEM]["documents"] == 33
    assert manifest["systems"][runner.BI_NEUTRAL_SYSTEM]["chunks"] == 33
    assert manifest["selection_definition"]["block_integrity_weight"] == 0
    system_path = args.output_dir / manifest["systems"][runner.BI_NEUTRAL_SYSTEM]["path"]
    assert runner._sha256_file(system_path) == manifest["systems"][runner.BI_NEUTRAL_SYSTEM]["sha256"]
    assert runner._sha256_file(args.output_dir / "folds.jsonl") == manifest["folds_sha256"]
    assert runner._sha256_file(args.output_dir / "adaptive_bi_neutral_selections.json") == manifest["selections_sha256"]
    rows = runner._read_jsonl(system_path)
    assert len({row["chunk_id"] for row in rows}) == len(rows)
    assert {row["method"] for row in rows} == {"our_recurs_1100"}


def test_bi_neutral_prepare_rejects_selection_drift(tmp_path: Path):
    args = _bi_neutral_fixture(tmp_path)
    locked = json.loads(args.phase_h_selections.read_text(encoding="utf-8"))
    locked["selections"]["doc-00"] = "page"
    runner._json_dump(args.phase_h_selections, locked)
    with pytest.raises(runner.PhaseGError, match="differ from Phase H"):
        runner.prepare_bi_neutral(args)


def test_bi_neutral_prepare_protects_frozen_input_directories(tmp_path: Path):
    args = _bi_neutral_fixture(tmp_path)
    args.output_dir = args.prepared_dir / "phase-j"
    with pytest.raises(runner.PhaseGError, match="outside frozen input directories"):
        runner.prepare_bi_neutral(args)


def test_bi_neutral_prepare_rejects_source_hash_and_duplicate_identity(tmp_path: Path):
    args = _bi_neutral_fixture(tmp_path)
    source = args.prepared_dir / "systems" / "processed__our_recurs_1100.jsonl"
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(runner.PhaseGError, match="checksum mismatch"):
        runner.prepare_bi_neutral(args)

    args = _bi_neutral_fixture(tmp_path / "duplicate")
    source = args.prepared_dir / "systems" / "processed__our_recurs_1100.jsonl"
    rows = runner._read_jsonl(source)
    rows[1]["chunk_id"] = rows[0]["chunk_id"]
    runner._write_jsonl(source, rows)
    base_manifest_path = args.prepared_dir / "prepare_manifest.json"
    base = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base["systems"]["processed__our_recurs_1100"]["sha256"] = runner._sha256_file(source)
    runner._json_dump(base_manifest_path, base)
    with pytest.raises(runner.PhaseGError, match="duplicate chunk identities"):
        runner.prepare_bi_neutral(args)


def test_bi_neutral_prepare_requires_every_document(tmp_path: Path):
    args = _bi_neutral_fixture(tmp_path)
    source = args.prepared_dir / "systems" / "processed__our_recurs_1100.jsonl"
    rows = runner._read_jsonl(source)
    runner._write_jsonl(source, rows[:-1])
    base_manifest_path = args.prepared_dir / "prepare_manifest.json"
    base = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base["systems"]["processed__our_recurs_1100"]["sha256"] = runner._sha256_file(source)
    runner._json_dump(base_manifest_path, base)
    with pytest.raises(runner.PhaseGError, match="incomplete document coverage"):
        runner.prepare_bi_neutral(args)


def test_bi_neutral_evaluation_accepts_single_system(tmp_path: Path):
    prepared = tmp_path / "prepared"
    qa_dir = tmp_path / "qa"
    retrieval = tmp_path / "retrieval"
    output = tmp_path / "evaluation"
    runner._write_jsonl(
        prepared / "folds.jsonl",
        [{"doc_id": "doc-a", "fold": 0}, {"doc_id": "doc-b", "fold": 1}],
    )
    runner._write_jsonl(
        prepared / "systems" / "adaptive_bi_neutral.jsonl",
        [
            {"doc_id": doc, "source_start": 0, "source_end": 10}
            for doc in ("doc-a", "doc-b")
        ],
    )
    runner._write_jsonl(
        qa_dir / "qa_frozen.jsonl",
        [
            {"qa_id": f"{doc}::q1", "doc_id": doc, "evidence": [{"source_start": 0, "source_end": 10}]}
            for doc in ("doc-a", "doc-b")
        ],
    )
    runner._write_jsonl(
        retrieval / "adaptive_bi_neutral.jsonl",
        [
            {"qa_id": f"{doc}::q1", "results": [{"score": 1.0, "meta": {"doc_id": doc, "source_start": 0, "source_end": 10}}]}
            for doc in ("doc-a", "doc-b")
        ],
    )
    runner._json_dump(
        retrieval / "retrieval_manifest.json",
        {
            "source_commit": runner.SOURCE_COMMIT,
            "embedding_model": runner.EMBEDDING_MODEL,
            "embedding_revision": runner.EMBEDDING_REVISION,
            "reranker_model": runner.RERANKER_MODEL,
            "reranker_revision": runner.RERANKER_REVISION,
            "device": "cuda:0",
            "dtype": "bfloat16",
            "attention_implementation": "flash_attention_2",
            "query_prompt_sha256": runner._sha256_text(runner.QUERY_PROMPT),
        },
    )
    result = runner.evaluate_retrieval(
        argparse.Namespace(prepared_dir=prepared, qa_dir=qa_dir, retrieval_dir=retrieval, output_dir=output, systems="adaptive_bi_neutral")
    )
    assert result["qa_count"] == 2
    assert set(result["summary"]) == {"adaptive_bi_neutral"}
    assert result["summary"]["adaptive_bi_neutral"]["hit@1"] == 1.0
    assert len(runner._read_jsonl(output / "per_query_metrics.jsonl")) == 2
