"""Low-cost correctness checks for the Phase G retrieval benchmark."""

from __future__ import annotations

import importlib.util
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
