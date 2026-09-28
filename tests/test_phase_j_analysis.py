"""Synthetic, local-only checks for the exact Phase J analysis contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "research" / "week3" / "phase_j_analysis.py"
SPEC = importlib.util.spec_from_file_location("phase_j_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> argparse.Namespace:
    artifact_root = tmp_path / "artifacts"
    g = artifact_root / "evaluation-full"
    j = artifact_root / "phase-j-bi-neutral" / "phase-j-results"
    phase_g_prepared = artifact_root / "prepared"
    phase_j_prepared = artifact_root / "phase-j-bi-neutral" / "prepared"
    folds_path = phase_g_prepared / "folds.jsonl"
    folds = [
        {"doc_id": f"d{i:02d}", "doc_name": f"Document {i}", "domain": "synthetic", "fold": i % 5}
        for i in range(33)
    ]
    _write_jsonl(folds_path, folds)
    base_rows = []
    neutral_rows = []
    ranked_by_key = {}
    for fold_row in folds:
        doc = fold_row["doc_id"]
        fold = fold_row["fold"]
        for q in range(3):
            qa_id = f"{doc}::q{q + 1}"
            for system in (analysis.ADAPTIVE, *analysis.FIXED_SYSTEMS, analysis.NEUTRAL):
                if system == "raw__page":
                    relevant_rank = 1 if fold == 0 else 2
                elif system == "raw__langch_recurs_default":
                    relevant_rank = 2
                elif system == analysis.NEUTRAL:
                    relevant_rank = 1 if doc == "d00" and q == 2 else 3
                elif system == analysis.ADAPTIVE:
                    relevant_rank = None if doc == "d00" else 3
                else:
                    relevant_rank = 4
                ranked = [
                    {"doc_id": doc, "source_start": 200, "source_end": 300}
                    for _ in range(10)
                ]
                if relevant_rank is not None:
                    ranked[relevant_rank - 1] = {
                        "doc_id": doc, "source_start": 0, "source_end": 100
                    }
                ranked_by_key[(system, qa_id)] = ranked
                metrics = analysis._query_metrics(
                    doc,
                    [{"source_start": 0, "source_end": 100}],
                    ranked,
                    [{"doc_id": doc, "source_start": 0, "source_end": 100}],
                )
                row = {"qa_id": qa_id, "doc_id": doc, "fold": fold, "system_id": system,
                       **metrics}
                (neutral_rows if system == analysis.NEUTRAL else base_rows).append(row)
    _write_jsonl(g / "per_query_metrics.jsonl", base_rows)
    _write_jsonl(j / "evaluation" / "per_query_metrics.jsonl", neutral_rows)
    _write_jsonl(phase_j_prepared / "folds.jsonl", folds)
    identity = {
        "source_commit": analysis.SOURCE_COMMIT, "relevance": "source-character interval overlap",
        "embedding_model": "Qwen/Qwen3-Embedding-4B",
        "embedding_revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
        "reranker_model": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "reranker_revision": "ac6544c8a46e00af67e330e85a9028c66b8cfd9a",
        "device": "cuda:0", "dtype": "bfloat16", "attention_implementation": "flash_attention_2",
        "query_prompt_sha256": "prompt-hash", "input_qa_sha256": "qa-hash",
        "input_folds_sha256": analysis._sha256(folds_path),
    }
    harness_hash = "harness-hash"

    qa_rows = [
        {"qa_id": f"d{i:02d}::q{q + 1}", "doc_id": f"d{i:02d}",
         "evidence": [{"source_start": 0, "source_end": 100}]}
        for i in range(33) for q in range(3)
    ]
    qa_path = artifact_root / "qa-full" / "qa_frozen.jsonl"
    _write_jsonl(qa_path, qa_rows)
    qa_hash = analysis._sha256(qa_path)
    identity["input_qa_sha256"] = qa_hash
    (artifact_root / "qa-full" / "qa_frozen_manifest.json").write_text(json.dumps({
        "status": "complete", "mode": "full", "qa_count": 99, "documents": 33,
        "qa_sha256": qa_hash,
    }), encoding="utf-8")

    def write_pipeline(prepared_dir: Path, index_dir: Path, retrieval_dir: Path,
                       systems: tuple[str, ...], source_rows: list[dict]) -> tuple[Path, dict[str, str]]:
        prepared_systems = {}
        index_systems = {}
        retrieval_systems = {}
        retrieval_hashes = {}
        for system in systems:
            prepared_file = prepared_dir / "systems" / f"{system}.jsonl"
            _write_jsonl(prepared_file, [
                {"system_id": system, "doc_id": row["doc_id"],
                 "source_start": 0, "source_end": 100}
                for row in folds
            ])
            prepared_systems[system] = {
                "path": f"systems/{system}.jsonl", "sha256": analysis._sha256(prepared_file)
            }
            store = index_dir / system / "document_store.json"
            store.parent.mkdir(parents=True, exist_ok=True)
            store.write_text("{}", encoding="utf-8")
            index_systems[system] = {"sha256": analysis._sha256(store)}
            rows_path = retrieval_dir / f"{system}.jsonl"
            system_rows = [
                {"qa_id": row["qa_id"], "system_id": system,
                 "results": [
                     {"score": 1.0 / rank, "meta": meta}
                     for rank, meta in enumerate(ranked_by_key[(system, row["qa_id"])], 1)
                 ]}
                for row in source_rows if row["system_id"] == system
            ]
            _write_jsonl(rows_path, system_rows)
            retrieval_hashes[system] = analysis._sha256(rows_path)
            retrieval_systems[system] = {"sha256": retrieval_hashes[system], "queries": 99}
        prepare_path = prepared_dir / "prepare_manifest.json"
        prepare_path.write_text(json.dumps({
            "status": "complete", "source_commit": analysis.SOURCE_COMMIT,
            "documents": 33, "folds": 5, "fold_seed": 2026, "systems": prepared_systems,
        }), encoding="utf-8")
        index_path = index_dir / "index_manifest.json"
        index_path.write_text(json.dumps({
            "status": "complete", "mode": "full", "qa_count": 99, "source_documents": 33,
            "harness_sha256": harness_hash, "input_prepare_manifest_sha256": analysis._sha256(prepare_path),
            "systems": index_systems,
            **{key: value for key, value in identity.items() if key not in (
                "relevance", "reranker_model", "reranker_revision", "input_folds_sha256")},
        }), encoding="utf-8")
        retrieval_path = retrieval_dir / "retrieval_manifest.json"
        retrieval_path.write_text(json.dumps({
            "status": "complete", "harness_sha256": harness_hash,
            "input_index_manifest_sha256": analysis._sha256(index_path),
            "systems": retrieval_systems, "retrieval": analysis.RETRIEVAL_SETTINGS,
            **{key: value for key, value in identity.items() if key not in (
                "relevance", "input_folds_sha256")},
        }), encoding="utf-8")
        return retrieval_path, retrieval_hashes

    phase_g_systems = (analysis.ADAPTIVE, *analysis.FIXED_SYSTEMS)
    g_retrieval_path, g_retrieval_hashes = write_pipeline(
        phase_g_prepared, artifact_root / "index-full", artifact_root / "retrieval-full",
        phase_g_systems, base_rows,
    )
    j_retrieval_path, j_retrieval_hashes = write_pipeline(
        phase_j_prepared, j / "index", j / "retrieval", (analysis.NEUTRAL,), neutral_rows,
    )
    base = pd.DataFrame(base_rows)
    neutral = pd.DataFrame(neutral_rows)
    virtual, _ = analysis._derive_virtual_comparators(base)
    base_all = pd.concat([base, virtual], ignore_index=True)
    for path, frame, systems in (
        (g / "retrieval_evaluation.json", base_all, list(phase_g_systems)),
        (j / "evaluation" / "retrieval_evaluation.json", neutral, [analysis.NEUTRAL]),
    ):
        manifest = {
            "schema_version": 1, "status": "complete", "mode": "full", "qa_count": 99,
            "systems": systems, "harness_sha256": harness_hash, **identity,
            "summary": {
                system: {metric: float(rows[metric].mean()) for metric in analysis.METRICS}
                for system, rows in frame.groupby("system_id")
            },
        }
        if systems == [analysis.NEUTRAL]:
            manifest["input_retrieval_manifest_sha256"] = analysis._sha256(j_retrieval_path)
            manifest["input_retrieval_sha256"] = j_retrieval_hashes
        else:
            manifest["input_retrieval_manifest_sha256"] = analysis._sha256(g_retrieval_path)
            manifest["input_retrieval_sha256"] = g_retrieval_hashes
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return argparse.Namespace(
        phase_g_evaluation_dir=g,
        phase_j_result_dir=j,
        folds_path=folds_path,
        output_dir=tmp_path / "analysis",
    )


def test_exact_analysis_aggregates_qa_to_documents_and_reproduces_cv(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    manifest = analysis.run(args)
    documents = pd.read_csv(args.output_dir / "document_metrics.csv")
    comparisons = pd.read_csv(args.output_dir / "paired_comparisons.csv")
    summary = pd.read_csv(args.output_dir / "system_summary.csv")
    assert len(documents) == 33 * 6
    assert len(summary) == 10 * 6
    assert len(comparisons) == 10 * 5
    assert set(documents.groupby("system_id").size()) == {33}
    assert manifest["statistical_unit"] == "document"
    assert manifest["bootstrap_samples"] == 10_000
    assert manifest["analysis_script_sha256"] == analysis._sha256(MODULE_PATH)
    assert isinstance(manifest["analysis_worktree_dirty"], bool)
    assert manifest["best_fixed_by_held_out_fold"] == {
        "0": "raw__langch_recurs_default", **{str(i): "raw__page" for i in range(1, 5)}
    }
    d00 = documents[(documents["system_id"] == analysis.NEUTRAL) & (documents["doc_id"] == "d00")]
    assert d00.iloc[0]["ndcg@10"] == pytest.approx(2 / 3)
    delta = comparisons[(comparisons["metric"] == "ndcg@10") &
                        (comparisons["comparator"] == analysis.ADAPTIVE)].iloc[0]
    assert delta["mean_difference"] == pytest.approx((2 / 3) / 33)
    assert delta["documents"] == 33
    assert 0 <= delta["wilcoxon_p"] <= delta["holm_p"] <= 1
    assert "## Paper says" in (args.output_dir / "SUMMARY.md").read_text(encoding="utf-8")
    for name, checksum in manifest["outputs"].items():
        assert analysis._sha256(args.output_dir / name) == checksum


def test_rejects_different_qa_or_fold_identity(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    path = args.phase_j_result_dir / "evaluation" / "per_query_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["fold"] = 4
    _write_jsonl(path, rows)
    with pytest.raises(analysis.PhaseJAnalysisError, match="QA/document/fold identity"):
        analysis.run(args)
    assert not args.output_dir.exists()


def test_rejects_mismatched_model_revision_before_writing(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    path = args.phase_j_result_dir / "evaluation" / "retrieval_evaluation.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["reranker_revision"] = "changed"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(analysis.PhaseJAnalysisError, match="reranker_revision"):
        analysis.run(args)
    assert not args.output_dir.exists()


def test_holm_five_comparison_family_and_zero_difference() -> None:
    assert analysis._holm([0.01, 0.04, 0.03, 1.0, 0.001]) == pytest.approx(
        [0.04, 0.09, 0.09, 1.0, 0.005]
    )
    assert analysis._rank_biserial(pd.Series([0.0, 0.0]).to_numpy()) == 0.0


def test_rejects_tampered_retrieval_chain(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    path = args.phase_j_result_dir / "retrieval" / f"{analysis.NEUTRAL}.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(analysis.PhaseJAnalysisError, match="checksum"):
        analysis.run(args)


def test_rejects_per_query_metric_swap_that_preserves_aggregate_mean(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    path = args.phase_j_result_dir / "evaluation" / "per_query_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for metric in analysis.METRICS:
        rows[2][metric], rows[9][metric] = rows[9][metric], rows[2][metric]
    _write_jsonl(path, rows)
    with pytest.raises(analysis.PhaseJAnalysisError, match="per-query metric"):
        analysis.run(args)
