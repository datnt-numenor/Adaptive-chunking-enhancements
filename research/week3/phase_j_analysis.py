"""Local, document-paired analysis of the exact Phase J BI-neutral retrieval run.

No model, GPU, or provider client is loaded. The Phase G fixed-system rows are
used only to reconstruct its locked best-fixed CV and nDCG oracle comparators.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, wilcoxon


SCHEMA_VERSION = 1
SOURCE_COMMIT = "ea87ce8e1a97888f3f179e7f1359ff7f43fb179d"
SEED = 2026
BOOTSTRAP_SAMPLES = 10_000
DOCUMENTS = 33
QA_COUNT = 99
FOLDS = 5
NEUTRAL = "adaptive_bi_neutral"
ADAPTIVE = "adaptive"
FIXED_SYSTEMS = (
    "processed__our_recurs_1100",
    "processed__our_recurs_600",
    "processed__page",
    "processed__llm_regex",
    "raw__langch_recurs_1100",
    "raw__langch_recurs_default",
    "raw__page",
    "raw__semantic",
    "raw__sentence",
)
COMPARATORS = (
    ADAPTIVE,
    "raw__page",
    "raw__langch_recurs_default",
    "best_fixed_cv",
    "oracle",
)
METRICS = tuple(
    f"{kind}@{k}" for k in (1, 3, 5, 10) for kind in ("hit", "recall")
) + ("mrr@10", "ndcg@10")
LOCKED_IDENTITY = (
    "source_commit",
    "relevance",
    "embedding_model",
    "embedding_revision",
    "reranker_model",
    "reranker_revision",
    "device",
    "dtype",
    "attention_implementation",
    "query_prompt_sha256",
    "input_qa_sha256",
    "input_folds_sha256",
)
LOCKED_VALUES = {
    "embedding_model": "Qwen/Qwen3-Embedding-4B",
    "embedding_revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
    "reranker_model": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "reranker_revision": "ac6544c8a46e00af67e330e85a9028c66b8cfd9a",
    "relevance": "source-character interval overlap",
    "device": "cuda:0",
    "dtype": "bfloat16",
    "attention_implementation": "flash_attention_2",
}
RETRIEVAL_SETTINGS = {
    "bm25_top_k": 50,
    "dense_top_k": 50,
    "reranker_top_k": 10,
    "reranker_strategy": "bi_encoder_cosine",
    "reranker_query_prompt": "model_prompt_name:query",
}


class PhaseJAnalysisError(RuntimeError):
    """An input cannot support the locked, paired Phase J comparison."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PhaseJAnalysisError(f"Missing required input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PhaseJAnalysisError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise PhaseJAnalysisError(f"Missing required input: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise PhaseJAnalysisError(f"Empty JSONL input: {path}")
    return pd.DataFrame(rows)


def _jsonl_rows(path: Path):
    if not path.is_file():
        raise PhaseJAnalysisError(f"Missing required input: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _frozen_qa(qa_dir: Path, expected_hash: str) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    qa_path = qa_dir / "qa_frozen.jsonl"
    manifest_path = qa_dir / "qa_frozen_manifest.json"
    manifest = _read_json(manifest_path)
    if (manifest.get("status"), manifest.get("mode"), manifest.get("qa_count"),
            manifest.get("documents")) != ("complete", "full", QA_COUNT, DOCUMENTS):
        raise PhaseJAnalysisError("Frozen QA manifest is not a complete 99-QA full set")
    if manifest.get("qa_sha256") != expected_hash or _sha256(qa_path) != expected_hash:
        raise PhaseJAnalysisError("Frozen QA checksum differs from both evaluations")
    rows = list(_jsonl_rows(qa_path))
    qa = {row["qa_id"]: row for row in rows}
    if len(rows) != QA_COUNT or len(qa) != QA_COUNT or len({row["doc_id"] for row in rows}) != DOCUMENTS:
        raise PhaseJAnalysisError("Frozen QA rows do not cover 99 unique QA and 33 documents")
    for row in rows:
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise PhaseJAnalysisError(f"Missing evidence spans for {row['qa_id']}")
        if any(int(item["source_end"]) <= int(item["source_start"]) for item in evidence):
            raise PhaseJAnalysisError(f"Invalid evidence span for {row['qa_id']}")
    return qa, {"frozen_qa": qa_path, "frozen_qa_manifest": manifest_path}


def _check_manifest_pair(base: dict[str, Any], neutral: dict[str, Any], folds_path: Path) -> None:
    for label, manifest, systems in (
        ("Phase G", base, {ADAPTIVE, *FIXED_SYSTEMS}),
        ("Phase J", neutral, {NEUTRAL}),
    ):
        if (manifest.get("status"), manifest.get("mode"), manifest.get("qa_count")) != (
            "complete", "full", QA_COUNT
        ):
            raise PhaseJAnalysisError(f"{label} evaluation is not a complete full 99-QA run")
        if set(manifest.get("systems", [])) != systems or len(manifest["systems"]) != len(systems):
            raise PhaseJAnalysisError(f"{label} has unexpected systems")
        if manifest.get("source_commit") != SOURCE_COMMIT:
            raise PhaseJAnalysisError(f"{label} source commit differs from the locked experiment")
        if manifest.get("input_folds_sha256") != _sha256(folds_path):
            raise PhaseJAnalysisError(f"{label} fold hash differs from the supplied frozen folds")
    for key in LOCKED_IDENTITY:
        if not base.get(key) or base[key] != neutral.get(key):
            raise PhaseJAnalysisError(f"Phase G and J evaluation identity differs: {key}")
    for key, expected in LOCKED_VALUES.items():
        if neutral[key] != expected:
            raise PhaseJAnalysisError(f"Locked Phase J configuration differs: {key}")


def _check_pipeline(
    label: str,
    prepared_dir: Path,
    index_dir: Path,
    retrieval_dir: Path,
    evaluation: dict[str, Any],
    systems: Sequence[str],
    folds_path: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    """Verify prepared → indexed → retrieved → evaluated checksums for a run."""

    prefix = "phase_g" if label == "Phase G" else "phase_j"
    prepared_path = prepared_dir / "prepare_manifest.json"
    index_path = index_dir / "index_manifest.json"
    retrieval_path = retrieval_dir / "retrieval_manifest.json"
    prepared = _read_json(prepared_path)
    index = _read_json(index_path)
    retrieval = _read_json(retrieval_path)
    expected = set(systems)
    if (prepared.get("status") != "complete" or prepared.get("source_commit") != SOURCE_COMMIT or
            prepared.get("documents") != DOCUMENTS or prepared.get("folds") != FOLDS or
            prepared.get("fold_seed") != SEED or not expected <= set(prepared.get("systems", {}))):
        raise PhaseJAnalysisError(f"{label} prepared manifest differs from the locked experiment")
    prepared_folds = prepared_dir / "folds.jsonl"
    if _sha256(prepared_folds) != _sha256(folds_path):
        raise PhaseJAnalysisError(f"{label} prepared folds differ from the frozen folds")
    if (index.get("status"), index.get("mode"), index.get("qa_count"),
            index.get("source_documents")) != ("complete", "full", QA_COUNT, DOCUMENTS):
        raise PhaseJAnalysisError(f"{label} index manifest is not a complete full run")
    if set(index.get("systems", {})) != expected or set(retrieval.get("systems", {})) != expected:
        raise PhaseJAnalysisError(f"{label} index/retrieval systems differ from the evaluation")
    if index.get("input_prepare_manifest_sha256") != _sha256(prepared_path):
        raise PhaseJAnalysisError(f"{label} index does not reference its prepared manifest")
    if retrieval.get("status") != "complete" or retrieval.get("retrieval") != RETRIEVAL_SETTINGS:
        raise PhaseJAnalysisError(f"{label} retrieval manifest differs from the locked protocol")
    if retrieval.get("input_index_manifest_sha256") != _sha256(index_path):
        raise PhaseJAnalysisError(f"{label} retrieval does not reference its index manifest")
    if evaluation.get("input_retrieval_manifest_sha256") != _sha256(retrieval_path):
        raise PhaseJAnalysisError(f"{label} evaluation does not reference its retrieval manifest")
    for field in ("source_commit", "embedding_model", "embedding_revision", "device", "dtype",
                  "attention_implementation", "query_prompt_sha256", "input_qa_sha256"):
        if index.get(field) != evaluation.get(field):
            raise PhaseJAnalysisError(f"{label} index identity differs from evaluation: {field}")
    for field in ("source_commit", "embedding_model", "embedding_revision", "reranker_model",
                  "reranker_revision", "device", "dtype", "attention_implementation",
                  "query_prompt_sha256", "input_qa_sha256"):
        if retrieval.get(field) != evaluation.get(field):
            raise PhaseJAnalysisError(f"{label} retrieval identity differs from evaluation: {field}")
    if len({index.get("harness_sha256"), retrieval.get("harness_sha256"), evaluation.get("harness_sha256")}) != 1:
        raise PhaseJAnalysisError(f"{label} harness hashes differ across pipeline stages")
    files = {
        f"{prefix}_prepared_manifest": prepared_path,
        f"{prefix}_prepared_folds": prepared_folds,
        f"{prefix}_index_manifest": index_path,
        f"{prefix}_retrieval_manifest": retrieval_path,
    }
    prepared_files: dict[str, Path] = {}
    retrieval_rows: dict[str, Path] = {}
    for system in systems:
        prepared_record = prepared["systems"][system]
        prepared_file = prepared_dir / prepared_record["path"]
        if _sha256(prepared_file) != prepared_record.get("sha256"):
            raise PhaseJAnalysisError(f"{label} prepared system checksum mismatch: {system}")
        store = index_dir / system / "document_store.json"
        if _sha256(store) != index["systems"][system].get("sha256"):
            raise PhaseJAnalysisError(f"{label} index store checksum mismatch: {system}")
        retrieved = retrieval_dir / f"{system}.jsonl"
        retrieved_hash = _sha256(retrieved)
        if (retrieved_hash != retrieval["systems"][system].get("sha256") or
                retrieved_hash != evaluation.get("input_retrieval_sha256", {}).get(system) or
                retrieval["systems"][system].get("queries") != QA_COUNT):
            raise PhaseJAnalysisError(f"{label} retrieval rows fail checksum or query-count validation: {system}")
        files[f"{prefix}_prepared_{system}"] = prepared_file
        files[f"{prefix}_index_{system}"] = store
        files[f"{prefix}_retrieval_{system}"] = retrieved
        prepared_files[system] = prepared_file
        retrieval_rows[system] = retrieved
    return files, prepared_files, retrieval_rows


def _check_retrieval_rows(
    label: str,
    retrieval_rows: dict[str, Path],
    qa: dict[str, dict[str, Any]],
) -> None:
    """Verify that every system has one top-10 result row for every frozen QA."""

    expected_qa = set(qa)
    for system, path in retrieval_rows.items():
        rows = list(_jsonl_rows(path))
        qa_ids = [row.get("qa_id") for row in rows]
        if (len(rows) != QA_COUNT or len(set(qa_ids)) != QA_COUNT or
                set(qa_ids) != expected_qa):
            raise PhaseJAnalysisError(
                f"{label} retrieval rows do not cover all {QA_COUNT} frozen QA: {system}"
            )
        if any(row.get("system_id") != system for row in rows):
            raise PhaseJAnalysisError(f"{label} retrieval rows contain the wrong system id: {system}")
        if any(not isinstance(row.get("results"), list) or len(row["results"]) != 10 for row in rows):
            raise PhaseJAnalysisError(f"{label} retrieval is not top-10 for every QA: {system}")


def _interval_overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _union_length(intervals: Sequence[tuple[int, int]]) -> int:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _query_metrics(
    evidence_doc_id: str,
    evidence: Sequence[dict[str, Any]],
    ranked_chunks: Sequence[dict[str, Any]],
    corpus_chunks: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Independently reproduce Phase G source-span retrieval metrics."""

    evidence_spans = [
        (int(item["source_start"]), int(item["source_end"])) for item in evidence
    ]
    evidence_total = sum(end - start for start, end in evidence_spans)
    if evidence_total <= 0:
        raise PhaseJAnalysisError("Evidence spans must be non-empty")

    def relevance(chunk: dict[str, Any]) -> float:
        if chunk.get("doc_id") != evidence_doc_id:
            return 0.0
        span = (int(chunk["source_start"]), int(chunk["source_end"]))
        return max(
            _interval_overlap(span, evidence_span) /
            (evidence_span[1] - evidence_span[0])
            for evidence_span in evidence_spans
        )

    ranked_relevance = [relevance(chunk) for chunk in ranked_chunks]
    corpus_relevance = sorted((relevance(chunk) for chunk in corpus_chunks), reverse=True)
    result: dict[str, float] = {}
    for k in (1, 3, 5, 10):
        coverage = 0
        for evidence_start, evidence_end in evidence_spans:
            overlaps = []
            for chunk in ranked_chunks[:k]:
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
    dcg = sum(
        value / math.log2(rank + 1)
        for rank, value in enumerate(ranked_relevance[:10], 1)
    )
    idcg = sum(
        value / math.log2(rank + 1)
        for rank, value in enumerate(corpus_relevance[:10], 1)
    )
    result["ndcg@10"] = dcg / idcg if idcg else 0.0
    return result


def _verify_per_query_metrics(
    label: str,
    metrics: pd.DataFrame,
    retrieval_rows: dict[str, Path],
    prepared_files: dict[str, Path],
    qa: dict[str, dict[str, Any]],
) -> None:
    """Recompute every saved metric from frozen evidence and retrieved chunks."""

    saved = metrics.set_index(["system_id", "qa_id"])
    for system, retrieval_path in retrieval_rows.items():
        corpus_by_doc: dict[str, list[dict[str, Any]]] = {}
        for chunk in _jsonl_rows(prepared_files[system]):
            corpus_by_doc.setdefault(chunk["doc_id"], []).append(chunk)
        for row in _jsonl_rows(retrieval_path):
            qa_row = qa[row["qa_id"]]
            ranked = [result["meta"] for result in row["results"]]
            recomputed = _query_metrics(
                qa_row["doc_id"], qa_row["evidence"], ranked,
                corpus_by_doc[qa_row["doc_id"]],
            )
            recorded = saved.loc[(system, row["qa_id"])]
            for metric, expected in recomputed.items():
                if not np.isclose(float(recorded[metric]), expected, atol=1e-12):
                    raise PhaseJAnalysisError(
                        f"{label} per-query metric differs from retrieved evidence: "
                        f"{system}/{row['qa_id']}/{metric}"
                    )


def _check_rows(base: pd.DataFrame, neutral: pd.DataFrame, folds: pd.DataFrame) -> None:
    required = {"qa_id", "doc_id", "fold", "system_id", *METRICS}
    for label, frame, systems in (
        ("Phase G", base, {ADAPTIVE, *FIXED_SYSTEMS}),
        ("Phase J", neutral, {NEUTRAL}),
    ):
        if required - set(frame.columns):
            raise PhaseJAnalysisError(f"{label} rows lack columns: {sorted(required - set(frame.columns))}")
        if len(frame) != QA_COUNT * len(systems) or set(frame["system_id"]) != systems:
            raise PhaseJAnalysisError(f"{label} row count or systems differ from the full protocol")
        if frame.duplicated(["system_id", "qa_id"]).any():
            raise PhaseJAnalysisError(f"{label} contains duplicate system/QA rows")
        counts = frame.groupby("system_id")["qa_id"].nunique()
        if set(counts) != {QA_COUNT}:
            raise PhaseJAnalysisError(f"{label} does not cover all {QA_COUNT} QA for every system")
        values = frame[list(METRICS)].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < -1e-12) | (values > 1 + 1e-12)).any():
            raise PhaseJAnalysisError(f"{label} contains invalid retrieval metrics")
    if {"doc_id", "doc_name", "domain", "fold"} - set(folds.columns):
        raise PhaseJAnalysisError("Frozen fold rows are incomplete")
    if len(folds) != DOCUMENTS or folds["doc_id"].nunique() != DOCUMENTS or folds["doc_name"].nunique() != DOCUMENTS:
        raise PhaseJAnalysisError("Frozen folds must contain exactly 33 distinct documents")
    if folds["fold"].nunique() != FOLDS or set(folds["fold"]) != set(range(FOLDS)):
        raise PhaseJAnalysisError("Frozen folds must be 0 through 4")
    key_columns = ["qa_id", "doc_id", "fold"]
    base_keys = base[base["system_id"] == ADAPTIVE][key_columns].sort_values("qa_id").reset_index(drop=True)
    neutral_keys = neutral[key_columns].sort_values("qa_id").reset_index(drop=True)
    if not base_keys.equals(neutral_keys):
        raise PhaseJAnalysisError("Phase J QA/document/fold identity differs from Phase G")
    if base_keys["qa_id"].nunique() != QA_COUNT or base_keys["doc_id"].nunique() != DOCUMENTS:
        raise PhaseJAnalysisError("The frozen QA set must contain 99 QA from 33 documents")
    if set(base_keys.groupby("doc_id").size()) != {3}:
        raise PhaseJAnalysisError("Every document must contribute exactly three QA")
    fold_map = folds.set_index("doc_id")["fold"]
    if set(base_keys["doc_id"]) != set(fold_map.index) or any(
        int(row.fold) != int(fold_map.loc[row.doc_id]) for row in base_keys.itertuples(index=False)
    ):
        raise PhaseJAnalysisError("QA folds do not match the frozen document folds")
    expected_keys = base_keys.set_index("qa_id")[["doc_id", "fold"]]
    for system, rows in base.groupby("system_id"):
        actual = rows.set_index("qa_id")[["doc_id", "fold"]].sort_index()
        if not actual.equals(expected_keys.sort_index()):
            raise PhaseJAnalysisError(f"Phase G QA/document/fold identity differs for {system}")


def _derive_virtual_comparators(base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Reproduce Phase G's held-out-fold best fixed and per-query nDCG oracle."""

    fixed = base[base["system_id"].isin(FIXED_SYSTEMS)]
    best_rows: list[pd.DataFrame] = []
    chosen: dict[str, str] = {}
    for fold in range(FOLDS):
        training = fixed[fixed["fold"] != fold]
        means = training.groupby("system_id")["ndcg@10"].mean()
        # Phase G uses insertion order for exact ties; FIXED_SYSTEMS preserves it.
        winner = max(FIXED_SYSTEMS, key=lambda system: means.loc[system])
        chosen[str(fold)] = winner
        best_rows.append(fixed[(fixed["fold"] == fold) & (fixed["system_id"] == winner)].copy())
    best = pd.concat(best_rows, ignore_index=True)
    best["system_id"] = "best_fixed_cv"
    # Phase G's oracle takes the complete row of the fixed system with the best
    # nDCG@10 for each QA. Other oracle metrics are descriptive, not upper bounds.
    order = {system: index for index, system in enumerate(FIXED_SYSTEMS)}
    sorted_fixed = fixed.assign(_tie_order=fixed["system_id"].map(order)).sort_values(
        ["qa_id", "ndcg@10", "_tie_order"], ascending=[True, False, True]
    )
    oracle = sorted_fixed.drop_duplicates("qa_id").drop(columns="_tie_order").copy()
    oracle["system_id"] = "oracle"
    return pd.concat([best, oracle], ignore_index=True), chosen


def _document_metrics(rows: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    docs = rows.groupby(["system_id", "doc_id"], as_index=False)[list(METRICS)].mean()
    docs = docs.merge(folds[["doc_id", "doc_name", "domain", "fold"]], on="doc_id", validate="many_to_one")
    if set(docs.groupby("system_id").size()) != {DOCUMENTS}:
        raise PhaseJAnalysisError("Document aggregation lost a system or document")
    return docs.sort_values(["system_id", "doc_id"]).reset_index(drop=True)


def _bootstrap_ci(values: Sequence[float], rng: np.random.Generator, samples: int) -> tuple[float, float]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) != DOCUMENTS or not np.isfinite(vector).all():
        raise PhaseJAnalysisError("Bootstrap expects 33 finite document values")
    indices = rng.integers(0, len(vector), size=(samples, len(vector)))
    return tuple(float(x) for x in np.quantile(vector[indices].mean(axis=1), [0.025, 0.975]))


def _holm(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        result[index] = running
    return result


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def _summaries(docs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    indexed = {system: rows.set_index("doc_id").sort_index() for system, rows in docs.groupby("system_id")}
    for metric in METRICS:
        for system in (NEUTRAL, *COMPARATORS):
            values = indexed[system][metric].to_numpy(dtype=float)
            low, high = _bootstrap_ci(values, rng, BOOTSTRAP_SAMPLES)
            summary_rows.append({
                "system_id": system, "metric": metric, "documents": DOCUMENTS,
                "mean": float(values.mean()), "std": float(values.std(ddof=1)),
                "ci95_low": low, "ci95_high": high,
            })
        metric_comparisons = []
        reference = indexed[NEUTRAL]
        for comparator in COMPARATORS:
            other = indexed[comparator]
            if not reference.index.equals(other.index):
                raise PhaseJAnalysisError(f"Unpaired documents: {NEUTRAL} vs {comparator}")
            delta = reference[metric].to_numpy(dtype=float) - other[metric].to_numpy(dtype=float)
            low, high = _bootstrap_ci(delta, rng, BOOTSTRAP_SAMPLES)
            p = 1.0 if np.all(delta == 0) else float(
                wilcoxon(delta, zero_method="pratt", alternative="two-sided", method="auto").pvalue
            )
            metric_comparisons.append({
                "metric": metric, "reference": NEUTRAL, "comparator": comparator,
                "documents": DOCUMENTS, "mean_difference": float(delta.mean()),
                "median_difference": float(np.median(delta)), "ci95_low": low,
                "ci95_high": high, "wilcoxon_p": p,
                "rank_biserial": _rank_biserial(delta),
                "reference_wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
                "reference_losses": int((delta < 0).sum()),
            })
        for row, adjusted in zip(metric_comparisons, _holm([row["wilcoxon_p"] for row in metric_comparisons])):
            row["holm_p"] = adjusted
            row["significant_0_05"] = adjusted < 0.05
        comparison_rows.extend(metric_comparisons)
    return pd.DataFrame(summary_rows), pd.DataFrame(comparison_rows)


def _write_summary(path: Path, summary: pd.DataFrame, comparisons: pd.DataFrame, best_fixed: dict[str, str]) -> None:
    means = summary[summary["metric"] == "ndcg@10"].set_index("system_id")
    paired = comparisons[comparisons["metric"] == "ndcg@10"].set_index("comparator")
    lines = [
        "# Phase J — Exact BI-neutral mixed-index retrieval analysis", "",
        "## Paper says", "",
        "- Sections 2.4 and 3.5 and Table 5 compare Adaptive with raw LangChain recursive default and raw page for downstream answer quality.",
        "- The paper does not report this BI-neutral mixed-index experiment.", "",
        "## Code and artifacts show", "",
        "- Frozen Phase G and exact Phase J evaluations share the same 99 QA, 33 documents, folds, models, retrieval definition, and source commit.",
        "- Each document contributes three QA, averaged before bootstrap intervals and paired tests.",
        "- Best fixed is chosen by training-fold nDCG@10; held-out documents are evaluated with that fixed system.",
        f"- Best-fixed choices by held-out fold: {json.dumps(best_fixed, sort_keys=True)}.",
        "- Oracle chooses the best fixed-system nDCG@10 row for each QA. It is an upper bound for nDCG@10 only.",
        "- All ten retrieval metrics are in `system_summary.csv` and `paired_comparisons.csv`.", "",
        "| System | Mean nDCG@10 |", "|---|---:|",
    ]
    for system in (NEUTRAL, *COMPARATORS):
        lines.append(f"| `{system}` | {means.loc[system, 'mean']:.4f} |")
    lines.extend(["", "| BI-neutral minus comparator | Mean difference | 95% document-bootstrap CI | Holm p |",
                  "|---|---:|---:|---:|"])
    for system in COMPARATORS:
        row = paired.loc[system]
        lines.append(f"| `{system}` | {row['mean_difference']:+.4f} | "
                     f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['holm_p']:.4g} |")
    lines.extend([
        "", "## Analyst inference", "",
        "- These are paired retrieval comparisons for the new mixed index. They do not measure answer quality.",
        "- The per-QA oracle uses outcome information and is a diagnostic upper bound, not a deployable selector.",
        "- Two-sided Wilcoxon tests use Pratt handling for zero differences; Holm correction covers the five comparators separately within each retrieval metric.",
        "- A confidence interval and a corrected p value answer different questions and need not give the same threshold decision.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_dir = args.phase_g_evaluation_dir
    neutral_dir = args.phase_j_result_dir / "evaluation"
    base_manifest_path = base_dir / "retrieval_evaluation.json"
    neutral_manifest_path = neutral_dir / "retrieval_evaluation.json"
    base_rows_path = base_dir / "per_query_metrics.jsonl"
    neutral_rows_path = neutral_dir / "per_query_metrics.jsonl"
    folds_path = args.folds_path
    base_manifest = _read_json(base_manifest_path)
    neutral_manifest = _read_json(neutral_manifest_path)
    _check_manifest_pair(base_manifest, neutral_manifest, folds_path)
    base = _read_jsonl(base_rows_path)
    neutral = _read_jsonl(neutral_rows_path)
    folds = _read_jsonl(folds_path)
    _check_rows(base, neutral, folds)
    phase_g_root = base_dir.parent
    phase_j_prepared_dir = args.phase_j_result_dir.parent / "prepared"
    qa, qa_paths = _frozen_qa(phase_g_root / "qa-full", neutral_manifest["input_qa_sha256"])
    phase_g_paths, phase_g_prepared_files, phase_g_retrieval_rows = _check_pipeline(
        "Phase G",
        phase_g_root / "prepared",
        phase_g_root / "index-full",
        phase_g_root / "retrieval-full",
        base_manifest,
        (ADAPTIVE, *FIXED_SYSTEMS),
        folds_path,
    )
    phase_j_paths, phase_j_prepared_files, phase_j_retrieval_rows = _check_pipeline(
        "Phase J",
        phase_j_prepared_dir,
        args.phase_j_result_dir / "index",
        args.phase_j_result_dir / "retrieval",
        neutral_manifest,
        (NEUTRAL,),
        folds_path,
    )
    _check_retrieval_rows("Phase G", phase_g_retrieval_rows, qa)
    _check_retrieval_rows("Phase J", phase_j_retrieval_rows, qa)
    _verify_per_query_metrics(
        "Phase G", base, phase_g_retrieval_rows, phase_g_prepared_files, qa
    )
    _verify_per_query_metrics(
        "Phase J", neutral, phase_j_retrieval_rows, phase_j_prepared_files, qa
    )
    derived, best_fixed = _derive_virtual_comparators(base)
    # The base and virtual comparators must reproduce Phase G's published means.
    for system in (*FIXED_SYSTEMS, ADAPTIVE, "best_fixed_cv", "oracle"):
        source = derived if system in {"best_fixed_cv", "oracle"} else base
        rows = source[source["system_id"] == system]
        for metric in METRICS:
            if not np.isclose(rows[metric].mean(), base_manifest["summary"][system][metric], atol=1e-12):
                raise PhaseJAnalysisError(f"Phase G summary mismatch: {system}/{metric}")
    for metric in METRICS:
        if not np.isclose(neutral[metric].mean(), neutral_manifest["summary"][NEUTRAL][metric], atol=1e-12):
            raise PhaseJAnalysisError(f"Phase J summary mismatch: {metric}")
    combined = pd.concat([base[base["system_id"] == ADAPTIVE],
                          base[base["system_id"].isin(("raw__page", "raw__langch_recurs_default"))],
                          neutral, derived], ignore_index=True)
    documents = _document_metrics(combined, folds)
    summary, comparisons = _summaries(documents)
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise PhaseJAnalysisError(f"Analysis output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    documents.to_csv(output / "document_metrics.csv", index=False)
    summary.to_csv(output / "system_summary.csv", index=False)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    _write_summary(output / "SUMMARY.md", summary, comparisons, best_fixed)
    paths = {
        "phase_g_evaluation_manifest": base_manifest_path,
        "phase_g_per_query_metrics": base_rows_path,
        "phase_j_evaluation_manifest": neutral_manifest_path,
        "phase_j_per_query_metrics": neutral_rows_path,
        "frozen_folds": folds_path,
        **qa_paths,
        **phase_g_paths,
        **phase_j_paths,
    }
    script_path = Path(__file__).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    script_status = subprocess.run(
        ["git", "status", "--porcelain", "--", str(script_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": SOURCE_COMMIT, "analysis_commit": commit,
        "analysis_script_sha256": _sha256(script_path),
        "analysis_worktree_dirty": bool(script_status),
        "statistical_unit": "document", "documents": DOCUMENTS, "qa": QA_COUNT,
        "qa_per_document": 3, "folds": FOLDS, "fold_seed": SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": SEED,
        "confidence_interval": "95% percentile bootstrap over documents (paired differences for comparisons)",
        "paired_test": "two-sided Wilcoxon signed-rank with Pratt zeros",
        "multiple_testing": "Holm correction across five comparators separately for each metric",
        "comparators": list(COMPARATORS), "metrics": list(METRICS),
        "best_fixed_cv_metric": "training-fold ndcg@10", "best_fixed_by_held_out_fold": best_fixed,
        "oracle_definition": "best fixed-system per-QA ndcg@10; upper bound only for ndcg@10",
        "inputs": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in paths.items()},
        "outputs": {name: _sha256(output / name) for name in (
            "document_metrics.csv", "paired_comparisons.csv", "system_summary.csv", "SUMMARY.md"
        )},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-g-evaluation-dir", type=Path, required=True)
    parser.add_argument("--phase-j-result-dir", type=Path, required=True)
    parser.add_argument("--folds-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
