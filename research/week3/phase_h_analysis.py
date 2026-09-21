"""Phase H: document-level retrieval statistics and Block Integrity ablation.

This stage is intentionally local and deterministic.  It consumes the frozen
Phase G artifacts; it does not call an API, load an embedding model, or create
new infrastructure.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


SCHEMA_VERSION = 1
SEED = 2026
BOOTSTRAP_SAMPLES = 10_000
RETRIEVAL_METRICS = (
    "hit@1",
    "recall@1",
    "hit@3",
    "recall@3",
    "hit@5",
    "recall@5",
    "hit@10",
    "recall@10",
    "mrr@10",
    "ndcg@10",
)
ADAPTIVE_CANDIDATES = (
    "our_recurs_1100",
    "our_recurs_600",
    "page",
    "llm_regex",
)
INTRINSIC_METRICS = (
    "size_compliance",
    "block_integrity",
    "intrachunk_cohesion",
    "document_contextual_coherence",
    "references_completeness",
)
PAGE_NEUTRAL_METRICS = tuple(
    metric for metric in INTRINSIC_METRICS if metric != "block_integrity"
)
METHOD_TO_SYSTEM = {
    "our_recurs_1100": "processed__our_recurs_1100",
    "our_recurs_600": "processed__our_recurs_600",
    "page": "processed__page",
    "llm_regex": "processed__llm_regex",
}


class PhaseHError(RuntimeError):
    """Raised when an input violates the frozen Phase H protocol."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile confidence interval from document-level bootstrap samples."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a non-empty finite vector")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    draws = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[draws].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm family-wise error correction preserving the original order."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("p-values must be a finite vector")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[original_index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(values), dtype=float)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return adjusted.tolist()


def rank_biserial(differences: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation; positive favors the first system."""

    values = np.asarray(differences, dtype=float)
    nonzero = values[values != 0]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def select_methods(
    metrics: pd.DataFrame,
    scoring_metrics: Sequence[str],
) -> tuple[dict[str, str], pd.DataFrame]:
    """Select one candidate per document with NaN-aware equal weighting."""

    required = {"doc_name", "chunking_method", "metric_name", "score"}
    missing = required - set(metrics.columns)
    if missing:
        raise PhaseHError(f"Metric table is missing columns: {sorted(missing)}")
    filtered = metrics[
        metrics["chunking_method"].isin(ADAPTIVE_CANDIDATES)
        & metrics["metric_name"].isin(scoring_metrics)
    ]
    selections: dict[str, str] = {}
    score_rows: list[dict[str, Any]] = []
    for doc_name, rows in filtered.groupby("doc_name", sort=True):
        pivot = rows.pivot(index="metric_name", columns="chunking_method", values="score")
        absent = set(ADAPTIVE_CANDIDATES) - set(pivot.columns)
        if absent:
            raise PhaseHError(f"{doc_name} is missing candidate methods: {sorted(absent)}")
        method_scores: dict[str, float] = {}
        for method in pivot.columns:
            values = pivot[method].reindex(scoring_metrics).dropna()
            if values.empty:
                raise PhaseHError(f"{doc_name}/{method} has no usable scoring metrics")
            method_scores[str(method)] = float(values.mean())
        best = max(method_scores, key=method_scores.get)
        selections[str(doc_name)] = best
        for method, score in method_scores.items():
            score_rows.append(
                {
                    "doc_name": doc_name,
                    "method": method,
                    "score": score,
                    "selected": method == best,
                }
            )
    return selections, pd.DataFrame(score_rows)


def audit_page_boundaries(data_dir: Path, tolerance_chars: int = 5) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Measure overlap between page ends and parser-provided split points."""

    parsed_dir = data_dir / "adi_parsed" if (data_dir / "adi_parsed").is_dir() else data_dir
    rows: list[dict[str, Any]] = []
    for path in sorted(parsed_dir.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        pages = list(document["pages"].values())
        page_boundaries = np.cumsum([len(str(page)) for page in pages]).tolist()[:-1]
        split_points = [int(point) for point in document["split_points"]]
        matched = sum(
            any(abs(boundary - split) <= tolerance_chars for split in split_points)
            for boundary in page_boundaries
        )
        rows.append(
            {
                "doc_name": path.stem,
                "page_boundaries": len(page_boundaries),
                "matched_parser_splits": matched,
                "match_rate": matched / len(page_boundaries) if page_boundaries else math.nan,
            }
        )
    frame = pd.DataFrame(rows)
    total = int(frame["page_boundaries"].sum())
    matches = int(frame["matched_parser_splits"].sum())
    summary = {
        "documents": len(frame),
        "tolerance_chars": tolerance_chars,
        "page_boundaries": total,
        "matched_parser_splits": matches,
        "match_rate": matches / total if total else None,
    }
    return frame, summary


def _validate_per_query(frame: pd.DataFrame) -> None:
    required = {"qa_id", "doc_id", "fold", "system_id", *RETRIEVAL_METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise PhaseHError(f"Per-query metrics are missing columns: {sorted(missing)}")
    counts = frame.groupby("system_id")["qa_id"].nunique()
    if len(counts) != 10 or set(counts) != {99}:
        raise PhaseHError(f"Expected 99 unique QA for 10 systems, observed {counts.to_dict()}")
    if frame.duplicated(["system_id", "qa_id"]).any():
        raise PhaseHError("Duplicate system/QA rows in retrieval metrics")
    qa_docs = frame[["qa_id", "doc_id"]].drop_duplicates()
    if len(qa_docs) != 99 or qa_docs["doc_id"].nunique() != 33:
        raise PhaseHError("Expected 99 QA across 33 documents")
    if set(qa_docs.groupby("doc_id").size()) != {3}:
        raise PhaseHError("Every document must contribute exactly three QA")
    if not np.isfinite(frame[list(RETRIEVAL_METRICS)].to_numpy(dtype=float)).all():
        raise PhaseHError("Retrieval metrics contain non-finite values")


def _document_metrics(per_query: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    doc_metrics = (
        per_query.groupby(["system_id", "doc_id"], as_index=False)[list(RETRIEVAL_METRICS)]
        .mean()
        .merge(folds[["doc_id", "doc_name", "domain", "fold"]], on="doc_id", validate="many_to_one")
    )
    if doc_metrics["doc_id"].nunique() != 33:
        raise PhaseHError("Document aggregation lost documents")
    return doc_metrics


def system_bootstrap_summary(
    doc_metrics: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for system_id, system_rows in doc_metrics.groupby("system_id", sort=True):
        system_rows = system_rows.sort_values("doc_id")
        for metric in RETRIEVAL_METRICS:
            values = system_rows[metric].to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, samples=samples, rng=rng)
            rows.append(
                {
                    "system_id": system_id,
                    "metric": metric,
                    "documents": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons(
    doc_metrics: pd.DataFrame,
    *,
    reference: str,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    """Paired document-level reference-minus-comparator comparisons."""

    systems = sorted(set(doc_metrics["system_id"]) - {reference})
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for metric in RETRIEVAL_METRICS:
        reference_rows = (
            doc_metrics[doc_metrics["system_id"] == reference]
            .set_index("doc_id")
            .sort_index()
        )
        metric_rows: list[dict[str, Any]] = []
        for comparator in systems:
            comparator_rows = (
                doc_metrics[doc_metrics["system_id"] == comparator]
                .set_index("doc_id")
                .sort_index()
            )
            if list(reference_rows.index) != list(comparator_rows.index):
                raise PhaseHError(f"Document mismatch: {reference} vs {comparator}")
            differences = (
                reference_rows[metric].to_numpy(dtype=float)
                - comparator_rows[metric].to_numpy(dtype=float)
            )
            low, high = bootstrap_mean_ci(differences, samples=samples, rng=rng)
            if np.all(differences == 0):
                p_value = 1.0
            else:
                p_value = float(
                    wilcoxon(differences, zero_method="pratt", alternative="two-sided", method="auto").pvalue
                )
            metric_rows.append(
                {
                    "metric": metric,
                    "reference": reference,
                    "comparator": comparator,
                    "documents": len(differences),
                    "mean_difference": float(differences.mean()),
                    "median_difference": float(np.median(differences)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "wilcoxon_p": p_value,
                    "rank_biserial": rank_biserial(differences),
                    "reference_wins": int((differences > 0).sum()),
                    "ties": int((differences == 0).sum()),
                    "reference_losses": int((differences < 0).sum()),
                }
            )
        adjusted = holm_adjust([row["wilcoxon_p"] for row in metric_rows])
        for row, corrected in zip(metric_rows, adjusted):
            row["holm_p"] = corrected
            row["significant_0_05"] = corrected < 0.05
        rows.extend(metric_rows)
    return pd.DataFrame(rows)


def grouped_summary(doc_metrics: pd.DataFrame, group: str) -> pd.DataFrame:
    return (
        doc_metrics.groupby([group, "system_id"], as_index=False)[list(RETRIEVAL_METRICS)]
        .mean()
        .sort_values([group, "ndcg@10", "system_id"], ascending=[True, False, True])
    )


def replay_selector(
    per_query: pd.DataFrame,
    folds: pd.DataFrame,
    selections: dict[str, str],
    label: str,
) -> pd.DataFrame:
    """Replay fixed-index query rows chosen by document-level selector decisions.

    This is a counterfactual diagnostic, not an exact mixed-index retrieval run,
    because distractor documents were chunked with the same fixed method in each
    source retrieval output.
    """

    doc_names = folds.set_index("doc_id")["doc_name"].to_dict()
    rows: list[dict[str, Any]] = []
    indexed = per_query.set_index(["system_id", "qa_id"])
    if not indexed.index.is_unique:
        raise PhaseHError("Fixed-system replay input contains duplicate system/QA rows")
    qa_docs = per_query[["qa_id", "doc_id"]].drop_duplicates().sort_values("qa_id")
    for qa in qa_docs.itertuples(index=False):
        doc_name = doc_names[qa.doc_id]
        method = selections[doc_name]
        system = METHOD_TO_SYSTEM[method]
        source = indexed.loc[(system, qa.qa_id)]
        rows.append(
            {
                "qa_id": qa.qa_id,
                "doc_id": qa.doc_id,
                "fold": int(source["fold"]),
                "system_id": label,
                "selected_method": method,
                **{metric: float(source[metric]) for metric in RETRIEVAL_METRICS},
            }
        )
    return pd.DataFrame(rows)


def _replay_comparison(
    original: pd.DataFrame,
    page_neutral: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    combined = pd.concat([original, page_neutral], ignore_index=True)
    docs = _document_metrics(combined, folds)
    comparisons = paired_comparisons(
        docs,
        reference="page_neutral_replay",
        samples=samples,
        seed=seed,
    )
    return comparisons


def _write_summary(
    path: Path,
    *,
    page_audit: dict[str, Any],
    original_selections: dict[str, str],
    neutral_selections: dict[str, str],
    comparisons: pd.DataFrame,
    replay_summary: pd.DataFrame,
    replay_comparison: pd.DataFrame,
) -> None:
    changes = sum(original_selections[doc] != neutral_selections[doc] for doc in original_selections)
    ndcg = comparisons[comparisons["metric"] == "ndcg@10"].set_index("comparator")
    replay_ndcg = replay_comparison[replay_comparison["metric"] == "ndcg@10"].iloc[0]
    replay_pivot = replay_summary[replay_summary["metric"] == "ndcg@10"].set_index("system_id")
    lines = [
        "# Phase H — Statistical analysis and BI page-neutral ablation",
        "",
        "## Paper says",
        "",
        "- Table 3 reports a Wilcoxon signed-rank result for Adaptive versus individual intrinsic-method scores.",
        "- Table 5 reports a Wilcoxon signed-rank result for Retrieval Completeness.",
        "- The paper does not specify a page-neutral Block Integrity ablation.",
        "",
        "## Code and artifacts show",
        "",
        f"- Phase G contains 99 QA across 33 documents and 10 retrieval systems.",
        f"- {page_audit['matched_parser_splits']}/{page_audit['page_boundaries']} page boundaries "
        f"({page_audit['match_rate']:.2%}) coincide with parser split points within "
        f"{page_audit['tolerance_chars']} characters.",
        f"- Removing BI from selector scoring changes {changes}/33 document decisions.",
        f"- Original selector counts: {dict(Counter(original_selections.values()))}.",
        f"- BI-neutral selector counts: {dict(Counter(neutral_selections.values()))}.",
        "",
        "### Primary document-level nDCG@10 comparisons (Adaptive minus comparator)",
        "",
        "| Comparator | Mean difference | 95% bootstrap CI | Holm p |",
        "|---|---:|---:|---:|",
    ]
    for comparator in ("raw__langch_recurs_default", "raw__page", "raw__langch_recurs_1100"):
        row = ndcg.loc[comparator]
        lines.append(
            f"| `{comparator}` | {row['mean_difference']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['holm_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Page-neutral selector replay",
            "",
            "This replay chooses, for each QA, the fixed-system result matching the method selected for that QA's document. "
            "It isolates selector changes but is **not** an exact mixed-index retrieval rerun because distractor documents "
            "come from fixed-method indexes.",
            "",
            f"- Original-selection replay nDCG@10: {replay_pivot.loc['original_selection_replay', 'mean']:.4f}.",
            f"- BI-neutral replay nDCG@10: {replay_pivot.loc['page_neutral_replay', 'mean']:.4f}.",
            f"- Paired document delta: {replay_ndcg['mean_difference']:+.4f}, 95% CI "
            f"[{replay_ndcg['ci95_low']:+.4f}, {replay_ndcg['ci95_high']:+.4f}], "
            f"Holm p={replay_ndcg['holm_p']:.4g}.",
            "",
            "## Analyst inference",
            "",
            "- A high page-boundary/parser-boundary overlap makes BI non-independent of the page representation; "
            "dropping BI is a conservative sensitivity analysis, not a replacement definition for structural integrity.",
            "- Statistical conclusions use paired document-level units, so the three QA from one document are not treated "
            "as independent samples.",
            "- Exact downstream claims for the BI-neutral selector require one new mixed-index retrieval run. The replay "
            "is diagnostic evidence only.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = args.evaluation_dir / "per_query_metrics.jsonl"
    evaluation_manifest_path = args.evaluation_dir / "retrieval_evaluation.json"
    folds_path = args.prepared_dir / "folds.jsonl"
    for path in (per_query_path, evaluation_manifest_path, folds_path, args.metrics_path, args.adaptive_selections):
        if not path.is_file():
            raise PhaseHError(f"Missing required input: {path}")

    evaluation = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
    if evaluation.get("status") != "complete" or evaluation.get("mode") != "full" or evaluation.get("qa_count") != 99:
        raise PhaseHError("Phase G evaluation manifest is not a complete 99-QA full run")

    per_query = pd.DataFrame(_read_jsonl(per_query_path))
    _validate_per_query(per_query)
    folds = pd.DataFrame(_read_jsonl(folds_path))
    if folds["doc_id"].nunique() != 33 or folds["fold"].nunique() != 5:
        raise PhaseHError("Frozen folds must contain 33 documents in five folds")
    if folds.duplicated("doc_id").any():
        raise PhaseHError("A document appears in multiple fold rows")

    doc_metrics = _document_metrics(per_query, folds)
    summary = system_bootstrap_summary(doc_metrics, samples=args.bootstrap_samples, seed=args.seed)
    comparisons = paired_comparisons(
        doc_metrics,
        reference="adaptive",
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    fold_summary = grouped_summary(doc_metrics, "fold")
    domain_summary = grouped_summary(doc_metrics, "domain")

    metric_rows = pd.read_parquet(args.metrics_path)
    original_recomputed, original_scores = select_methods(metric_rows, INTRINSIC_METRICS)
    locked = json.loads(args.adaptive_selections.read_text(encoding="utf-8"))
    locked_selections = {str(key): str(value) for key, value in locked["selections"].items()}
    if original_recomputed != locked_selections:
        differences = {
            doc: {"recomputed": original_recomputed.get(doc), "locked": locked_selections.get(doc)}
            for doc in set(original_recomputed) | set(locked_selections)
            if original_recomputed.get(doc) != locked_selections.get(doc)
        }
        raise PhaseHError(f"Could not reproduce locked Adaptive selections: {differences}")
    neutral_selections, neutral_scores = select_methods(metric_rows, PAGE_NEUTRAL_METRICS)
    score_table = original_scores.rename(columns={"score": "original_score", "selected": "original_selected"}).merge(
        neutral_scores.rename(columns={"score": "page_neutral_score", "selected": "page_neutral_selected"}),
        on=["doc_name", "method"],
        validate="one_to_one",
    )
    changes = pd.DataFrame(
        [
            {
                "doc_name": doc,
                "domain": doc.split(" ", 1)[0],
                "original_method": locked_selections[doc],
                "page_neutral_method": neutral_selections[doc],
                "changed": locked_selections[doc] != neutral_selections[doc],
            }
            for doc in sorted(locked_selections)
        ]
    )

    page_rows, page_audit = audit_page_boundaries(args.data_dir)
    original_replay = replay_selector(per_query, folds, locked_selections, "original_selection_replay")
    neutral_replay = replay_selector(per_query, folds, neutral_selections, "page_neutral_replay")
    actual_adaptive = per_query[per_query["system_id"] == "adaptive"].copy()
    replay_rows = pd.concat([actual_adaptive, original_replay, neutral_replay], ignore_index=True)
    replay_doc_metrics = _document_metrics(replay_rows, folds)
    replay_summary = system_bootstrap_summary(
        replay_doc_metrics,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    replay_comparison = _replay_comparison(
        original_replay,
        neutral_replay,
        folds,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )

    summary.to_csv(args.output_dir / "system_document_bootstrap.csv", index=False)
    comparisons.to_csv(args.output_dir / "adaptive_paired_comparisons.csv", index=False)
    fold_summary.to_csv(args.output_dir / "fold_summary.csv", index=False)
    domain_summary.to_csv(args.output_dir / "domain_summary.csv", index=False)
    score_table.to_csv(args.output_dir / "selector_scores.csv", index=False)
    changes.to_csv(args.output_dir / "page_neutral_selection_changes.csv", index=False)
    page_rows.to_csv(args.output_dir / "page_boundary_audit.csv", index=False)
    replay_summary.to_csv(args.output_dir / "page_neutral_replay_summary.csv", index=False)
    replay_comparison.to_csv(args.output_dir / "page_neutral_replay_comparison.csv", index=False)
    _write_json(
        args.output_dir / "page_neutral_selections.json",
        {
            "schema_version": SCHEMA_VERSION,
            "definition": "exclude block_integrity and equally weight the four remaining intrinsic metrics",
            "scoring_metrics": list(PAGE_NEUTRAL_METRICS),
            "weights": {metric: 1 / len(PAGE_NEUTRAL_METRICS) for metric in PAGE_NEUTRAL_METRICS},
            "selections": neutral_selections,
        },
    )
    _write_summary(
        args.output_dir / "SUMMARY.md",
        page_audit=page_audit,
        original_selections=locked_selections,
        neutral_selections=neutral_selections,
        comparisons=comparisons,
        replay_summary=replay_summary,
        replay_comparison=replay_comparison,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": _utc_now(),
        "source_commit": _git_commit(),
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "statistical_unit": "document",
        "confidence_interval": "paired or unpaired percentile bootstrap at 95%",
        "paired_test": "two-sided Wilcoxon signed-rank with Pratt zeros",
        "multiple_testing": "Holm correction within each metric family",
        "documents": int(folds["doc_id"].nunique()),
        "qa": int(per_query["qa_id"].nunique()),
        "systems": int(per_query["system_id"].nunique()),
        "folds": int(folds["fold"].nunique()),
        "page_neutral_definition": "block_integrity weight set to zero; four remaining metrics equally weighted",
        "selection_changes": int(changes["changed"].sum()),
        "page_boundary_audit": page_audit,
        "replay_limitation": (
            "Counterfactual replay selects fixed-index per-query rows by document; "
            "it is not an exact mixed-index retrieval rerun."
        ),
        "inputs": {
            "per_query_metrics": {"path": str(per_query_path), "sha256": _sha256(per_query_path)},
            "evaluation_manifest": {"path": str(evaluation_manifest_path), "sha256": _sha256(evaluation_manifest_path)},
            "folds": {"path": str(folds_path), "sha256": _sha256(folds_path)},
            "intrinsic_metrics": {"path": str(args.metrics_path), "sha256": _sha256(args.metrics_path)},
            "adaptive_selections": {"path": str(args.adaptive_selections), "sha256": _sha256(args.adaptive_selections)},
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    _write_json(args.output_dir / "phase_h_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--adaptive-selections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
