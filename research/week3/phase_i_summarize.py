"""Validate and summarize the completed Phase I Table 5 judge outputs.

The raw DeepEval JSON files are treated as immutable evidence.  This module
only normalizes floating-point round-off within ``SCORE_TOLERANCE`` when it
writes the derived summary (for example, 1.0000000000000002 becomes 1.0).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ABSTAIN = "I don't know based on the provided context."
SYSTEMS = {
    "adaptive": "Adaptive",
    "raw__langch_recurs_default": "Raw LangChain recursive default",
    "raw__page": "Raw page",
}
METRICS = ("Retrieval Completeness", "Correctness [GEval]")
SCORE_TOLERANCE = 1e-12


class SummaryError(RuntimeError):
    """Raised when judge outputs do not satisfy the locked Table 5 shape."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_score(value: Any, *, location: str) -> tuple[float, bool]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{location}: score is not numeric: {value!r}")
    score = float(value)
    if not math.isfinite(score):
        raise SummaryError(f"{location}: score is not finite: {score!r}")
    if score < -SCORE_TOLERANCE or score > 1.0 + SCORE_TOLERANCE:
        raise SummaryError(f"{location}: score is outside [0, 1]: {score!r}")
    normalized = min(1.0, max(0.0, score))
    return normalized, normalized != score


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(input_dir: Path, output_dir: Path, *, expected_queries: int = 99) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    system_rows: dict[str, list[dict[str, Any]]] = {}
    input_hashes: dict[str, str] = {}

    for system in SYSTEMS:
        path = input_dir / system / "generated_questions_evaluation_results.json"
        if not path.is_file():
            raise SummaryError(f"Missing judge output: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != expected_queries:
            raise SummaryError(
                f"{system}: expected {expected_queries} rows, found "
                f"{len(rows) if isinstance(rows, list) else type(rows).__name__}"
            )
        query_ids = [row.get("query_id") for row in rows]
        if any(not isinstance(query_id, str) or not query_id for query_id in query_ids):
            raise SummaryError(f"{system}: every row must have a non-empty query_id")
        if len(set(query_ids)) != expected_queries:
            raise SummaryError(f"{system}: query_id values are not unique")
        system_rows[system] = rows
        input_hashes[system] = _sha256(path)

    reference_ids = {row["query_id"] for row in system_rows["adaptive"]}
    for system, rows in system_rows.items():
        if {row["query_id"] for row in rows} != reference_ids:
            raise SummaryError(f"{system}: query_id set differs from Adaptive")

    summaries: list[dict[str, Any]] = []
    total_clipped = 0
    for system, rows in system_rows.items():
        scores: dict[str, list[float]] = {metric: [] for metric in METRICS}
        clipped = 0
        abstentions = 0

        for row in rows:
            query_id = row["query_id"]
            metrics = row.get("metrics")
            if not isinstance(metrics, list):
                raise SummaryError(f"{system}/{query_id}: metrics must be a list")
            names = [metric.get("name") for metric in metrics]
            if len(names) != len(set(names)):
                raise SummaryError(f"{system}/{query_id}: duplicate metric names")
            if any(name not in METRICS for name in names):
                raise SummaryError(f"{system}/{query_id}: unexpected metrics {names!r}")
            if "Retrieval Completeness" not in names:
                raise SummaryError(f"{system}/{query_id}: missing Retrieval Completeness")

            is_abstention = row.get("generated_output") == ABSTAIN
            has_correctness = "Correctness [GEval]" in names
            if is_abstention != (not has_correctness):
                raise SummaryError(
                    f"{system}/{query_id}: abstention and correctness presence disagree"
                )
            abstentions += int(is_abstention)

            for metric in metrics:
                name = metric["name"]
                if metric.get("error") is not None:
                    raise SummaryError(
                        f"{system}/{query_id}/{name}: metric error {metric['error']!r}"
                    )
                score, was_clipped = _normalized_score(
                    metric.get("score"), location=f"{system}/{query_id}/{name}"
                )
                scores[name].append(score)
                clipped += int(was_clipped)

        retrieval = scores["Retrieval Completeness"]
        correctness = scores["Correctness [GEval]"]
        flat = retrieval + correctness
        summaries.append(
            {
                "system": system,
                "label": SYSTEMS[system],
                "retrieval_completeness_mean_pct": statistics.fmean(retrieval) * 100,
                "retrieval_completeness_std_pct": _sample_std(retrieval) * 100,
                "answer_correctness_mean_pct": statistics.fmean(correctness) * 100,
                "answer_correctness_std_pct": _sample_std(correctness) * 100,
                "final_score_pct": statistics.fmean(flat) * 100,
                "coverage_adjusted_correctness_pct": sum(correctness) / expected_queries * 100,
                "answered_queries": len(correctness),
                "abstentions": abstentions,
                "total_queries": expected_queries,
                "normalized_roundoff_scores": clipped,
            }
        )
        total_clipped += clipped

    result = {
        "schema_version": 1,
        "status": "ok",
        "input_dir": str(input_dir),
        "expected_queries_per_system": expected_queries,
        "systems": len(SYSTEMS),
        "rows": expected_queries * len(SYSTEMS),
        "same_query_ids": True,
        "score_policy": {
            "raw_outputs_modified": False,
            "accepted_roundoff_tolerance": SCORE_TOLERANCE,
            "normalized_roundoff_scores": total_clipped,
        },
        "input_sha256": input_hashes,
        "summary": summaries,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table5_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = list(summaries[0])
    with (output_dir / "table5_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, default=99)
    args = parser.parse_args()
    result = summarize(args.input_dir, args.output_dir, expected_queries=args.expected_queries)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
