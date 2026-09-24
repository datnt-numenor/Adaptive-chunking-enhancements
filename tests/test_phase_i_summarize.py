import json
import sys
from pathlib import Path

import pytest


WEEK3 = Path(__file__).resolve().parents[1] / "research" / "week3"
sys.path.insert(0, str(WEEK3))

from phase_i_summarize import ABSTAIN, SYSTEMS, SummaryError, summarize  # noqa: E402


def _write_system(root: Path, system: str, ids: tuple[str, ...]) -> None:
    rows = []
    for index, query_id in enumerate(ids):
        abstain = system == "raw__page" and index == 1
        metrics = [
            {
                "name": "Retrieval Completeness",
                "score": 1.0,
                "error": None,
            }
        ]
        if not abstain:
            metrics.append(
                {
                    "name": "Correctness [GEval]",
                    "score": 1.0000000000000002 if index == 0 else 0.5,
                    "error": None,
                }
            )
        rows.append(
            {
                "query_id": query_id,
                "generated_output": ABSTAIN if abstain else "Grounded answer",
                "metrics": metrics,
            }
        )
    path = root / system / "generated_questions_evaluation_results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_summarize_validates_and_normalizes_roundoff(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "summary"
    for system in SYSTEMS:
        _write_system(input_dir, system, ("q1", "q2"))

    result = summarize(input_dir, output_dir, expected_queries=2)

    assert result["status"] == "ok"
    assert result["rows"] == 6
    assert result["score_policy"]["normalized_roundoff_scores"] == 3
    page = next(row for row in result["summary"] if row["system"] == "raw__page")
    assert page["answered_queries"] == 1
    assert page["abstentions"] == 1
    assert page["answer_correctness_mean_pct"] == 100.0
    assert page["coverage_adjusted_correctness_pct"] == 50.0
    assert (output_dir / "table5_summary.json").is_file()
    assert (output_dir / "table5_summary.csv").is_file()


def test_summarize_rejects_query_id_mismatch(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    for system in SYSTEMS:
        ids = ("q1", "different") if system == "raw__page" else ("q1", "q2")
        _write_system(input_dir, system, ids)

    with pytest.raises(SummaryError, match="query_id set differs"):
        summarize(input_dir, tmp_path / "summary", expected_queries=2)
