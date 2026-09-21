"""Deterministic unit tests for Phase H statistical analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "research" / "week3" / "phase_h_analysis.py"
SPEC = importlib.util.spec_from_file_location("phase_h_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def test_bootstrap_mean_ci_is_deterministic_and_contains_mean():
    values = [0.1, 0.2, 0.3, 0.4]
    first = analysis.bootstrap_mean_ci(
        values, samples=2_000, rng=np.random.default_rng(2026)
    )
    second = analysis.bootstrap_mean_ci(
        values, samples=2_000, rng=np.random.default_rng(2026)
    )
    assert first == second
    assert first[0] <= np.mean(values) <= first[1]


def test_holm_adjust_matches_known_order_and_is_monotone():
    adjusted = analysis.holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_rank_biserial_direction():
    assert analysis.rank_biserial([1.0, 2.0, 3.0]) == 1.0
    assert analysis.rank_biserial([-1.0, -2.0, -3.0]) == -1.0
    assert analysis.rank_biserial([0.0, 0.0]) == 0.0


def test_page_neutral_selector_excludes_block_integrity():
    rows = []
    for method in analysis.ADAPTIVE_CANDIDATES:
        scores = {
            "size_compliance": 0.8,
            "block_integrity": 0.0,
            "intrachunk_cohesion": 0.8,
            "document_contextual_coherence": 0.8,
            "references_completeness": 0.8,
        }
        if method == "page":
            scores["block_integrity"] = 1.0
            scores["size_compliance"] = 0.7
            scores["intrachunk_cohesion"] = 0.7
            scores["document_contextual_coherence"] = 0.7
            scores["references_completeness"] = 0.7
        for metric, score in scores.items():
            rows.append(
                {
                    "doc_name": "doc",
                    "chunking_method": method,
                    "metric_name": metric,
                    "score": score,
                }
            )
    frame = pd.DataFrame(rows)
    original, _ = analysis.select_methods(frame, analysis.INTRINSIC_METRICS)
    neutral, _ = analysis.select_methods(frame, analysis.PAGE_NEUTRAL_METRICS)
    assert original["doc"] == "page"
    assert neutral["doc"] == "llm_regex"


def test_per_query_validation_rejects_document_leakage_shape():
    frame = pd.DataFrame(
        [
            {
                "qa_id": "q1",
                "doc_id": "d1",
                "fold": 0,
                "system_id": "s1",
                **{metric: 1.0 for metric in analysis.RETRIEVAL_METRICS},
            }
        ]
    )
    with pytest.raises(analysis.PhaseHError, match="99 unique QA"):
        analysis._validate_per_query(frame)
