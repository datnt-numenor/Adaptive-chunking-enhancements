"""Local-only guard tests for the paper Table 5 answer stage."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).parents[1] / "research" / "week3" / "phase_i_table5.py"
SPEC = importlib.util.spec_from_file_location("phase_i_table5", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
table5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = table5
SPEC.loader.exec_module(table5)
sys.path.insert(0, str(MODULE_PATH.parent))
from phase_i_judge_preflight import JudgePreflightError, inspect_smoke


def test_prompt_matches_upstream_context_format():
    text = table5._prompt("Why?", [{"content": "Example.", "meta": {"doc_name": "Doc A"}}])
    assert text == table5.ANSWER_PROMPT.format(
        context_str="<Document, ID = 0, name = Doc A>\nContent: Example.\n</Document, ID = 0, name = Doc A>",
        question_str="Why?",
    )


def test_frozen_qa_accepts_human_edits_and_rejects_duplicates():
    rows = []
    for doc in range(33):
        for question in range(3):
            rows.append({"qa_id": f"d{doc}::q{question}", "qa_hash": f"hash-{doc}-{question}",
                         "doc_id": f"d{doc}", "validation_status": "valid_after_edit" if question == 1 else "valid",
                         "question": f"Question {doc}-{question}?", "reference_answer": "Answer."})
    assert len(table5._validate_qa(rows)) == 99
    rows[-1]["qa_hash"] = rows[0]["qa_hash"]
    with pytest.raises(table5.PhaseIError, match="99 unique"):
        table5._validate_qa(rows)
    rows[-1]["qa_hash"] = "unique-again"
    rows[-1]["question"] = rows[0]["question"]
    with pytest.raises(table5.PhaseIError, match="99 unique"):
        table5._validate_qa(rows)


def test_retrieval_rejects_duplicate_chunk_and_nonconsecutive_rank():
    ids = {f"q{x}" for x in range(99)}
    def make_row(qa_id):
        return {"qa_id": qa_id, "system_id": "adaptive", "results": [
            {"rank": n, "content": "text", "meta": {"chunk_id": f"{qa_id}-{n}"}}
            for n in range(1, 11)]}
    rows = [make_row(value) for value in sorted(ids)]
    table5._validate_retrieval(rows, ids, "adaptive")
    rows[0]["results"][1]["meta"]["chunk_id"] = rows[0]["results"][0]["meta"]["chunk_id"]
    with pytest.raises(table5.PhaseIError, match="Duplicate"):
        table5._validate_retrieval(rows, ids, "adaptive")
    rows[0] = make_row(rows[0]["qa_id"])
    rows[0]["results"][0]["rank"] = 2
    with pytest.raises(table5.PhaseIError, match="ranks"):
        table5._validate_retrieval(rows, ids, "adaptive")


def test_unresolved_paid_attempt_keeps_full_upper_bound(tmp_path):
    journal = tmp_path / "journal.jsonl"
    journal.write_text("\n".join(json.dumps(event) for event in [
        {"type": "attempt", "id": "a", "request_hash": "x", "upper_usd": 0.7},
        {"type": "attempt", "id": "b", "request_hash": "y", "upper_usd": 0.3},
        {"type": "complete", "id": "b", "request_hash": "y", "actual_usd": 0.2},
    ]) + "\n", encoding="utf-8")
    liability, _, pending = table5._journal_liability(journal)
    assert liability == pytest.approx(0.9)
    assert pending == {"x"}


def test_confirmed_preconnect_failure_can_be_retried_without_liability(tmp_path):
    journal = tmp_path / "journal.jsonl"
    journal.write_text("\n".join(json.dumps(event) for event in [
        {"type": "attempt", "id": "a", "request_hash": "x", "upper_usd": 0.7},
        {"type": "aborted_preconnect", "id": "a", "reason": "WinError 10013"},
    ]) + "\n", encoding="utf-8")
    liability, _, pending = table5._journal_liability(journal)
    assert liability == 0
    assert pending == set()


def test_explicit_rate_limit_rejection_is_not_billed(tmp_path):
    journal = tmp_path / "journal.jsonl"
    journal.write_text("\n".join(json.dumps(event) for event in [
        {"type": "attempt", "id": "a", "request_hash": "x", "upper_usd": 0.7},
        {"type": "rejected_rate_limit", "id": "a", "request_hash": "x"},
    ]) + "\n", encoding="utf-8")
    liability, _, pending = table5._journal_liability(journal)
    assert liability == 0
    assert pending == set()


def test_only_completed_lower_cap_answer_is_reused(tmp_path):
    prompt_hash = "p" * 64
    old_hash = table5._hash_json({"model": table5.MODEL, "prompt_sha256": prompt_hash,
                                 "max_output_tokens": 512, "temperature": 0, "top_p": 1})
    request = {"prompt_sha256": prompt_hash, "request_hash": "new-hash"}
    plan = {"previous_output_cap": 512, "max_output_tokens": 1024}
    table5._write_json(tmp_path / f"{old_hash}.json",
                       {"request_hash": old_hash, "prompt_sha256": prompt_hash,
                        "answer": "Complete answer", "response_id": "response-1"})
    assert not table5._reuse_completed_lower_cap_answer(request, plan, tmp_path, set())
    assert table5._reuse_completed_lower_cap_answer(request, plan, tmp_path, {old_hash})
    cached = json.loads((tmp_path / "new-hash.json").read_text(encoding="utf-8"))
    assert cached["reused_from_request_hash"] == old_hash
    assert cached["max_output_tokens_used"] == 512


def test_paid_command_refuses_missing_explicit_gate(tmp_path):
    with pytest.raises(table5.PhaseIError, match="allow-paid-run"):
        table5.generate(argparse.Namespace(output_dir=tmp_path, allow_paid_run=False, budget_usd=10))


def test_judge_preflight_validates_three_complete_smoke_exports(tmp_path):
    input_dir = tmp_path / "judge-input-smoke"
    plan_path = tmp_path / "generation_plan.json"
    table5._write_json(plan_path, {
        "requests_sha256": "prepared-requests", "model": table5.MODEL,
        "price": {"input_usd_per_million": 2.0, "output_usd_per_million": 8.0},
    })
    hashes = {}
    for system in table5.PAPER_SYSTEMS:
        rows = [{
            "query_id": f"q{index}", "query_text": f"Question {index}?",
            "reference_answer": "Reference answer", "generated_output": "Generated answer",
            "context_data": [{"content": f"Context {chunk}"} for chunk in range(10)],
        } for index in range(6)]
        path = input_dir / system / "generated_questions_generation_results.json"
        table5._write_json(path, rows)
        hashes[system] = table5._sha256(path)
    table5._write_json(input_dir / "judge_input_manifest.json", {
        "status": "ready_for_judge", "scope": "smoke",
        "systems": list(table5.PAPER_SYSTEMS), "qa_per_system": 6,
        "requests_sha256": "prepared-requests", "answer_files_sha256": hashes,
    })
    judge_path = MODULE_PATH.parents[2] / "src" / "adaptive_chunking" / "paper" / "rag_eval.py"
    report = inspect_smoke(input_dir, plan_path, judge_path)
    assert report["judge_calls_expected"] == {
        "retrieval_completeness": 18, "correctness_geval": 18, "total": 36,
    }
    assert report["completeness_prompt_tokens"] > 0
    path = input_dir / table5.PAPER_SYSTEMS[0] / "generated_questions_generation_results.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(JudgePreflightError, match="changed after export"):
        inspect_smoke(input_dir, plan_path, judge_path)
