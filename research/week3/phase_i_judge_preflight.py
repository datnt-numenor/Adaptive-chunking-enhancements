"""Offline validation and cost planning for the original Table 5 judge.

This module never imports DeepEval, loads an API key, or calls a provider.
Its estimate is illustrative: DeepEval's internal G-Eval prompt and output
length are not bounded by the upstream evaluation code.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import tiktoken

from phase_i_table5 import ABSTAIN, PAPER_SYSTEMS, _sha256, _write_json


class JudgePreflightError(RuntimeError):
    pass


def _original_judge_definitions(path: Path) -> tuple[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    prompt = None
    steps = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RetrievalCompletenessMetric":
            for item in node.body:
                if isinstance(item, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "RETRIEVAL_COMPLETENESS_PROMPT"
                    for target in item.targets
                ):
                    prompt = ast.literal_eval(item.value)
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_rag_results_generated_questions":
            for item in ast.walk(node):
                if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == "GEval":
                    for keyword in item.keywords:
                        if keyword.arg == "evaluation_steps":
                            steps = ast.literal_eval(keyword.value)
    if not isinstance(prompt, str) or not isinstance(steps, list) or len(steps) != 3:
        raise JudgePreflightError("Could not locate the original judge prompt and G-Eval steps")
    return prompt, steps


def inspect_smoke(input_dir: Path, generation_plan: Path, original_judge: Path) -> dict[str, Any]:
    manifest_path = input_dir / "judge_input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = json.loads(generation_plan.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready_for_judge" or manifest.get("scope") != "smoke":
        raise JudgePreflightError("Judge input is not a completed smoke export")
    if manifest.get("requests_sha256") != plan.get("requests_sha256"):
        raise JudgePreflightError("Judge export belongs to another generation plan")
    if manifest.get("systems") != list(PAPER_SYSTEMS) or manifest.get("qa_per_system") != 6:
        raise JudgePreflightError("Judge export must contain six QA for each paper system")

    encoding = tiktoken.get_encoding("o200k_base")
    completeness_template, correctness_steps = _original_judge_definitions(original_judge)
    qa_ids: set[str] | None = None
    total_completeness_tokens = 0
    total_correctness_payload_tokens = 0
    row_stats: list[dict[str, Any]] = []
    correctness_calls = 0
    for system in PAPER_SYSTEMS:
        path = input_dir / system / "generated_questions_generation_results.json"
        if _sha256(path) != manifest.get("answer_files_sha256", {}).get(system):
            raise JudgePreflightError(f"Answer file changed after export: {system}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        ids = {item["query_id"] for item in rows}
        if len(rows) != 6 or len(ids) != 6 or (qa_ids is not None and ids != qa_ids):
            raise JudgePreflightError(f"Mismatched or duplicate QA in {system}")
        qa_ids = ids
        for item in rows:
            contexts = item["context_data"]
            if len(contexts) != 10 or any(not value.get("content") for value in contexts):
                raise JudgePreflightError(f"Invalid top-10 context: {system}/{item['query_id']}")
            if not item.get("reference_answer") or not item.get("generated_output"):
                raise JudgePreflightError(f"Missing answer: {system}/{item['query_id']}")
            prompt = completeness_template.format(
                context="\n\n".join(chunk["content"] for chunk in contexts),
                reference_answer=item["reference_answer"],
            )
            completeness_tokens = len(encoding.encode(prompt))
            total_completeness_tokens += completeness_tokens
            abstained = ABSTAIN in item["generated_output"]
            if not abstained:
                correctness_calls += 1
                # DeepEval wraps these fields in its own template. This is
                # payload-only, NOT an exact G-Eval prompt token count.
                payload = "\n".join(correctness_steps + [item["reference_answer"], item["generated_output"]])
                total_correctness_payload_tokens += len(encoding.encode(payload))
            row_stats.append({
                "system_id": system,
                "qa_id": item["query_id"],
                "answer_chars": len(item["generated_output"]),
                "abstained": abstained,
                "completeness_prompt_tokens": completeness_tokens,
            })

    calls = len(row_stats) + correctness_calls
    input_rate = plan["price"]["input_usd_per_million"]
    output_rate = plan["price"]["output_usd_per_million"]
    # The 256-token assumptions are planning examples, not API output caps.
    illustrative_input = total_completeness_tokens + total_correctness_payload_tokens + calls * 256
    illustrative_cost = (illustrative_input * input_rate + calls * 256 * output_rate) / 1_000_000
    return {
        "status": "offline_preflight_only",
        "judge_calls_expected": {"retrieval_completeness": len(row_stats),
                                 "correctness_geval": correctness_calls, "total": calls},
        "qa_per_system": 6,
        "model_intended": plan["model"],
        "input_hashes": {"generation_plan": _sha256(generation_plan),
                         "judge_input_manifest": _sha256(manifest_path),
                         "original_judge_source": _sha256(original_judge)},
        "completeness_prompt_sha256": hashlib.sha256(completeness_template.encode()).hexdigest(),
        "correctness_steps_sha256": hashlib.sha256(json.dumps(correctness_steps).encode()).hexdigest(),
        "completeness_prompt_tokens": total_completeness_tokens,
        "correctness_payload_only_tokens": total_correctness_payload_tokens,
        "illustrative_cost_usd_256_output_tokens_per_call": round(illustrative_cost, 6),
        "cost_warning": "Not a hard upper bound: upstream DeepEval G-Eval template, calls, and output are unbounded here.",
        "rows": row_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--generation-plan", type=Path, required=True)
    parser.add_argument("--original-judge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_smoke(args.input_dir, args.generation_plan, args.original_judge)
    _write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
