"""Budgeted, resumable execution of the paper's two Table 5 judge metrics.

The metric definitions and G-Eval template come from the original rag_eval.py
and pinned DeepEval. Calls are serialized, capped, and journaled before sending.
The 512-token response ceiling and serial execution are cost-control deviations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
from typing import Any
import uuid

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"

import tiktoken
from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from openai.types.chat import ChatCompletion

from phase_i_judge_preflight import _original_judge_definitions
from phase_i_table5 import (
    ABSTAIN, INPUT_USD_PER_MILLION, MODEL, OUTPUT_USD_PER_MILLION,
    PAPER_SYSTEMS, _append_journal, _hash_json, _journal_liability,
    _read_jsonl, _sha256, _write_json,
)

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_JUDGE = ROOT / "src" / "adaptive_chunking" / "paper" / "rag_eval.py"
ENCODING = tiktoken.get_encoding("o200k_base")
FRAMING_RESERVE = 1024


class JudgeError(RuntimeError):
    pass


def _load_original_judge():
    """Load rag_eval.py unchanged, without unrelated chunking dependencies."""
    source_root = ROOT / "src" / "adaptive_chunking"
    package = types.ModuleType("adaptive_chunking")
    package.__path__ = [str(source_root)]
    paper = types.ModuleType("adaptive_chunking.paper")
    paper.__path__ = [str(source_root / "paper")]
    sys.modules[package.__name__] = package
    sys.modules[paper.__name__] = paper
    name = "adaptive_chunking.paper.rag_eval"
    spec = importlib.util.spec_from_file_location(name, ORIGINAL_JUDGE)
    if spec is None or spec.loader is None:
        raise JudgeError("Cannot load original judge source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class BudgetedGPTModel(GPTModel):
    def __init__(self, output_dir: Path, budget_usd: float, max_output_tokens: int,
                 delay_seconds: float, client: OpenAI | None = None):
        # DeepEval 3.5.9 accepts the base alias, not its immutable snapshot.
        # The actual API request below always uses the pinned snapshot.
        super().__init__(model="gpt-4.1", temperature=0)
        self.model_name = MODEL
        self.output_dir = output_dir
        self.budget_usd = budget_usd
        self.max_output_tokens = max_output_tokens
        self.delay_seconds = delay_seconds
        self.client = client or OpenAI(max_retries=0)
        self.journal = output_dir / "judge_journal.jsonl"
        self.cache_dir = output_dir / "judge_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.generation_journal = output_dir / "generation_journal.jsonl"
        self.fatal_error: Exception | None = None

    def _call(self, prompt: str, kind: str, schema: type | None = None,
              top_logprobs: int | None = None) -> tuple[Any, float]:
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        key = _hash_json({"model": MODEL, "prompt_sha256": prompt_sha,
                          "kind": kind, "schema": schema.__name__ if schema else None,
                          "top_logprobs": top_logprobs,
                          "max_output_tokens": self.max_output_tokens})
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("request_hash") != key or cached.get("prompt_sha256") != prompt_sha:
                raise JudgeError(f"Judge cache collision: {key}")
            if kind == "structured":
                return schema.model_validate(cached["parsed"]), cached["actual_usd"]
            return ChatCompletion.model_validate(cached["completion"]), cached["actual_usd"]

        generation_liability, _, _ = _journal_liability(self.generation_journal)
        judge_liability, _, pending = _journal_liability(self.journal)
        if key in pending:
            raise JudgeError(f"Pending judge attempt for {key}; inspect before retry")
        estimated_input = len(ENCODING.encode(prompt)) + FRAMING_RESERVE
        upper = (estimated_input * INPUT_USD_PER_MILLION
                 + self.max_output_tokens * OUTPUT_USD_PER_MILLION) / 1_000_000
        if generation_liability + judge_liability + upper > self.budget_usd:
            raise JudgeError("Combined Table 5 budget guard: next judge call exceeds "
                             f"${self.budget_usd:.2f} (reserved "
                             f"${generation_liability + judge_liability:.4f})")
        attempt_id = uuid.uuid4().hex
        _append_journal(self.journal, {"type": "attempt", "id": attempt_id,
                                      "request_hash": key, "kind": kind,
                                      "prompt_sha256": prompt_sha,
                                      "upper_usd": upper,
                                      "time": datetime.now(timezone.utc).isoformat()})
        try:
            kwargs: dict[str, Any] = {
                "model": MODEL, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_completion_tokens": self.max_output_tokens,
            }
            if kind == "structured":
                completion = self.client.beta.chat.completions.parse(
                    **kwargs, response_format=schema)
            else:
                completion = self.client.chat.completions.create(
                    **kwargs, logprobs=True, top_logprobs=top_logprobs)
        except RateLimitError as exc:
            if exc.status_code == 429 and "rate_limit_exceeded" in str(exc):
                _append_journal(self.journal, {"type": "rejected_rate_limit",
                                              "id": attempt_id, "request_hash": key,
                                              "time": datetime.now(timezone.utc).isoformat()})
                self.fatal_error = JudgeError("Judge TPM rate limit; wait and resume")
                raise self.fatal_error from None
            self.fatal_error = exc
            raise
        except Exception as exc:
            self.fatal_error = exc
            raise

        if (len(completion.choices) != 1 or completion.choices[0].finish_reason != "stop"
                or completion.usage is None):
            self.fatal_error = JudgeError(f"Incomplete judge response for {key}; inspect before retry")
            raise self.fatal_error
        cost = (completion.usage.prompt_tokens * INPUT_USD_PER_MILLION
                + completion.usage.completion_tokens * OUTPUT_USD_PER_MILLION) / 1_000_000
        record: dict[str, Any] = {"request_hash": key, "prompt_sha256": prompt_sha,
                                  "kind": kind, "model": completion.model,
                                  "response_id": completion.id,
                                  "prompt_tokens": completion.usage.prompt_tokens,
                                  "completion_tokens": completion.usage.completion_tokens,
                                  "actual_usd": cost}
        if kind == "structured":
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                self.fatal_error = JudgeError(f"Missing structured judge output for {key}")
                raise self.fatal_error
            record["parsed"] = parsed.model_dump()
            result: Any = parsed
        else:
            record["completion"] = completion.model_dump(mode="json")
            result = completion
        _write_json(cache_path, record)
        _append_journal(self.journal, {"type": "complete", "id": attempt_id,
                                      "request_hash": key, "actual_usd": cost,
                                      "time": datetime.now(timezone.utc).isoformat()})
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return result, cost

    def generate(self, prompt: str, schema: type | None = None):
        if schema is None:
            raise JudgeError("Unexpected unstructured judge generation")
        return self._call(prompt, "structured", schema=schema)

    def generate_raw_response(self, prompt: str, top_logprobs: int = 5):
        return self._call(prompt, "logprobs", top_logprobs=top_logprobs)


def _metric_result(name: str, metric: Any) -> dict[str, Any]:
    return {"name": name, "score": metric.score, "threshold": metric.threshold,
            "success": metric.is_successful(), "reason": metric.reason,
            "error": getattr(metric, "error", None)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_paid_run or args.budget_usd <= 0:
        raise JudgeError("Paid judging requires --allow-paid-run and --budget-usd")
    if importlib.metadata.version("deepeval") != "3.5.9":
        raise JudgeError("Expected pinned DeepEval 3.5.9")
    if args.max_output_tokens != 512:
        raise JudgeError("Only the audited 512-token judge cap is supported")
    input_dir = args.output_dir / ("judge-input-smoke" if args.smoke else "judge-input-full")
    manifest_path = input_dir / "judge_input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = 6 if args.smoke else 99
    if (manifest.get("status") != "ready_for_judge"
            or manifest.get("scope") != ("smoke" if args.smoke else "full")
            or manifest.get("systems") != list(PAPER_SYSTEMS)
            or manifest.get("qa_per_system") != expected):
        raise JudgeError("Invalid judge input manifest")
    plan = json.loads((args.output_dir / "generation_plan.json").read_text(encoding="utf-8"))
    if manifest.get("requests_sha256") != plan.get("requests_sha256"):
        raise JudgeError("Judge inputs do not match generation plan")
    source_hash = _sha256(ORIGINAL_JUDGE)
    _, steps = _original_judge_definitions(ORIGINAL_JUDGE)
    run_dir = args.output_dir / ("judge-smoke" if args.smoke else "judge-full")
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {"scope": "smoke" if args.smoke else "full", "model": MODEL,
                "deepeval": "3.5.9", "max_output_tokens": args.max_output_tokens,
                "input_manifest_sha256": _sha256(manifest_path),
                "original_judge_sha256": source_hash,
                "judge_harness_sha256": _sha256(Path(__file__))}
    identity_path = run_dir / "run_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise JudgeError("Judge run identity changed; use a new output directory")
    else:
        _write_json(identity_path, identity)

    load_dotenv(ROOT / ".env", override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise JudgeError("OPENAI_API_KEY is absent")
    original = _load_original_judge()
    model = BudgetedGPTModel(args.output_dir, args.budget_usd,
                             args.max_output_tokens, args.delay_seconds)
    completeness = original.RetrievalCompletenessMetric(model)
    correctness = GEval(name="Correctness", model=model,
                        evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT,
                                           LLMTestCaseParams.ACTUAL_OUTPUT],
                        evaluation_steps=steps, async_mode=False)

    for system in PAPER_SYSTEMS:
        input_path = input_dir / system / "generated_questions_generation_results.json"
        if _sha256(input_path) != manifest["answer_files_sha256"][system]:
            raise JudgeError(f"Changed answer input: {system}")
        rows = json.loads(input_path.read_text(encoding="utf-8"))
        if len(rows) != expected or len({row["query_id"] for row in rows}) != expected:
            raise JudgeError(f"Incomplete or duplicate QA in {system}")
        output_path = run_dir / system / "generated_questions_evaluation_results.json"
        done = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else []
        done_ids = {row["query_id"] for row in done}
        if len(done_ids) != len(done) or not done_ids <= {row["query_id"] for row in rows}:
            raise JudgeError(f"Invalid resumed results: {system}")
        for row in rows:
            if row["query_id"] in done_ids:
                continue
            case = LLMTestCase(input=row["query_text"], actual_output=row["generated_output"],
                               expected_output=row["reference_answer"],
                               retrieval_context=[item["content"] for item in row["context_data"]])
            model.fatal_error = None
            completeness.measure(case)
            if model.fatal_error is not None:
                raise model.fatal_error
            metrics = [_metric_result("Retrieval Completeness", completeness)]
            if ABSTAIN not in row["generated_output"]:
                correctness.measure(case, _show_indicator=False)
                if model.fatal_error is not None:
                    raise model.fatal_error
                metrics.append(_metric_result("Correctness [GEval]", correctness))
            done.append({**row, "metrics": metrics,
                         "success": all(item["success"] for item in metrics)})
            _write_json(output_path, done)

    total = sum(len(json.loads((run_dir / system / "generated_questions_evaluation_results.json")
                               .read_text(encoding="utf-8"))) for system in PAPER_SYSTEMS)
    generation_liability, _, _ = _journal_liability(args.output_dir / "generation_journal.jsonl")
    judge_liability, _, pending = _journal_liability(args.output_dir / "judge_journal.jsonl")
    result = {**identity, "status": "complete" if total == expected * len(PAPER_SYSTEMS) else "partial",
              "rows": total, "judge_liability_usd": round(judge_liability, 6),
              "combined_liability_usd": round(generation_liability + judge_liability, 6),
              "pending_requests": len(pending)}
    _write_json(run_dir / "result_manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--allow-paid-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--delay-seconds", type=float, default=10)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
