"""Prepare and resume paper Table 5 answer generation from frozen Phase G data.

This stage does not rerun chunking, indexing, or retrieval. The ``prepare``
command is local-only. ``generate`` requires an explicit paid-run flag and a
request-level budget. DeepEval judging is deliberately a separate stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PAPER_SYSTEMS = ("adaptive", "raw__langch_recurs_default", "raw__page")
MODEL = "gpt-4.1-2025-04-14"
ABSTAIN = "I don't know based on the provided context."
ANSWER_PROMPT = """You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context does not contain enough information to answer the question, respond with "I don't know based on the provided context."

Context:
{context_str}

Question: {question_str}

Answer:"""
INPUT_USD_PER_MILLION = 2.0
OUTPUT_USD_PER_MILLION = 8.0
PRICING_URL = "https://developers.openai.com/api/docs/models/gpt-4.1"
CHAT_FRAMING_RESERVE_TOKENS = 256


class PhaseIError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PhaseIError(f"Missing input: {path}")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_journal(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _prompt(question: str, results: list[dict[str, Any]]) -> str:
    context_str = ""
    for index, chunk in enumerate(results):
        doc_name = chunk["meta"]["doc_name"]
        context_str += f"<Document, ID = {index}, name = {doc_name}>\n"
        context_str += f"Content: {chunk['content']}\n"
        context_str += f"</Document, ID = {index}, name = {doc_name}>\n\n"
    return ANSWER_PROMPT.format(context_str=context_str.strip(), question_str=question.strip())


def _validate_qa(qas: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ids = [row["qa_id"] for row in qas]
    hashes = [row["qa_hash"] for row in qas]
    questions = [row["question"] for row in qas]
    per_doc = Counter(row["doc_id"] for row in qas)
    # The upstream DeepEval routine resumes by query_text, so repeated question
    # text would silently conflate otherwise distinct QA IDs.
    if (len(qas) != 99 or len(set(ids)) != 99 or len(set(hashes)) != 99
            or len(set(questions)) != 99
            or len(per_doc) != 33 or set(per_doc.values()) != {3}):
        raise PhaseIError("Frozen QA must have 99 unique IDs, hashes, and questions; 3 per each of 33 documents")
    if any(row.get("validation_status") not in {"valid", "valid_after_edit"} or not row.get("reference_answer")
           or not row.get("question") for row in qas):
        raise PhaseIError("Frozen QA contains an invalid or empty item")
    return {row["qa_id"]: row for row in qas}


def _validate_retrieval(rows: list[dict[str, Any]], qa_ids: set[str], system_id: str) -> None:
    if len(rows) != 99 or {row.get("qa_id") for row in rows} != qa_ids:
        raise PhaseIError(f"Incomplete or duplicate QA retrieval for {system_id}")
    for row in rows:
        results = row.get("results", [])
        if row.get("system_id") != system_id or len(results) != 10:
            raise PhaseIError(f"Invalid top-10 retrieval for {system_id}/{row.get('qa_id')}")
        if [item.get("rank") for item in results] != list(range(1, 11)):
            raise PhaseIError(f"Invalid ranks for {system_id}/{row['qa_id']}")
        chunk_ids = [item.get("meta", {}).get("chunk_id") for item in results]
        if len(set(chunk_ids)) != 10 or any(not item.get("content") for item in results):
            raise PhaseIError(f"Duplicate or empty retrieved chunk for {system_id}/{row['qa_id']}")


def build_requests(qa_path: Path, retrieval_dir: Path, evaluation_path: Path,
                   max_output_tokens: int = 512) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_output_tokens < 128:
        raise PhaseIError("max_output_tokens must be at least 128")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("status") != "complete" or evaluation.get("qa_count") != 99:
        raise PhaseIError("Full retrieval evaluation is not complete")
    if evaluation.get("input_qa_sha256") != _sha256(qa_path):
        raise PhaseIError("QA hash differs from the frozen retrieval evaluation")
    qas = _validate_qa(_read_jsonl(qa_path))
    import tiktoken
    encoding = tiktoken.get_encoding("o200k_base")
    requests: list[dict[str, Any]] = []
    input_hashes = {"qa": _sha256(qa_path), "evaluation": _sha256(evaluation_path)}
    for system_id in PAPER_SYSTEMS:
        path = retrieval_dir / f"{system_id}.jsonl"
        if _sha256(path) != evaluation.get("input_retrieval_sha256", {}).get(system_id):
            raise PhaseIError(f"Retrieval hash mismatch: {system_id}")
        input_hashes[system_id] = _sha256(path)
        rows = _read_jsonl(path)
        _validate_retrieval(rows, set(qas), system_id)
        for row in sorted(rows, key=lambda value: value["qa_id"]):
            qa = qas[row["qa_id"]]
            prompt = _prompt(qa["question"], row["results"])
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            requests.append({
                "system_id": system_id,
                "qa_id": qa["qa_id"],
                "doc_id": qa["doc_id"],
                "doc_name": qa["doc_name"],
                "domain": qa["domain"],
                "qa_hash": qa["qa_hash"],
                "question": qa["question"],
                "reference_answer": qa["reference_answer"],
                "context_data": row["results"],
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "context_sha256": _hash_json(row["results"]),
                "input_tokens_estimate": len(encoding.encode(prompt)),
                "request_hash": _hash_json({"model": MODEL, "prompt_sha256": prompt_hash,
                                             "max_output_tokens": max_output_tokens,
                                             "temperature": 0, "top_p": 1}),
            })
    if len(requests) != 297 or len({(r["system_id"], r["qa_id"]) for r in requests}) != 297:
        raise PhaseIError("Expected exactly 297 system-query generation requests")
    return requests, input_hashes


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    requests, input_hashes = build_requests(args.qa, args.retrieval_dir,
                                            args.evaluation, args.max_output_tokens)
    output = args.output_dir
    requests_path = output / "generation_requests.jsonl"
    previous_output_cap = None
    if requests_path.exists():
        old_manifest_path = output / "generation_plan.json"
        if not old_manifest_path.is_file():
            raise PhaseIError("Existing requests have no manifest; choose a new output directory")
        old = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        if old.get("input_hashes") != input_hashes:
            raise PhaseIError("Existing plan belongs to a different input or configuration")
        if old.get("max_output_tokens") != args.max_output_tokens:
            previous_output_cap = old.get("max_output_tokens")
            if (not getattr(args, "allow_output_cap_increase", False)
                    or previous_output_cap != 512 or args.max_output_tokens != 1024):
                raise PhaseIError("Output cap change requires an explicit 512-to-1024 increase")
    _write_jsonl(requests_path, requests)
    total_input = sum(row["input_tokens_estimate"] for row in requests)
    doc_tokens = Counter()
    doc_domains = {}
    for row in requests:
        doc_tokens[row["doc_id"]] += row["input_tokens_estimate"]
        doc_domains[row["doc_id"]] = row["domain"]
    largest = sorted(doc_tokens, key=lambda doc: (-doc_tokens[doc], doc))[0]
    second = next(doc for doc in sorted(doc_tokens, key=lambda doc: (-doc_tokens[doc], doc))
                  if doc_domains[doc] != doc_domains[largest])
    smoke_docs = [largest, second]
    smoke_requests = [row for row in requests if row["doc_id"] in smoke_docs]
    smoke_input = sum(row["input_tokens_estimate"] for row in smoke_requests)
    smoke_upper = ((smoke_input + len(smoke_requests) * CHAT_FRAMING_RESERVE_TOKENS) * INPUT_USD_PER_MILLION +
                   len(smoke_requests) * args.max_output_tokens * OUTPUT_USD_PER_MILLION) / 1_000_000
    upper = ((total_input + len(requests) * CHAT_FRAMING_RESERVE_TOKENS) * INPUT_USD_PER_MILLION +
             len(requests) * args.max_output_tokens * OUTPUT_USD_PER_MILLION) / 1_000_000
    result = {
        "schema_version": 1, "status": "prepared", "created_at": datetime.now(timezone.utc).isoformat(),
        "harness_sha256": _sha256(Path(__file__)),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version.split()[0],
        "model": MODEL, "systems": list(PAPER_SYSTEMS), "qa": 99, "requests": 297,
        "max_output_tokens": args.max_output_tokens, "input_hashes": input_hashes,
        "previous_output_cap": previous_output_cap or old.get("previous_output_cap") if requests_path.exists() else None,
        "chat_framing_reserve_tokens_per_request": CHAT_FRAMING_RESERVE_TOKENS,
        "requests_sha256": _sha256(requests_path),
        "answer_prompt_sha256": hashlib.sha256(ANSWER_PROMPT.encode()).hexdigest(),
        "input_tokens_estimate": total_input,
        "generation_upper_usd": round(upper, 6),
        "smoke_doc_ids": smoke_docs,
        "smoke_requests": len(smoke_requests),
        "smoke_input_tokens_estimate": smoke_input,
        "smoke_generation_upper_usd": round(smoke_upper, 6),
        "price": {"input_usd_per_million": INPUT_USD_PER_MILLION,
                  "output_usd_per_million": OUTPUT_USD_PER_MILLION,
                  "source": PRICING_URL, "checked_on": "2026-09-23"},
        "deviations": [f"A {args.max_output_tokens}-token output cap is added as a cost guard; the upstream generator leaves output uncapped.",
                       "Frozen human-reviewed QA replaces the paper's undocumented QA sample."],
        "judge_status": "not_started",
    }
    _write_json(output / "generation_plan.json", result)
    return result


def _journal_liability(path: Path) -> tuple[float, dict[str, dict[str, Any]], set[str]]:
    if not path.exists():
        return 0.0, {}, set()
    events = _read_jsonl(path)
    attempts: dict[str, dict[str, Any]] = {}
    completed: dict[str, dict[str, Any]] = {}
    nonbillable: set[str] = set()
    for event in events:
        if event["type"] == "attempt":
            if event["id"] in attempts:
                raise PhaseIError("Duplicate journal attempt id")
            attempts[event["id"]] = event
        elif event["type"] == "complete":
            if event["id"] not in attempts:
                raise PhaseIError("Journal completion without attempt")
            completed[event["id"]] = event
        elif event["type"] in {"aborted_preconnect", "rejected_rate_limit"}:
            if event["id"] not in attempts or event["id"] in completed:
                raise PhaseIError("Invalid nonbillable rejection")
            nonbillable.add(event["id"])
        else:
            raise PhaseIError("Unknown journal event")
    liability = sum(completed[key]["actual_usd"] if key in completed else value["upper_usd"]
                    for key, value in attempts.items() if key not in nonbillable)
    pending = {value["request_hash"] for key, value in attempts.items()
               if key not in completed and key not in nonbillable}
    return liability, completed, pending


def _reuse_completed_lower_cap_answer(request: dict[str, Any], plan: dict[str, Any],
                                      cache_dir: Path, completed_hashes: set[str]) -> bool:
    """Reuse only an already-complete answer from the same prompt and model.

    A response with finish_reason=stop is independent of a larger output ceiling.
    The smaller ceiling was only a cost guard, not a prompt change.
    """
    old_cap = plan.get("previous_output_cap")
    if old_cap != 512 or plan["max_output_tokens"] != 1024:
        return False
    old_hash = _hash_json({"model": MODEL, "prompt_sha256": request["prompt_sha256"],
                           "max_output_tokens": old_cap, "temperature": 0, "top_p": 1})
    old_path = cache_dir / f"{old_hash}.json"
    if old_hash not in completed_hashes or not old_path.is_file():
        return False
    old = json.loads(old_path.read_text(encoding="utf-8"))
    if (old.get("request_hash") != old_hash
            or old.get("prompt_sha256") != request["prompt_sha256"]
            or not old.get("answer") or not old.get("response_id")):
        raise PhaseIError(f"Invalid completed lower-cap cache: {old_path}")
    _write_json(cache_dir / f"{request['request_hash']}.json",
                {**old, "request_hash": request["request_hash"],
                 "reused_from_request_hash": old_hash, "max_output_tokens_used": old_cap})
    return True


def generate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_paid_run or args.budget_usd <= 0:
        raise PhaseIError("Paid generation requires --allow-paid-run and --budget-usd > 0")
    plan_path = args.output_dir / "generation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    requests_path = args.output_dir / "generation_requests.jsonl"
    if plan["requests_sha256"] != _sha256(requests_path):
        raise PhaseIError("Generation requests differ from the prepared manifest")
    if plan["harness_sha256"] != _sha256(Path(__file__)):
        raise PhaseIError("Harness changed since preparation; prepare a new plan")
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise PhaseIError("OPENAI_API_KEY is absent; load the local .env outside this script")
    from openai import OpenAI, RateLimitError
    requests = _read_jsonl(requests_path)
    if not args.smoke:
        selected = requests
    else:
        selected = [row for row in requests if row["doc_id"] in plan["smoke_doc_ids"]]
    cache_dir = args.output_dir / "generation_cache"
    cache_dir.mkdir(exist_ok=True)
    journal = args.output_dir / "generation_journal.jsonl"
    liability, completed, pending = _journal_liability(journal)
    completed_hashes = {event["request_hash"] for event in completed.values()}
    client = OpenAI(max_retries=0)
    count = 0
    for request in selected:
        key = request["request_hash"]
        cache_path = cache_dir / f"{key}.json"
        if not cache_path.is_file():
            _reuse_completed_lower_cap_answer(request, plan, cache_dir, completed_hashes)
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("request_hash") != key or cached.get("prompt_sha256") != request["prompt_sha256"]:
                raise PhaseIError(f"Cache collision: {key}")
            continue
        if key in pending:
            raise PhaseIError(f"Pending attempt for {key}; inspect the provider before retrying")
        upper = ((request["input_tokens_estimate"] + CHAT_FRAMING_RESERVE_TOKENS) * INPUT_USD_PER_MILLION +
                 plan["max_output_tokens"] * OUTPUT_USD_PER_MILLION) / 1_000_000
        if liability + upper > args.budget_usd:
            raise PhaseIError(f"Budget guard: estimated liability {liability:.4f} + {upper:.4f} exceeds ${args.budget_usd:.2f}")
        attempt_id = uuid.uuid4().hex
        _append_journal(journal, {"type": "attempt", "id": attempt_id, "request_hash": key,
                                  "upper_usd": upper, "time": datetime.now(timezone.utc).isoformat()})
        liability += upper
        try:
            response = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": request["prompt"]}],
                temperature=0, top_p=1, max_completion_tokens=plan["max_output_tokens"],
            )
        except RateLimitError as exc:
            if exc.status_code == 429 and "rate_limit_exceeded" in str(exc):
                _append_journal(journal, {"type": "rejected_rate_limit", "id": attempt_id,
                                          "request_hash": key,
                                          "time": datetime.now(timezone.utc).isoformat()})
                raise PhaseIError("TPM rate limit rejected this request; wait and resume") from None
            raise
        if len(response.choices) != 1 or response.choices[0].finish_reason != "stop":
            raise PhaseIError(f"Incomplete response for {key}; inspect before retry")
        answer = response.choices[0].message.content
        if not answer or not answer.strip() or response.usage is None:
            raise PhaseIError(f"Empty answer or missing token usage for {key}")
        cost = (response.usage.prompt_tokens * INPUT_USD_PER_MILLION +
                response.usage.completion_tokens * OUTPUT_USD_PER_MILLION) / 1_000_000
        _write_json(cache_path, {"request_hash": key, "prompt_sha256": request["prompt_sha256"],
                                 "answer": answer, "model": response.model,
                                 "response_id": response.id,
                                 "prompt_tokens": response.usage.prompt_tokens,
                                 "completion_tokens": response.usage.completion_tokens,
                                 "actual_usd": cost})
        _append_journal(journal, {"type": "complete", "id": attempt_id, "request_hash": key,
                                  "actual_usd": cost, "time": datetime.now(timezone.utc).isoformat()})
        liability += cost - upper
        count += 1
        if args.delay_seconds:
            time.sleep(args.delay_seconds)
    return {"status": "partial" if len(selected) < len(requests) else "generation_cached",
            "new_requests": count, "selected_requests": len(selected), "budget_liability_usd": round(liability, 6)}


def export_answers(args: argparse.Namespace) -> dict[str, Any]:
    """Create the original rag_eval input shape from verified cached answers."""
    plan = json.loads((args.output_dir / "generation_plan.json").read_text(encoding="utf-8"))
    request_path = args.output_dir / "generation_requests.jsonl"
    if plan["requests_sha256"] != _sha256(request_path):
        raise PhaseIError("Generation requests differ from the prepared manifest")
    requests = _read_jsonl(request_path)
    selected = ([row for row in requests if row["doc_id"] in plan["smoke_doc_ids"]]
                if args.smoke else requests)
    by_system: dict[str, list[dict[str, Any]]] = {system: [] for system in PAPER_SYSTEMS}
    for row in selected:
        cache_path = args.output_dir / "generation_cache" / f"{row['request_hash']}.json"
        if not cache_path.is_file():
            raise PhaseIError(f"Missing generated answer: {row['system_id']}/{row['qa_id']}")
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if (cache.get("request_hash") != row["request_hash"]
                or cache.get("prompt_sha256") != row["prompt_sha256"]
                or not cache.get("answer")):
            raise PhaseIError(f"Invalid answer cache: {cache_path}")
        by_system[row["system_id"]].append({
            "query_id": row["qa_id"], "query_text": row["question"],
            "reference_answer": row["reference_answer"],
            "reference_answer_doc_name": row["doc_name"],
            "generated_output": cache["answer"], "context_data": row["context_data"],
        })
    size = 6 if args.smoke else 99
    if any(len(rows) != size for rows in by_system.values()):
        raise PhaseIError("Answer export does not contain the same complete QA set for all systems")
    out_dir = args.output_dir / ("judge-input-smoke" if args.smoke else "judge-input-full")
    for system_id, rows in by_system.items():
        _write_json(out_dir / system_id / "generated_questions_generation_results.json", rows)
    manifest = {"status": "ready_for_judge", "scope": "smoke" if args.smoke else "full",
                "systems": list(PAPER_SYSTEMS), "qa_per_system": size,
                "requests_sha256": plan["requests_sha256"],
                "answer_files_sha256": {
                    system_id: _sha256(out_dir / system_id / "generated_questions_generation_results.json")
                    for system_id in PAPER_SYSTEMS}}
    _write_json(out_dir / "judge_input_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("prepare", help="Validate frozen inputs and estimate generation cost")
    pre.add_argument("--qa", type=Path, required=True)
    pre.add_argument("--retrieval-dir", type=Path, required=True)
    pre.add_argument("--evaluation", type=Path, required=True)
    pre.add_argument("--output-dir", type=Path, required=True)
    pre.add_argument("--max-output-tokens", type=int, default=512)
    pre.add_argument("--allow-output-cap-increase", action="store_true",
                     help="Permit only the audited 512-to-1024 cap increase after a length stop")
    gen = sub.add_parser("generate", help="Resume paid GPT-4.1 answer generation")
    gen.add_argument("--output-dir", type=Path, required=True)
    gen.add_argument("--allow-paid-run", action="store_true")
    gen.add_argument("--budget-usd", type=float, required=True)
    gen.add_argument("--smoke", action="store_true",
                     help="Generate all three systems for two representative documents")
    gen.add_argument("--delay-seconds", type=float, default=1.0)
    exp = sub.add_parser("export", help="Validate cached answers and export original rag_eval input")
    exp.add_argument("--output-dir", type=Path, required=True)
    exp.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "generate":
        result = generate(args)
    else:
        result = export_answers(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
