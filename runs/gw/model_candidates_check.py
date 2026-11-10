"""Chon model cho buoc sinh cau tra loi / cham diem Table 5 khi PHAT TRIEN cai tien (khong so voi cot cua paper).

Dung 18 prompt smoke THAT cua harness (cung prompt, temperature 0, top_p 1, toi da 512 token ra) va do cho moi model:
  - phien ban tra ve (response.model) co dung ten da ghim khong
  - do on dinh: chay 2 lan -> ty le trung nguyen van, do giong, cung tu choi
  - ty le tu choi ("I don't know based on the provided context.")
  - F1 tu voi dap an chuan cua bo QA (chi la chi bao tho, khong thay the judge)
  - do giong voi cau tra loi GPT-4.1 chinh thuc da co (doc cache, khong tra tien lai)
  - co ho tro logprobs (G-Eval cua DeepEval can) va json_schema (Retrieval Completeness can) khong
  - chi phi thuc te va do tre
Moi phan hoi duoc cache trong --output-dir; chay lai khong tra tien lan nua.

  python -X utf8 model_candidates_check.py --plan-dir %R%\\t5_gpt41_official
      --reference-dirs %R%\\t5_gpt41_official %R%\\t5_gpt41_official_rerun
      --env %K%\\official.env --models gpt-4.1-mini-2025-04-14 gpt-4.1-nano-2025-04-14 gpt-4o-mini-2024-07-18
      --output-dir %R%\\model_candidates                       (chi uoc tinh chi phi)
  ... them --allow-paid-run --budget-usd 1                     (goi API that)
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ABSTAIN = "I don't know based on the provided context."
MAX_OUTPUT_TOKENS = 512
FRAMING = 256
# Gia tham khao (USD / 1 trieu token: vao, ra) theo bang gia OpenAI - KIEM TRA LAI tren trang OpenAI;
# co the ghi de bang --price model=vao,ra. Chi dung cho rao chan ngan sach va uoc tinh.
DEFAULT_PRICES = {"gpt-4.1": (2.00, 8.00), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40),
                  "gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm(text: str) -> str:
    return " ".join(text.split())


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def f1(answer: str, reference: str) -> float:
    a, r = words(answer), words(reference)
    if not a or not r:
        return 0.0
    common = sum(min(a.count(w), r.count(w)) for w in set(a))
    if common == 0:
        return 0.0
    precision, recall = common / len(a), common / len(r)
    return 2 * precision * recall / (precision + recall)


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def price_of(model: str, overrides: dict[str, tuple[float, float]]) -> tuple[float, float]:
    if model in overrides:
        return overrides[model]
    base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    if base in DEFAULT_PRICES:
        return DEFAULT_PRICES[base]
    sys.exit(f"Khong biet gia cua {model}: them --price {model}=VAO,RA (USD/1M token)")


def pair_stats(first: list[str], second: list[str]) -> dict:
    return {"exact": sum(norm(a) == norm(b) for a, b in zip(first, second)),
            "similarity": statistics.mean(similarity(a, b) for a, b in zip(first, second)),
            "abstain_agree": sum((ABSTAIN in a) == (ABSTAIN in b) for a, b in zip(first, second))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--reference-dirs", type=Path, nargs="*", default=[],
                        help="1-2 thu muc da co cau tra loi GPT-4.1 chinh thuc cho cung ke hoach")
    parser.add_argument("--env", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="ten model co ngay phien ban")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--price", action="append", default=[], help="model=vao,ra (USD/1M token)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compare-output-dir", type=Path,
                        help="thu muc ket qua cua script nay da chay voi API khac (vd chinh thuc) cho CUNG model: "
                             "so cau tra loi cung model giua hai nha cung cap")
    parser.add_argument("--allow-paid-run", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=0.0)
    args = parser.parse_args()
    if args.runs < 2:
        sys.exit("Can it nhat --runs 2 de do do on dinh")

    overrides = {}
    for item in args.price:
        name, values = item.split("=", 1)
        overrides[name] = tuple(float(v) for v in values.split(","))
    plan = json.loads((args.plan_dir / "generation_plan.json").read_text(encoding="utf-8"))
    requests = [json.loads(line) for line in (args.plan_dir / "generation_requests.jsonl").open(encoding="utf-8")
                if line.strip()]
    smoke = sorted((r for r in requests if r["doc_id"] in plan["smoke_doc_ids"]),
                   key=lambda r: (r["system_id"], r["qa_id"]))
    if len(smoke) != 18:
        sys.exit(f"Can 18 prompt smoke, co {len(smoke)}")

    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(model: str, run: int, kind: str, prompt_sha: str) -> Path:
        return cache_dir / f"{sha(json.dumps([model, run, kind, prompt_sha, MAX_OUTPUT_TOKENS]))}.json"

    todo, upper = [], 0.0
    for model in args.models:
        pin, pout = price_of(model, overrides)
        for run in range(args.runs):
            for r in smoke:
                if not cache_path(model, run, "answer", r["prompt_sha256"]).exists():
                    todo.append((model, run, r))
                    upper += ((r["input_tokens_estimate"] + FRAMING) * pin + MAX_OUTPUT_TOKENS * pout) / 1e6
        for kind in ("logprobs", "json_schema"):
            if not cache_path(model, 0, kind, kind).exists():
                todo.append((model, 0, kind))
                upper += (200 * pin + 50 * pout) / 1e6
    print(f"{len(args.models)} model x {args.runs} lan x 18 prompt; con {len(todo)} yeu cau chua co cache; "
          f"chi phi toi da ~${upper:.3f} (theo gia tham khao)")
    if todo and not args.allow_paid_run:
        print("Chi uoc tinh. Them --allow-paid-run --budget-usd <tran> de goi API.")
        return
    if todo and upper > args.budget_usd:
        sys.exit(f"Chi phi toi da ${upper:.3f} vuot --budget-usd {args.budget_usd}")

    host, client = None, None
    if todo:                                     # chi nap key khi thuc su goi API
        from dotenv import dotenv_values
        from openai import OpenAI
        env = {k: v for k, v in dotenv_values(args.env).items() if v}
        host = urlparse(env["OPENAI_BASE_URL"]).netloc
        print(f"Goi API qua {host} (khong in key)")
        client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env["OPENAI_BASE_URL"], max_retries=2, timeout=180)
    unavailable_path = args.output_dir / "unavailable_models.json"
    unavailable = json.loads(unavailable_path.read_text(encoding="utf-8")) if unavailable_path.exists() else {}
    for i, item in enumerate(todo, 1):
        model, run, payload = item
        if model in unavailable:
            continue
        started = time.time()
        if payload == "logprobs":
            try:
                resp = client.chat.completions.create(model=model, temperature=0, max_completion_tokens=5,
                                                      messages=[{"role": "user", "content": "Reply with the digit 7."}],
                                                      logprobs=True, top_logprobs=5)
                content = resp.choices[0].logprobs.content if resp.choices[0].logprobs else None
                ok = bool(content) and bool(content[0].top_logprobs)
                record = {"supported": ok, "model": resp.model, "usage": resp.usage.model_dump()}
            except Exception as error:  # noqa: BLE001 - ghi lai loi de danh gia kha nang
                record = {"supported": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
            path = cache_path(model, 0, "logprobs", "logprobs")
        elif payload == "json_schema":
            schema = {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": {
                "type": "object", "additionalProperties": False, "required": ["level", "reason"],
                "properties": {"level": {"type": "integer"}, "reason": {"type": "string"}}}}}
            try:
                resp = client.chat.completions.create(model=model, temperature=0, max_completion_tokens=50,
                                                      messages=[{"role": "user", "content": "Rate 2 and give a reason."}],
                                                      response_format=schema)
                parsed = json.loads(resp.choices[0].message.content)
                record = {"supported": isinstance(parsed.get("level"), int), "model": resp.model,
                          "usage": resp.usage.model_dump()}
            except Exception as error:  # noqa: BLE001
                record = {"supported": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
            path = cache_path(model, 0, "json_schema", "json_schema")
        else:
            r = payload
            import openai
            try:
                resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": r["prompt"]}],
                                                      temperature=0, top_p=1, max_completion_tokens=MAX_OUTPUT_TOKENS)
            except (openai.NotFoundError, openai.PermissionDeniedError, openai.BadRequestError) as error:
                unavailable[model] = f"{type(error).__name__}: {str(error)[:300]}"
                unavailable_path.write_text(json.dumps(unavailable, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  [{i}/{len(todo)}] {model}: KHONG DUNG DUOC ({type(error).__name__}) - bo qua model nay")
                continue
            record = {"request_hash": r["request_hash"], "prompt_sha256": r["prompt_sha256"], "model": resp.model,
                      "answer": resp.choices[0].message.content or "", "finish_reason": resp.choices[0].finish_reason,
                      "system_fingerprint": resp.system_fingerprint, "usage": resp.usage.model_dump(),
                      "latency_s": round(time.time() - started, 2)}
            path = cache_path(model, run, "answer", r["prompt_sha256"])
        record.update({"requested_model": model, "run": run, "base_url_host": host,
                       "time": datetime.now(timezone.utc).isoformat()})
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        print(f"  [{i}/{len(todo)}] {model} lan {run} {payload if isinstance(payload, str) else payload['qa_id']} xong")

    # --- Tong hop ---
    references = []
    for ref_dir in args.reference_dirs:
        cached = {}
        for p in (ref_dir / "generation_cache").glob("*.json"):
            row = json.loads(p.read_text(encoding="utf-8"))
            cached[row["prompt_sha256"]] = row
        if all(r["prompt_sha256"] in cached for r in smoke):
            references.append([cached[r["prompt_sha256"]] for r in smoke])
        else:
            print(f"Bo qua {ref_dir}: chua du 18 cau tra loi smoke")
    rows = []
    if references:
        ref_answers = [[c["answer"] for c in ref] for ref in references]
        pin, pout = DEFAULT_PRICES["gpt-4.1"]
        row = {"model": "gpt-4.1 (chinh thuc, tu cache)", "returned": sorted({c["model"] for c in references[0]}),
               "abstain_rate": sum(ABSTAIN in a for a in ref_answers[0]) / 18,
               "f1_vs_reference": statistics.mean(f1(a, r["reference_answer"]) for a, r in zip(ref_answers[0], smoke)),
               "cost_usd_per_18": sum((c["prompt_tokens"] * pin + c["completion_tokens"] * pout) / 1e6 for c in references[0]),
               "logprobs": "da dung trong harness", "json_schema": "da dung trong harness"}
        if len(references) > 1:
            s = pair_stats(ref_answers[0], ref_answers[1])
            row.update({"self_exact": s["exact"], "self_similarity": s["similarity"], "self_abstain_agree": s["abstain_agree"]})
        rows.append(row)
    unavailable_path = args.output_dir / "unavailable_models.json"
    unavailable = json.loads(unavailable_path.read_text(encoding="utf-8")) if unavailable_path.exists() else {}
    for model in args.models:
        complete = all(cache_path(model, k, "answer", r["prompt_sha256"]).exists() for k in range(args.runs) for r in smoke)
        if not complete:
            rows.append({"model": model, "returned": "KHONG DUNG DUOC: " + unavailable.get(model, "thieu ket qua - chay lai"),
                         "abstain_rate": None, "f1_vs_reference": None, "logprobs": "-", "json_schema": "-"})
            continue
        runs = [[json.loads(cache_path(model, k, "answer", r["prompt_sha256"]).read_text(encoding="utf-8"))
                 for r in smoke] for k in range(args.runs)]
        answers = [[c["answer"] for c in run] for run in runs]
        s = pair_stats(answers[0], answers[1])
        pin, pout = price_of(model, overrides)
        probes = {k: json.loads(cache_path(model, 0, k, k).read_text(encoding="utf-8")) for k in ("logprobs", "json_schema")}
        row = {"model": model, "returned": sorted({c["model"] for run in runs for c in run}),
               "self_exact": s["exact"], "self_similarity": s["similarity"], "self_abstain_agree": s["abstain_agree"],
               "abstain_rate": sum(ABSTAIN in a for a in answers[0]) / 18,
               "f1_vs_reference": statistics.mean(f1(a, r["reference_answer"]) for a, r in zip(answers[0], smoke)),
               "truncated": sum(c["finish_reason"] != "stop" for run in runs for c in run),
               "latency_s": statistics.mean(c["latency_s"] for c in runs[0]),
               "cost_usd_per_18": sum((c["usage"]["prompt_tokens"] * pin + c["usage"]["completion_tokens"] * pout) / 1e6
                                      for c in runs[0]),
               "logprobs": probes["logprobs"].get("supported"), "json_schema": probes["json_schema"].get("supported")}
        if references:
            g = pair_stats(answers[0], [c["answer"] for c in references[0]])
            row.update({"vs_gpt41_similarity": g["similarity"], "vs_gpt41_abstain_agree": g["abstain_agree"]})
        if args.compare_output_dir:
            other = [args.compare_output_dir / "cache" / cache_path(model, k, "answer", r["prompt_sha256"]).name
                     for k in range(2) for r in smoke]
            if all(path.exists() for path in other):
                o = [json.loads(path.read_text(encoding="utf-8")) for path in other]
                o0, o1 = [c["answer"] for c in o[:18]], [c["answer"] for c in o[18:]]
                cross = pair_stats(answers[0], o0)
                own = pair_stats(o0, o1)
                row.update({"other_returned": sorted({c["model"] for c in o}),
                            "other_self_exact": own["exact"], "other_self_similarity": own["similarity"],
                            "cross_exact": cross["exact"], "cross_similarity": cross["similarity"],
                            "cross_abstain_agree": cross["abstain_agree"],
                            "cross_prompt_tokens_match": sum(a["usage"]["prompt_tokens"] == b["usage"]["prompt_tokens"]
                                                             for a, b in zip(runs[0], o[:18]))})
        rows.append(row)

    for row in rows:
        if "cost_usd_per_18" in row:
            row["cost_usd_full_generation_est"] = row["cost_usd_per_18"] * 297 / 18
    fields = ["model", "returned", "self_exact", "self_similarity", "self_abstain_agree", "abstain_rate",
              "f1_vs_reference", "vs_gpt41_similarity", "vs_gpt41_abstain_agree", "truncated", "latency_s",
              "cost_usd_per_18", "cost_usd_full_generation_est", "logprobs", "json_schema", "other_returned",
              "other_self_exact", "other_self_similarity", "cross_exact", "cross_similarity", "cross_abstain_agree",
              "cross_prompt_tokens_match"]
    with (args.output_dir / "candidates_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "candidates_summary.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(), "plan_requests_sha256": plan["requests_sha256"],
        "smoke_doc_ids": plan["smoke_doc_ids"], "runs": args.runs, "max_output_tokens": MAX_OUTPUT_TOKENS,
        "prices_usd_per_million": {m: price_of(m, overrides) for m in args.models}, "rows": rows,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2, ensure_ascii=False,
        default=str), encoding="utf-8")

    def fmt(value, digits=3):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value if value is not None else "-")
    print("\nmodel | tra ve | on dinh (trung/18, giong, cung tu choi) | ty le tu choi | F1 voi dap an | "
          "giong GPT-4.1 | cat cut | tre (s) | $/18 | $ full sinh | logprobs | json_schema")
    for row in rows:
        print(f"{row['model']} | {row['returned']} | {row.get('self_exact', '-')}/18, {fmt(row.get('self_similarity'))}, "
              f"{row.get('self_abstain_agree', '-')}/18 | {fmt(row['abstain_rate'], 2)} | {fmt(row['f1_vs_reference'])} | "
              f"{fmt(row.get('vs_gpt41_similarity'))} | {row.get('truncated', '-')} | {fmt(row.get('latency_s'), 1)} | "
              f"{fmt(row.get('cost_usd_per_18'), 4)} | {fmt(row.get('cost_usd_full_generation_est'), 2)} | "
              f"{row['logprobs']} | {row['json_schema']}")
    if args.compare_output_dir:
        print(f"\nDoi chieu cung model voi {args.compare_output_dir} (ben kia):")
        for row in rows:
            if "cross_exact" in row:
                print(f"  {row['model']}: ben kia tra ve {row['other_returned']} | ben kia tu trung {row['other_self_exact']}/18 "
                      f"(giong {row['other_self_similarity']:.3f}) | day tu trung {row['self_exact']}/18 "
                      f"(giong {row['self_similarity']:.3f}) | GIUA HAI BEN trung {row['cross_exact']}/18 "
                      f"(giong {row['cross_similarity']:.3f}), cung tu choi {row['cross_abstain_agree']}/18, "
                      f"prompt_tokens trung {row['cross_prompt_tokens_match']}/18")
            elif row.get("returned") and "KHONG" not in str(row["returned"]) and "chinh thuc" not in row["model"]:
                print(f"  {row['model']}: ben kia chua co du 2 lan chay cho model nay")
    print(f"\nDa ghi {args.output_dir / 'candidates_summary.csv'}")


if __name__ == "__main__":
    main()
