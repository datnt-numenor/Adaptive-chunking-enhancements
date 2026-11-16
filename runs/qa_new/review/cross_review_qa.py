"""Cham cheo bo QA bang GPT-4.1 (model khac ho voi Claude da sinh va da review QA).

GPT-4.1 cham MU (khong thay nhan cua Claude) theo cung bang tieu chi Phan B, tren 2 ban:
  v1 = runs/qa_new/qa_new_frozen.jsonl          -> do dong thuan voi qa_audit.jsonl cua Claude (kappa)
  v2 = runs/qa_new/review/candidate_v2.jsonl    -> ban de xuat sau khi sua: GPT co chap nhan khong
Dau ra (runs/qa_new/review/):
  cross_review_gpt.jsonl, cross_review_summary.md, adjudication_queue.csv, cross_review_manifest.json
  cross_review_cache/ (moi phan hoi tra phi, chay lai khong goi API lan nua), cross_review_journal.jsonl
Khoa API: OPENAI_API_KEY_OFFICIAL neu co, khong thi OPENAI_API_KEY (doc tu .env, khong in ra).
Mac dinh goi THANG api.openai.com (khong qua gateway), du .env co OPENAI_BASE_URL.

Chay tu thu muc goc repo:
  python -X utf8 runs\\qa_new\\review\\cross_review_qa.py --smoke                     (chi uoc tinh)
  python -X utf8 runs\\qa_new\\review\\cross_review_qa.py --smoke --allow-paid-run    (6 cau x 2 ban)
  python -X utf8 runs\\qa_new\\review\\cross_review_qa.py --allow-paid-run            (99 cau x 2 ban)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MODEL = "gpt-4.1-2025-04-14"
INPUT_USD_PER_MILLION, OUTPUT_USD_PER_MILLION = 2.0, 8.0   # cung gia voi research/week3/phase_i_table5.py
MAX_OUTPUT_TOKENS = 800
CONTEXT_CHARS = 1500
OFFICIAL_BASE_URL = "https://api.openai.com/v1"

QA_DIR = Path("runs/qa_new")
REVIEW = QA_DIR / "review"
DATA = Path("data/clair/adi_parsed")

LABELS = {
    "grounded": ["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"],
    "missing_info_location": ["none", "adjacent", "same_page", "other_page", "not_in_document"],
    "actual_type": ["MULTI-PART", "STRUCTURE", "PARAPHRASED", "SIMPLE"],
    "unique_answer": ["UNIQUE", "AMBIGUOUS"],
    "translation": ["OK", "ISSUE", "N/A"],
    "chunking_sensitivity": ["EASY", "MEDIUM", "HARD"],
    "verdict": ["KEEP", "FIX", "DROP"],
}
BOOLEANS = ["evidence_complete", "self_contained", "forbidden_source"]
COMPARED = ["grounded", "evidence_complete", "self_contained", "actual_type", "unique_answer",
            "translation", "chunking_sensitivity", "forbidden_source", "verdict"]

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "qa_audit", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                **{k: {"type": "string", "enum": v} for k, v in LABELS.items()},
                **{k: {"type": "boolean"} for k in BOOLEANS},
                "unsupported_parts": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string"},
            },
            "required": [*LABELS, *BOOLEANS, "unsupported_parts", "reasoning"],
        },
    },
}

INSTRUCTIONS = """You are auditing ONE item of a question-answering set used to evaluate document retrieval.
Judge ONLY from the material shown. Do not use outside knowledge.

Requirements for a good item:
- The answer must be fully supported by the EVIDENCE QUOTES (verbatim excerpts of the document). The quotes are
  the ground truth used to score retrieval, so a statement that appears only in the surrounding context and not
  inside any quote counts as missing evidence.
- The question must be self-contained: a reader who has not seen the document must know which law, organisation,
  standard or report it is about. "the report"/"this document" is acceptable only if that report is named earlier
  in the same question.
- The question must have one correct answer within the document.
- The item must not rely on page headers, page footers or the table of contents. Footnotes are normal content.
- The document may be in French while question and answer are in English; then the translation must be faithful.

Labels (use exactly these values):
- grounded: SUPPORTED = the core answer to the question is fully supported by the quotes; PARTIAL = part of the
  core answer is not in the quotes; UNSUPPORTED = the core answer is not supported; CONTRADICTED = quotes say otherwise.
- evidence_complete: true only if every statement in the answer, including secondary details, is inside a quote.
- missing_info_location: where the statements missing from the quotes can be found in the material shown:
  none (nothing missing) | adjacent (immediately next to a quote) | same_page | other_page | not_in_document
  (not visible anywhere in the material shown).
- self_contained: true/false as defined above.
- actual_type: MULTI-PART = needs facts from at least two separate places; STRUCTURE = depends on a table, list,
  definition, or a pronoun/abbreviation defined elsewhere; PARAPHRASED = asks a specific detail using wording
  different from the document; SIMPLE = none of these.
- unique_answer: UNIQUE, or AMBIGUOUS if another passage shown gives a different or conflicting answer.
- translation: OK | ISSUE | N/A (the quotes are in English).
- chunking_sensitivity: EASY = one short contiguous passage sharing many words with the question, which any
  reasonable chunking keeps together; HARD = needs two or more distant passages, a table together with its header,
  or a pronoun/abbreviation defined in another block; MEDIUM = in between. Use the EVIDENCE LAYOUT facts.
- forbidden_source: true if the item relies on page headers, page footers or the table of contents.
- verdict: KEEP = all requirements met; FIX = a fixable problem (PARTIAL, incomplete evidence, not self-contained,
  ambiguous, translation issue, MULTI-PART supported by a single quote, ...); DROP = UNSUPPORTED or CONTRADICTED
  and not fixable by choosing other quotes, or forbidden_source.
- unsupported_parts: the exact answer fragments that are not supported by the quotes (empty list if none).
- reasoning: one to three sentences in English explaining the verdict.
"""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


_DOCS: dict[str, dict] = {}


def load_doc(name: str) -> dict:
    if name not in _DOCS:
        doc = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
        offsets, pos = {}, 0
        for page in sorted(doc["pages"], key=int):
            offsets[page] = pos
            pos += len(doc["pages"][page])
        if "".join(doc["pages"][p] for p in sorted(doc["pages"], key=int)) != doc["full_text"]:
            raise SystemExit(f"{name}: full_text khac cac trang noi lien - khong tinh duoc vi tri")
        doc["_offsets"] = offsets
        _DOCS[name] = doc
    return _DOCS[name]


def locate(doc: dict, item: dict, ev: dict) -> tuple[str, int, int]:
    page = str(ev["page"])
    text = doc["pages"].get(page)
    if text is None or text.count(ev["quote"]) != 1:
        raise SystemExit(f"{item['qa_id']}: quote khong xuat hien dung 1 lan tren trang {page}")
    start = text.find(ev["quote"])
    return page, start, start + len(ev["quote"])


def build_prompt(item: dict) -> str:
    doc = load_doc(item["doc_name"])
    spans = []                                      # (page, local_start, local_end, global_start, global_end)
    for ev in item["evidence"]:
        page, s, e = locate(doc, item, ev)
        g = doc["_offsets"][page] + s
        spans.append((page, s, e, g, g + len(ev["quote"])))

    lines = [INSTRUCTIONS, f"DOCUMENT TITLE: {item['doc_name'].split('_', 1)[-1]}",
             f"QUESTION: {item['question']}", f"ANSWER: {item['answer']}", "EVIDENCE QUOTES:"]
    for i, ev in enumerate(item["evidence"], 1):
        lines.append(f'[{i}] page {ev["page"]}: "{ev["quote"]}"')

    ordered = sorted(spans, key=lambda x: x[3])
    splits = doc["split_points"]
    inside = sum(1 for _, _, _, g, h in ordered for p in splits if g < p < h)
    gaps = [ordered[i + 1][3] - ordered[i][4] for i in range(len(ordered) - 1)]
    between = sum(1 for i in range(len(ordered) - 1) for p in splits if ordered[i][4] <= p <= ordered[i + 1][3])
    lines += ["EVIDENCE LAYOUT (computed from the parser):",
              f"- quotes: {len(spans)}; pages: {sorted({int(s[0]) for s in spans})}",
              f"- characters between consecutive quotes: {gaps if gaps else 'n/a'}",
              f"- parser block boundaries inside quotes: {inside}; between quotes: {between}"]

    lines.append(f"CONTEXT AROUND EACH QUOTE (+/-{CONTEXT_CHARS} characters, quotes marked <<< >>>):")
    by_page: dict[str, list] = {}
    for page, s, e, _, _ in spans:
        by_page.setdefault(page, []).append((s, e))
    for page in sorted(by_page, key=int):
        text = doc["pages"][page]
        windows = []
        for s, e in sorted(by_page[page]):
            lo, hi = max(0, s - CONTEXT_CHARS), min(len(text), e + CONTEXT_CHARS)
            if windows and lo <= windows[-1][1]:
                windows[-1][1] = max(windows[-1][1], hi)
            else:
                windows.append([lo, hi])
        for lo, hi in windows:
            marks = sorted((s, e) for s, e in by_page[page] if lo <= s and e <= hi)
            out, cur = [], lo
            for s, e in marks:
                out += [text[cur:s], "<<<", text[s:e], ">>>"]
                cur = e
            out.append(text[cur:hi])
            lines += [f"--- page {page} ---", "".join(out)]

    terms = re.findall(r"\d+(?:[.,]\d+)*", item["answer"]) + re.findall(r"\b[A-Z]{2,}\b", item["answer"])
    terms = [t for t in dict.fromkeys(terms) if len(t) >= 2]
    full, taken, snippets = doc["full_text"], [(g, h) for *_, g, h in spans], []
    for term in terms:
        per_term = 0
        for m in re.finditer(re.escape(term), full):
            lo, hi = max(0, m.start() - 250), min(len(full), m.end() + 250)
            if any(lo < h and g < hi for g, h in taken):
                continue
            taken.append((lo, hi))
            snippets.append(f'[term "{term}"] ...{full[lo:hi]}...')
            per_term += 1
            if per_term == 2 or len(snippets) == 4:
                break
        if len(snippets) == 4:
            break
    lines.append("OTHER PASSAGES OF THE DOCUMENT CONTAINING NUMBERS/ACRONYMS FROM THE ANSWER (for the uniqueness check):")
    lines += snippets or ["(none found)"]
    return "\n".join(lines)


def count_tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:                                            # noqa: BLE001 - uoc tinh tho neu thieu tiktoken
        return len(text) // 3


def make_client(args):
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    env_name = "OPENAI_API_KEY_OFFICIAL" if os.environ.get("OPENAI_API_KEY_OFFICIAL") else "OPENAI_API_KEY"
    if not os.environ.get(env_name):
        raise SystemExit("Khong tim thay OPENAI_API_KEY_OFFICIAL hoac OPENAI_API_KEY trong .env")
    print(f"Dung khoa tu bien {env_name} (khong in gia tri); base URL: {args.base_url}")
    return OpenAI(api_key=os.environ[env_name], base_url=args.base_url, max_retries=0, timeout=120)


def call_with_retry(client, request: dict):
    import openai
    for attempt, wait in enumerate([2, 4, 8, 16, 32, None], 1):
        try:
            return client.chat.completions.create(**request)
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                openai.InternalServerError) as error:
            if wait is None:
                raise
            print(f"  loi tam thoi ({type(error).__name__}), thu lai sau {wait}s (lan {attempt})")
            time.sleep(wait)


def last_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for row in read_jsonl(path):
        rows[row["qa_id"]] = row                                   # dong cuoi cung cua moi qa_id thang
    return rows


def kappa(a: list, b: list) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def fmt(x: float) -> str:
    return "n/a" if isinstance(x, float) and math.isnan(x) else f"{x:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", default="v1,v2")
    parser.add_argument("--smoke", action="store_true", help="chi 2 tai lieu dau (6 cau moi ban)")
    parser.add_argument("--allow-paid-run", action="store_true")
    parser.add_argument("--base-url", default=OFFICIAL_BASE_URL)
    parser.add_argument("--request-delay", type=float, default=0.5)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()

    frozen, manifest = QA_DIR / "qa_new_frozen.jsonl", QA_DIR / "qa_new_frozen_manifest.json"
    v1_sha = sha256_file(frozen)
    if v1_sha != json.loads(manifest.read_text(encoding="utf-8"))["sha256"]:
        raise SystemExit("sha256 cua qa_new_frozen.jsonl khong khop manifest - dung lai")
    sources = {"v1": frozen, "v2": REVIEW / "candidate_v2.jsonl"}
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    for t in targets:
        if t not in sources or not sources[t].exists():
            raise SystemExit(f"Thieu dau vao cho {t}: {sources.get(t)}")
    items = {t: read_jsonl(sources[t]) for t in targets}
    if args.smoke:
        keep_docs = sorted({r["qa_id"].split("-")[0] for r in items[targets[0]]})[:2]
        items = {t: [r for r in rows if r["qa_id"].split("-")[0] in keep_docs] for t, rows in items.items()}

    cache = REVIEW / "cross_review_cache"
    cache.mkdir(parents=True, exist_ok=True)
    plan = []
    for t in targets:
        for item in sorted(items[t], key=lambda r: r["qa_id"]):
            prompt = build_prompt(item)
            request = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                       "top_p": 1, "max_completion_tokens": MAX_OUTPUT_TOKENS, "response_format": SCHEMA}
            key = sha256_text(json.dumps(request, sort_keys=True, ensure_ascii=False))
            plan.append({"target": t, "item": item, "request": request, "key": key,
                         "path": cache / f"{key}.json", "tokens": count_tokens(prompt) + 400})

    todo = [p for p in plan if not p["path"].exists()]
    upper = sum(p["tokens"] * INPUT_USD_PER_MILLION + MAX_OUTPUT_TOKENS * OUTPUT_USD_PER_MILLION
                for p in todo) / 1e6
    print(f"{len(plan)} yeu cau ({', '.join(f'{t}: {len(items[t])}' for t in targets)}); da co cache: "
          f"{len(plan) - len(todo)}; can goi: {len(todo)}; ~{sum(p['tokens'] for p in todo):,} token vao; "
          f"chi phi toi da ~${upper:.2f}")
    if todo and not args.allow_paid_run:
        print("Chi uoc tinh. Them --allow-paid-run de goi API.")
        return

    journal = REVIEW / "cross_review_journal.jsonl"
    client = make_client(args) if todo else None
    for i, p in enumerate(todo, 1):
        completion = call_with_retry(client, p["request"])
        data = completion.model_dump()
        p["path"].write_text(json.dumps({"target": p["target"], "qa_id": p["item"]["qa_id"], "request_key": p["key"],
                                         "time": datetime.now(timezone.utc).isoformat(), "response": data},
                                        ensure_ascii=False), encoding="utf-8")
        usage = data.get("usage") or {}
        append_jsonl(journal, {"time": datetime.now(timezone.utc).isoformat(), "target": p["target"],
                               "qa_id": p["item"]["qa_id"], "request_key": p["key"], "model": data.get("model"),
                               "prompt_tokens": usage.get("prompt_tokens"),
                               "completion_tokens": usage.get("completion_tokens")})
        print(f"  [{i}/{len(todo)}] {p['target']} {p['item']['qa_id']} xong")
        time.sleep(args.request_delay)

    # --- Tong hop tu cache ---
    results, invalid, models, fingerprints, cost = {}, [], Counter(), Counter(), 0.0
    for p in plan:
        cached = json.loads(p["path"].read_text(encoding="utf-8"))["response"]
        models[cached.get("model")] += 1
        fingerprints[cached.get("system_fingerprint")] += 1
        usage = cached.get("usage") or {}
        cost += ((usage.get("prompt_tokens") or 0) * INPUT_USD_PER_MILLION
                 + (usage.get("completion_tokens") or 0) * OUTPUT_USD_PER_MILLION) / 1e6
        choice = cached["choices"][0]
        try:
            labels = json.loads(choice["message"]["content"])
            bad = [k for k, v in LABELS.items() if labels.get(k) not in v] + \
                  [k for k in BOOLEANS if not isinstance(labels.get(k), bool)]
            if bad or choice.get("finish_reason") != "stop":
                raise ValueError(f"nhan sai: {bad}, finish_reason={choice.get('finish_reason')}")
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            invalid.append((p["target"], p["item"]["qa_id"], str(error)))
            continue
        results[(p["target"], p["item"]["qa_id"])] = labels

    with (REVIEW / "cross_review_gpt.jsonl").open("w", encoding="utf-8") as handle:
        for (t, qa_id), labels in sorted(results.items()):
            handle.write(json.dumps({"target": t, "qa_id": qa_id, "reviewer": MODEL, **labels},
                                    ensure_ascii=False) + "\n")

    claude = last_rows(REVIEW / "qa_audit.jsonl")
    v1_rows = {r["qa_id"]: r for r in read_jsonl(frozen)}
    md = ["# Cham cheo bo QA bang GPT-4.1", "",
          f"- Thoi gian: {started} -> {datetime.now(timezone.utc).isoformat()}",
          f"- Model yeu cau: `{MODEL}`; model tra ve: {dict(models)}; system_fingerprint: {dict(fingerprints)}",
          f"- sha256 v1: `{v1_sha}`" + (f"; candidate_v2: `{sha256_file(sources['v2'])}`" if "v2" in targets else ""),
          f"- So yeu cau: {len(plan)}; phan hoi hop le: {len(results)}; khong hop le: {len(invalid)}",
          f"- Chi phi thuc te (tinh tu usage): ${cost:.4f}", "- GPT-4.1 cham mu: khong duoc thay nhan cua Claude.",
          "- Kappa = n/a khi ca hai ben chi dung mot gia tri nhan (khong tinh duoc).", ""]

    if "v1" in targets:
        ids = [q for (t, q) in results if t == "v1" and q in claude]
        md += ["## v1: dong thuan giua Claude va GPT-4.1 (cung mot ban QA)", "",
               f"So cau so sanh duoc: {len(ids)}", "", "| Tieu chi | Trung nhan | Kappa |", "|---|---:|---:|"]
        for field in COMPARED:
            a = [claude[q].get(field) for q in ids]
            b = [results[("v1", q)].get(field) for q in ids]
            agree = sum(x == y for x, y in zip(a, b))
            md.append(f"| {field} | {agree}/{len(ids)} | {fmt(kappa(a, b))} |")
        a = [claude[q].get("verdict") == "KEEP" for q in ids]
        b = [results[("v1", q)].get("verdict") == "KEEP" for q in ids]
        md += ["", f"KEEP so voi can sua (FIX/DROP): trung {sum(x == y for x, y in zip(a, b))}/{len(ids)}, "
                   f"kappa {fmt(kappa(a, b))}", "",
               "Bang cheo verdict (hang = Claude, cot = GPT-4.1):", "", "| Claude \\ GPT | KEEP | FIX | DROP |",
               "|---|---:|---:|---:|"]
        for c in LABELS["verdict"]:
            row = [sum(1 for q in ids if claude[q].get("verdict") == c and results[("v1", q)]["verdict"] == g)
                   for g in LABELS["verdict"]]
            md.append(f"| {c} | " + " | ".join(map(str, row)) + " |")
        md.append("")

    queue = []
    if "v2" in targets:
        v2_ids = [q for (t, q) in results if t == "v2"]
        verdicts = Counter(results[("v2", q)]["verdict"] for q in v2_ids)
        md += ["## v2 de xuat: GPT-4.1 co chap nhan ban da sua khong", "",
               f"KEEP {verdicts['KEEP']}, FIX {verdicts['FIX']}, DROP {verdicts['DROP']} / {len(v2_ids)}", ""]
        sens_c, sens_g = [], []
        for q in v2_ids:
            row = claude.get(q, {})
            sens_c.append(row.get("chunking_sensitivity_after_fix") or row.get("chunking_sensitivity"))
            sens_g.append(results[("v2", q)]["chunking_sensitivity"])
        md += [f"Do nhay voi chunking (B7) tren v2: Claude {dict(Counter(sens_c))}; GPT-4.1 {dict(Counter(sens_g))}; "
               f"trung {sum(x == y for x, y in zip(sens_c, sens_g))}/{len(v2_ids)}, kappa {fmt(kappa(sens_c, sens_g))}",
               ""]
        v2_rows = {r["qa_id"]: r for r in read_jsonl(sources["v2"])}
        for q in sorted(r["qa_id"] for r in items["v2"]):          # chi cac cau da dua vao lan chay nay
            reasons = []
            g = results.get(("v2", q))
            if g and g["verdict"] != "KEEP":
                reasons.append(f"gpt_v2_{g['verdict']}")
            old = v1_rows.get(q, {})
            for field in ("question", "answer"):
                if v2_rows[q].get(field) != old.get(field):
                    reasons.append(f"{field}_changed")
            if ("v2", q) not in results:
                reasons.append("gpt_v2_missing_or_invalid")
            if reasons:
                queue.append({"qa_id": q, "reasons": ";".join(reasons),
                              "gpt_verdict": g["verdict"] if g else "",
                              "gpt_unsupported_parts": " | ".join(g["unsupported_parts"]) if g else "",
                              "gpt_reasoning": g["reasoning"] if g else "",
                              "claude_verdict_v1": claude.get(q, {}).get("verdict", ""),
                              "claude_reasoning": claude.get(q, {}).get("reasoning", "")})
        with (REVIEW / "adjudication_queue.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["qa_id", "reasons", "gpt_verdict", "gpt_unsupported_parts", "gpt_reasoning",
                      "claude_verdict_v1", "claude_reasoning"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(queue)
        md += [f"## Hang doi phan xu: {len(queue)} cau (adjudication_queue.csv)", "",
               "Gom: cau GPT-4.1 khong chap nhan tren v2, cau doi chu question/answer, cau thieu phan hoi hop le.", ""]
        md += [f"- {r['qa_id']}: {r['reasons']}" for r in queue]
        md.append("")

    if invalid:
        md += ["## Phan hoi khong hop le", ""] + [f"- {t} {q}: {e}" for t, q, e in invalid]
    (REVIEW / "cross_review_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (REVIEW / "cross_review_manifest.json").write_text(json.dumps({
        "status": "complete" if not invalid and len(results) == len(plan) else "incomplete",
        "mode": "smoke" if args.smoke else "full", "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "model_requested": MODEL, "models_returned": dict(models), "system_fingerprints": dict(fingerprints),
        "params": {"temperature": 0, "top_p": 1, "max_completion_tokens": MAX_OUTPUT_TOKENS,
                   "context_chars": CONTEXT_CHARS, "base_url": args.base_url},
        "instructions_sha256": sha256_text(INSTRUCTIONS), "schema_sha256": sha256_text(json.dumps(SCHEMA)),
        "script_sha256": sha256_file(Path(__file__)), "inputs": {t: {"path": str(sources[t]),
                                                                      "sha256": sha256_file(sources[t])} for t in targets},
        "claude_audit_sha256": sha256_file(REVIEW / "qa_audit.jsonl"),
        "requests": len(plan), "valid": len(results), "invalid": len(invalid), "cost_usd": round(cost, 6),
        "adjudication_queue": len(queue),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    if sha256_file(frozen) != v1_sha:
        raise SystemExit("CANH BAO: qa_new_frozen.jsonl da thay doi trong luc chay!")
    print(f"Hop le {len(results)}/{len(plan)}; chi phi ~${cost:.4f}; hang doi phan xu: {len(queue)} cau")
    print(f"Xem {REVIEW / 'cross_review_summary.md'}")


if __name__ == "__main__":
    main()
