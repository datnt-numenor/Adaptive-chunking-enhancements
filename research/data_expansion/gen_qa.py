"""Build the QA test set for the new documents (data/clair_ext), following the QA v1/v2 procedure.

Steps (each writes into --work, never overwrites a frozen file):
  export    excerpt per document (<= 45k chars of consecutive pages, random start, seed 2026, as
            runs/export_for_chat.py) + prompt (runs/qa_new/prompt.txt, asking for 6 QAs = 2 per type);
            prints token counts and a cost estimate for --price-in/--price-out.     [free]
  generate  call the model (OpenAI-compatible, key from --env file, never printed); responses cached.   [PAID]
  validate  quotes must occur exactly once on their page (same tolerant matching as runs/validate_qa.py),
            no vague wording; spans in full_text are computed.                                          [free]
  review    a second model checks each QA against the excerpt (answerable, correct, evidence sufficient,
            self-contained, not about headers/TOC). Only "keep" QAs survive.                          [PAID]
  freeze    qa_frozen.jsonl (+ sha256 manifest) in the harness format (qa_id, doc_id, question, answer,
            evidence[{page, quote, source_start, source_end}]).                                          [free]

Usage (cmd):
  python -X utf8 research\\data_expansion\\gen_qa.py export   --work runs\\qa_ext_v1 --price-in 2 --price-out 8
  python -X utf8 research\\data_expansion\\gen_qa.py generate --work runs\\qa_ext_v1 --model gpt-4.1 ^
         --env D:\\Project\\keys\\gateway.env --price-in 2 --price-out 8 --budget-usd 5 --allow-paid-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("o200k_base")
MAX_CHARS, SEED, N_QA = 45_000, 2026, 6
VAGUE = re.compile(r"\b(this|the) (document|report|paper|text|section|article|study)\b|\babove\b", re.I)
VARIANTS = ["'’‘`´ʼ", '"“”«»„', "-‐‑–—"]


def ntok(s: str) -> int:
    return len(ENC.encode(s, disallowed_special=()))


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def doc_title(stem: str) -> str:
    return stem.split("_", 1)[-1]


def prompt_text() -> str:
    base = Path("runs/qa_new/prompt.txt").read_text(encoding="utf-8")
    base = base.replace("Write exactly 3 question-answer pairs", f"Write exactly {N_QA} question-answer pairs")
    base = base.replace("Use these three types, one each:", "Use these three types, TWO of each (in this order):")
    # the JSON example in the template shows 3 items; state the count explicitly
    return base.replace("Return ONLY this JSON, no commentary:",
                        f"Return ONLY this JSON (with {N_QA} items in \"qas\"), no commentary:")


# ---------------------------------------------------------------- export
def export(args) -> None:
    rng = random.Random(SEED)
    out = args.work / "export"
    out.mkdir(parents=True, exist_ok=True)
    prompt = prompt_text()
    (args.work / "prompt.txt").write_text(prompt, encoding="utf-8")
    index, tot_in = [], 0
    for path in sorted(args.parsed.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        nums = sorted(doc["pages"], key=int)
        total = sum(len(doc["pages"][n]) for n in nums)
        start = 0
        if total > MAX_CHARS:
            cands = [k for k in range(len(nums)) if sum(len(doc["pages"][n]) for n in nums[k:]) >= MAX_CHARS * 0.8]
            start = rng.choice(cands)
        chosen, size = [], 0
        for n in nums[start:]:
            if chosen and size + len(doc["pages"][n]) > MAX_CHARS:
                break
            chosen.append(n)
            size += len(doc["pages"][n])
        body = "".join(f"\n<<<PAGE {n}>>>\n{doc['pages'][n]}" for n in chosen)
        text = f"DOCUMENT: {doc_title(path.stem)}\n{body}"
        (out / f"{path.stem}.txt").write_text(text, encoding="utf-8")
        n_in = ntok(prompt + "\n" + text)
        tot_in += n_in
        index.append({"doc_name": path.stem, "pages": f"{chosen[0]}-{chosen[-1]}", "n_pages_total": len(nums),
                      "chars": size, "full_doc": total <= MAX_CHARS, "input_tokens": n_in})
    (args.work / "export_index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    out_tok = len(index) * 2_500
    cost = tot_in / 1e6 * args.price_in + out_tok / 1e6 * args.price_out
    print(f"{len(index)} documents, input {tot_in:,} tokens, output ~{out_tok:,} tokens (estimate)")
    print(f"generate cost estimate: ${cost:.2f} at ${args.price_in}/M in, ${args.price_out}/M out")
    print(f"review cost estimate (same prices, excerpt re-sent once per document): ~${cost * 1.1:.2f}")


# ---------------------------------------------------------------- API helpers
def client_from_env(env_path: Path):
    from dotenv import dotenv_values
    from openai import OpenAI

    vals = dotenv_values(env_path)
    if "OPENAI_API_KEY" not in vals:
        sys.exit("env file has no OPENAI_API_KEY")
    return OpenAI(api_key=vals["OPENAI_API_KEY"], base_url=vals.get("OPENAI_BASE_URL") or None, timeout=300)


def call_json(client, model: str, content: str, cache: Path, max_tokens: int) -> dict:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    last = None
    use_rf = model.startswith("gpt")  # response_format only for OpenAI models; others: fenced JSON is stripped
    for attempt in range(4):
        try:
            kw = {"response_format": {"type": "json_object"}} if use_rf else {}
            r = client.chat.completions.create(model=model, temperature=0, max_tokens=max_tokens,
                                               messages=[{"role": "user", "content": content}], **kw)
            text = (r.choices[0].message.content or "").strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
            if not text.startswith("{") and "{" in text:
                text = text[text.index("{"): text.rindex("}") + 1]
            rec = {"model_requested": model, "model_returned": r.model, "content": text,
                   "usage": {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens},
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}
            json.loads(rec["content"])  # must parse
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            return rec
        except Exception as e:  # retry transient errors / invalid JSON
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{cache.name}: {type(last).__name__}: {last}")


def spent(dirs: list[Path], pin: float, pout: float) -> float:
    total = 0.0
    for d in dirs:
        for f in d.glob("*.json") if d.exists() else []:
            u = json.loads(f.read_text(encoding="utf-8"))["usage"]
            total += u["in"] / 1e6 * pin + u["out"] / 1e6 * pout
    return total


def guard(args):
    if not args.allow_paid_run:
        sys.exit("paid step: add --allow-paid-run (after the user approved the estimate)")
    if not args.env:
        sys.exit("--env <file> required")


# ---------------------------------------------------------------- generate
def generate(args) -> None:
    guard(args)
    client = client_from_env(args.env)
    prompt = (args.work / "prompt.txt").read_text(encoding="utf-8")
    raw = args.work / "raw"
    docs = sorted((args.work / "export").glob("*.txt"))
    if args.limit:
        docs = docs[: args.limit]

    def one(p: Path):
        if spent([raw], args.price_in, args.price_out) > args.budget_usd:
            return f"SKIP budget {p.stem}"
        rec = call_json(client, args.model, prompt + "\n" + p.read_text(encoding="utf-8"), raw / f"{p.stem}.json", 6000)
        return f"ok {rec['model_returned']} in={rec['usage']['in']} out={rec['usage']['out']} {p.stem}"

    with ThreadPoolExecutor(args.workers) as ex:
        for msg in ex.map(one, docs):
            print(msg, flush=True)
    print(f"spent so far (generate): ${spent([raw], args.price_in, args.price_out):.2f}")


# ---------------------------------------------------------------- validate
def _char(ch):
    for g in VARIANTS:
        if ch in g:
            return "[" + re.escape(g) + "]"
    return re.escape(ch)


def locate(page_text: str, quote: str):
    hits = [m.start() for m in re.finditer(re.escape(quote), page_text)]
    if len(hits) == 1:
        return hits[0], quote
    pattern = r"\s+".join("".join(_char(c) for c in part) for part in quote.split())
    loose = list(re.finditer(pattern, page_text)) if pattern else []
    if len(loose) == 1:
        return loose[0].start(), loose[0].group(0)
    return None, f"quote occurs {len(hits) or len(loose)} times"


def validate(args) -> None:
    rows, problems = [], []
    for raw in sorted((args.work / "raw").glob("*.json")):
        doc = json.loads((args.parsed / f"{raw.stem}.json").read_text(encoding="utf-8"))
        offsets, pos = {}, 0
        for n in sorted(doc["pages"], key=int):
            offsets[int(n)] = pos
            pos += len(doc["pages"][n])
        try:
            qas = json.loads(json.loads(raw.read_text(encoding="utf-8"))["content"])["qas"]
        except Exception as e:
            problems.append(f"{raw.stem}: bad JSON {e}")
            continue
        for j, qa in enumerate(qas, 1):
            qid, ok, evs, why = f"{raw.stem}::{j}", True, [], []
            for ev in qa.get("evidence", []) or []:
                page = str(ev.get("page", ""))
                if page not in doc["pages"]:
                    ok, _ = False, why.append(f"page {page} missing")
                    continue
                start, exact = locate(doc["pages"][page], str(ev.get("quote", "")))
                if start is None:
                    ok, _ = False, why.append(f"p{page}: {exact}")
                    continue
                s = offsets[int(page)] + start
                evs.append({"page": int(page), "quote": exact, "source_start": s, "source_end": s + len(exact)})
            if not evs:
                ok, _ = False, why.append("no evidence")
            if VAGUE.search(qa.get("question", "")):
                ok, _ = False, why.append("vague question")
            if len(qa.get("question", "").split()) > 40:
                ok, _ = False, why.append("question too long")
            rows.append({"qid": qid, "doc_name": raw.stem, "ok": ok, "why": "; ".join(why),
                         "question": qa.get("question", ""), "answer": qa.get("answer", ""), "evidence": evs})
    with (args.work / "validated.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in rows if r["ok"]]
    print(f"{len(rows)} QAs from {len({r['doc_name'] for r in rows})} docs; valid {len(ok)} "
          f"({len({r['doc_name'] for r in ok})} docs)")
    for p in problems:
        print("  PROBLEM", p)
    for r in rows:
        if not r["ok"]:
            print(f"  drop {r['qid']}: {r['why']}")


# ---------------------------------------------------------------- review
REVIEW = """You are auditing question-answer pairs for a document-retrieval benchmark.
Below: an excerpt of one document, then numbered QA pairs with their evidence quotes.
For EACH pair decide "keep" or "drop". Drop if ANY of these holds:
- the answer is wrong, incomplete, or not fully supported by the evidence quotes;
- the question cannot be answered unambiguously from the document (several valid answers);
- the question is not self-contained (does not name the organisation/law/report it is about);
- the question is about headers, footers, table of contents, or bibliography;
- the evidence quotes are insufficient to support the answer.
Return ONLY JSON: {"verdicts": [{"n": 1, "decision": "keep"|"drop", "reason": "<short>"}, ...]}

DOCUMENT EXCERPT:
"""


def review(args) -> None:
    guard(args)
    client = client_from_env(args.env)
    rows = [json.loads(x) for x in (args.work / "validated.jsonl").read_text(encoding="utf-8").splitlines()]
    by_doc: dict[str, list] = {}
    for r in rows:
        if r["ok"]:
            by_doc.setdefault(r["doc_name"], []).append(r)
    rdir = args.work / f"review_{re.sub(r'[^A-Za-z0-9.-]', '_', args.model)}"

    def one(item):
        name, qas = item
        excerpt = (args.work / "export" / f"{name}.txt").read_text(encoding="utf-8")
        listing = "\n\n".join(
            f"[{k}] Q: {q['question']}\nA: {q['answer']}\nEVIDENCE: " + " || ".join(e["quote"] for e in q["evidence"])
            for k, q in enumerate(qas, 1))
        rec = call_json(client, args.model, REVIEW + excerpt + "\n\nQA PAIRS:\n" + listing, rdir / f"{name}.json", 16000)
        return f"ok {rec['model_returned']} {name}"

    with ThreadPoolExecutor(args.workers) as ex:
        for msg in ex.map(one, sorted(by_doc.items())):
            print(msg, flush=True)
    print(f"spent so far (review): ${spent([rdir], args.price_in, args.price_out):.2f}")


# ---------------------------------------------------------------- freeze
def freeze(args) -> None:
    out = args.work / "qa_frozen.jsonl"
    if out.exists():
        sys.exit(f"{out} exists - frozen sets are never overwritten")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    rows = [json.loads(x) for x in (args.work / "validated.jsonl").read_text(encoding="utf-8").splitlines()]
    verdicts = {}
    for rdir in args.review_dirs:
        for f in sorted(Path(rdir).glob("*.json")):
            v = json.loads(json.loads(f.read_text(encoding="utf-8"))["content"]).get("verdicts", [])
            qas = [r for r in rows if r["doc_name"] == f.stem and r["ok"]]
            for item in v:
                k = int(item.get("n", 0)) - 1
                if 0 <= k < len(qas):
                    verdicts.setdefault(qas[k]["qid"], []).append(item.get("decision", "drop"))
    doc_ids = json.loads((args.parsed.parent / "doc_ids.json").read_text(encoding="utf-8"))
    kept = [r for r in rows if r["ok"] and verdicts.get(r["qid"]) and all(d == "keep" for d in verdicts[r["qid"]])
            and len(verdicts[r["qid"]]) == len(args.review_dirs)]
    with out.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps({"qa_id": r["qid"], "doc_id": doc_ids[r["doc_name"]], "doc_name": r["doc_name"],
                                "question": r["question"], "answer": r["answer"], "evidence": r["evidence"]},
                               ensure_ascii=False) + "\n")
    man = {"qa_sha256": sha(out.read_bytes()), "n_qa": len(kept), "n_docs": len({r["doc_name"] for r in kept}),
           "n_validated": sum(r["ok"] for r in rows), "n_generated": len(rows),
           "review_dirs": [str(d) for d in args.review_dirs], "rule": "keep only if every reviewer says keep",
           "prompt_sha256": sha((args.work / "prompt.txt").read_bytes()), "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (args.work / "qa_frozen_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(json.dumps(man, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["export", "generate", "validate", "review", "freeze"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--parsed", type=Path, default=Path("data/clair_ext/parsed"))
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--env", type=Path)
    ap.add_argument("--price-in", type=float, default=2.0)
    ap.add_argument("--price-out", type=float, default=8.0)
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--allow-paid-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--review-dirs", nargs="*", type=Path, default=[])
    args = ap.parse_args()
    {"export": export, "generate": generate, "validate": validate, "review": review, "freeze": freeze}[args.step](args)


if __name__ == "__main__":
    main()
