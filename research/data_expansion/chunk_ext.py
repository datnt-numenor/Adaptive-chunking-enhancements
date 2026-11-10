"""Chunk a new parsed corpus with the paper's pipeline (replicate.run_chunking), restricted to the methods needed:
Adaptive candidates (page, our_recurs_1100, our_recurs_600, llm_regex) + baselines (langch_recurs_default, page raw).
Same splitter settings, same post-processing (split oversized > 1100 for page/llm_regex, merge < 100 to 1150).

llm_regex calls OpenAI-compatible chat (model gpt-4o, temperature 0, same prompt as the paper). Key from --env
(never printed). Every response is cached in --cache (hash of model+messages+temperature), so reruns are free.

Usage (cmd):
  python -X utf8 research\\data_expansion\\chunk_ext.py --data-dir data\\clair_ext_set --out runs\\ext_v1 ^
     --env D:\\Project\\keys\\gateway.env --allow-paid-run          (omit --allow-paid-run: estimate only)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

import tiktoken


def estimate(parsed: Path, price_in: float, price_out: float) -> float:
    from adaptive_chunking.paper.replicate import _build_few_shot_prompt

    enc = tiktoken.get_encoding("o200k_base")
    base = len(enc.encode(_build_few_shot_prompt()))
    tot_in = n_req = 0
    for p in sorted(parsed.glob("*.json")):
        n = len(enc.encode(json.loads(p.read_text(encoding="utf-8"))["full_text"], disallowed_special=()))
        reqs = max(1, -(-n // 8000))  # context_tokens=8000 windows (upper bound)
        n_req += reqs
        tot_in += n + reqs * base
    cost = tot_in / 1e6 * price_in + n_req * 100 / 1e6 * price_out
    print(f"llm_regex: ~{n_req} requests, ~{tot_in:,} input tokens -> ~${cost:.2f} "
          f"(gpt-4o list price ${price_in}/M in, ${price_out}/M out)")
    return cost


async def run(args) -> None:
    import openai
    from dotenv import dotenv_values
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from adaptive_chunking.paper.replicate import SEPARATORS, _build_few_shot_prompt, count_tokens_func
    from adaptive_chunking.paper.splitters import LLMRegexSplitter
    from adaptive_chunking.postprocessing import (merge_small_chunks_from_df, merge_small_chunks_to_neighbours,
                                                  split_oversized_chunks, split_oversized_chunks_from_df)
    from adaptive_chunking.split_documents import split_documents_from_dir
    from adaptive_chunking.splitters import RecursiveSplitter

    vals = dotenv_values(args.env) if args.env else {"OPENAI_API_KEY": "unused"}
    client = openai.AsyncOpenAI(api_key=vals["OPENAI_API_KEY"], base_url=vals.get("OPENAI_BASE_URL") or None,
                                timeout=600)
    args.cache.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    async def llm(prompt: str) -> str:
        req = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
        key = hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        f = args.cache / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))["content"]
        async with sem:
            for attempt in range(5):
                try:
                    r = await client.chat.completions.create(**req)
                    break
                except Exception as e:  # rate limit / transient
                    print(f"[llm] retry {attempt + 1}: {type(e).__name__}", flush=True)
                    await asyncio.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError("llm_regex request failed 5 times")
        rec = {"content": r.choices[0].message.content, "model_returned": r.model,
               "usage": {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens},
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        f.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        print(f"[llm] {rec['model_returned']} in={rec['usage']['in']} out={rec['usage']['out']}", flush=True)
        return rec["content"]

    rec_kw = dict(separators=SEPARATORS, chunk_overlap=0, is_separator_regex=True, attach_separator_to="start",
                  length_function=count_tokens_func, merging="to_chunk_size", merging_order="forward")
    wanted = set(args.methods.split(","))
    sync = {
        "page": None,
        "langch_recurs_default": RecursiveCharacterTextSplitter(),
        "our_recurs_1100": RecursiveSplitter(chunk_size=1100, **rec_kw),
        "our_recurs_600": RecursiveSplitter(chunk_size=600, **rec_kw),
    }
    asyn = {"llm_regex": LLMRegexSplitter(base_prompt=_build_few_shot_prompt(), async_client_completion_func=llm,
                                          count_tokens_func=count_tokens_func, context_tokens=8000)}
    sync = {k: v for k, v in sync.items() if k in wanted}
    asyn = {k: v for k, v in asyn.items() if k in wanted}
    parsed = args.data_dir / "adi_parsed"
    raw, nos, sm = (args.out / "chunks" / s for s in ("raw", "no_oversizing", "small_merged"))
    await split_documents_from_dir(parsed_docs_dir=parsed, sync_splitters=sync, async_splitters=asyn, output_dir=raw,
                                   count_tokens_func=count_tokens_func, skip_non_english=False,
                                   replace_all_results=True)
    oversize = RecursiveSplitter(separators=SEPARATORS, chunk_size=1100, chunk_overlap=0, is_separator_regex=True,
                                 attach_separator_to="start", length_function=count_tokens_func,
                                 merging="to_chunk_size")
    split_oversized_chunks_from_df(
        parsed_docs_dir=parsed, chunks_path=raw / "chunks.parquet", output_dir=nos,
        methods_to_be_regularized={"page", "llm_regex"} & (set(sync) | set(asyn)),
        split_oversized_func=lambda ch: split_oversized_chunks(ch, oversize, count_tokens_func, max_chunk_tokens=1100),
        count_tokens_func=count_tokens_func, replace_all_results=True)
    merge_small_chunks_from_df(
        parsed_docs_dir=parsed, chunks_path=nos / "chunks.parquet", output_dir=sm,
        methods_to_be_regularized=(set(sync) | set(asyn)) - {"langch_recurs_default"},  # baseline used at raw stage only (LC chunks overlap/strip: not mappable for merging)
        merge_small_chunks_func=lambda ch: merge_small_chunks_to_neighbours(ch, count_tokens_func, min_limit=100,
                                                                             max_limit=1150, merge_to="next"),
        count_tokens_func=count_tokens_func, replace_all_results=True)
    (args.out / "chunking_manifest.json").write_text(json.dumps({
        "data_dir": str(args.data_dir), "methods": sorted(set(sync) | set(asyn)), "llm_model": args.model,
        "llm_cache": str(args.cache), "postprocessing": "replicate.run_chunking 1b/1c",
        "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
    print("done ->", args.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--env", type=Path)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--cache", type=Path, default=Path("runs/openai_cache_ext"))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--price-in", type=float, default=2.5)
    ap.add_argument("--price-out", type=float, default=10.0)
    ap.add_argument("--allow-paid-run", action="store_true")
    ap.add_argument("--methods", default="page,langch_recurs_default,our_recurs_1100,our_recurs_600,llm_regex")
    args = ap.parse_args()
    paid = "llm_regex" in args.methods.split(",")
    if paid:
        estimate(args.data_dir / "adi_parsed", args.price_in, args.price_out)
    if paid and not args.allow_paid_run:
        return
    if args.out.exists():
        sys.exit(f"{args.out} exists - choose a new folder")
    if paid and not args.env:
        sys.exit("--env required")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
