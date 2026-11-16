"""Buoc 5b: chay chunking cua repo goc, nhung goi OpenAI tuan tu va co cache.

Khong sua code goc. Script chi boc ham goi OpenAI de:
  - gui tung request mot, cach nhau LLM_MIN_INTERVAL giay (tranh loi 429 rate limit);
  - luu moi phan hoi vao runs/openai_cache/ -> chay lai khong ton tien lan nua;
  - in so token cua moi request de tinh chi phi.
Cach dung: python -X utf8 runs\run_chunking_cached.py <cac tham so nhu replicate>
"""
import asyncio
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion

load_dotenv()
CACHE = Path(os.environ.get("LLM_CACHE_DIR", "runs/openai_cache"))
CACHE.mkdir(parents=True, exist_ok=True)
MIN_INTERVAL = float(os.environ.get("LLM_MIN_INTERVAL", "65"))
USAGE_LOG = CACHE / "usage.jsonl"

AsyncCompletions = type(openai.AsyncOpenAI(api_key="unused").chat.completions)
_original_create = AsyncCompletions.create
_lock = asyncio.Lock()
_last_call = [0.0]


async def _cached_create(self, *args, **kwargs):
    request = {k: kwargs.get(k) for k in ("model", "messages", "temperature")}
    key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        print(f"[openai] cache hit {key[:12]}", flush=True)
        return ChatCompletion.model_validate_json(path.read_text(encoding="utf-8"))

    async with _lock:
        wait = _last_call[0] + MIN_INTERVAL - time.monotonic()
        if _last_call[0] and wait > 0:
            print(f"[openai] cho {wait:.0f}s truoc request tiep theo...", flush=True)
            await asyncio.sleep(wait)
        response = await _original_create(self, *args, **kwargs)
        _last_call[0] = time.monotonic()

    path.write_text(response.model_dump_json(), encoding="utf-8")
    usage = {"key": key, "model": response.model, "input_tokens": response.usage.prompt_tokens,
             "output_tokens": response.usage.completion_tokens, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with USAGE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(usage) + "\n")
    print(f"[openai] {usage['model']} in={usage['input_tokens']} out={usage['output_tokens']}", flush=True)
    return response


AsyncCompletions.create = _cached_create

if __name__ == "__main__":
    sys.argv = ["replicate", *sys.argv[1:]]
    runpy.run_module("adaptive_chunking.paper.replicate", run_name="__main__")