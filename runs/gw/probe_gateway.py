"""Tham do vi sao gateway tra 429: TPM thap (gui cham la duoc) hay kenh model bi chan han.

Gui 4 yeu cau (chi phi rat nho, khong ghi cache, khong anh huong harness):
  1. prompt ngan  -> gpt-4.1-2025-04-14      3. prompt ngan -> gpt-4.1-mini
  2. prompt ngan  -> gpt-4.1 (alias)         4. MOT prompt smoke that (~8-9k token) -> gpt-4.1-2025-04-14
In: ma trang thai, thong bao loi, cac header x-ratelimit-*, model tra ve. Khong in key.
  python -X utf8 probe_gateway.py D:\\Project\\keys\\gateway.env D:\\Project\\adaptive-chunking\\runs\\table5_v2_answers_gateway
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values
from openai import APIStatusError, OpenAI


def show(client: OpenAI, label: str, model: str, prompt: str) -> None:
    started = time.time()
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0, top_p=1,
            max_completion_tokens=20)
        r = raw.parse()
        headers = {k: v for k, v in raw.headers.items() if k.lower().startswith("x-ratelimit")}
        print(f"[OK ] {label}: model tra ve={r.model}, prompt_tokens={r.usage.prompt_tokens}, "
              f"{time.time() - started:.1f}s, header={headers or 'khong co'}")
    except APIStatusError as error:
        headers = {k: v for k, v in error.response.headers.items()
                   if k.lower().startswith(("x-ratelimit", "retry-after"))}
        try:
            body = error.response.json()
        except ValueError:
            body = error.response.text[:500]
        print(f"[LOI] {label}: HTTP {error.status_code}, header={headers or 'khong co'}\n      noi dung: "
              f"{json.dumps(body, ensure_ascii=False)[:600]}")
    except Exception as error:  # noqa: BLE001 - in loi mang de chan doan
        print(f"[LOI] {label}: {type(error).__name__}: {str(error)[:300]}")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    env = dotenv_values(sys.argv[1])
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env["OPENAI_BASE_URL"], max_retries=0, timeout=120)
    print(f"Gateway: {urlparse(env['OPENAI_BASE_URL']).netloc}")
    plan_dir = Path(sys.argv[2])
    plan = json.loads((plan_dir / "generation_plan.json").read_text(encoding="utf-8"))
    smoke = next(json.loads(line) for line in (plan_dir / "generation_requests.jsonl").open(encoding="utf-8")
                 if json.loads(line)["doc_id"] in plan["smoke_doc_ids"])
    short = "Reply with the single word: OK"
    show(client, "1. ngan  -> gpt-4.1-2025-04-14", "gpt-4.1-2025-04-14", short)
    time.sleep(3)
    show(client, "2. ngan  -> gpt-4.1", "gpt-4.1", short)
    time.sleep(3)
    show(client, "3. ngan  -> gpt-4.1-mini", "gpt-4.1-mini", short)
    time.sleep(3)
    show(client, f"4. prompt smoke that (~{smoke['input_tokens_estimate']} token) -> gpt-4.1-2025-04-14",
         "gpt-4.1-2025-04-14", smoke["prompt"])


if __name__ == "__main__":
    main()
