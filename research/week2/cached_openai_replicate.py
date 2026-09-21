"""Run the upstream reproduction CLI with a durable Chat Completions cache."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
from pathlib import Path
from types import SimpleNamespace


CACHE_SCHEMA_VERSION = 1


def _request_payload(args: tuple, kwargs: dict) -> dict:
    if args:
        raise RuntimeError("Cached Chat Completions requires keyword arguments")
    return {"endpoint": "chat.completions.create", "parameters": kwargs}


def _cache_key(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def install_cache(cache_dir: Path, openai_module=None, chat_completion_type=None) -> None:
    if openai_module is None or chat_completion_type is None:
        import openai as openai_module
        from openai.types.chat import ChatCompletion as chat_completion_type

    original_client = openai_module.AsyncOpenAI

    class CachedCompletions:
        def __init__(self, completions):
            self._completions = completions

        async def create(self, *args, **kwargs):
            request = _request_payload(args, kwargs)
            digest = _cache_key(request)
            cache_path = cache_dir / f"{digest}.json"
            if cache_path.is_file():
                record = json.loads(cache_path.read_text(encoding="utf-8"))
                if record.get("schema_version") != CACHE_SCHEMA_VERSION:
                    raise RuntimeError(f"Unsupported API cache schema: {cache_path}")
                if record.get("request") != request:
                    raise RuntimeError(f"API cache hash collision: {cache_path}")
                print(f"OPENAI_CACHE_HIT {digest}", flush=True)
                return chat_completion_type.model_validate(record["response"])

            print(f"OPENAI_CACHE_MISS {digest}", flush=True)
            response = await self._completions.create(*args, **kwargs)
            record = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "request_sha256": digest,
                "request": request,
                "response": response.model_dump(mode="json"),
            }
            _write_json_atomic(cache_path, record)
            print(f"OPENAI_CACHE_SAVED {digest}", flush=True)
            return response

    class CachedAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            # Avoid hidden SDK retries. The shard controller stops on failure
            # so the user can inspect an uncertain request before continuing.
            kwargs.setdefault("max_retries", 0)
            self._client = original_client(*args, **kwargs)
            self.chat = SimpleNamespace(
                completions=CachedCompletions(self._client.chat.completions)
            )

        def __getattr__(self, name):
            return getattr(self._client, name)

    openai_module.AsyncOpenAI = CachedAsyncOpenAI


def main() -> None:
    raw_cache_dir = os.environ.get("ADAPTIVE_OPENAI_CACHE_DIR")
    if not raw_cache_dir:
        raise RuntimeError("ADAPTIVE_OPENAI_CACHE_DIR is required")
    install_cache(Path(raw_cache_dir))
    runpy.run_module("adaptive_chunking.paper.replicate", run_name="__main__")


if __name__ == "__main__":
    main()
