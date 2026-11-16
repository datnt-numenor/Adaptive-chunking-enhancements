"""List model ids offered by an OpenAI-compatible endpoint (free GET /models). Never prints the key.
Usage: python -X utf8 research\data_expansion\list_models.py D:\Project\keys\gateway.env"""
import sys
from urllib.parse import urlparse

from dotenv import dotenv_values
from openai import OpenAI

v = dotenv_values(sys.argv[1])
c = OpenAI(api_key=v["OPENAI_API_KEY"], base_url=v.get("OPENAI_BASE_URL") or None)
print("host:", urlparse(v.get("OPENAI_BASE_URL") or "https://api.openai.com").netloc)
ids = sorted(m.id for m in c.models.list())
print(len(ids), "models")
print("\n".join(i for i in ids if any(k in i.lower() for k in ("gpt-4", "gpt-5", "claude", "gemini", "o3", "o4", "deepseek", "qwen"))))
