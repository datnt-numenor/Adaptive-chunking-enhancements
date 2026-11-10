"""Chay mot script cua harness voi OPENAI_API_KEY / OPENAI_BASE_URL lay tu MOT file .env rieng.

Muc dich: chuyen giua API chinh thuc va gateway ma khong go key vao dong lenh (khong luu vao lich su cmd).
Harness goi load_dotenv(override=False) nen gia tri dat o day duoc giu nguyen.
  python -X utf8 run_with_env.py D:\\Project\\keys\\gateway.env research\\week3\\phase_i_table5.py generate ...
File .env can co: OPENAI_API_KEY=...  va  OPENAI_BASE_URL=https://.../v1   (khong in key ra man hinh).
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values

OPENAI_VARS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_ORGANIZATION", "OPENAI_PROJECT_ID")


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    env_file, script, *rest = sys.argv[1:]
    values = {k: v for k, v in dotenv_values(env_file).items() if v}
    missing = [k for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL") if k not in values]
    if missing:
        sys.exit(f"{env_file} thieu {missing}")
    for name in OPENAI_VARS:                     # xoa bien OpenAI cu de khong lan sang nha cung cap khac
        os.environ.pop(name, None)
    os.environ.update({k: v for k, v in values.items() if k in OPENAI_VARS})
    print(f"[run_with_env] {Path(env_file).name}: base URL host = {urlparse(values['OPENAI_BASE_URL']).netloc}",
          flush=True)
    script_path = Path(script).resolve()
    sys.path.insert(0, str(script_path.parent))
    sys.argv = [str(script_path), *rest]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
