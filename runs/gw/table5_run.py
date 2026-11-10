"""Chay harness Table 5 cua nhom (phase_i_table5 / phase_i_judge / phase_i_summarize) voi ten model tuy chon
va key lay tu mot file .env rieng - KHONG sua file harness tren dia.

Cach lam: import module harness, ghi de hang so MODEL trong bo nho, roi goi main() voi cac tham so con lai.
File harness giu nguyen nen kiem tra harness_sha256 cua nhom van hoat dong. Ten model duoc ghi vao
generation_plan.json ("model"), vao run_identity.json cua judge, va vao launcher_log.jsonl cua thu muc dau ra.

Chay tu thu muc goc repo nhom:
  python -X utf8 table5_run.py --model gpt-4.1 table5 prepare --qa ... --retrieval-dir ... --evaluation ... --output-dir OUT
  python -X utf8 table5_run.py --model gpt-4.1 --env D:\\Project\\keys\\official.env table5 generate --output-dir OUT --allow-paid-run --budget-usd 4 --smoke
  python -X utf8 table5_run.py --model gpt-4.1 table5 export --output-dir OUT --smoke
  python -X utf8 table5_run.py --model gpt-4.1 --env D:\\Project\\keys\\official.env judge --output-dir OUT --budget-usd 4 --allow-paid-run --smoke
  python -X utf8 table5_run.py --model gpt-4.1 summarize --input-dir OUT\\judge-full --output-dir OUT\\summary
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

MODULES = {"table5": "phase_i_table5", "judge": "phase_i_judge", "summarize": "phase_i_summarize"}
OPENAI_VARS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_ORGANIZATION", "OPENAI_PROJECT_ID")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def option(rest: list[str], name: str) -> str | None:
    for i, item in enumerate(rest):
        if item == name and i + 1 < len(rest):
            return rest[i + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return None


def load_env(path: str) -> str:
    from dotenv import dotenv_values
    values = {k: v for k, v in dotenv_values(path).items() if v}
    missing = [k for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL") if k not in values]
    if missing:
        sys.exit(f"{path} thieu {missing}")
    for name in OPENAI_VARS:
        os.environ.pop(name, None)
    os.environ.update({k: v for k, v in values.items() if k in OPENAI_VARS})
    return urlparse(values["OPENAI_BASE_URL"]).netloc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="vd gpt-4.1 (nhu code goc) hoac gpt-4.1-2025-04-14")
    parser.add_argument("--price-in", type=float, help="USD / 1 trieu token vao cua model (bat buoc neu khong phai gpt-4.1)")
    parser.add_argument("--price-out", type=float, help="USD / 1 trieu token ra")
    parser.add_argument("--env", help="file .env chua OPENAI_API_KEY va OPENAI_BASE_URL (can cho generate/judge)")
    parser.add_argument("--harness-dir", default="research/week3")
    parser.add_argument("stage", choices=MODULES)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    rest = args.rest
    if (args.price_in is None) != (args.price_out is None):
        sys.exit("Can ca --price-in va --price-out")
    if args.price_in is None and args.model not in {"gpt-4.1", "gpt-4.1-2025-04-14"}:
        sys.exit(f"Harness ghim gia GPT-4.1 ($2/$8). Voi {args.model} phai them --price-in VAO --price-out RA "
                 "(USD/1M token, lay tu trang gia OpenAI) de tran ngan sach va chi phi ghi lai dung")
    sub = rest[0] if rest and args.stage == "table5" else ""

    calls_api = args.stage == "judge" or (args.stage == "table5" and sub == "generate")
    host = None
    if args.env:
        host = load_env(args.env)
    elif calls_api:
        sys.exit("Buoc nay goi API: can --env <file .env>")
    else:
        for name in OPENAI_VARS:                     # buoc khong goi API: khong de key nao lot vao
            os.environ.pop(name, None)

    out = option(rest, "--output-dir")
    out_dir = Path(out) if out else None
    if out_dir is not None and args.stage in {"table5", "judge"} and sub != "prepare":
        plan_path = out_dir / "generation_plan.json"
        if not plan_path.is_file():
            sys.exit(f"Thieu {plan_path} - chay 'table5 prepare' truoc")
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        planned = plan_data.get("model")
        if args.price_in is not None:
            plan_price = plan_data.get("price", {})
            if (plan_price.get("input_usd_per_million"), plan_price.get("output_usd_per_million")) != (args.price_in, args.price_out):
                sys.exit(f"Gia trong ke hoach {plan_price} khac --price-in/--price-out - dung dung gia da dung khi prepare")
        if planned != args.model:
            sys.exit(f"Ke hoach trong {out_dir} duoc tao voi model {planned!r}, khong phai {args.model!r} "
                     "- dung dung --model hoac prepare thu muc moi")

    harness = Path(args.harness_dir).resolve()
    sys.path.insert(0, str(harness))
    module = importlib.import_module(MODULES[args.stage])
    for name in ("phase_i_table5", "phase_i_judge"):
        if name in sys.modules and hasattr(sys.modules[name], "MODEL"):
            sys.modules[name].MODEL = args.model
            if args.price_in is not None:
                sys.modules[name].INPUT_USD_PER_MILLION = args.price_in
                sys.modules[name].OUTPUT_USD_PER_MILLION = args.price_out
    print(f"[table5_run] model={args.model} | gia={args.price_in}/{args.price_out} | stage={args.stage} {sub} | "
          f"base URL host={host or '(khong goi API)'}", flush=True)

    record = {"time": datetime.now(timezone.utc).isoformat(), "stage": args.stage, "args": rest,
              "model": args.model, "price_usd_per_million": [args.price_in, args.price_out], "base_url_host": host, "launcher_sha256": sha(Path(__file__).resolve()),
              "harness_file": Path(module.__file__).name, "harness_sha256": sha(Path(module.__file__)),
              "python": platform.python_version()}
    sys.argv = [module.__file__, *rest]
    status = "error"
    try:
        result = module.main()
        status = "ok" if not result else f"exit {result}"
    except SystemExit as error:
        status = "ok" if not error.code else f"exit {error.code}"
        raise
    except BaseException as error:
        status = f"error: {type(error).__name__}: {str(error)[:300]}"
        raise
    finally:
        if out_dir is not None and out_dir.exists():
            with (out_dir / "launcher_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({**record, "status": status}, ensure_ascii=False) + "\n")
        if args.stage == "table5" and sub == "prepare" and status == "ok" and out_dir is not None:
            saved = json.loads((out_dir / "generation_plan.json").read_text(encoding="utf-8")).get("model")
            print(f"[table5_run] generation_plan.json ghi model = {saved}")


if __name__ == "__main__":
    main()
