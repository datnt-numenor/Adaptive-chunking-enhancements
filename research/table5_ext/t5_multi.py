"""Table 5 (answer quality) for ANY set of systems and ANY number of questions, with the group's harness UNCHANGED
on disk. phase_i_table5.py / phase_i_judge.py hard-code 3 systems x 99 questions; here their source is read, a few
literals are replaced IN MEMORY (each replacement must match exactly once, else abort), and the patched modules are
registered under their real names/paths (so harness_sha256 checks still refer to the untouched files). Then
runs/gw/table5_run.py (MODEL/price override, --env handling, launcher_log) runs as usual.

  set T5_SYSTEMS=adaptive,sdp_a_t600__doc   &  set T5_QA=99
  python -X utf8 research\table5_ext\t5_multi.py --model gpt-4o-mini-2024-07-18 --price-in 0.15 --price-out 0.60 ^
      table5 prepare --qa ... --retrieval-dir ... --evaluation ... --output-dir OUT
  (then table5 generate / table5 export / judge / summarize exactly as with table5_run.py)
Run from the group repo root (D:\Project\Adaptive-chunking-enhancements); judge needs .venv-judge.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve()
LAUNCHER = HERE.parents[2] / "runs" / "gw" / "table5_run.py"
HARNESS = Path("research/week3").resolve()

PATCHES = {
    "phase_i_table5": [
        ('PAPER_SYSTEMS = ("adaptive", "raw__langch_recurs_default", "raw__page")',
         'PAPER_SYSTEMS = tuple(__import__("os").environ["T5_SYSTEMS"].split(","))\n'
         '_T5_N = int(__import__("os").environ["T5_QA"])'),
        ("if (len(qas) != 99 or len(set(ids)) != 99 or len(set(hashes)) != 99",
         "if (len(qas) != _T5_N or len(set(ids)) != _T5_N or len(set(hashes)) != _T5_N"),
        ("or len(set(questions)) != 99", "or len(set(questions)) != _T5_N"),
        ("or len(per_doc) != 33 or set(per_doc.values()) != {3}):", "):  # T5_MULTI: any number of QA per document"),
        ("if len(rows) != 99 or {row.get", "if len(rows) != _T5_N or {row.get"),
        ('evaluation.get("qa_count") != 99', 'evaluation.get("qa_count") != _T5_N'),
        ('if len(requests) != 297 or len({(r["system_id"], r["qa_id"]) for r in requests}) != 297:',
         'if len(requests) != _T5_N * len(PAPER_SYSTEMS) or len({(r["system_id"], r["qa_id"]) for r in requests}) != _T5_N * len(PAPER_SYSTEMS):'),
        ('"qa": 99, "requests": 297,', '"qa": _T5_N, "requests": _T5_N * len(PAPER_SYSTEMS),'),
        ("size = 6 if args.smoke else 99", "size = 6 if args.smoke else _T5_N"),
    ],
    "phase_i_judge": [
        ("expected = 6 if args.smoke else 99", 'expected = 6 if args.smoke else int(os.environ["T5_QA"])'),
    ],
}


def load_patched(name: str) -> str:
    path = HARNESS / f"{name}.py"
    src = path.read_text(encoding="utf-8")
    for old, new in PATCHES[name]:
        if src.count(old) != 1:
            sys.exit(f"[t5_multi] patch target not found exactly once in {path}: {old[:60]!r}")
        src = src.replace(old, new)
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def main() -> None:
    for var in ("T5_SYSTEMS", "T5_QA"):
        if not os.environ.get(var):
            sys.exit(f"[t5_multi] set {var}")
    sys.path.insert(0, str(HARNESS))
    stage = next((a for a in sys.argv[1:] if a in ("table5", "judge", "summarize")), None)
    patched = {"phase_i_table5": load_patched("phase_i_table5")}
    if stage == "judge":
        patched["phase_i_judge"] = load_patched("phase_i_judge")
    print(f"[t5_multi] systems={os.environ['T5_SYSTEMS']} qa={os.environ['T5_QA']} in-memory patched sha256={patched}",
          flush=True)
    out = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--output-dir" and i + 1 < len(sys.argv)), None)
    if out and Path(out).exists():
        with (Path(out) / "t5_multi_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"argv": sys.argv[1:], "systems": os.environ["T5_SYSTEMS"], "qa": os.environ["T5_QA"],
                                "patched_source_sha256": patched, "launcher": str(LAUNCHER)}) + "\n")
    spec_globals = {"__name__": "__main__", "__file__": str(LAUNCHER)}
    sys.argv = [str(LAUNCHER), *sys.argv[1:]]
    exec(compile(LAUNCHER.read_text(encoding="utf-8"), str(LAUNCHER), "exec"), spec_globals)


if __name__ == "__main__":
    main()
