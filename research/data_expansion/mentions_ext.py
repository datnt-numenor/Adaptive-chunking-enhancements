"""Coreference mentions (for the RC metric) with the paper's CoreferenceSolver (maverick-mes-ontonotes).

replicate.py --steps mentions cannot run as shipped: it calls CoreferenceSolver(max_context_tokens=2000) without
the required `pre_splitter` argument (src/adaptive_chunking/paper/replicate.py:338, metrics.py:693). Here the
pre-splitter splits the text into LINES keeping every character (the solver accumulates len(block) as offsets and
overlaps `lines_overlap=2` blocks), which is what the parameter names imply. Validate with `--validate N` against
the shipped CLAIR mentions before trusting new outputs.

Run in D:\\Project\\.venv-coref (maverick-coref; installs on Windows with PYTHONUTF8=1):
  set PYTHONUTF8=1
  D:\\Project\\.venv-coref\\Scripts\\python.exe research\\data_expansion\\mentions_ext.py ^
      --parsed data\\clair_ext_set\\adi_parsed --out data\\clair_ext_set\\mentions
  D:\\Project\\.venv-coref\\Scripts\\python.exe research\\data_expansion\\mentions_ext.py --validate 2 --out runs\\mentions_validate
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
from pathlib import Path


class LineSplitter:
    def split_text(self, text: str) -> list[str]:
        parts = re.findall(r"[^\n]*\n|[^\n]+$", text)
        assert "".join(parts) == text
        return parts


def run(parsed: Path, out: Path, device: str) -> None:
    import spacy

    from adaptive_chunking.extract_mentions import find_mentions_per_origin
    from adaptive_chunking.metrics import CoreferenceSolver

    # torch>=2.6 loads checkpoints with weights_only=True; the official maverick checkpoint stores its hydra
    # config (omegaconf objects). Allow-list exactly those classes instead of disabling the safety check.
    import collections
    import typing

    import torch
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode
    torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode,
                                          typing.Any, dict, list, int, collections.defaultdict])
    solver = CoreferenceSolver(pre_splitter=LineSplitter(), device=device, max_context_tokens=2000)
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 10_000_000
    done = {p.stem for p in out.glob("*.parquet")} if out.exists() else set()
    todo = [p for p in sorted(parsed.glob("*.json")) if p.stem not in done]
    with tempfile.TemporaryDirectory() as tmp:  # find_mentions_per_origin processes a whole dir: one doc at a time
        for p in todo:
            d = Path(tmp) / p.stem
            d.mkdir()
            shutil.copy2(p, d / p.name)
            t0 = time.time()
            find_mentions_per_origin(parsed_docs_dir=d, models={"coref_solver": solver, "spacy_model": nlp},
                                     output_dir=out, skip_non_english=False)
            print(f"[mentions] {time.time() - t0:6.0f}s {p.stem}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", type=Path, default=Path("data/clair/adi_parsed"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--validate", type=int, default=0, help="run on the N smallest CLAIR docs and compare")
    args = ap.parse_args()
    if not args.validate:
        run(args.parsed, args.out, args.device)
        return
    import pandas as pd

    clair = sorted(Path("data/clair/adi_parsed").glob("*.json"),
                   key=lambda p: len(json.loads(p.read_text(encoding="utf-8"))["full_text"]))
    english = [p for p in clair if (Path("data/clair/mentions") / f"{p.stem}.parquet").exists()]
    pick = []
    for p in english:
        ref = pd.read_parquet(Path("data/clair/mentions") / f"{p.stem}.parquet")["entity_pron_mentions"].iloc[0]
        if len(ref) >= 5:
            pick.append(p)
        if len(pick) == args.validate:
            break
    with tempfile.TemporaryDirectory() as tmp:
        for p in pick:
            shutil.copy2(p, Path(tmp) / p.name)
        run(Path(tmp), args.out, args.device)
    for p in pick:
        ref = pd.read_parquet(Path("data/clair/mentions") / f"{p.stem}.parquet")["entity_pron_mentions"].iloc[0]
        mine = pd.read_parquet(args.out / f"{p.stem}.parquet")["entity_pron_mentions"].iloc[0]
        to_set = lambda xs: {json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) for x in xs}  # noqa: E731
        a, b = to_set(ref), to_set(mine)
        print(f"{p.stem[:60]:60} shipped={len(a):4d} mine={len(b):4d} common={len(a & b):4d} "
              f"jaccard={len(a & b) / max(len(a | b), 1):.2f}")


if __name__ == "__main__":
    main()
