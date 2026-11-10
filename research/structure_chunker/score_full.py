"""Score chunks with the paper's full 5 metrics (original compute_metrics_per_origin + jina-v3).

Modes:
    # score an SDP run (reads <run>/chunks/small_merged, writes <run>/metrics/)
    python -X utf8 research/structure_chunker/score_full.py run runs/sdp/A_default

    # sanity check: rescore existing baseline chunks on a few docs and diff against the Kaggle numbers
    python -X utf8 research/structure_chunker/score_full.py validate --docs 2 --methods our_recurs_1100 page

The embedder is loaded exactly like replicate._make_embedder (SentenceTransformer, trust_remote_code),
with the revisions pinned to the ones recorded for the Kaggle reproduction.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

JINA = "jinaai/jina-embeddings-v3"
JINA_REV = "ab036b023d30b4d1138c4c3bfa9f0c445ab455d6"
CODE_REV = "bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3"
DATA = Path("data/clair")
KAGGLE = Path("runs/full_kaggle")
FIVE = ["size_compliance", "block_integrity", "intrachunk_cohesion",
        "document_contextual_coherence", "references_completeness"]


def load_embedder(device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(JINA, trust_remote_code=True, revision=JINA_REV,
                                config_kwargs={"code_revision": CODE_REV})
    model = model.to(device)
    env = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0) if "cuda" in device else device,
           "jina_revision": JINA_REV, "code_revision": CODE_REV,
           "dtype": str(next(model.parameters()).dtype), "python": sys.version, "platform": platform.platform()}
    return model, env


def score(chunks_dir: Path, out_dir: Path, device: str, batch_size: int, data: Path = DATA) -> dict:
    from adaptive_chunking.compute_metrics import compute_metrics_per_origin

    model, env = load_embedder(device)
    t0 = time.time()
    compute_metrics_per_origin(chunks_dir=chunks_dir, mentions_dir=data / "mentions",
                               parsed_docs_dir=data / "adi_parsed", models={"sentence_embedder": model},
                               output_dir=str(out_dir), batch_size=batch_size)
    env["seconds"] = round(time.time() - t0, 1)
    return env


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("run_dir", type=Path)
    r.add_argument("--data-dir", type=Path, default=DATA)
    r.add_argument("--stage", default="small_merged")
    v = sub.add_parser("validate")
    v.add_argument("--docs", type=int, default=2)
    v.add_argument("--methods", nargs="+", default=["our_recurs_1100"])
    v.add_argument("--out", type=Path, default=Path("runs/sdp/_validate"))
    for p in (r, v):
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    if args.cmd == "run":
        out = args.run_dir / ("metrics" if args.stage == "small_merged" else f"metrics_{args.stage}")
        env = score(args.run_dir / "chunks" / args.stage, out, args.device, args.batch_size, args.data_dir)
        env["data_dir"] = str(args.data_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
        print(json.dumps(env, indent=2))
        return

    # validate: pick the smallest docs by token count so the check is quick
    chunks = pd.read_parquet(KAGGLE / "chunks" / "small_merged" / "chunks.parquet")
    sizes = chunks[chunks.method == "page"].groupby("doc_name").chunk_len.sum().sort_values()
    docs = list(sizes.index[: args.docs])
    subset = chunks[chunks.doc_name.isin(docs) & chunks.method.isin(args.methods)]
    with tempfile.TemporaryDirectory() as tmp:
        subset.to_parquet(Path(tmp) / "chunks.parquet")
        if args.out.exists():
            for f in args.out.glob("*.parquet"):
                f.unlink()
        env = score(Path(tmp), args.out, args.device, args.batch_size)
    mine = pd.read_parquet(args.out / "chunking_metrics.parquet")
    ref = pd.read_parquet(KAGGLE / "results" / "chunking_metrics.parquet")
    key = ["doc_name", "chunking_method", "metric_name"]
    cmp = mine.merge(ref, on=key, suffixes=("_local", "_kaggle"))
    cmp = cmp[cmp.metric_name.isin(FIVE)].copy()
    cmp["abs_diff"] = (cmp.score_local.astype(float) - cmp.score_kaggle.astype(float)).abs()
    pd.set_option("display.width", 200)
    print(cmp.assign(doc_name=cmp.doc_name.str[:35]).to_string(index=False))
    print(f"\nmax |local - kaggle| = {cmp.abs_diff.max():.2e}; seconds = {env['seconds']}; {env}")


if __name__ == "__main__":
    main()
