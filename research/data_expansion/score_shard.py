"""Compute the paper's 5 chunk metrics for several runs on ALL visible GPUs (one process per GPU), one document at a
time, with the same code as score_full.py (compute_metrics_per_origin + jina-v3 at the pinned revisions).

Work unit = (run, document): all methods of a document are scored on the same GPU, so the per-document Adaptive
selection never compares numbers from different machines. Output per run and GPU:
  <out-root>/<run_name>/gpu<i>/chunking_metrics.parquet (+ metrics_performance.parquet, environment.json)
Resumable (compute_metrics_per_origin skips documents already in the output parquet).

  python -X utf8 research/data_expansion/score_shard.py plan --run ext_v1=runs/ext_v1 --run sdp_a=runs/sdp_ext/A_target600 \
      --run sdp_b=runs/sdp_ext/B_default --data-dir data/clair_ext_set --skip skip.json --out-root metrics_out
  python -X utf8 research/data_expansion/score_shard.py launch --out-root metrics_out     # spawns one worker per GPU
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "structure_chunker"))


def n_gpus() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    return len([x for x in out.stdout.splitlines() if x.strip()])


def plan(args) -> None:
    skip = json.loads(Path(args.skip).read_text(encoding="utf-8")) if args.skip else {}
    items = []
    for spec in args.run:
        name, run = spec.split("=", 1)
        c = pd.read_parquet(Path(run) / "chunks" / "small_merged" / "chunks.parquet")
        # cost model: DCC embeds one <=3000-token window per chunk -> ~ number of chunks (all methods of the doc)
        cost = c.groupby("doc_name").size()
        for doc, w in cost.items():
            if doc not in set(skip.get(name, [])):
                items.append({"run": name, "run_dir": run, "doc": doc, "cost": int(w)})
    g = args.gpus or n_gpus()
    loads, shards = [0] * g, [[] for _ in range(g)]
    for it in sorted(items, key=lambda x: -x["cost"]):  # longest-processing-time first
        i = loads.index(min(loads))
        shards[i].append(it)
        loads[i] += it["cost"]
    prio = {spec.split("=", 1)[0]: k for k, spec in enumerate(args.run)}  # run order = priority (ext_v1 first)
    for s in shards:
        s.sort(key=lambda x: (prio[x["run"]], -x["cost"]))
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_text(json.dumps({"data_dir": args.data_dir, "batch_size": args.batch_size,
                                                "shards": shards, "loads": loads}, indent=1), encoding="utf-8")
    print(f"{len(items)} (run, doc) items on {g} GPUs; load per GPU (chunks): {loads}")


def launch(args) -> None:
    root = Path(args.out_root)
    p = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    procs = []
    for i in range(len(p["shards"])):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        log = open(root / f"worker{i}.log", "a", encoding="utf-8")
        procs.append(subprocess.Popen([sys.executable, "-u", "-X", "utf8", str(HERE), "worker", "--out-root", str(root),
                                       "--shard", str(i)], env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [pr.wait() for pr in procs]
    print("worker exit codes:", codes)
    if any(codes):
        sys.exit(1)


def worker(args) -> None:
    import torch

    from adaptive_chunking.compute_metrics import compute_metrics_per_origin
    from score_full import load_embedder

    root = Path(args.out_root)
    p = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    items = p["shards"][args.shard]
    data = Path(p["data_dir"])
    model, env = load_embedder(os.environ.get("SHARD_DEVICE", "cuda:0"))
    env["gpu_index"] = args.shard
    chunks_cache: dict[str, pd.DataFrame] = {}
    t_all = time.time()
    for k, it in enumerate(items, 1):
        out = root / it["run"] / f"gpu{args.shard}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
        if it["run"] not in chunks_cache:
            chunks_cache[it["run"]] = pd.read_parquet(Path(it["run_dir"]) / "chunks" / "small_merged" / "chunks.parquet")
        df = chunks_cache[it["run"]]
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            df[df.doc_name == it["doc"]].to_parquet(Path(tmp) / "chunks.parquet")
            compute_metrics_per_origin(chunks_dir=tmp, mentions_dir=data / "mentions",
                                       parsed_docs_dir=data / "adi_parsed", models={"sentence_embedder": model},
                                       output_dir=str(out), batch_size=p["batch_size"])
        torch.cuda.empty_cache()
        print(f"[shard {args.shard}] {k}/{len(items)} {time.time() - t0:6.0f}s total {time.time() - t_all:6.0f}s "
              f"{it['run']} {it['doc'][:70]}", flush=True)
    print(f"[shard {args.shard}] DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--run", action="append", required=True, help="name=run_dir (order = priority)")
    a.add_argument("--data-dir", required=True)
    a.add_argument("--skip", help="json {run_name: [doc,...]} already scored elsewhere")
    a.add_argument("--gpus", type=int, default=0)
    a.add_argument("--batch-size", type=int, default=8)  # same as score_full.py
    a.add_argument("--out-root", required=True)
    b = sub.add_parser("launch")
    b.add_argument("--out-root", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--out-root", required=True)
    w.add_argument("--shard", type=int, required=True)
    args = ap.parse_args()
    {"plan": plan, "launch": launch, "worker": worker}[args.cmd](args)


if __name__ == "__main__":
    main()
