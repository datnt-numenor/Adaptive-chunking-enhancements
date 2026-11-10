"""Run the SDP chunker on the CLAIR corpus and score the CPU-only metrics (SC, BI, RC).

Writes, under --out:
    config.json                         chunker configuration + environment
    chunks/raw/chunks.parquet           raw SDP chunks (same schema as the pipeline)
    chunks/small_merged/chunks.parquet  after merge_small_chunks_to_neighbours (min 100, max 1150),
                                        i.e. the same post-processing stage the paper scores (`*` rows)
    cheap_metrics.parquet               SC / BI / RC per document, computed with the original metrics.py

Usage:
    python -X utf8 research/structure_chunker/run_sdp.py --out runs/sdp/A_default
    python -X utf8 research/structure_chunker/run_sdp.py --out runs/sdp/A_t05 --set tau=0.5
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptive_chunking.chunking_utils import count_tokens  # noqa: E402
from adaptive_chunking.metrics import (  # noqa: E402
    compute_block_integrity,
    compute_filtered_missing_ref_error,
    compute_size_compliance,
)
from adaptive_chunking.paper.replicate import SEPARATORS  # noqa: E402
from adaptive_chunking.postprocessing import (  # noqa: E402
    check_chunk_gaps,
    get_page_info,
    get_title_info,
    merge_small_chunks_to_neighbours,
)
from adaptive_chunking.splitters import RecursiveSplitter  # noqa: E402
from sdp_chunker import SDPConfig, build_units, sdp_chunk  # noqa: E402


def parse_set(values: list[str]) -> dict:
    out = {}
    for item in values:
        key, raw = item.split("=", 1)
        out[key] = json.loads(raw)
    return out


def load_chunker_embedder(name: str, device: str):
    """Embedder used to *choose* cut points. 'jina' = the metric's own model (pinned like score_full.py);
    'qwen0.6b' = an independent embedder, to check the gain is not just optimising the metric's model."""
    from sentence_transformers import SentenceTransformer

    if name == "jina":
        from score_full import CODE_REV, JINA, JINA_REV
        m = SentenceTransformer(JINA, trust_remote_code=True, revision=JINA_REV,
                                config_kwargs={"code_revision": CODE_REV}).to(device)
        return m, {"name": JINA, "revision": JINA_REV}
    # local_dir snapshot (Windows cannot create the HF-cache symlinks for this repo), pinned commit below
    path = "D:/Project/models/Qwen3-Embedding-0.6B@97b0c61"
    m = SentenceTransformer(path, device=device,
                            model_kwargs={"torch_dtype": "bfloat16"}, tokenizer_kwargs={"padding_side": "left"})
    m.max_seq_length = 2048
    return m, {"name": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
               "path": path, "note": "documents encoded without instruction prompt (same as semantic chunking)"}


def records_for(doc_name, method, stage, chunks, doc):
    titles = get_title_info(titles=doc["titles"], chunks=chunks, text=doc["full_text"])
    pages = get_page_info(pages=doc["pages"], chunks=chunks, text=doc["full_text"])
    return [
        {"doc_name": doc_name, "method": method, "type": stage, "chunk_index": i,
         "chunk_text": c, "chunk_pages": pages[i], "titles_context": titles[i],
         "chunk_len": count_tokens(c)}
        for i, c in enumerate(chunks)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/clair"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method", default="sdp_a")
    ap.add_argument("--set", nargs="*", default=[], help="SDPConfig overrides, e.g. tau=0.5")
    ap.add_argument("--docs", nargs="*", default=None, help="only these doc stems (smoke runs)")
    ap.add_argument("--smallest", type=int, default=0, help="only the N smallest docs (smoke runs)")
    ap.add_argument("--embed", action="store_true", help="version B: embedding score + Dinkelbach DP")
    ap.add_argument("--embed-model", default="jina", choices=["jina", "qwen0.6b"])
    ap.add_argument("--struct-weight", type=float, default=0.1)
    ap.add_argument("--icc-weight", type=float, default=1.0)
    ap.add_argument("--dcc-weight", type=float, default=1.0)
    ap.add_argument("--window-tokens", type=int, default=3000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = SDPConfig(**parse_set(args.set))
    oversize = RecursiveSplitter(
        separators=SEPARATORS, chunk_size=cfg.max_tokens, chunk_overlap=0, is_separator_regex=True,
        attach_separator_to="start", length_function=count_tokens, merging="to_chunk_size",
    )
    merge = lambda ch: merge_small_chunks_to_neighbours(  # noqa: E731  same as replicate.py step 1c
        ch, count_tokens, min_limit=100, max_limit=1150, merge_to="next")

    mentions = {}
    for pq in (args.data_dir / "mentions").glob("*.parquet"):
        m = pd.read_parquet(pq)
        mentions[m["doc_name"].iloc[0]] = m["entity_pron_mentions"].iloc[0]

    paths = sorted((args.data_dir / "adi_parsed").glob("*.json"))
    if args.docs:
        paths = [p for p in paths if p.stem in set(args.docs)]
    if args.smallest:
        paths = sorted(paths, key=lambda p: len(json.loads(p.read_text(encoding="utf-8"))["full_text"]))[: args.smallest]

    model, emb_dir, extra = None, None, {}
    if args.embed:
        import numpy as np

        from sdp_embed import EmbedScorer, dinkelbach_segment, embed_units
        model, model_info = load_chunker_embedder(args.embed_model, args.device)
        emb_dir = Path("runs/sdp/_unit_emb") / f"{args.embed_model}_max{cfg.max_tokens}"
        emb_dir.mkdir(parents=True, exist_ok=True)
        extra = {"embed_model": model_info, "struct_weight": args.struct_weight, "icc_weight": args.icc_weight,
                 "dcc_weight": args.dcc_weight, "window_tokens": args.window_tokens, "dinkelbach": {}}

    raw_rows, merged_rows, metric_rows = [], [], []
    t0 = time.time()
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        name, text = path.stem, doc["full_text"]
        units = build_units(text, doc["split_points"], count_tokens, cfg.max_tokens, oversize.split_text)
        if args.embed:
            cache = emb_dir / f"{name}.npy"
            if cache.exists():
                emb = np.load(cache)
                assert len(emb) == len(units), f"stale embedding cache {cache}"
            else:
                emb = embed_units(text, units, model)
                np.save(cache, emb)
            scorer = EmbedScorer(units, emb, args.window_tokens, args.icc_weight, args.dcc_weight)
            starts, hist = dinkelbach_segment(units, scorer, cfg, struct_weight=args.struct_weight)
            extra["dinkelbach"][name] = [round(h, 5) for h in hist]
            offsets = [0] + [units[s].start for s in starts[1:]] + [len(text)]
            chunks = [text[a:b] for a, b in zip(offsets, offsets[1:]) if b > a]
        else:
            chunks = sdp_chunk(text, doc["split_points"], count_tokens, cfg, units=units)
        assert "".join(chunks) == text, name
        merged = merge(chunks)
        assert check_chunk_gaps(merged, text), name
        raw_rows += records_for(name, args.method, "raw", chunks, doc)
        merged_rows += records_for(name, args.method, "small_merged", merged, doc)

        pairs = mentions.get(name)
        rc_err = compute_filtered_missing_ref_error(text, merged, pairs) if pairs is not None and len(pairs) else None
        scores = {
            "size_compliance": compute_size_compliance(merged, max_tokens=1100, min_tokens=100),
            "block_integrity": compute_block_integrity(merged, doc["split_points"], text, tolerance_chars=5),
            "references_completeness": None if rc_err is None else 1 - rc_err,
            "num_chunks": len(merged),
            "avg_chunk_tokens": sum(count_tokens(c) for c in merged) / len(merged),
            "num_units": len(units),
            "num_raw_chunks": len(chunks),
        }
        metric_rows += [{"doc_name": name, "chunking_method": args.method, "metric_name": k, "score": v}
                        for k, v in scores.items()]
        print(f"{name[:60]:60} units={len(units):5d} chunks={len(merged):4d} "
              f"SC={scores['size_compliance']:.3f} BI={scores['block_integrity']:.3f} "
              f"RC={scores['references_completeness'] if scores['references_completeness'] is not None else float('nan'):.3f}")

    for stage, rows in (("raw", raw_rows), ("small_merged", merged_rows)):
        d = args.out / "chunks" / stage
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(d / "chunks.parquet")
    pd.DataFrame(metric_rows).to_parquet(args.out / "cheap_metrics.parquet")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cfg_record = {k: v for k, v in asdict(cfg).items() if k != "extra_chunk_score"}
    (args.out / "config.json").write_text(json.dumps({
        "method": args.method, "config": cfg_record, "overrides": args.set,
        "docs": [p.stem for p in paths], "version_b": extra or None,
        "postprocessing": "merge_small_chunks_to_neighbours(min=100,max=1150,next)",
        "git_commit": commit, "python": sys.version, "platform": platform.platform(),
        "seconds": round(time.time() - t0, 1), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
