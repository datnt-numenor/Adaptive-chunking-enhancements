"""Test set: Adaptive selection, Table-3 scores, and retrieval system files (harness format) for clair_ext.

Inputs
  --data-dir   data/clair_ext_set (adi_parsed/, doc_ids.json)
  --chunks     run from chunk_ext.py, with metrics/ from score_full.py (small_merged stage)
  --sdp        name=run_dir (run_sdp.py on the same data dir, with metrics/), repeatable
Outputs (--out, new folder)
  table3_per_doc.csv, table3_summary.csv, table3_comparisons.csv   intrinsic scores (5 metrics, equal weights)
  adaptive_selections.json                                          per-document choice (paper rule)
  prepared/systems/<system>.jsonl + prepare_manifest.json + documents.jsonl   (plain and __doc variants)

Adaptive = per document the candidate (page, our_recurs_1100, our_recurs_600, llm_regex at small_merged stage)
with the best equal-weight mean of the 5 metrics, NaN metrics ignored (adaptive_chunking.paper.analysis.find_best_method),
exactly as phase_g_runner._select_adaptive_methods.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["size_compliance", "block_integrity", "intrachunk_cohesion", "document_contextual_coherence",
           "references_completeness"]
CANDIDATES = ["page", "our_recurs_1100", "our_recurs_600", "llm_regex"]
SEED, N_BOOT = 2026, 10_000


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def find_spans(frame: pd.DataFrame, docs: dict, method_label: str) -> list[dict]:
    """Same logic as phase_g_runner._find_chunk_spans (sequential find, empty chunks skipped)."""
    out = []
    for doc_name, g in frame.groupby("doc_name", sort=False):
        d = docs[doc_name]
        text, cursor = d["full_text"], 0
        for row in g.sort_values("chunk_index").itertuples(index=False):
            raw = str(row.chunk_text)
            if not raw.strip():
                continue
            pos, matched = -1, ""
            for cand in [raw] + ([raw.strip()] if raw.strip() != raw else []):
                pos = text.find(cand, cursor)
                if pos < 0:
                    pos = text.find(cand)
                if pos >= 0:
                    matched = cand
                    break
            if pos < 0:
                raise SystemExit(f"cannot map chunk {doc_name}/{method_label}/{row.chunk_index}")
            cursor = pos + len(matched)
            pages = getattr(row, "chunk_pages", [])
            out.append({"chunk_id": f"{d['doc_id']}::{method_label}::{int(row.chunk_index):06d}",
                        "doc_id": d["doc_id"], "doc_name": doc_name, "method": method_label,
                        "chunk_index": int(row.chunk_index), "chunk_text": raw.strip(),
                        "titles_context": str(getattr(row, "titles_context", "")),
                        "chunk_pages": [int(x) for x in (list(pages) if pages is not None else [])],
                        "source_start": pos, "source_end": cursor})
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def per_doc_score(metrics: pd.DataFrame) -> pd.DataFrame:
    m = metrics[metrics.metric_name.isin(METRICS)].copy()
    m["score"] = m["score"].astype(float)
    return m.pivot_table(index=["doc_name", "chunking_method"], columns="metric_name", values="score")


def main() -> None:
    from adaptive_chunking.paper.analysis import find_best_method

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--sdp", action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-metrics", type=Path, help="chunking_metrics.parquet (default <chunks>/metrics/...)")
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} exists - choose a new folder")

    ids = json.loads((args.data_dir / "doc_ids.json").read_text(encoding="utf-8"))
    docs = {}
    for p in sorted((args.data_dir / "adi_parsed").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        docs[p.stem] = {"doc_id": ids[p.stem], "full_text": d["full_text"], "doc_name": p.stem}

    # ---------- intrinsic metrics, Adaptive selection
    base_metrics = pd.read_parquet(args.base_metrics or args.chunks / "metrics" / "chunking_metrics.parquet")
    table = per_doc_score(base_metrics)
    sdp_frames = {}
    for item in args.sdp:
        name, run = item.split("=", 1)
        run, _, mpath = run.partition("::")  # name=run_dir[::metrics.parquet]
        run = Path(run)
        mt = pd.read_parquet(Path(mpath) if mpath else run / "metrics" / "chunking_metrics.parquet")
        mt["chunking_method"] = name
        table = pd.concat([table, per_doc_score(mt)])
        sdp_frames[name] = pd.read_parquet(run / "chunks" / "small_merged" / "chunks.parquet")
    table = table.reindex(columns=METRICS)
    selections = {}
    for doc in docs:
        sub = table.loc[doc].loc[CANDIDATES].T  # metrics x candidates
        selections[doc], _ = find_best_method(sub, {m: 0.2 for m in METRICS})
    eq = table.apply(lambda r: r.dropna().mean(), axis=1).rename("table3")  # equal weights, NaN ignored
    per_doc = eq.unstack("chunking_method")
    per_doc["adaptive"] = [per_doc.loc[d, selections[d]] for d in per_doc.index]
    args.out.mkdir(parents=True)
    (args.out / "adaptive_selections.json").write_text(json.dumps(selections, indent=1, ensure_ascii=False), encoding="utf-8")
    per_doc.to_csv(args.out / "table3_per_doc.csv")
    full = table.copy()
    full["table3"] = eq
    summ = (full.groupby(level="chunking_method").mean() * 100).round(2)
    ad_rows = pd.DataFrame([full.loc[(d, selections[d])] for d in docs]).mean() * 100
    summ.loc["adaptive"] = ad_rows.round(2)
    summ.to_csv(args.out / "table3_summary.csv")
    print(summ.to_string())
    print("Adaptive choices:", pd.Series(selections).value_counts().to_dict())

    rng = np.random.default_rng(SEED)
    comp = []
    for c in [c for c in per_doc.columns if c not in ("adaptive",)]:
        d = (per_doc[c] - per_doc["adaptive"]).dropna().to_numpy() * 100
        idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
        lo, hi = np.quantile(d[idx].mean(axis=1), [0.025, 0.975])
        comp.append({"system": c, "diff_vs_adaptive": d.mean(), "ci_low": lo, "ci_high": hi,
                     "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                     "p_wilcoxon": wilcoxon(d, zero_method="pratt").pvalue if np.any(d) else 1.0})
    pd.DataFrame(comp).to_csv(args.out / "table3_comparisons.csv", index=False)
    print(pd.DataFrame(comp).round(3).to_string(index=False))

    # ---------- retrieval systems
    sm = pd.read_parquet(args.chunks / "chunks" / "small_merged" / "chunks.parquet")
    raw = pd.read_parquet(args.chunks / "chunks" / "raw" / "chunks.parquet")
    frames = {
        "adaptive": pd.concat([sm[(sm.doc_name == d) & (sm.method == m)] for d, m in selections.items()]),
        "raw__langch_recurs_default": raw[raw.method == "langch_recurs_default"],
        "raw__page": raw[raw.method == "page"],
        **{f"processed__{m}": sm[sm.method == m] for m in CANDIDATES},
        **sdp_frames,
    }
    systems = {}
    for sid, fr in frames.items():
        rows = find_spans(fr, docs, sid)
        if {r["doc_name"] for r in rows} != set(docs):
            sys.exit(f"{sid} does not cover all documents")
        for variant, rr in (("", rows), ("__doc", [dict(r) for r in rows])):
            if variant:
                for r in rr:
                    title = r["doc_name"].split("_", 1)[-1]
                    r["chunk_text"] = f"Document: {title}\n\n" + r["chunk_text"]
                    r["chunk_id"] = r["chunk_id"].replace(f"::{sid}::", f"::{sid}__doc::")
                    r["method"] = sid + "__doc"
            path = args.out / "prepared" / "systems" / f"{sid}{variant}.jsonl"
            write_jsonl(path, rr)
            systems[sid + variant] = {"path": f"systems/{sid}{variant}.jsonl", "sha256": sha_file(path),
                                      "chunks": len(rr)}
        print(f"{sid:32} {len(rows):6d} chunks")
    write_jsonl(args.out / "prepared" / "documents.jsonl",
                [{"doc_id": v["doc_id"], "doc_name": k} for k, v in docs.items()])
    (args.out / "prepared" / "prepare_manifest.json").write_text(json.dumps({
        "data_dir": str(args.data_dir), "chunks_run": str(args.chunks), "sdp": args.sdp, "systems": systems,
        "adaptive_rule": "find_best_method, 5 metrics x 0.2, candidates " + ",".join(CANDIDATES)},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved ->", args.out)


if __name__ == "__main__":
    main()
