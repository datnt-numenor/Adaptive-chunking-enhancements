"""Local PROXY of the group's Table-5 retrieval (phase_g_runner.py), to screen many variants on the laptop GPU.

Same pipeline as the harness (rag_utils.create_retrieval_pipeline of the group repo):
    BM25 top-50 (haystack InMemoryBM25Retriever, BM25L)  +  dense top-50  ->  union  ->
    Snowflake arctic-embed-l-v2.0 bi-encoder cosine rerank (query prompt "query")  ->  top-10
Only difference: the dense retriever uses Qwen3-Embedding-0.6B instead of 4B (4B does not fit in 8 GB).
Metrics: the harness's own `query_metrics` (character-span coverage).
Final numbers must be confirmed on the A5000 with the real harness; this is a screening tool.

Chunk *representation* variants (text that is indexed, embedded and reranked; spans are unchanged):
    plain    chunk_text (= harness)
    doc      "Document: <title>\\n\\n" + chunk_text           title = doc_name without the CLAIR category prefix
    docpath  "Document: <title>\\nSection: <h1 > h2 > h3>\\n\\n" + chunk_text   (heading path at chunk start)

Output layout mirrors the harness so analyze_sdp_retrieval.py works unchanged:
    <out>/evaluation-full/per_query_metrics.jsonl, <out>/retrieval-full/<system>@<repr>.jsonl, <out>/manifest.json

Usage (cmd):
  python -X utf8 research\\structure_chunker\\retrieval\\proxy_retrieval.py --prepared runs\\table5_sdp_v1\\prepared ^
     --systems adaptive,raw__langch_recurs_default,raw__page,sdp_b --reprs plain,doc --out runs\\proxy\\r1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HARNESS = Path(os.environ.get("AC_HARNESS", "D:/Project/Adaptive-chunking-enhancements"))  # pod: copy of the harness file
QWEN = "D:/Project/models/Qwen3-Embedding-0.6B@97b0c61"
SNOW = "D:/Project/models/snowflake-arctic-embed-l-v2.0@ac6544c"
CACHE = Path("runs/proxy/_emb_cache")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def doc_title(doc_name: str) -> str:
    return doc_name.split("_", 1)[1] if "_" in doc_name else doc_name


class HeadingIndex:
    """Heading path (h1..h3) in force at a character offset of the source text."""

    def __init__(self, text: str):
        self.items = [(m.start(), len(m.group(1)), m.group(2).strip("# ").strip()) for m in HEADING.finditer(text)]

    def path(self, offset: int, max_level: int = 3) -> str:
        cur: dict[int, str] = {}
        for pos, lvl, title in self.items:
            if pos > offset:
                break
            cur = {k: v for k, v in cur.items() if k < lvl}
            cur[lvl] = title
        return " > ".join(cur[k] for k in sorted(cur) if k <= max_level)


def represent(row: dict, mode: str, headings: dict[str, HeadingIndex]) -> str:
    text = row["chunk_text"]
    if mode == "plain":
        return text
    head = f"Document: {doc_title(row['doc_name'])}\n"
    if mode == "docpath":
        # heading in force just after the chunk start (a chunk that starts with a heading uses that heading)
        path = headings[row["doc_id"]].path(int(row["source_start"]) + 1)
        if path:
            head += f"Section: {path}\n"
    elif mode != "doc":
        raise ValueError(mode)
    return head + "\n" + text


class Encoder:
    """SentenceTransformer with an on-disk cache keyed by text hash (cache is per model)."""

    def __init__(self, path: str, name: str, device: str):
        from sentence_transformers import SentenceTransformer

        kw = {"tokenizer_kwargs": {"padding_side": "left"}} if "qwen" in path.lower() else {}
        self.m = SentenceTransformer(path, device=device, model_kwargs={"torch_dtype": torch.bfloat16}, **kw)
        if "qwen" in path.lower():
            self.m.max_seq_length = 8192
        self.dir = CACHE / name
        self.dir.mkdir(parents=True, exist_ok=True)

    def docs(self, texts: list[str], tag: str) -> np.ndarray:
        f = self.dir / f"{tag}.npz"
        keys = [h(t) for t in texts]
        if f.exists():
            z = np.load(f)
            if list(z["keys"]) == keys:
                return z["emb"]
        order = np.argsort([-len(t) for t in texts])  # length-sorted batches: less padding
        emb = np.zeros((len(texts), self.m.get_sentence_embedding_dimension()), dtype=np.float32)
        out = self.m.encode([texts[i] for i in order], batch_size=8, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
        emb[order] = out
        np.savez(f, keys=np.array(keys), emb=emb)
        return emb

    def query(self, q: str) -> np.ndarray:
        return self.m.encode([q], prompt_name="query", normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=False)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--qa", type=Path, default=Path("runs/table5_v2/qa-full/qa_frozen.jsonl"))
    ap.add_argument("--systems", required=True)
    ap.add_argument("--reprs", default="plain")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-dir", type=Path, default=Path("data/clair"))
    ap.add_argument("--dense-model", default=QWEN, help="path/id of the dense embedder (harness: Qwen3-Embedding-4B)")
    ap.add_argument("--rerank-model", default=SNOW)
    ap.add_argument("--cache-tag", default="", help="prefix for embedding-cache files (use for a non-CLAIR corpus)")
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} exists - choose a new folder")

    sys.path.insert(0, str(HARNESS / "research" / "week3"))
    import phase_g_runner as pg  # harness metric code (read only)
    from haystack import Document
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    from adaptive_chunking.chunking_utils import count_tokens  # noqa: F401  (same package env as harness)

    qas = read_jsonl(args.qa)
    qa_docs = {q["doc_id"] for q in qas}
    headings = {}
    if "docpath" in args.reprs:
        if args.data_dir == Path("data/clair"):
            sources = pg._load_source_documents(args.data_dir)
            headings = {d["doc_id"]: HeadingIndex(d["full_text"]) for d in sources.values()}
        else:
            ids = json.loads((args.data_dir / "doc_ids.json").read_text(encoding="utf-8"))
            for stem, did in ids.items():
                text = json.loads((args.data_dir / "adi_parsed" / f"{stem}.json").read_text(encoding="utf-8"))["full_text"]
                headings[did] = HeadingIndex(text)
    manifest = json.loads((args.prepared / "prepare_manifest.json").read_text(encoding="utf-8"))

    t0 = time.time()
    dense_name = "qwen0.6b" if args.dense_model == QWEN else Path(args.dense_model).name.replace("/", "_")
    qwen = Encoder(args.dense_model, dense_name, args.device)
    snow = Encoder(args.rerank_model, "snowflake", args.device)
    q_qwen = {q["qa_id"]: qwen.query(q["question"]) for q in qas}
    q_snow = {q["qa_id"]: snow.query(q["question"]) for q in qas}

    (args.out / "evaluation-full").mkdir(parents=True)
    (args.out / "retrieval-full").mkdir()
    per_query, summary = [], {}
    for system in [s for s in args.systems.split(",") if s]:
        rows = [r for r in read_jsonl(args.prepared / manifest["systems"][system]["path"]) if r["doc_id"] in qa_docs]
        meta_keys = ("chunk_id", "doc_id", "doc_name", "method", "chunk_index", "chunk_pages", "source_start", "source_end")
        for mode in [m for m in args.reprs.split(",") if m]:
            sid = f"{system}@{mode}"
            texts = [represent(r, mode, headings) for r in rows]
            e_qwen = qwen.docs(texts, f"{args.cache_tag}{system}__{mode}")
            e_snow = snow.docs(texts, f"{args.cache_tag}{system}__{mode}")
            store = InMemoryDocumentStore()
            store.write_documents([Document(id=r["chunk_id"], content=t, meta={k: r[k] for k in meta_keys},
                                            embedding=e.tolist()) for r, t, e in zip(rows, texts, e_qwen)])
            bm25 = InMemoryBM25Retriever(document_store=store, top_k=50)
            dense = InMemoryEmbeddingRetriever(document_store=store, top_k=50)
            pos = {r["chunk_id"]: i for i, r in enumerate(rows)}
            corpus = [{k: r[k] for k in ("doc_id", "source_start", "source_end")} for r in rows]
            out_rows = []
            for q in qas:
                cands = bm25.run(query=q["question"])["documents"] + \
                    dense.run(query_embedding=q_qwen[q["qa_id"]].tolist())["documents"]
                seen, uniq = set(), []
                for d in cands:
                    if d.id not in seen:
                        seen.add(d.id)
                        uniq.append(d)
                idx = np.array([pos[d.id] for d in uniq])
                scores = e_snow[idx] @ q_snow[q["qa_id"]]
                order = sorted(range(len(uniq)), key=lambda i: (-scores[i], i))[:10]
                ranked = [uniq[i] for i in order]
                metrics = pg.query_metrics(q["doc_id"], q["evidence"], [d.meta for d in ranked], corpus)
                per_query.append({"qa_id": q["qa_id"], "system_id": sid, "doc_id": q["doc_id"], **metrics})
                out_rows.append({"qa_id": q["qa_id"], "system_id": sid, "results": [
                    {"rank": k + 1, "score": float(scores[order[k]]), "content": d.content, "meta": d.meta}
                    for k, d in enumerate(ranked)]})
            with (args.out / "retrieval-full" / f"{sid}.jsonl").open("w", encoding="utf-8") as f:
                for r in out_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            mine = [p for p in per_query if p["system_id"] == sid]
            summary[sid] = {m: float(np.mean([p[m] for p in mine])) for m in ("recall@10", "recall@5", "ndcg@10", "hit@10")}
            print(f"{sid:45} " + " ".join(f"{k}={v:.4f}" for k, v in summary[sid].items()) +
                  f"  chunks={len(rows)}  t={time.time() - t0:.0f}s", flush=True)

    with (args.out / "evaluation-full" / "per_query_metrics.jsonl").open("w", encoding="utf-8") as f:
        for r in per_query:
            f.write(json.dumps(r) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({
        "proxy_of": "phase_g_runner retrieve+evaluate (hybrid BM25 + dense + Snowflake bi-encoder rerank)",
        "dense": {"model": args.dense_model,
                  "note": "default = Qwen3-Embedding-0.6B@97b0c61 (proxy); harness uses Qwen3-Embedding-4B"},
        "reranker": {"model": "Snowflake/snowflake-arctic-embed-l-v2.0", "revision": "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"},
        "prepared": str(args.prepared), "qa": str(args.qa), "systems": args.systems, "reprs": args.reprs,
        "summary": summary, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
