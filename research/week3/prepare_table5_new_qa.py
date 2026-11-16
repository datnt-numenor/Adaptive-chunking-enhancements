"""Tao dau vao Table 5 (3 he thong) tu ket qua chunking/metrics cua ban va bo QA moi da freeze.

Dung lai dung ham cua phase_g_runner.py (khong sua code nhom), nhung khong doi Week 2 artifacts
(validation_report / metrics_merge_manifest) va khong can semantic, vi Table 5 chi dung:
    adaptive (4 ung vien, trong so bang nhau), raw__langch_recurs_default, raw__page
Ghi ra <output-root>/:
    prepared/   systems/*.jsonl, documents.jsonl, folds.jsonl, adaptive_selections.json, prepare_manifest.json
    qa-full/    qa_frozen.jsonl + qa_frozen_manifest.json   (99 QA, 33 tai lieu)
    qa-smoke/   qa_frozen.jsonl + qa_frozen_manifest.json   (6 QA, 2 tai lieu - bat buoc chay truoc full)
Chay tu thu muc goc repo nhom:
    python -X utf8 research/week3/prepare_table5_new_qa.py --run-dir <...>/runs/full_kaggle
        --data-dir data/clair --qa-frozen <...>/qa_new_frozen.jsonl
        --qa-manifest <...>/qa_new_frozen_manifest.json --output-root research/week3/artifacts/table5-qa-new
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase_g_runner as pg  # noqa: E402

SYSTEMS = {"adaptive": None, **{s: pg.FIXED_SYSTEMS[s] for s in ("raw__langch_recurs_default", "raw__page")}}


def sha256(path: Path) -> str:
    return pg._sha256_file(path)


def build_prepared(args, documents, root: Path):
    processed = pd.read_parquet(args.run_dir / "chunks" / "small_merged" / "chunks.parquet")
    raw = pd.read_parquet(args.run_dir / "chunks" / "raw" / "chunks.parquet")
    metrics_path = args.run_dir / "results" / "chunking_metrics.parquet"
    metrics = pd.read_parquet(metrics_path)
    selections = pg._select_adaptive_methods(metrics)

    frames = {"adaptive": pd.concat(
        [processed[(processed["doc_name"] == d) & (processed["method"] == m)] for d, m in selections.items()],
        ignore_index=True)}
    for system_id, spec in SYSTEMS.items():
        if spec is None:
            continue
        stage, method = spec
        source = processed if stage == "small_merged" else raw
        frames[system_id] = source[source["method"] == method].copy()
    for system_id, frame in frames.items():
        if frame["doc_name"].nunique() != 33:
            raise pg.PhaseGError(f"{system_id} khong phu du 33 tai lieu")

    out = root / "prepared"
    records = {}
    for system_id, frame in frames.items():
        rows = pg._find_chunk_spans(frame, documents)
        path = out / "systems" / f"{system_id}.jsonl"
        pg._write_jsonl(path, rows)
        records[system_id] = {
            "path": path.relative_to(out).as_posix(), "sha256": sha256(path), "chunks": len(rows),
            "excluded_empty_chunks": int(frame["chunk_text"].fillna("").str.strip().eq("").sum()),
            "documents": len({r["doc_id"] for r in rows}),
            "stage": "mixed_processed_selection" if system_id == "adaptive" else SYSTEMS[system_id][0],
            "method": "per_document_equal_weight" if system_id == "adaptive" else SYSTEMS[system_id][1],
        }
    document_rows = [
        {k: v for k, v in doc.items() if k not in {"full_text", "page_spans"}}
        | {"page_spans": [{k: v for k, v in p.items() if k != "text"} for p in doc["page_spans"]]}
        for doc in sorted(documents.values(), key=lambda item: item["doc_id"])
    ]
    pg._write_jsonl(out / "documents.jsonl", document_rows)
    pg._json_dump(out / "adaptive_selections.json",
                  {"candidate_methods": list(pg.ADAPTIVE_CANDIDATES), "weights": pg.WEIGHTS, "selections": selections})
    pg._write_jsonl(out / "folds.jsonl", pg._build_folds(documents))
    manifest = {
        "schema_version": pg.SCHEMA_VERSION, "harness_sha256": pg._harness_sha256(),
        "adapter": Path(__file__).name, "adapter_sha256": sha256(Path(__file__).resolve()),
        "status": "complete", "created_at": pg._utc_now(), "source_commit": pg.SOURCE_COMMIT,
        "note": "Tao tu ket qua chay doc lap (khong co semantic); chi gom 3 he thong cua Table 5.",
        "input_run_dir": str(args.run_dir),
        "input_sha256": {"processed_chunks": sha256(args.run_dir / "chunks" / "small_merged" / "chunks.parquet"),
                         "raw_chunks": sha256(args.run_dir / "chunks" / "raw" / "chunks.parquet"),
                         "processed_metrics": sha256(metrics_path)},
        "documents": 33, "folds": 5, "fold_seed": pg.FOLD_SEED,
        "adaptive_candidates": list(pg.ADAPTIVE_CANDIDATES),
        "paper_protocol_systems": list(pg.PAPER_PROTOCOL_SYSTEMS), "systems": records,
    }
    pg._json_dump(out / "prepare_manifest.json", manifest)
    return selections, frames, records


def convert_qa(args, documents):
    expected = json.loads(args.qa_manifest.read_text(encoding="utf-8"))
    if sha256(args.qa_frozen) != expected["sha256"]:
        raise pg.PhaseGError("SHA256 cua qa_new_frozen.jsonl khong khop manifest - file da bi sua?")
    rows = []
    for line in args.qa_frozen.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        doc = documents[item["doc_name"]]
        qa = {
            "qa_id": f"qnew-{item['qa_id']}", "doc_id": doc["doc_id"], "doc_name": item["doc_name"],
            "domain": doc["domain"], "question": item["question"].strip(),
            "reference_answer": item["answer"].strip(),
            "evidence": pg._map_evidence(doc, item["evidence"]),
            "validation_status": "valid", "validation_error": "",
            "review": {"status": "accept", "reviewer": "validate_qa.py + human", "notes": ""},
        }
        qa["qa_hash"] = pg._qa_content_hash(qa)
        rows.append(qa)
    per_doc = Counter(r["doc_id"] for r in rows)
    if (len(rows) != 99 or len(per_doc) != 33 or set(per_doc.values()) != {3}
            or len({r["qa_hash"] for r in rows}) != 99 or len({r["question"] for r in rows}) != 99):
        raise pg.PhaseGError("Can dung 99 QA duy nhat, 3 cau cho moi tai lieu trong 33 tai lieu")
    return sorted(rows, key=lambda r: r["qa_id"]), expected


def write_qa(directory: Path, rows, mode, expected):
    path = directory / "qa_frozen.jsonl"
    pg._write_jsonl(path, rows)
    pg._json_dump(directory / "qa_frozen_manifest.json", {
        "schema_version": pg.SCHEMA_VERSION, "harness_sha256": pg._harness_sha256(), "status": "complete",
        "mode": mode, "created_at": pg._utc_now(), "qa_count": len(rows),
        "documents": len({r["doc_id"] for r in rows}), "qa_sha256": sha256(path),
        "generation_model": expected.get("generator"), "source_qa_sha256": expected["sha256"],
        "source_prompt_sha256": expected.get("prompt_sha256"), "human_review_required": True,
    })
    return sha256(path)


def smoke_documents(frames, documents):
    """Cung quy tac voi prepare(): tai lieu co chunk dai nhat, roi tai lieu dai nhat thuoc domain khac."""
    chars = pd.concat([f[["doc_name", "chunk_text"]] for f in frames.values()], ignore_index=True)
    chars["n"] = chars["chunk_text"].map(lambda v: len(str(v)))
    order = chars.groupby("doc_name")["n"].max().sort_values(ascending=False).index.tolist()
    first = order[0]
    second = next(d for d in order[1:] if documents[d]["domain"] != documents[first]["domain"])
    return [first, second]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--qa-frozen", type=Path, required=True)
    parser.add_argument("--qa-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        sys.exit(f"{args.output_root} da ton tai - chon thu muc moi, khong ghi de ket qua cu")
    # Ghi vao thu muc tam, chi doi ten khi moi buoc thanh cong -> lan chay loi khong de lai ket qua do dang
    root = args.output_root.with_name(args.output_root.name + ".partial")
    if root.exists():
        sys.exit(f"{root} con sot tu lan chay loi truoc - xem roi xoa thu cong")

    documents = pg._load_source_documents(args.data_dir)
    qas, expected = convert_qa(args, documents)          # kiem tra QA truoc khi ghi bat cu thu gi
    selections, frames, records = build_prepared(args, documents, root)
    full_sha = write_qa(root / "qa-full", qas, "full", expected)
    smoke_names = smoke_documents(frames, documents)
    smoke_ids = {documents[d]["doc_id"] for d in smoke_names}
    smoke_sha = write_qa(root / "qa-smoke", [q for q in qas if q["doc_id"] in smoke_ids], "smoke", expected)

    root.rename(args.output_root)
    print("Adaptive chon:", dict(Counter(selections.values())))
    for system_id, record in records.items():
        print(f"  {system_id:28} {record['chunks']:6d} chunk, {record['documents']} tai lieu, "
              f"bo {record['excluded_empty_chunks']} chunk rong")
    print(f"QA full : 99 cau, sha256 {full_sha}")
    print(f"QA smoke: 6 cau tu {smoke_names}, sha256 {smoke_sha}")
    print(f"Da ghi vao {args.output_root}")


if __name__ == "__main__":
    main()
