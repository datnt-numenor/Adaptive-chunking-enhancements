"""Build a NEW retrieval input folder that adds SDP systems to the frozen Table-5 v2 inputs.

Nothing under runs/table5_v2 is modified: its prepared/ files, qa-full/ and qa-smoke/ are copied byte-for-byte
(sha256 checked), and new system files are written next to the copies. System rows are produced with the
harness's own helpers (phase_g_runner._load_source_documents / _find_chunk_spans), exactly as
prepare_table5_new_qa.py did for the existing systems.

A system can be
  * one SDP run:          --system sdp_a=runs/sdp/A_target800
  * a CV-selected mixture: --cv-system sdp_cv=runs/sdp/tune_A_targets   (uses manifest.json from tune.py: each
    document takes the chunks of the config chosen on the folds that do NOT contain it)

Usage (cmd):
  python -X utf8 research\\structure_chunker\\retrieval\\prepare_sdp_retrieval.py ^
     --harness D:\\Project\\Adaptive-chunking-enhancements ^
     --base runs\\table5_v2 --out runs\\table5_sdp_v1 ^
     --cv-system sdp_cv=runs\\sdp\\tune_A_targets --system sdp_b=runs\\sdp\\B_default
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA = Path("data/clair")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_harness(repo: Path):
    sys.path.insert(0, str(repo / "research" / "week3"))
    import phase_g_runner  # noqa: E402

    return phase_g_runner


def chunks_of_run(run: Path) -> pd.DataFrame:
    df = pd.read_parquet(run / "chunks" / "small_merged" / "chunks.parquet")
    assert df["method"].nunique() == 1, run
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", type=Path, required=True, help="clone of the group repo (read only)")
    ap.add_argument("--base", type=Path, default=Path("runs/table5_v2"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--system", action="append", default=[], help="name=run_dir")
    ap.add_argument("--cv-system", action="append", default=[], help="name=tune_dir")
    ap.add_argument("--kaggle-system", action="append", default=[],
                    help="name=stage:method from runs/full_kaggle/chunks, e.g. proc_recurs1100=small_merged:our_recurs_1100")
    ap.add_argument("--kaggle-run", type=Path, default=Path("runs/full_kaggle"))
    ap.add_argument("--doc-variants", default="",
                    help="comma list of systems (base or new) that also get a '<name>__doc' copy whose chunk_text is "
                         "prefixed with 'Document: <title>' (same as proxy_retrieval.py repr 'doc'); spans unchanged")
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} already exists - choose a new output folder (CLAUDE.md rule 2)")

    pg = load_harness(args.harness)
    documents = pg._load_source_documents(DATA)

    # 1. byte-for-byte copies of the frozen inputs
    for part in ("prepared", "qa-full", "qa-smoke"):
        shutil.copytree(args.base / part, args.out / part)
    for part in ("qa-full", "qa-smoke"):
        m = json.loads((args.out / part / "qa_frozen_manifest.json").read_text(encoding="utf-8"))
        assert sha(args.out / part / "qa_frozen.jsonl") == m["qa_sha256"], part
    base_manifest = json.loads((args.base / "prepared" / "prepare_manifest.json").read_text(encoding="utf-8"))
    for sid, rec in base_manifest["systems"].items():
        assert sha(args.out / "prepared" / rec["path"]) == rec["sha256"], sid

    # 2. new systems
    new_systems: dict[str, dict] = {}
    frames: dict[str, tuple[pd.DataFrame, dict]] = {}
    for item in args.system:
        name, run = item.split("=", 1)
        frames[name] = (chunks_of_run(Path(run)), {"source": run, "kind": "single_run"})
    for item in args.kaggle_system:
        name, spec = item.split("=", 1)
        stage, method = spec.split(":", 1)
        df = pd.read_parquet(args.kaggle_run / "chunks" / stage / "chunks.parquet")
        df = df[df["method"] == method].copy()
        assert len(df), item
        frames[name] = (df, {"source": f"{args.kaggle_run}/chunks/{stage}", "kind": "kaggle_fixed", "method": method})
    for item in args.cv_system:
        name, tune_dir = item.split("=", 1)
        tm = json.loads((Path(tune_dir) / "manifest.json").read_text(encoding="utf-8"))
        folds = pd.read_json(args.base / "prepared" / "folds.jsonl", lines=True).set_index("doc_name")["fold"]
        run_by_name = {Path(r).name: Path(r) for r in tm["runs"]}
        parts, choice = [], {}
        for doc_name, fold in folds.items():
            cfg = tm["chosen_per_fold"][str(int(fold))]["config"]
            df = chunks_of_run(run_by_name[cfg])
            parts.append(df[df.doc_name == doc_name])
            choice[doc_name] = cfg
        merged = pd.concat(parts, ignore_index=True)
        merged["method"] = name  # one method label for the mixture
        frames[name] = (merged, {"source": tune_dir, "kind": "cv_selected", "per_document_config": choice})

    for name, (df, info) in frames.items():
        if name in base_manifest["systems"]:
            sys.exit(f"system name {name} already exists in the base manifest")
        df = df.copy()
        df["method"] = name
        rows = pg._find_chunk_spans(df, documents)
        if {r["doc_name"] for r in rows} != set(documents):
            sys.exit(f"{name}: does not cover all 33 documents")
        path = args.out / "prepared" / "systems" / f"{name}.jsonl"
        pg._write_jsonl(path, rows)
        new_systems[name] = {"path": f"systems/{name}.jsonl", "sha256": sha(path), "chunks": len(rows),
                             "documents": len({r['doc_id'] for r in rows}), "stage": "small_merged", **info}
        print(f"{name:12} {len(rows):5d} chunks <- {info['source']}")

    for name in [x for x in args.doc_variants.split(",") if x]:
        src = new_systems.get(name) or base_manifest["systems"].get(name)
        if src is None:
            sys.exit(f"--doc-variants: unknown system {name}")
        rows = pg._read_jsonl(args.out / "prepared" / src["path"])
        for r in rows:
            title = r["doc_name"].split("_", 1)[1] if "_" in r["doc_name"] else r["doc_name"]
            r["chunk_text"] = f"Document: {title}\n\n" + r["chunk_text"]
            r["chunk_id"] = r["chunk_id"].replace(f"::{r['method']}::", f"::{name}__doc::")
            r["method"] = f"{name}__doc"
        vname = f"{name}__doc"
        path = args.out / "prepared" / "systems" / f"{vname}.jsonl"
        pg._write_jsonl(path, rows)
        new_systems[vname] = {"path": f"systems/{vname}.jsonl", "sha256": sha(path), "chunks": len(rows),
                              "documents": len({r['doc_id'] for r in rows}), "stage": src.get("stage", "base"),
                              "kind": "doc_title_prefix", "base_system": name,
                              "prefix": "Document: <doc_name without category>\n\n"}
        print(f"{vname:12} {len(rows):5d} chunks (doc-title prefix of {name})")

    manifest = dict(base_manifest)
    manifest["systems"] = {**base_manifest["systems"], **new_systems}
    manifest["sdp_extension"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_prepared_manifest_sha256": sha(args.base / "prepared" / "prepare_manifest.json"),
        "harness_repo": str(args.harness), "harness_sha256": sha(args.harness / "research/week3/phase_g_runner.py"),
        "new_systems": sorted(new_systems),
        "note": "Base files copied unchanged; new systems appended. Launcher registers their names in ALL_SYSTEMS.",
    }
    (args.out / "prepared" / "prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
