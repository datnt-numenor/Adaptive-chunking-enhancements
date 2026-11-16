"""Buoc 6 (Kaggle): gop ket qua cac shard thanh runs/full_kaggle de chay Buoc 7.

Giai nen MOI lan chay Kaggle vao mot thu muc rieng trong runs\kaggle\ (vd shard0, shard1, shard1_raw, ...)
roi chay:  python -X utf8 runs\merge_kaggle_shards.py
Mot shard co the ghep tu 2 lan chay: lay metrics tu lan chay co metrics thanh cong, raw_metrics tu lan
chay co raw_metrics thanh cong. Script khong ghi de runs/full; chunks duoc chep tu runs/full/chunks.
"""
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

KAGGLE = Path("runs/kaggle")
CHUNKS = Path("runs/full/chunks")
DEST = Path("runs/full_kaggle")
KEY = ["doc_name", "chunking_method", "metric_name"]
PARTS = {"metrics": ("proc", "results"), "raw_metrics": ("raw", "results_raw")}

chosen = defaultdict(dict)   # shard -> step -> thu muc out
shard_docs = {}
for o in sorted(m.parent for m in KAGGLE.rglob("shard_manifest.json")):
    manifest = json.loads((o / "shard_manifest.json").read_text(encoding="utf-8"))
    shard = manifest["shard"]
    if shard in shard_docs and shard_docs[shard] != manifest["docs"]:
        sys.exit(f"DUNG LAI: shard {shard} co 2 danh sach tai lieu khac nhau ({o})")
    shard_docs[shard] = manifest["docs"]
    status_path = o / "run_status.json"
    if not status_path.exists():
        print(f"Bo qua {o}: thieu run_status.json")
        continue
    for step, job in json.loads(status_path.read_text(encoding="utf-8"))["jobs"].items():
        ok = job["returncode"] == 0
        print(f"  {o.parent.name}: shard {shard} {step}: {'OK' if ok else 'LOI'} ({job['minutes']} phut)")
        if ok and step not in chosen[shard]:
            chosen[shard][step] = o

problems = [f"shard {s}: chua co {step} thanh cong" for s in sorted(shard_docs)
            for step in PARTS if step not in chosen[s]]
docs = [d for s in shard_docs.values() for d in s]
expected = {p.stem for p in Path("data/clair/adi_parsed").glob("*.json")}
if len(docs) != len(set(docs)):
    problems.append("co tai lieu nam o 2 shard khac nhau")
if set(docs) != expected:
    problems.append(f"thieu {len(expected - set(docs))} tai lieu: {sorted(expected - set(docs))}")
if problems:
    sys.exit("DUNG LAI:\n  " + "\n  ".join(problems))
if DEST.exists():
    sys.exit(f"{DEST} da ton tai - doi ten hoac xoa no truoc khi gop lai")

shutil.copytree(CHUNKS, DEST / "chunks")
for step, (kind, sub) in PARTS.items():
    groups = defaultdict(list)
    for shard in sorted(chosen):
        base = chosen[shard][step] / kind / sub
        for f in base.rglob("*"):
            if f.is_file():
                groups[f.relative_to(base)].append(f)
    for rel, files in sorted(groups.items()):
        target = DEST / sub / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.suffix == ".parquet":
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            if set(KEY) <= set(df.columns):
                dup = int(df.duplicated(KEY).sum())
                print(f"{sub}/{rel}: {len(df):,} dong, {df['doc_name'].nunique()} tai lieu, "
                      f"{df['chunking_method'].nunique()} phuong phap, {df['metric_name'].nunique()} metric, trung lap: {dup}")
            df.to_parquet(target)
        elif len(files) == 1:
            shutil.copy2(files[0], target)
        else:
            print(f"Bo qua {sub}/{rel}: {len(files)} file khong phai parquet trung ten")
print(f"\nDa gop vao {DEST}")