"""Structural quality of the cuts (CPU): what the Table-3 metrics do not see."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy_sweep as ps

RUNS = {"A_target600": "runs/sdp/A_target600", "B_default": "runs/sdp/B_default", "sw0": "runs/sdp_sweep_v2/sw0",
        "rc1": "runs/sdp_sweep_v1/rc1", "sw0_rc1": "runs/sdp_sweep_v2/sw0_rc1", "sw0_dcc1.5_rc1": "runs/sdp_sweep_v2/sw0_dcc1.5_rc1"}
if len(sys.argv) > 1:
    RUNS = {Path(a).name: a for a in sys.argv[1:]}
docs = ps.load_docs()
print(f"{'run':16}{'chunks':>7}{'cuts':>6}{'before heading':>16}{'mid-paragraph':>15}{'starts w/ page artefact':>25}{'heading left at chunk end':>27}{'pairs split':>13}")
for name, run in RUNS.items():
    ch = pd.read_parquet(f"{run}/chunks/small_merged/chunks.parquet").sort_values(["doc_name", "chunk_index"])
    n = cuts = head = lower = art = orphan = 0
    split = total = 0
    for doc, g in ch.groupby("doc_name"):
        d = docs[doc]; u = d["units"]
        b = ps.unit_bounds(u, list(g.chunk_text))
        n += len(b)
        for i, _ in b[1:]:
            cuts += 1
            head += u[i].heading_level > 0
            lower += u[i].starts_lower
            art += u[i].is_artifact
            prev = next((u[k] for k in range(i - 1, -1, -1) if not u[k].is_artifact), None)
            orphan += bool(prev and prev.heading_level > 0)
        if d["pairs"] is not None:
            starts = sorted(u[i].start for i, _ in b[1:])
            for a, c in d["pairs"]:
                lo, hi = (a, c) if a[0] <= c[0] else (c, a)
                split += any(lo[0] < s <= hi[1] for s in starts); total += 1
    print(f"{name:16}{n:7d}{cuts:6d}{head:10d} ({head/cuts:4.0%}){lower:9d} ({lower/cuts:4.0%}){art:19d} ({art/cuts:4.0%}){orphan:21d} ({orphan/cuts:4.0%}){split:7d}/{total}")
