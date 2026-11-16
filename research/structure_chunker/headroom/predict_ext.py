import sys, numpy as np, pandas as pd
sys.path.insert(0, "research/structure_chunker/headroom")
from analyze_headroom import wide, oracle, CAND, boot
M = ["RC", "ICC", "DCC", "BI", "SC"]
# 1. proxy error measured on CLAIR for the same setting (real - predicted, per document)
real = wide("runs/sdp_sweep_v2/sw0_rc1/metrics/chunking_metrics.parquet").droplevel(0)
sw = pd.read_csv("runs/sdp_sweep_v2/sweep_per_doc.csv"); p = sw[sw.setting == "sw0_rc1"].set_index("doc_name").reindex(real.index)
e_icc = (real.ICC - p.ICC_pred).dropna(); e_dcc = (real.DCC - p.DCC_pred).dropna()
print(f"CLAIR proxy error (real - predicted): ICC {e_icc.mean():+.2f} (sd {e_icc.std():.2f}), DCC {e_dcc.mean():+.2f} (sd {e_dcc.std():.2f}, n={len(e_dcc)})")
# 2. clair_ext: Adaptive and SDP-B real, sw0_rc1 = real SC/BI/RC + proxy ICC/DCC
ext = wide("runs/ext_metrics_merged_v1/ext_v1/chunking_metrics.parquet"); ada = oracle(ext, CAND)
B = wide("runs/ext_metrics_merged_v1/sdp_b/chunking_metrics.parquet").droplevel(0).reindex(ada.index)
s = pd.read_csv("runs/sdp_sweep_ext_v1/sweep_per_doc.csv"); s = s[s.setting == "sw0_rc1"].set_index("doc_name").reindex(ada.index)
def frame(dcc_shift, icc_shift=0.0):
    f = pd.DataFrame({"RC": s.RC * 100, "ICC": s.ICC_pred + icc_shift, "DCC": s.DCC_pred + dcc_shift, "BI": s.BI * 100, "SC": s.SC * 100})
    f["mean5"] = f[M].mean(axis=1); return f
print(f"\n{'':34}" + "".join(f"{m:>8}" for m in M) + f"{'T':>8}{'per-doc':>9}   vs Adaptive [CI], higher in   vs SDP-B")
for n, t in (("Adaptive (real)", ada), ("SDP-B (real)", B)):
    print(f"{n:34}" + "".join(f"{t[m].mean():8.2f}" for m in M) + f"{t[M].mean().mean():8.2f}{t.mean5.mean():9.2f}")
for n, sh in (("sw0_rc1 raw proxy", 0.0), ("sw0_rc1 with CLAIR DCC correction", e_dcc.mean()), ("sw0_rc1 pessimistic (2x corr.)", 2 * e_dcc.mean())):
    t = frame(sh); d = (t.mean5 - ada.mean5).to_numpy(); lo, hi = boot(d); db = (t.mean5 - B.mean5).mean()
    print(f"{n:34}" + "".join(f"{t[m].mean():8.2f}" for m in M) + f"{t[M].mean().mean():8.2f}{t.mean5.mean():9.2f}   {d.mean():+.2f} [{lo:+.2f},{hi:+.2f}] {int((d > 0).sum())}/{len(d)}   {db:+.2f}")
print("\nAdaptive sd over documents on clair_ext:", {m: round(ada[m].std(), 1) for m in M + ["mean5"]})
print("RC is real here: sw0_rc1", round((s.RC * 100).mean(), 2), "vs SDP-B", round(B.RC.mean(), 2), "vs Adaptive", round(ada.RC.mean(), 2))
