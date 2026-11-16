import sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, "research/structure_chunker/headroom")
from analyze_headroom import wide, oracle, CAND, boot
ada = oracle(wide("runs/full_kaggle/results/chunking_metrics.parquet"), CAND)
B = wide("runs/sdp/B_default/metrics/chunking_metrics.parquet").droplevel(0).reindex(ada.index)
S = wide("runs/sdp_sweep_v2/sw0_rc1/metrics/chunking_metrics.parquet").droplevel(0).reindex(ada.index)
for name, ref in (("Adaptive", ada), ("SDP-B", B)):
    d = (S.mean5 - ref.mean5).to_numpy(); lo, hi = boot(d)
    print(f"sw0_rc1 - {name}: {d.mean():+.2f} [{lo:+.2f},{hi:+.2f}] wins {int((d>0).sum())} ties {int((d==0).sum())} losses {int((d<0).sum())} / {len(d)}  Wilcoxon p={wilcoxon(d, zero_method='pratt').pvalue:.2g}  sd of diff {d.std(ddof=1):.2f}")
    for m in ("ICC", "DCC", "RC"):
        x = (S[m] - ref[m]).dropna().to_numpy(); lo, hi = boot(x)
        print(f"     {m}: {x.mean():+.2f} [{lo:+.2f},{hi:+.2f}] wins {int((x>0).sum())}/{len(x)}")
bad = (S.mean5 - ada.mean5).sort_values().head(3)
print("lowest per-document differences vs Adaptive:"); print(bad.round(2).to_string())
k = bad.index[0]
print(pd.DataFrame({"Adaptive": ada.loc[k], "SDP-B": B.loc[k], "sw0_rc1": S.loc[k]}).round(2).to_string())
