# independent re-check, no helper imports
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
F = ["references_completeness", "intrachunk_cohesion", "document_contextual_coherence", "block_integrity", "size_compliance"]
def tab(path, method=None):
    d = pd.read_parquet(path)
    if method: d = d[d.chunking_method == method]
    d = d[d.metric_name.isin(F)]
    assert not d.duplicated(["doc_name", "metric_name"]).any()
    return d.pivot(index="doc_name", columns="metric_name", values="score")[F].astype(float) * 100
base = pd.read_parquet("runs/full_kaggle/results/chunking_metrics.parquet")
cands = {m: tab("runs/full_kaggle/results/chunking_metrics.parquet", m) for m in ["llm_regex", "our_recurs_1100", "our_recurs_600", "page"]}
docs = sorted(cands["page"].index)
ada = pd.DataFrame([cands[max(cands, key=lambda m: cands[m].loc[d].mean())].loc[d] for d in docs], index=docs)
B = tab("runs/sdp/B_default/metrics/chunking_metrics.parquet").reindex(docs)
raw = pd.read_parquet("runs/sdp_sweep_v2/sw0_rc1/metrics/chunking_metrics.parquet")
S = tab("runs/sdp_sweep_v2/sw0_rc1/metrics/chunking_metrics.parquet").reindex(docs)
print("sw0_rc1 metrics file: methods =", list(raw.chunking_method.unique()), "| documents =", raw.doc_name.nunique(), "| same 33 docs as baseline:", set(raw.doc_name) == set(docs))
ch = pd.read_parquet("runs/sdp_sweep_v2/sw0_rc1/chunks/small_merged/chunks.parquet")
print("chunks scored:", len(ch), "chunks,", ch.doc_name.nunique(), "docs; num_chunks metric sum =", int(raw[raw.metric_name == "num_chunks"].score.sum()))
print("documents with a value, per metric (RC ICC DCC BI SC):", S.notna().sum().tolist(), "| Adaptive:", ada.notna().sum().tolist(), "| SDP-B:", B.notna().sum().tolist())
print(f"\n{'row':26}{'RC':>8}{'ICC':>8}{'DCC':>8}{'BI':>8}{'SC':>8}{'T':>8}{'per-doc mean':>14}")
for n, t in (("Adaptive, rerun", ada), ("SDP-B", B), ("SDP-B optimised (sw0_rc1)", S)):
    m = t.mean()
    print(f"{n:26}" + "".join(f"{m[c]:8.2f}" for c in F) + f"{m.mean():8.2f}{t.mean(axis=1).mean():14.2f}")
rng = np.random.default_rng(2026)
for n, t in (("SDP-B", B), ("sw0_rc1", S)):
    d = (t.mean(axis=1) - ada.mean(axis=1)).to_numpy()
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    print(f"{n} - Adaptive: mean {d.mean():+.3f}  CI [{np.quantile(bs, .025):+.2f}, {np.quantile(bs, .975):+.2f}]  higher in {int((d > 0).sum())}/{len(d)}  Wilcoxon p = {wilcoxon(d, zero_method='pratt').pvalue:.2g};  T difference {t.mean().mean() - ada.mean().mean():+.2f}")

print('T exact:', round(ada.mean().mean(), 4), round(B.mean().mean(), 4), round(S.mean().mean(), 4), '| per-doc exact:', round(ada.mean(axis=1).mean(), 4), round(S.mean(axis=1).mean(), 4))
print('doc without DCC in sw0_rc1:', list(S.index[S.document_contextual_coherence.isna()]), '| docs without RC:', list(S.index[S.references_completeness.isna()]))
