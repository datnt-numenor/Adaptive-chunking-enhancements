"""Headroom analysis on EXISTING artifacts only (no GPU, no API).

A. Table 3 on CLAIR and on clair_ext: per-method scores, average chunk size, per-document oracles
   (best method per document = what the paper's Adaptive selector would do with a larger candidate pool)
   and the "utopia" bound (per-document best value of every metric taken independently - not attainable by
   one chunking, only an upper limit).
B. Does a higher Table-3 score go with better retrieval? Per-document differences (system - Adaptive) of the
   intrinsic metrics against the differences in recall@10 / nDCG@10 (Spearman, document = unit).
C. Table 5: where the missing points are (refusals vs answer quality), per-question oracle over systems,
   refusals split by recall@10 of the same question.

Usage:  python -X utf8 research/structure_chunker/headroom/analyze_headroom.py --out runs/headroom_analysis_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FIVE = {"size_compliance": "SC", "block_integrity": "BI", "intrachunk_cohesion": "ICC",
        "document_contextual_coherence": "DCC", "references_completeness": "RC"}
CAND = ["llm_regex", "our_recurs_1100", "our_recurs_600", "page"]
LINES: list[str] = []


def say(*parts) -> None:
    line = " ".join(str(p) for p in parts)
    print(line)
    LINES.append(line)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def wide(path: Path) -> pd.DataFrame:
    """(method, doc) x [SC, BI, ICC, DCC, RC, tok, n] in points."""
    df = pd.read_parquet(path)
    keep = list(FIVE) + ["avg_chunk_tokens", "num_chunks"]
    p = df[df.metric_name.isin(keep)].pivot_table(index=["chunking_method", "doc_name"], columns="metric_name",
                                                  values="score", dropna=False)
    p = p.reindex(columns=keep)
    p[list(FIVE)] *= 100
    p.columns = list(FIVE.values()) + ["tok", "n"]
    p["mean5"] = p[list(FIVE.values())].mean(axis=1)
    return p


def oracle(tab: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    sub = tab.loc[methods]
    pick = sub["mean5"].groupby("doc_name").idxmax()
    out = sub.loc[pick.values].copy()
    out.index = pd.MultiIndex.from_tuples(out.index)
    return out.droplevel(0)


def boot(d: np.ndarray, n: int = 10000, seed: int = 2026) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def table3_block(name: str, tab: pd.DataFrame, pools: dict[str, list[str]], out: Path) -> dict[str, pd.DataFrame]:
    cols = list(FIVE.values()) + ["mean5", "tok"]
    say(f"\n=== A. Table 3 - {name} ({tab.index.get_level_values(1).nunique()} docs) ===")
    say(f"{'method':30}" + "".join(f"{c:>8}" for c in cols))
    for m in tab.index.get_level_values(0).unique():
        r = tab.loc[m][cols].mean()
        say(f"{m:30}" + "".join(f"{r[c]:8.2f}" for c in cols))
    res = {}
    ada = oracle(tab, CAND)
    say("-- per-document oracle (argmax of the 5-metric mean; = Adaptive selector on that pool)")
    for label, methods in pools.items():
        o = oracle(tab, methods)
        res[label] = o
        d = (o["mean5"] - ada["mean5"]).to_numpy()
        lo, hi = boot(d)
        r = o[cols].mean()
        say(f"{label:30}" + "".join(f"{r[c]:8.2f}" for c in cols) + f"   vs Adaptive {d.mean():+.2f} [{lo:+.2f},{hi:+.2f}]")
    ut = tab[list(FIVE.values())].groupby("doc_name").max()
    say(f"{'utopia (each metric max)':30}" + "".join(f"{ut[c].mean():8.2f}" for c in FIVE.values())
        + f"{ut.mean(axis=1).mean():8.2f}" + "   <- upper limit, not one chunking")
    # size trade-off: within document, across methods
    rows = []
    for doc, g in tab.groupby(level=1):
        if len(g) >= 5:
            rows.append({"doc": doc, "tok~ICC": spearmanr(g.tok, g.ICC).statistic, "tok~DCC": spearmanr(g.tok, g.DCC).statistic,
                         "ICC~DCC": spearmanr(g.ICC, g.DCC).statistic, "tok~RC": spearmanr(g.tok, g.RC).statistic})
    if rows:
        tr = pd.DataFrame(rows).set_index("doc")
        say("-- within-document Spearman across methods (median over documents; share of documents with rho<0)")
        for c in tr:
            say(f"   {c:8} median {tr[c].median():+.2f}   negative in {int((tr[c] < 0).sum())}/{tr[c].notna().sum()} docs")
        tr.to_csv(out / f"tradeoff_{name}.csv")
    tab.reset_index().to_csv(out / f"table3_{name}.csv", index=False)
    return res


def relation_block(name: str, tab_by_system: dict[str, pd.DataFrame], retr: pd.DataFrame, ref: str, out: Path) -> None:
    """tab_by_system: system -> per-doc frame (index doc_name) with intrinsic columns; retr: system_id, doc_name, metrics."""
    say(f"\n=== B. Intrinsic vs retrieval, per-document differences vs {ref} - {name} ===")
    say(f"{'system':22}{'n':>4}{'d_mean5':>9}{'d_ICC':>8}{'d_DCC':>8}{'d_tok':>8}{'d_rec@10':>10}{'d_ndcg':>8}"
        f"   rho(d_mean5,d_rec) rho(d_ICC,d_rec) rho(d_DCC,d_rec) rho(d_tok,d_rec) rho(d_mean5,d_ndcg)")
    r0 = retr[retr.system_id == ref].set_index("doc_name")
    pooled = []
    for s, t in tab_by_system.items():
        if s == ref:
            continue
        r1 = retr[retr.system_id == s].set_index("doc_name")
        j = pd.DataFrame({
            "d_mean5": t["mean5"] - tab_by_system[ref]["mean5"], "d_ICC": t["ICC"] - tab_by_system[ref]["ICC"],
            "d_DCC": t["DCC"] - tab_by_system[ref]["DCC"], "d_tok": t["tok"] - tab_by_system[ref]["tok"],
            "d_rec": r1["recall@10"] - r0["recall@10"], "d_ndcg": r1["ndcg@10"] - r0["ndcg@10"]}).dropna()
        j["system"] = s
        pooled.append(j)

        def rho(a, b):
            x = spearmanr(j[a], j[b])
            return f"{x.statistic:+.2f} (p={x.pvalue:.2f})"
        say(f"{s:22}{len(j):4d}{j.d_mean5.mean():9.2f}{j.d_ICC.mean():8.2f}{j.d_DCC.mean():8.2f}{j.d_tok.mean():8.0f}"
            f"{j.d_rec.mean():10.3f}{j.d_ndcg.mean():8.3f}   {rho('d_mean5', 'd_rec'):>17} {rho('d_ICC', 'd_rec'):>16} "
            f"{rho('d_DCC', 'd_rec'):>16} {rho('d_tok', 'd_rec'):>16} {rho('d_mean5', 'd_ndcg'):>18}")
    if pooled:
        allj = pd.concat(pooled)
        allj.reset_index().to_csv(out / f"relation_{name}.csv", index=False)
        nz = allj[allj.d_rec != 0]
        say(f"   documents with any change in recall@10: {len(nz)}/{len(allj)} "
            f"(the rest are ties: the chunker did not change recall@10 at all)")


def t5_block(name: str, per_query: Path, retr_pq: pd.DataFrame | None, sysmap: dict[str, str], out: Path) -> None:
    say(f"\n=== C. Table 5 - {name} ===")
    q = pd.read_csv(per_query)
    say(f"{'system':28}{'n':>5}{'cov':>8}{'ans-only':>10}{'refuse':>8}{'cov if no refusal':>20}{'gap':>7}")
    for s, g in q.groupby("system", sort=False):
        ans = g[g.abstain == 0].correct_ans.mean() * 100
        say(f"{s:28}{len(g):5d}{g.correct_cov.mean() * 100:8.1f}{ans:10.1f}{int(g.abstain.sum()):8d}{ans:20.1f}"
            f"{ans - g.correct_cov.mean() * 100:7.1f}")
    piv = q.pivot(index="query_id", columns="system", values="correct_cov")
    ab = q.pivot(index="query_id", columns="system", values="abstain")
    say(f"per-question oracle over {piv.shape[1]} systems (best system chosen per question, NOT attainable): "
        f"{piv.max(axis=1).mean() * 100:.1f}")
    say(f"questions refused by ALL systems: {int((ab.min(axis=1) == 1).sum())}/{len(ab)};  by at least one: "
        f"{int((ab.max(axis=1) == 1).sum())};  by none: {int((ab.max(axis=1) == 0).sum())}")
    if retr_pq is not None:
        say("refusals by recall@10 of the same question and system:")
        for s, rs in sysmap.items():
            g = q[q.system == s].merge(retr_pq[retr_pq.system_id == rs][["qa_id", "recall@10"]],
                                       left_on="query_id", right_on="qa_id", how="inner")
            if g.empty:
                continue
            z, part, full = g[g["recall@10"] == 0], g[(g["recall@10"] > 0) & (g["recall@10"] < 1)], g[g["recall@10"] == 1]
            say(f"   {s:26} recall=0: {int(z.abstain.sum())}/{len(z)} refused | 0<recall<1: {int(part.abstain.sum())}/{len(part)}"
                f" | recall=1: {int(full.abstain.sum())}/{len(full)} refused, answered-only correctness {full[full.abstain == 0].correct_ans.mean() * 100:.1f}")
    piv.to_csv(out / f"t5_per_question_{name}.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists - choose a new folder")
    args.out.mkdir(parents=True)

    # ---------- A. CLAIR ----------
    base = wide(Path("runs/full_kaggle/results/chunking_metrics.parquet"))
    sdp_runs = ["A_default", "A_target600", "A_target800", "A_target1000", "B_default", "B_qwen"]
    parts = [base.loc[CAND + ["langch_recurs_default"]]]
    names = {}
    for r in sdp_runs:
        t = wide(Path(f"runs/sdp/{r}/metrics/chunking_metrics.parquet"))
        m = t.index.get_level_values(0).unique()[0]
        t = t.rename(index={m: r}, level=0)
        names[r] = m
        parts.append(t)
    clair = pd.concat(parts)
    a_runs = [r for r in sdp_runs if r.startswith("A_")]
    o_clair = table3_block("CLAIR", clair, {
        "Adaptive (4 cand.)": CAND, "SDP-A pool only": a_runs, "SDP-B pool (B_default,B_qwen)": ["B_default", "B_qwen"],
        "4 cand + SDP-A pool": CAND + a_runs, "all SDP (A+B)": sdp_runs, "4 cand + all SDP": CAND + sdp_runs}, args.out)

    # ---------- A. ext ----------
    ext = pd.concat([wide(Path("runs/ext_metrics_merged_v1/ext_v1/chunking_metrics.parquet")),
                     wide(Path("runs/ext_metrics_merged_v1/sdp_a/chunking_metrics.parquet")),
                     wide(Path("runs/ext_metrics_merged_v1/sdp_b/chunking_metrics.parquet"))])
    ext_sdp = [m for m in ext.index.get_level_values(0).unique() if m not in CAND]
    o_ext = table3_block("clair_ext", ext, {
        "Adaptive (4 cand.)": CAND, "SDP pool (A_t600,B)": ext_sdp, "4 cand + SDP pool": CAND + ext_sdp}, args.out)

    # ---------- B. relation ----------
    docmap = {d["doc_id"]: d["doc_name"] for d in read_jsonl(Path("runs/table5_v2/prepared/documents.jsonl"))}
    r1 = pd.read_csv("runs/table5_sdp_analysis/document_metrics.csv")
    r2 = pd.read_csv("runs/table5_repr_analysis_vs_adaptive/document_metrics.csv")
    retr = pd.concat([r1, r2[r2.system_id == "sdp_a_t600"]])
    retr["doc_name"] = retr.doc_id.map(docmap)
    relation_block("CLAIR_A5000", {
        "adaptive": o_clair["Adaptive (4 cand.)"], "sdp_a": clair.loc["A_target800"], "sdp_a_t600": clair.loc["A_target600"],
        "sdp_b": clair.loc["B_default"], "sdp_bq": clair.loc["B_qwen"]}, retr, "adaptive", args.out)

    docmap_e = {d["doc_id"]: d["doc_name"] for d in read_jsonl(Path("runs/ext_prepared_v1/prepared/documents.jsonl"))}
    re_ = pd.read_csv("runs/ext_4b_analysis_vs_adaptive/document_metrics.csv")
    re_["doc_name"] = re_.doc_id.map(docmap_e)
    a_name = [m for m in ext_sdp if "a" in m.replace("sdp_", "")][0] if any("a" in m.replace("sdp_", "") for m in ext_sdp) else ext_sdp[0]
    b_name = [m for m in ext_sdp if m != a_name][0]
    relation_block("clair_ext_4B", {
        "adaptive@plain": o_ext["Adaptive (4 cand.)"], "sdp_a_t600@plain": ext.loc[a_name], "sdp_b@plain": ext.loc[b_name]},
        re_, "adaptive@plain", args.out)

    # ---------- C. Table 5 ----------
    pq_c = pd.DataFrame(read_jsonl(Path("runs/table5_repr_results/table5_sdp/evaluation-full/per_query_metrics.jsonl")))
    pq_c2 = pd.DataFrame(read_jsonl(Path("runs/table5_sdp_results/table5_sdp/evaluation-full/per_query_metrics.jsonl")))
    pq_c = pd.concat([pq_c, pq_c2[~pq_c2.system_id.isin(pq_c.system_id.unique())]])
    say("\n(retrieval systems available for CLAIR per-query join: " + ", ".join(sorted(pq_c.system_id.unique())) + ")")
    t5_block("CLAIR_gateway", Path("runs/t5_analysis_v1/clair_gw/per_query.csv"), pq_c,
             {s: s for s in ["adaptive", "raw__langch_recurs_default", "raw__page", "adaptive__doc", "sdp_a_t600",
                             "sdp_a_t600__doc", "sdp_b"]}, args.out)
    pq_e = pd.DataFrame(read_jsonl(Path(
        "runs/gpu_retrieval_ext_4b_results/extracted_verified/out/ext_4b_part0/evaluation-full/per_query_metrics.jsonl")))
    t5_block("clair_ext_gateway", Path("runs/t5_analysis_v1/ext_gw/per_query.csv"), pq_e,
             {s: f"{s}@plain" for s in ["adaptive", "sdp_a_t600", "sdp_b", "raw__langch_recurs_default"]}, args.out)

    (args.out / "report.txt").write_text("\n".join(LINES), encoding="utf-8")
    (args.out / "manifest.json").write_text(json.dumps({
        "inputs": ["runs/full_kaggle/results", "runs/sdp/*/metrics", "runs/ext_metrics_merged_v1", "runs/table5_sdp_analysis",
                   "runs/table5_repr_analysis_vs_adaptive", "runs/ext_4b_analysis_vs_adaptive", "runs/t5_analysis_v1",
                   "runs/table5_*_results/*/evaluation-full", "runs/gpu_retrieval_ext_4b_results"],
        "sdp_method_names": names, "bootstrap": {"n": 10000, "seed": 2026}, "unit": "document"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
