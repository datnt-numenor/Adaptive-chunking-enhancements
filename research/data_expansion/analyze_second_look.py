"""Statistics for the second look at clair_ext (pre-registered in PREREGISTRATION_2026-11-12_table3_second_look.md).

Reads the pod output of run_metrics_remote_v2.sh (<pod>/ext/<run>/gpu*/chunking_metrics.parquet and <pod>/clair/...),
the frozen Adaptive / SDP-B test-set scores (runs/ext_metrics_merged_v1) and the laptop CLAIR scores.

Test set: per-document mean of the 5 metrics (metrics that cannot be computed for a document are left out, as in
compare.py); paired difference vs Adaptive and vs SDP-B; 10,000-sample document bootstrap (seed 2026); Wilcoxon (Pratt).
H4 (sw0_rc1 vs Adaptive) is the single primary test. H5 and the secondary comparisons are Holm-corrected together.

  python -X utf8 research/data_expansion/analyze_second_look.py --pod <extracted>/metrics_out --out runs/ext_second_look_analysis_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

FIVE = {"references_completeness": "RC", "intrachunk_cohesion": "ICC", "document_contextual_coherence": "DCC",
        "block_integrity": "BI", "size_compliance": "SC"}
CAND = ["llm_regex", "our_recurs_1100", "our_recurs_600", "page"]
SEED, N_BOOT = 2026, 10_000
PRIMARY = "sw0_rc1"
SECONDARY = ["rc1", "keepcont_rc1", "noart_rc1"]
LINES: list[str] = []


def say(*p) -> None:
    line = " ".join(str(x) for x in p)
    print(line)
    LINES.append(line)


def table(df: pd.DataFrame) -> pd.DataFrame:
    """doc x [RC ICC DCC BI SC mean5] in points; one chunking method per frame."""
    d = df[df.metric_name.isin(FIVE)]
    assert not d.duplicated(["doc_name", "metric_name"]).any(), "duplicate (doc, metric) rows"
    t = d.pivot(index="doc_name", columns="metric_name", values="score").reindex(columns=list(FIVE)).astype(float) * 100
    t.columns = list(FIVE.values())
    t["mean5"] = t[list(FIVE.values())].mean(axis=1)
    return t


def pod_run(root: Path, run: str) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted((root / run).glob("gpu*/chunking_metrics.parquet"))]
    if not parts:
        raise SystemExit(f"no pod output for {run} under {root}")
    return table(pd.concat(parts, ignore_index=True))


def adaptive(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    cands = {m: table(df[df.chunking_method == m]) for m in CAND}
    docs = sorted(set.intersection(*(set(t.index) for t in cands.values())))
    return pd.DataFrame([cands[max(cands, key=lambda m: cands[m].loc[d, "mean5"])].loc[d] for d in docs], index=docs)


def holm(p: list[float]) -> list[float]:
    order, out, run = np.argsort(p), [0.0] * len(p), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        out[i] = run
    return out


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str = "mean5") -> dict:
    j = pd.concat([a[col], b[col]], axis=1, join="inner").dropna()
    d = (j.iloc[:, 0] - j.iloc[:, 1]).to_numpy()
    rng = np.random.default_rng(SEED)
    bs = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    return {"diff": float(d.mean()), "ci_low": float(np.quantile(bs, .025)), "ci_high": float(np.quantile(bs, .975)),
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()), "n": len(d),
            "p": float(wilcoxon(d, zero_method="pratt").pvalue) if np.any(d) else 1.0, "sd_diff": float(d.std(ddof=1))}


def show(label: str, t: pd.DataFrame) -> None:
    m = t.mean()
    say(f"{label:26}" + "".join(f"{m[c]:8.2f}" for c in FIVE.values()) + f"{m[list(FIVE.values())].mean():8.2f}{m['mean5']:10.2f}"
        f"   sd(mean5) {t['mean5'].std():.2f}   n={len(t)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", type=Path, required=True, help="extracted metrics_out of run_metrics_remote_v2.sh")
    ap.add_argument("--ext-base", type=Path, default=Path("runs/ext_metrics_merged_v1"))
    ap.add_argument("--primary", default=PRIMARY)
    ap.add_argument("--secondary", nargs="*", default=SECONDARY)
    ap.add_argument("--skip-clair", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists - choose a new folder")

    ada = adaptive(a.ext_base / "ext_v1" / "chunking_metrics.parquet")
    sdpb = table(pd.read_parquet(a.ext_base / "sdp_b" / "chunking_metrics.parquet"))
    runs = {r: pod_run(a.pod / "ext", r) for r in [a.primary] + list(a.secondary)}
    for r, t in runs.items():
        missing = sorted(set(ada.index) - set(t.index))
        if missing:
            raise SystemExit(f"{r}: {len(missing)} test documents missing in the pod output, e.g. {missing[:3]}")
    say(f"=== Test set clair_ext, {len(ada)} documents (points) ===")
    say(f"{'system':26}" + "".join(f"{c:>8}" for c in FIVE.values()) + f"{'T':>8}{'per-doc':>10}")
    show("Adaptive (4 cand.)", ada)
    show("SDP-B", sdpb.reindex(ada.index))
    for r, t in runs.items():
        show(r, t.reindex(ada.index))

    rows = []
    h4 = {"hypothesis": "H4 (primary)", "system": a.primary, "reference": "Adaptive", **paired(runs[a.primary], ada)}
    h4["p_holm"] = h4["p"]
    rows.append(h4)
    fam = [{"hypothesis": "H5", "system": a.primary, "reference": "SDP-B", **paired(runs[a.primary], sdpb)}]
    for r in a.secondary:
        fam.append({"hypothesis": "secondary", "system": r, "reference": "Adaptive", **paired(runs[r], ada)})
        fam.append({"hypothesis": "secondary", "system": r, "reference": "SDP-B", **paired(runs[r], sdpb)})
    for f, ph in zip(fam, holm([f["p"] for f in fam])):
        f["p_holm"] = ph
    rows += fam
    comp = pd.DataFrame(rows)
    say("\n=== Paired comparisons of the per-document 5-metric mean ===")
    for _, r in comp.iterrows():
        say(f"{r.hypothesis:13} {r.system:14} - {r.reference:9} {r['diff']:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}]  "
            f"wins/ties/losses {r.wins}/{r.ties}/{r.losses}  p = {r.p:.2g}  Holm p = {r.p_holm:.2g}  sd of diff {r.sd_diff:.2f}")
    say("\n=== Per-metric gain of the primary system over Adaptive (and Adaptive's sd across documents) ===")
    met = []
    for c in list(FIVE.values()) + ["mean5"]:
        x = paired(runs[a.primary], ada, c)
        sd = float(ada[c].std())
        met.append({"metric": c, **x, "adaptive_sd": sd, "exceeds_sd": bool(x["diff"] > sd)})
        say(f"   {c:6} {x['diff']:+.2f} [{x['ci_low']:+.2f}, {x['ci_high']:+.2f}]  wins {x['wins']}/{x['n']}   "
            f"Adaptive sd {sd:.1f}   gain > sd: {'YES' if x['diff'] > sd else 'no'}")

    clair_rows = []
    if not a.skip_clair and (a.pod / "clair").exists():
        say("\n=== CLAIR on the pod ===")
        lap = table(pd.read_parquet("runs/sdp_sweep_v2/sw0_rc1/metrics/chunking_metrics.parquet"))
        pod = pod_run(a.pod / "clair", "sw0_rc1").reindex(lap.index)
        diff = (pod[list(FIVE.values())] - lap[list(FIVE.values())]).abs() / 100
        say(f"sw0_rc1, pod vs laptop (same chunks): max |difference| = {np.nanmax(diff.to_numpy()):.2e} (0-1 scale); "
            f"mean5 {pod['mean5'].mean():.3f} vs {lap['mean5'].mean():.3f}")
        base = pd.read_parquet("runs/full_kaggle/results/chunking_metrics.parquet")
        cands = {m: table(base[base.chunking_method == m]) for m in CAND}
        docs = sorted(cands["page"].index)
        ada_c = pd.DataFrame([cands[max(cands, key=lambda m: cands[m].loc[d, "mean5"])].loc[d] for d in docs], index=docs)
        for r in ["sw0_rc1", "keepcont_rc1", "noart_rc1"]:
            t = pod_run(a.pod / "clair", r)
            x = paired(t, ada_c)
            clair_rows.append({"system": r, **x, "T": float(t[list(FIVE.values())].mean().mean())})
            show(r, t.reindex(docs))
            say(f"      - Adaptive: {x['diff']:+.2f} [{x['ci_low']:+.2f}, {x['ci_high']:+.2f}]  wins {x['wins']}/{x['n']}  p = {x['p']:.2g}")

    a.out.mkdir(parents=True)
    comp.to_csv(a.out / "comparisons.csv", index=False)
    pd.DataFrame(met).to_csv(a.out / "primary_by_metric.csv", index=False)
    if clair_rows:
        pd.DataFrame(clair_rows).to_csv(a.out / "clair_pod.csv", index=False)
    per_doc = pd.DataFrame({"adaptive": ada["mean5"], "sdp_b": sdpb["mean5"], **{r: t["mean5"] for r, t in runs.items()}})
    per_doc.to_csv(a.out / "per_document_mean5.csv")
    (a.out / "report.txt").write_text("\n".join(LINES), encoding="utf-8")
    (a.out / "manifest.json").write_text(json.dumps({
        "pod": str(a.pod), "ext_base": str(a.ext_base), "primary": a.primary, "secondary": a.secondary,
        "bootstrap": {"n": N_BOOT, "seed": SEED}, "unit": "document"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
