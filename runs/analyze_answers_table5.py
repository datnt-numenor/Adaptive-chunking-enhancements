"""Thong ke chat luong cau tra loi Table 5 theo tai lieu + noi voi retrieval (cung phuong phap Phase H/J).

Don vi = tai lieu (trung binh 3 cau). 95% CI bootstrap 10.000 mau seed 2026; Wilcoxon 2 phia (pratt); Holm tren
cac doi chung trong moi metric; rank-biserial; thang/hoa/thua; MDE 80%.
Metric theo cau: Retrieval Completeness (judge), Correctness tinh ca do phu (tu choi = 0), tu choi (0/1).
Tu choi = cau tra loi CHUA chuoi tu choi (nhu phase_i_judge.py va rag_eval.py); dem rieng "tu choi kem chu".
Kiem tra: tinh lai trung binh tung he thong va doi chieu table5_summary.json (lech > 1e-9 -> dung).
Cau hoi audit: noi per_query_metrics (recall@10, nDCG@10) voi ket qua cham cua CUNG cau.

  python -X utf8 analyze_answers_table5.py --judge-dir %OUT%\\judge-full --summary %OUT%\\summary\\table5_summary.json
     --retrieval-eval D:\\Project\\adaptive-chunking\\runs\\table5_v2_results\\table5_v2\\evaluation-full
     --prepared D:\\Project\\adaptive-chunking\\runs\\table5_v2\\prepared
     --label "gpt-4o-mini-2024-07-18 qua gateway (phat trien)" --output-dir D:\\Project\\adaptive-chunking\\runs\\t5_dev_gw_4omini_analysis
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, spearmanr, wilcoxon

SEED, BOOT = 2026, 10_000
ABSTAIN = "I don't know based on the provided context."
SYSTEMS = ["adaptive", "raw__langch_recurs_default", "raw__page"]
REFERENCE = "adaptive"
METRICS = {"completeness": "Retrieval Completeness (judge)", "correct_cov": "Correctness tinh do phu (tu choi = 0)",
           "abstain": "Ty le tu choi (thap hon = tot hon)"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def boot_ci(values, rng) -> tuple[float, float]:
    v = np.asarray(values, float)
    idx = rng.integers(0, len(v), size=(BOOT, len(v)))
    lo, hi = np.quantile(v[idx].mean(axis=1), [0.025, 0.975])
    return float(lo), float(hi)


def holm(p: list[float]) -> list[float]:
    order, out, run = sorted(range(len(p)), key=lambda i: p[i]), [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        out[i] = run
    return out


def rank_biserial(d: np.ndarray) -> float:
    nz = d[d != 0]
    if not len(nz):
        return 0.0
    r = rankdata(np.abs(nz))
    pos, neg = float(r[nz > 0].sum()), float(r[nz < 0].sum())
    return (pos - neg) / (pos + neg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge-dir", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True, help="table5_summary.json do phase_i_summarize tao")
    ap.add_argument("--retrieval-eval", type=Path, required=True, help="thu muc evaluation-full (per_query_metrics.jsonl)")
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--label", required=True, help="mo ta model/nha cung cap, in vao bao cao")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        sys.exit(f"{args.output_dir} da ton tai - khong ghi de")
    started = datetime.now(timezone.utc).isoformat()

    folds = pd.DataFrame(read_jsonl(args.prepared / "folds.jsonl"))
    by_name = folds.set_index("doc_name")
    rows, contexts = [], {}
    for system in SYSTEMS:
        path = args.judge_dir / system / "generated_questions_evaluation_results.json"
        for row in json.loads(path.read_text(encoding="utf-8")):
            metrics = {m["name"]: m for m in row["metrics"]}
            if any(m.get("error") for m in metrics.values()):
                sys.exit(f"{system}/{row['query_id']}: metric co loi - kiem tra lai buoc cham")
            out = row["generated_output"] or ""
            abstain = ABSTAIN in out
            if abstain == ("Correctness [GEval]" in metrics):
                sys.exit(f"{system}/{row['query_id']}: tu choi va Correctness khong nhat quan")
            correct = None if abstain else min(max(float(metrics["Correctness [GEval]"]["score"]), 0.0), 1.0)
            doc_name = row["reference_answer_doc_name"]
            rows.append({"system_id": system, "qa_id": row["query_id"], "doc_name": doc_name,
                         "doc_id": by_name.loc[doc_name, "doc_id"], "domain": by_name.loc[doc_name, "domain"],
                         "completeness": min(max(float(metrics["Retrieval Completeness"]["score"]), 0.0), 1.0),
                         "correct": correct, "correct_cov": 0.0 if abstain else correct, "abstain": float(abstain),
                         "partial_abstain": abstain and out.strip() != ABSTAIN})
            contexts[(system, row["query_id"])] = json.dumps([c.get("content") for c in row["context_data"]],
                                                             ensure_ascii=False)
    q = pd.DataFrame(rows)
    if q.groupby("system_id")["qa_id"].nunique().to_dict() != {s: 99 for s in SYSTEMS}:
        sys.exit("Can 99 cau cho moi he thong")

    # --- 1. Doi chieu voi table5_summary.json cua harness ---
    summary = {s["system"]: s for s in json.loads(args.summary.read_text(encoding="utf-8"))["summary"]}
    gap = 0.0
    for s in SYSTEMS:
        part, answered = q[q.system_id == s], q[(q.system_id == s) & (q.abstain == 0)]
        mine = {"retrieval_completeness_mean_pct": part.completeness.mean() * 100,
                "answer_correctness_mean_pct": answered.correct.mean() * 100,
                "coverage_adjusted_correctness_pct": part.correct_cov.mean() * 100,
                "abstentions": part.abstain.sum()}
        gap = max(gap, *(abs(float(mine[k]) - float(summary[s][k])) for k in mine))
    if gap > 1e-9:
        sys.exit(f"Khong tai lap duoc table5_summary.json (lech {gap:.2e})")

    # --- 2. Cap cau co ngu canh giong het nhau (bat buoc hoa) ---
    identical = {f"{a}~{b}": sum(contexts[(a, i)] == contexts[(b, i)] for i in q.qa_id.unique())
                 for a, b in [("adaptive", "raw__langch_recurs_default"), ("adaptive", "raw__page"),
                              ("raw__langch_recurs_default", "raw__page")]}

    # --- 3. Theo tai lieu + so sanh cap doi ---
    docs = q.groupby(["system_id", "doc_id"], as_index=False)[list(METRICS)].mean()
    idx = {s: docs[docs.system_id == s].set_index("doc_id").sort_index() for s in SYSTEMS}
    rng = np.random.default_rng(SEED)
    summ, comps = [], []
    for metric in METRICS:
        for s in SYSTEMS:
            v = idx[s][metric].to_numpy(float)
            lo, hi = boot_ci(v, rng)
            summ.append({"system_id": s, "metric": metric, "mean": v.mean(), "ci95_low": lo, "ci95_high": hi})
        block = []
        for comp in SYSTEMS[1:]:
            d = idx[REFERENCE][metric].to_numpy(float) - idx[comp][metric].to_numpy(float)
            lo, hi = boot_ci(d, rng)
            p = 1.0 if np.all(d == 0) else float(wilcoxon(d, zero_method="pratt", alternative="two-sided").pvalue)
            sd = d.std(ddof=1)
            block.append({"metric": metric, "comparator": comp, "mean_difference": d.mean(), "ci95_low": lo,
                          "ci95_high": hi, "wilcoxon_p": p, "rank_biserial": rank_biserial(d),
                          "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
                          "mde_80pct_power": 2.8 * sd / np.sqrt(len(d))})
        for r, adj in zip(block, holm([r["wilcoxon_p"] for r in block])):
            r["holm_p"] = adj
        comps += block
    summ, comps = pd.DataFrame(summ), pd.DataFrame(comps)

    # --- 4. Retrieval -> cau tra loi (cau hoi audit) ---
    pq = pd.DataFrame(read_jsonl(args.retrieval_eval / "per_query_metrics.jsonl"))
    link = q.merge(pq[["qa_id", "system_id", "recall@10", "ndcg@10"]], on=["qa_id", "system_id"], how="left",
                   validate="one_to_one")
    if link["recall@10"].isna().any():
        sys.exit("Khong noi duoc retrieval voi ket qua cham (qa_id khac nhau?)")
    link["recall_bucket"] = pd.cut(link["recall@10"], [-0.01, 0.0, 0.999, 1.0], labels=["=0", "0-1", "=1"])
    buckets = link.groupby(["recall_bucket"], observed=False).agg(
        queries=("qa_id", "size"), abstain_rate=("abstain", "mean"), completeness=("completeness", "mean"),
        correct_cov=("correct_cov", "mean")).reset_index()
    rho_rc, p_rc = spearmanr(link["recall@10"], link["completeness"])
    rho_ab, p_ab = spearmanr(link["recall@10"], link["abstain"])
    rdocs = link.groupby(["system_id", "doc_id"])[["recall@10"]].mean().reset_index()
    ridx = {s: rdocs[rdocs.system_id == s].set_index("doc_id").sort_index() for s in SYSTEMS}
    deltas = []
    for comp in SYSTEMS[1:]:
        dr = ridx[REFERENCE]["recall@10"] - ridx[comp]["recall@10"]
        dc = idx[REFERENCE]["correct_cov"] - idx[comp]["correct_cov"]
        rho, p = spearmanr(dr, dc)
        deltas.append({"comparator": comp, "spearman_delta_recall10_vs_delta_correct_cov": rho, "p": p,
                       "docs_recall_up": int((dr > 0).sum()), "docs_recall_up_and_answer_up": int(((dr > 0) & (dc > 0)).sum())})

    # --- 5. Ghi ---
    out = args.output_dir
    out.mkdir(parents=True)
    q.to_csv(out / "answer_query_metrics.csv", index=False)
    docs.to_csv(out / "answer_document_metrics.csv", index=False)
    summ.to_csv(out / "answer_system_summary.csv", index=False)
    comps.to_csv(out / "answer_comparisons.csv", index=False)
    buckets.to_csv(out / "retrieval_answer_buckets.csv", index=False)
    pd.DataFrame(deltas).to_csv(out / "retrieval_answer_deltas.csv", index=False)

    f = lambda x, d=3: "n/a" if pd.isna(x) else f"{x:.{d}f}"  # noqa: E731
    md = [f"# Table 5 - chat luong cau tra loi theo tai lieu ({args.label})", "",
          f"Sinh boi `analyze_answers_table5.py` luc {started}. So lieu tu file ket qua [Code/Artifact]; dong [Suy luan] la dien giai.",
          "**Chi dung de so sanh noi bo giua cac he thong cung model; khong so voi Table 5 cua paper.**", "",
          f"- Tai lap table5_summary.json: lech lon nhat {gap:.1e}.",
          f"- Tu choi kem them chu (dem la tu choi): " + ", ".join(
              f"{s} {int(q[(q.system_id == s)].partial_abstain.sum())}" for s in SYSTEMS) + ".",
          f"- Cau co 10 chunk ngu canh giong het nhau giua hai he thong (bat buoc hoa): {identical}.", "",
          "## Trung binh theo tai lieu (33 tai lieu, 95% CI bootstrap)", "",
          "| He thong | " + " | ".join(METRICS[m] for m in METRICS) + " |", "|---|" + "---:|" * len(METRICS)]
    for s in SYSTEMS:
        cells = [f"{f(r['mean'])} [{f(r['ci95_low'])}, {f(r['ci95_high'])}]" for _, r in
                 summ[summ.system_id == s].set_index("metric").loc[list(METRICS)].iterrows()]
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    md += ["", "## Adaptive tru doi chung (cap doi theo tai lieu; voi ty le tu choi, am = Adaptive tu choi it hon)", "",
           "| Metric | Doi chung | Hieu TB | 95% CI | Thang/Hoa/Thua | Rank-biserial | Holm p | MDE (80%) |",
           "|---|---|---:|---|---|---:|---:|---:|"]
    for _, r in comps.iterrows():
        md.append(f"| {r['metric']} | {r['comparator']} | {r['mean_difference']:+.4f} | [{r['ci95_low']:+.4f}, "
                  f"{r['ci95_high']:+.4f}] | {r['wins']}/{r['ties']}/{r['losses']} | {r['rank_biserial']:+.2f} | "
                  f"{f(r['holm_p'], 4)} | {f(r['mde_80pct_power'])} |")
    md += ["", "Holm hieu chinh tren 2 doi chung trong moi metric. Voi 'abstain', 'thang' = Adaptive co ty le tu choi CAO hon.",
           "", "## Retrieval -> cau tra loi (cau hoi audit; muc cau hoi, gop 3 he thong)", "",
           f"- Spearman recall@10 vs Retrieval Completeness (judge): rho = {rho_rc:.3f} (p = {p_rc:.2g}).",
           f"- Spearman recall@10 vs tu choi: rho = {rho_ab:.3f} (p = {p_ab:.2g}).", "",
           "| recall@10 | So cau | Ty le tu choi | Completeness | Correctness tinh do phu |", "|---|---:|---:|---:|---:|"]
    for _, r in buckets.iterrows():
        md.append(f"| {r['recall_bucket']} | {r['queries']} | {f(r['abstain_rate'])} | {f(r['completeness'])} | {f(r['correct_cov'])} |")
    md += ["", "Theo tai lieu: hieu recall@10 (Adaptive - doi chung) co di kem hieu Correctness tinh do phu khong?", "",
           "| Doi chung | Spearman | p | Tai lieu recall tang | ... va cau tra loi cung tang |", "|---|---:|---:|---:|---:|"]
    for r in deltas:
        md.append(f"| {r['comparator']} | {f(r['spearman_delta_recall10_vs_delta_correct_cov'])} | {f(r['p'], 3)} | "
                  f"{r['docs_recall_up']} | {r['docs_recall_up_and_answer_up']} |")
    md += ["", "## Gioi han", "",
           "- 33 tai lieu; chenh lech nho hon MDE khong ket luan duoc co hay khong co khac biet.",
           "- Gop 3 he thong o muc cau hoi: cac cau cung tai lieu khong doc lap; Spearman chi mo ta.",
           "- Judge va generator la cung model; chat luong cham bi gioi han boi model nay [Suy luan]."]
    (out / "SUMMARY_answers.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    inputs = [args.summary, args.retrieval_eval / "per_query_metrics.jsonl", args.prepared / "folds.jsonl"] + \
             [args.judge_dir / s / "generated_questions_evaluation_results.json" for s in SYSTEMS]
    (out / "analysis_manifest.json").write_text(json.dumps({
        "status": "complete", "label": args.label, "started_at": started, "command": sys.argv,
        "script_sha256": sha(Path(__file__).resolve()), "inputs": {str(p): sha(p) for p in inputs},
        "seed": SEED, "bootstrap_samples": BOOT, "abstention_rule": "substring", "summary_reproduction_gap": gap,
        "identical_context_pairs": identical,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__}}, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print((out / "SUMMARY_answers.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
