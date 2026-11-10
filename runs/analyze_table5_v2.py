"""Thong ke retrieval Table 5 (bo QA v2) theo tai lieu - cung phuong phap voi Phase H/J cua nhom.

Don vi thong ke = tai lieu (trung binh 3 cau/tai lieu truoc khi kiem dinh).
  - 95% CI: percentile bootstrap tren tai lieu, 10.000 mau, seed 2026 (hieu cap doi cho so sanh).
  - Wilcoxon signed-rank 2 phia (zero_method="pratt"), Holm tren cac doi chung trong moi metric.
  - Effect size: hieu trung binh, trung vi, rank-biserial, so tai lieu thang/hoa/thua.
  - Kiem tra lai: tinh lai trung binh moi he thong, best_fixed_cv, oracle tu per_query_metrics.jsonl
    va doi chieu voi retrieval_evaluation.json (sai lech > 1e-9 -> dung).
  - Phan nhom (chi mo ta, CI, khong kiem dinh vi it tai lieu): linh vuc, ngon ngu, phuong phap Adaptive chon,
    loai cau hoi (theo vi tri -1/-2/-3).

Chay tu bat ky dau (Windows):
  python -X utf8 analyze_table5_v2.py
     --results D:\\Project\\adaptive-chunking\\runs\\table5_v2_results\\table5_v2
     --prepared D:\\Project\\adaptive-chunking\\runs\\table5_v2\\prepared
     --audit D:\\Project\\adaptive-chunking\\runs\\qa_new\\review\\qa_audit.jsonl
     --output-dir D:\\Project\\adaptive-chunking\\runs\\table5_v2_analysis
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
from scipy.stats import rankdata, wilcoxon

SEED = 2026
BOOTSTRAP_SAMPLES = 10_000
METRICS = ["ndcg@10", "recall@10", "recall@5", "recall@1", "mrr@10", "hit@10"]
PRIMARY = "ndcg@10"
REFERENCE = "adaptive"
QUESTION_TYPE = {"1": "MULTI-PART", "2": "STRUCTURE", "3": "PARAPHRASED"}


class AnalysisError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise AnalysisError(f"Thieu file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bootstrap_ci(values, rng: np.random.Generator) -> tuple[float, float]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        raise AnalysisError("Bootstrap can vector huu han, khong rong")
    idx = rng.integers(0, len(vector), size=(BOOTSTRAP_SAMPLES, len(vector)))
    low, high = np.quantile(vector[idx].mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    out, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        out[i] = running
    return out


def rank_biserial(delta: np.ndarray) -> float:
    nonzero = delta[delta != 0]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    pos, neg = float(ranks[nonzero > 0].sum()), float(ranks[nonzero < 0].sum())
    return (pos - neg) / (pos + neg)


def derive_virtual(per_query: pd.DataFrame, systems: list[str]) -> tuple[pd.DataFrame, dict]:
    """Tai lap best_fixed_cv va oracle dung nhu phase_g_runner.evaluate_retrieval (thu tu hoa = thu tu --systems)."""
    fixed = [s for s in systems if s != REFERENCE]
    rows, chosen = [], {}
    for fold in sorted(per_query["fold"].unique()):
        train = per_query[(per_query["fold"] != fold) & per_query["system_id"].isin(fixed)]
        means = {s: train.loc[train["system_id"] == s, "ndcg@10"].mean() for s in fixed}
        winner = max(means, key=means.get)                        # max dau tien theo thu tu he thong
        chosen[int(fold)] = winner
        rows.append(per_query[(per_query["fold"] == fold) & (per_query["system_id"] == winner)])
    best = pd.concat(rows).assign(system_id="best_fixed_cv")
    order = {s: i for i, s in enumerate(fixed)}
    oracle = (per_query[per_query["system_id"].isin(fixed)]
              .assign(_o=lambda d: d["system_id"].map(order))
              .sort_values(["qa_id", "ndcg@10", "_o"], ascending=[True, False, True])
              .drop_duplicates("qa_id").drop(columns="_o").assign(system_id="oracle"))
    return pd.concat([best, oracle], ignore_index=True), chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="thu muc table5_v2 da giai nen")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--audit", type=Path, help="qa_audit.jsonl (de lay ngon ngu tai lieu); tuy chon")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        sys.exit(f"{args.output_dir} da ton tai - chon thu muc moi, khong ghi de ket qua cu")
    started = datetime.now(timezone.utc).isoformat()

    eval_dir = args.results / "evaluation-full"
    manifest = json.loads((eval_dir / "retrieval_evaluation.json").read_text(encoding="utf-8"))
    if manifest.get("mode") != "full" or manifest.get("qa_count") != 99 or manifest.get("status") != "complete":
        raise AnalysisError("retrieval_evaluation.json khong phai lan chay full 99 cau hoan chinh")
    systems = list(manifest["systems"])
    if REFERENCE not in systems:
        raise AnalysisError("Khong co he thong adaptive")
    per_query = pd.DataFrame(read_jsonl(eval_dir / "per_query_metrics.jsonl"))
    folds = pd.DataFrame(read_jsonl(args.prepared / "folds.jsonl"))
    if sha(args.prepared / "folds.jsonl") != manifest.get("input_folds_sha256"):
        raise AnalysisError("folds.jsonl khong phai ban da dung khi danh gia (sha256 khac)")
    selections = json.loads((args.prepared / "adaptive_selections.json").read_text(encoding="utf-8"))["selections"]

    # --- 1. Kiem tra toan ven va tai lap ket qua cua harness ---
    counts = per_query.groupby("system_id")["qa_id"].nunique()
    if set(counts.index) != set(systems) or set(counts) != {99}:
        raise AnalysisError(f"per_query_metrics khong du 99 cau cho moi he thong: {counts.to_dict()}")
    if per_query.groupby("system_id")["doc_id"].nunique().min() != 33:
        raise AnalysisError("Thieu tai lieu")
    virtual, chosen = derive_virtual(per_query, systems)
    rows = pd.concat([per_query, virtual], ignore_index=True)
    max_gap = 0.0
    for system, values in manifest["summary"].items():
        mine = rows[rows["system_id"] == system]
        for metric, value in values.items():
            max_gap = max(max_gap, abs(float(mine[metric].mean()) - float(value)))
    if max_gap > 1e-9:
        raise AnalysisError(f"Khong tai lap duoc summary cua harness (lech {max_gap:.2e})")

    # --- 2. Gop theo tai lieu ---
    meta = folds[["doc_id", "doc_name", "domain", "fold"]].copy()
    meta["selected_method"] = meta["doc_name"].map(selections)
    meta["domain_label"] = meta["doc_name"].str.split(" ", n=1).str[0]
    language = {}
    if args.audit:
        audit = {}
        for row in read_jsonl(args.audit):
            audit[row["qa_id"]] = row                                     # dong cuoi cung thang
        for row in audit.values():
            if row.get("doc_language") and row.get("doc_name"):
                language.setdefault(row["doc_name"], []).append(row["doc_language"])
        language = {doc: max(set(v), key=v.count) for doc, v in language.items()}
    meta["language"] = meta["doc_name"].map(language).fillna("unknown")
    docs = rows.groupby(["system_id", "doc_id"], as_index=False)[METRICS].mean().merge(meta, on="doc_id")
    all_systems = [*systems, "best_fixed_cv", "oracle"]
    indexed = {s: docs[docs["system_id"] == s].set_index("doc_id").sort_index() for s in all_systems}
    if any(len(v) != 33 for v in indexed.values()):
        raise AnalysisError("Gop theo tai lieu bi thieu")

    # --- 3. Tom tat tung he thong + so sanh cap doi ---
    rng = np.random.default_rng(SEED)
    summary, comparisons = [], []
    comparators = [s for s in systems if s != REFERENCE] + ["best_fixed_cv"]
    for metric in METRICS:
        for s in all_systems:
            values = indexed[s][metric].to_numpy(float)
            low, high = bootstrap_ci(values, rng)
            summary.append({"system_id": s, "metric": metric, "documents": 33, "mean": values.mean(),
                            "std": values.std(ddof=1), "ci95_low": low, "ci95_high": high})
        block = []
        for comp in comparators + ["oracle"]:
            delta = indexed[REFERENCE][metric].to_numpy(float) - indexed[comp][metric].to_numpy(float)
            low, high = bootstrap_ci(delta, rng)
            tested = comp != "oracle"                                      # oracle la can tren: chi mo ta
            p = (1.0 if np.all(delta == 0) else
                 float(wilcoxon(delta, zero_method="pratt", alternative="two-sided", method="auto").pvalue)) \
                if tested else float("nan")
            sd = delta.std(ddof=1)
            block.append({"metric": metric, "reference": REFERENCE, "comparator": comp, "documents": 33,
                          "mean_difference": delta.mean(), "median_difference": float(np.median(delta)),
                          "ci95_low": low, "ci95_high": high, "wilcoxon_p": p,
                          "rank_biserial": rank_biserial(delta), "wins": int((delta > 0).sum()),
                          "ties": int((delta == 0).sum()), "losses": int((delta < 0).sum()),
                          "sd_difference": sd, "mde_80pct_power": 2.8 * sd / np.sqrt(33), "tested": tested})
        tested_rows = [r for r in block if r["tested"]]
        for r, adj in zip(tested_rows, holm([r["wilcoxon_p"] for r in tested_rows])):
            r["holm_p"], r["significant_0_05"] = adj, adj < 0.05
        for r in block:
            r.setdefault("holm_p", float("nan"))
            r.setdefault("significant_0_05", False)
        comparisons += block
    summary, comparisons = pd.DataFrame(summary), pd.DataFrame(comparisons)

    # --- 4. Phan nhom (mo ta) ---
    groups = []
    for column in ["domain_label", "language", "selected_method"]:
        for value, part in indexed[REFERENCE].groupby(column):
            ids = part.index
            for comp in comparators:
                delta = (indexed[REFERENCE].loc[ids, PRIMARY] - indexed[comp].loc[ids, PRIMARY]).to_numpy(float)
                low, high = bootstrap_ci(delta, rng) if len(delta) >= 3 else (float("nan"), float("nan"))
                groups.append({"group_by": column, "group": value, "documents": len(ids), "comparator": comp,
                               "metric": PRIMARY, "adaptive_mean": indexed[REFERENCE].loc[ids, PRIMARY].mean(),
                               "comparator_mean": indexed[comp].loc[ids, PRIMARY].mean(),
                               "mean_difference": delta.mean(), "ci95_low": low, "ci95_high": high})
    groups = pd.DataFrame(groups)
    rows["question_type"] = rows["qa_id"].str.rsplit("-", n=1).str[-1].map(QUESTION_TYPE)
    qtype = rows.groupby(["question_type", "system_id"])[METRICS].mean().reset_index()

    # --- 5. Ghi ket qua ---
    out = args.output_dir
    out.mkdir(parents=True)
    docs.to_csv(out / "document_metrics.csv", index=False)
    summary.to_csv(out / "system_summary.csv", index=False)
    comparisons.to_csv(out / "comparisons.csv", index=False)
    groups.to_csv(out / "subgroups.csv", index=False)
    qtype.to_csv(out / "question_type_summary.csv", index=False)

    def f(x, d=3):
        return "n/a" if pd.isna(x) else f"{x:.{d}f}"

    md = ["# Table 5 retrieval - bo QA v2 (99 cau, 33 tai lieu)", "",
          f"Sinh tu dong boi `analyze_table5_v2.py` luc {started}. Moi so lieu duoi day tinh tu file ket qua "
          "[Code/Artifact]; cac dong danh dau [Suy luan] la dien giai.", "",
          f"- Kiem tra: tinh lai trung binh, best_fixed_cv, oracle tu per-query khop retrieval_evaluation.json "
          f"(lech lon nhat {max_gap:.1e}).",
          f"- best_fixed_cv chon theo fold: {chosen} (chi chon giua cac he thong co dinh co trong lan chay: "
          f"{[s for s in systems if s != REFERENCE]}).",
          f"- Thiet bi: {manifest.get('device')}, {manifest.get('dtype')}; embedding {manifest.get('embedding_model')} "
          f"@ {manifest.get('embedding_revision')}; reranker {manifest.get('reranker_model')}.", "",
          "## Trung binh theo tai lieu (95% CI bootstrap)", "",
          "| He thong | " + " | ".join(METRICS) + " |", "|---|" + "---:|" * len(METRICS)]
    for s in all_systems:
        cells = []
        for metric in METRICS:
            r = summary[(summary["system_id"] == s) & (summary["metric"] == metric)].iloc[0]
            cells.append(f"{f(r['mean'])} [{f(r['ci95_low'])}, {f(r['ci95_high'])}]")
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    md += ["", "## Adaptive tru doi chung (cap doi theo tai lieu)", "",
           "| Metric | Doi chung | Hieu TB | 95% CI | Thang/Hoa/Thua | Rank-biserial | Holm p | MDE (80%) |",
           "|---|---|---:|---|---|---:|---:|---:|"]
    for _, r in comparisons.iterrows():
        md.append(f"| {r['metric']} | {r['comparator']} | {r['mean_difference']:+.4f} | "
                  f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {r['wins']}/{r['ties']}/{r['losses']} | "
                  f"{r['rank_biserial']:+.2f} | {f(r['holm_p'], 4)} | {f(r['mde_80pct_power'])} |")
    primary = comparisons[(comparisons["metric"] == PRIMARY) & comparisons["tested"]]
    sig = primary[primary["significant_0_05"]]
    md += ["", f"Metric chinh: {PRIMARY}. Holm hieu chinh tren {len(primary)} doi chung trong moi metric; "
           "oracle la can tren nen chi mo ta, khong kiem dinh.",
           f"So doi chung co y nghia (Holm p < 0.05) o {PRIMARY}: {len(sig)}/{len(primary)}.",
           "MDE (80%) = muc chenh lech toi thieu phat hien duoc voi power 80%, alpha 0.05, uoc tinh bang xap xi "
           "chuan tu do lech chuan cua hieu theo tai lieu [Suy luan: xap xi].", "",
           f"## Phan nhom ({PRIMARY}, chi mo ta - it tai lieu, khong kiem dinh)", "",
           "| Nhom theo | Nhom | So tai lieu | Doi chung | Adaptive | Doi chung | Hieu | 95% CI |",
           "|---|---|---:|---|---:|---:|---:|---|"]
    for _, r in groups.iterrows():
        md.append(f"| {r['group_by']} | {r['group']} | {r['documents']} | {r['comparator']} | {f(r['adaptive_mean'])} | "
                  f"{f(r['comparator_mean'])} | {r['mean_difference']:+.3f} | [{f(r['ci95_low'])}, {f(r['ci95_high'])}] |")
    md += ["", "## Theo loai cau hoi (theo vi tri -1/-2/-3, muc cau hoi, chi mo ta)", "",
           "| Loai | He thong | " + " | ".join(METRICS) + " |", "|---|---|" + "---:|" * len(METRICS)]
    for _, r in qtype.iterrows():
        md.append(f"| {r['question_type']} | {r['system_id']} | " + " | ".join(f(r[m]) for m in METRICS) + " |")
    md += ["", "## Gioi han", "",
           "- 33 tai lieu: CI rong; chenh lech nho hon MDE khong the ket luan la co hay khong co khac biet.",
           "- best_fixed_cv chi chon giua cac he thong co dinh co trong lan chay, chua phai moc 'best fixed' day du.",
           "- Phan nhom va loai cau hoi la mo ta, khong hieu chinh boi so sanh boi.",
           "- So voi Table 5 cua nhom: khac bo QA (v2) va lua chon Adaptive (llm_regex khac) [Code/Artifact]."]
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    inputs = {p.name: {"path": str(p), "sha256": sha(p)} for p in
              [eval_dir / "retrieval_evaluation.json", eval_dir / "per_query_metrics.jsonl",
               args.prepared / "folds.jsonl", args.prepared / "adaptive_selections.json"] +
              ([args.audit] if args.audit else [])}
    (out / "analysis_manifest.json").write_text(json.dumps({
        "status": "complete", "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "script_sha256": sha(Path(__file__).resolve()), "inputs": inputs,
        "seed": SEED, "bootstrap_samples": BOOTSTRAP_SAMPLES, "primary_metric": PRIMARY, "metrics": METRICS,
        "unit": "document (mean of its QA)", "test": "Wilcoxon signed-rank, two-sided, zero_method=pratt",
        "multiple_comparisons": "Holm across comparators within each metric", "best_fixed_cv_choices": chosen,
        "harness_reproduction_max_abs_gap": max_gap,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__},
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print((out / "SUMMARY.md").read_text(encoding="utf-8").split("## Phan nhom")[0])
    print(f"Da ghi ket qua vao {out}")


if __name__ == "__main__":
    try:
        main()
    except AnalysisError as error:
        sys.exit(f"LOI: {error}")
