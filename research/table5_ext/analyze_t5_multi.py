"""Phan tich Table 5 (chat luong cau tra loi) cho nhieu he thong / nhieu run, don vi = tai lieu.

Quy tac giong phase_i_summarize.py (repo nhom, da va "chua"): tu choi = cau tra loi CHUA chuoi ABSTAIN;
Correctness chi co o cau khong tu choi; diem clip ve [0,1] voi sai so 1e-12.
Metric theo cau: completeness (Retrieval Completeness), correct_cov (Correctness tinh do phu, tu choi = 0),
correct_ans (Correctness chi tren cau tra loi - KHONG ghep cap duoc theo cau, chi bao cao mo ta), abstain.
So sanh ghep cap theo tai lieu: delta trung binh, CI bootstrap 10.000 (seed 2026), Wilcoxon pratt 2 phia,
Holm trong tung (ho so sanh x metric), thang/hoa/thua.

  python -X utf8 analyze_t5_multi.py --spec spec.json --out runs/<new_dir>
spec.json: {"label": "...", "systems": {"name": "path/to/generated_questions_evaluation_results.json", ...},
            "families": {"vs_adaptive": [["sdp_a", "adaptive"], ...], ...},
            "check_summary": "path/table5_summary.json" (tuy chon; doi chieu trung binh)}
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ABSTAIN = "I don't know based on the provided context."
SEED, BOOT, TOL = 2026, 10_000, 1e-12
PAIRED = ["completeness", "correct_cov", "abstain"]


def _metrics(raw):
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return ast.literal_eval(raw)
    return raw


def load(path: Path, name: str) -> pd.DataFrame:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        ms = {m["name"]: m for m in _metrics(r["metrics"])}
        abst = ABSTAIN in (r.get("generated_output") or "")
        if abst == ("Correctness [GEval]" in ms):
            raise SystemExit(f"{name}/{r['query_id']}: tu choi va Correctness khong khop")
        for m in ms.values():
            if m.get("error") is not None:
                raise SystemExit(f"{name}/{r['query_id']}: loi metric {m['error']!r}")
            s = float(m["score"])
            if s < -TOL or s > 1 + TOL:
                raise SystemExit(f"{name}/{r['query_id']}: diem ngoai [0,1]")
        clip = lambda v: min(1.0, max(0.0, float(v)))
        corr = clip(ms["Correctness [GEval]"]["score"]) if not abst else np.nan
        out.append({"system": name, "query_id": r["query_id"], "doc": r["reference_answer_doc_name"],
                    "completeness": clip(ms["Retrieval Completeness"]["score"]),
                    "correct_ans": corr, "correct_cov": 0.0 if abst else corr, "abstain": float(abst),
                    "abstain_with_text": float(abst and r["generated_output"].strip() != ABSTAIN)})
    df = pd.DataFrame(out)
    if df.query_id.duplicated().any():
        raise SystemExit(f"{name}: query_id trung")
    return df


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def paired(a: pd.Series, b: pd.Series, rng) -> dict:
    d = (a - b).to_numpy()
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    bm = d[idx].mean(axis=1)
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d, zero_method="pratt").pvalue)
    return {"delta": float(d.mean()), "ci_lo": float(np.percentile(bm, 2.5)), "ci_hi": float(np.percentile(bm, 97.5)),
            "p": p, "win": int((d > 1e-12).sum()), "tie": int((np.abs(d) <= 1e-12).sum()), "loss": int((d < -1e-12).sum()),
            "n_docs": len(d)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    dfs, hashes = [], {}
    for name, p in spec["systems"].items():
        dfs.append(load(Path(p), name)); hashes[name] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    q = pd.concat(dfs)
    ref_ids = set(dfs[0].query_id)
    for d in dfs:
        if set(d.query_id) != ref_ids:
            raise SystemExit(f"{d.system.iloc[0]}: tap query_id khac")
    q.to_csv(out / "per_query.csv", index=False)

    desc = q.groupby("system", sort=False).agg(
        n=("query_id", "size"), completeness=("completeness", "mean"), correct_ans=("correct_ans", "mean"),
        correct_cov=("correct_cov", "mean"), abstain=("abstain", "sum"), abstain_with_text=("abstain_with_text", "sum"))
    for c in ["completeness", "correct_ans", "correct_cov"]:
        desc[c] = desc[c] * 100
    desc.to_csv(out / "descriptive.csv")

    check = {}
    if spec.get("check_summary"):
        sm = json.loads(Path(spec["check_summary"]).read_text(encoding="utf-8"))["summary"]
        for s in sm:
            if s["system"] in desc.index:
                r = desc.loc[s["system"]]
                diffs = {"completeness": abs(r.completeness - s["retrieval_completeness_mean_pct"]),
                         "correct_ans": abs(r.correct_ans - s["answer_correctness_mean_pct"]),
                         "correct_cov": abs(r.correct_cov - s["coverage_adjusted_correctness_pct"]),
                         "abstain": abs(r.abstain - s["abstentions"])}
                check[s["system"]] = max(diffs.values())
                if check[s["system"]] > 1e-9:
                    raise SystemExit(f"Doi chieu summary lech: {s['system']} {diffs}")

    per_doc = q.groupby(["system", "doc"])[PAIRED].mean()
    rng = np.random.default_rng(SEED)
    res = []
    for fam, pairs in spec["families"].items():
        for metric in PAIRED:
            rows = []
            for x, y in pairs:
                r = paired(per_doc.loc[x][metric], per_doc.loc[y][metric].reindex(per_doc.loc[x].index), rng)
                rows.append({"family": fam, "metric": metric, "system": x, "baseline": y, **r})
            for r, h in zip(rows, holm([r["p"] for r in rows])):
                r["p_holm"] = float(h)
            res.extend(rows)
    pd.DataFrame(res).to_csv(out / "paired.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(
        {"label": spec.get("label"), "spec": spec, "input_sha256": hashes, "summary_check_max_abs_diff": check,
         "seed": SEED, "boot": BOOT, "unit": "document", "abstain_rule": "contains"}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    pd.set_option("display.width", 200)
    print(spec.get("label")); print(desc.round(2).to_string()); print("check:", check)
    pr = pd.DataFrame(res)
    for c in ["delta", "ci_lo", "ci_hi"]:
        pr[c] = pr[c] * 100
    print(pr[["family", "metric", "system", "baseline", "delta", "ci_lo", "ci_hi", "p", "p_holm", "win", "tie", "loss", "n_docs"]]
          .round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
