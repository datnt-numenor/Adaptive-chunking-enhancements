"""Kiem tra: metric RC (coreference, chi tieng Anh) co lam lech lua chon Adaptive o tai lieu khong phai tieng Anh?

In ra cho moi tai lieu: ngon ngu (langdetect, 5000 ky tu dau), so cap thuc the-dai tu trong mentions/,
RC va diem tong (trung binh deu cac metric co gia tri) cua 4 ung vien, lua chon CO RC va KHONG RC.
Chi doc file, khong goi API. Chay tu thu muc goc repo goc:
  python -X utf8 check_rc_language.py --metrics runs\\full_kaggle\\results\\chunking_metrics.parquet
      --data-dir data\\clair --output runs\\rc_language_check.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect

METRICS = ["size_compliance", "block_integrity", "intrachunk_cohesion",
           "document_contextual_coherence", "references_completeness"]
CANDIDATES = ["our_recurs_1100", "our_recurs_600", "page", "llm_regex"]


def select(scores: pd.DataFrame, metrics: list[str]) -> pd.Series:
    """Trung binh deu cac metric co gia tri (bo NaN), roi chon ung vien cao nhat - nhu find_best_method."""
    return scores[metrics].mean(axis=1, skipna=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        sys.exit(f"{args.output} da ton tai - khong ghi de")

    DetectorFactory.seed = 0
    pairs = {}
    for f in (args.data_dir / "mentions").glob("*.parquet"):
        d = pd.read_parquet(f)
        pairs[d["doc_name"].iloc[0]] = len(d["entity_pron_mentions"].iloc[0])
    lang = {}
    for f in (args.data_dir / "adi_parsed").glob("*.json"):
        lang[f.stem] = detect(json.loads(f.read_text(encoding="utf-8"))["full_text"][:5000])

    m = pd.read_parquet(args.metrics)
    m = m[m["metric_name"].isin(METRICS) & m["chunking_method"].isin(CANDIDATES)]
    wide = m.pivot_table(index=["doc_name", "chunking_method"], columns="metric_name", values="score", aggfunc="first")
    wide = wide.reindex(columns=METRICS)

    rows = []
    for doc, part in wide.groupby(level=0):
        part = part.droplevel(0)
        with_rc, without_rc = select(part, METRICS), select(part, METRICS[:-1])
        rows.append({
            "doc_name": doc, "language": lang.get(doc, "?"), "entity_pron_pairs": pairs.get(doc, 0),
            **{f"rc_{c}": part.loc[c, "references_completeness"] if c in part.index else None for c in CANDIDATES},
            **{f"score_{c}": round(100 * with_rc.get(c, float("nan")), 2) for c in CANDIDATES},
            "selected_with_rc": with_rc.idxmax(), "selected_without_rc": without_rc.idxmax(),
            "margin_with_rc": round(100 * (with_rc.max() - with_rc.drop(with_rc.idxmax()).max()), 2),
        })
    out = pd.DataFrame(rows)
    out["selection_changes"] = out["selected_with_rc"] != out["selected_without_rc"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("Lua chon CO RC (phai khop prepare_table5: page 16, our_recurs_1100 10, llm_regex 6, our_recurs_600 1):")
    print(out["selected_with_rc"].value_counts().to_string(), "\n")
    for group, part in out.groupby(out["language"] == "en"):
        name = "tieng Anh" if group else "KHONG phai tieng Anh"
        print(f"{name}: {len(part)} tai lieu, so cap dai tu trung vi {part['entity_pron_pairs'].median():.0f}, "
              f"bo RC lam doi lua chon o {int(part['selection_changes'].sum())}/{len(part)} tai lieu")
    cols = ["doc_name", "language", "entity_pron_pairs", *[f"rc_{c}" for c in CANDIDATES],
            "selected_with_rc", "selected_without_rc", "margin_with_rc"]
    print("\nTai lieu khong phai tieng Anh:")
    with pd.option_context("display.width", 250, "display.max_colwidth", 45):
        print(out[out["language"] != "en"][cols].to_string(index=False))
    print(f"\nDa ghi {args.output}")


if __name__ == "__main__":
    main()
