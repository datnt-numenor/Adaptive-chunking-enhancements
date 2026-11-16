"""Buoc 8: audit Block Integrity (BI) - BI co thien vi cach cat theo trang khong?

Ba bien the cua bo chon Adaptive (cung 4 ung vien *, chunk da hau xu ly):
  A. goc          : 5 metric, trong so bang nhau (nhu paper)
  B. bo BI        : 4 metric con lai, trong so bang nhau
  C. BI trung lap : tinh lai BI sau khi bo cac ranh gioi block nam o cho ngat trang (+-5 ky tu quanh
                    dau "<!-- PageBreak -->" va quanh diem bat dau trang moi). Cho ngat trang khong con la
                    ranh gioi "mien phi": cat o do bi tinh la cat ngang block, voi MOI phuong phap nhu nhau.
Kem them: kiem tra BI tinh lai == BI trong file ket qua, va tuong quan Spearman (Figure 1).
Cach dung: python -X utf8 runs\bi_audit.py [thu_muc_ket_qua=runs/full_kaggle]
"""
import json
import sys
from pathlib import Path

import pandas as pd

from adaptive_chunking.metrics import compute_block_integrity

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/full_kaggle")
DATA = Path("data/clair/adi_parsed")
FIVE = ["references_completeness", "intrachunk_cohesion", "document_contextual_coherence",
        "block_integrity", "size_compliance"]
SHORT = dict(zip(FIVE, ["RC", "ICC", "DCC", "BI", "SC"]))
CANDIDATES = ["llm_regex", "our_recurs_1100", "our_recurs_600", "page"]
PAGE_BREAK, TOL = "<!-- PageBreak -->", 5
out_dir = RUN / "audit"
out_dir.mkdir(exist_ok=True)

metrics = pd.read_parquet(RUN / "results" / "chunking_metrics.parquet")
scores = metrics[metrics["metric_name"].isin(FIVE)].pivot_table(
    index=["doc_name", "chunking_method"], columns="metric_name", values="score")[FIVE]
chunks = pd.read_parquet(RUN / "chunks" / "small_merged" / "chunks.parquet")

# Tinh lai BI: ban goc (de kiem tra) va ban trung lap voi trang
rows = []
for doc_name, doc_chunks in chunks[chunks["method"].isin(CANDIDATES)].groupby("doc_name"):
    doc = json.loads((DATA / f"{doc_name}.json").read_text(encoding="utf-8"))
    text, splits = doc["full_text"], doc["split_points"]
    breaks, start = [], 0
    while (pos := text.find(PAGE_BREAK, start)) != -1:
        breaks.append(pos)
        start = pos + 1
    pages = [doc["pages"][k] for k in sorted(doc["pages"], key=int)]
    if "".join(pages) == text:   # diem bat dau moi trang (dung cho page chunker cat)
        offset = 0
        for page in pages[:-1]:
            offset += len(page)
            breaks.append(offset)
    else:
        print(f"Canh bao: pages khong ghep lai dung full_text o {doc_name}; chi dung vi tri PageBreak")
    neutral = [s for s in splits if not any(abs(s - b) <= TOL for b in breaks)]
    for method, mc in doc_chunks.groupby("method"):
        texts = mc.sort_values("chunk_index")["chunk_text"].tolist()
        rows.append({"doc_name": doc_name, "chunking_method": method,
                     "bi_recomputed": compute_block_integrity(texts, splits, text, tolerance_chars=TOL),
                     "bi_page_neutral": compute_block_integrity(texts, neutral, text, tolerance_chars=TOL),
                     "page_boundaries": len(pages) - 1, "removed_splits": len(splits) - len(neutral)})
bi = pd.DataFrame(rows).set_index(["doc_name", "chunking_method"])
cand = scores.join(bi, how="inner")
gap = (cand["bi_recomputed"] - cand["block_integrity"]).abs().max()
print(f"Kiem tra: BI tinh lai khac BI trong file toi da {gap:.2e} (phai ~0)")
print(f"Bien the C bo {bi.groupby('doc_name')['removed_splits'].first().sum()} ranh gioi block "
      f"nam o {bi.groupby('doc_name')['page_boundaries'].first().sum()} cho ngat trang")

variants = {
    "A. goc (5 metric)": cand[FIVE].mean(axis=1),
    "B. bo BI (4 metric)": cand[[m for m in FIVE if m != "block_integrity"]].mean(axis=1),
    "C. BI trung lap": cand[[m for m in FIVE if m != "block_integrity"] + ["bi_page_neutral"]].mean(axis=1),
}
selections = {}
for name, score in variants.items():
    selections[name] = score.groupby(level="doc_name").idxmax().map(lambda key: key[1])
sel = pd.DataFrame(selections)
sel.to_csv(out_dir / "bi_audit_selections.csv")

base = sel["A. goc (5 metric)"]
print(f"\n{'Bien the':22}" + "".join(f"{m:>17}" for m in CANDIDATES) + f"{'doi so voi A':>14}")
for name in sel:
    counts = sel[name].value_counts()
    print(f"{name:22}" + "".join(f"{int(counts.get(m, 0)):>17}" for m in CANDIDATES)
          + f"{int((sel[name] != base).sum()):>14}")

print("\nBI trung binh cua tung ung vien: goc -> trung lap voi trang")
summary = cand.groupby(level="chunking_method")[["block_integrity", "bi_page_neutral"]].mean() * 100
for method, r in summary.loc[CANDIDATES].iterrows():
    print(f"  {method:18} {r['block_integrity']:6.1f} -> {r['bi_page_neutral']:6.1f}")

# Figure 1: Spearman giua 5 metric tren moi cap (tai lieu, phuong phap) cua ket qua da hau xu ly
paper_fig1 = {("RC", "ICC"): -0.21, ("RC", "DCC"): 0.24, ("ICC", "DCC"): -0.44, ("RC", "BI"): 0.31,
              ("ICC", "BI"): -0.34, ("DCC", "BI"): 0.31, ("RC", "SC"): 0.04, ("ICC", "SC"): 0.12,
              ("DCC", "SC"): -0.16, ("BI", "SC"): 0.02}
rho = scores.dropna().rename(columns=SHORT).corr(method="spearman")
print(f"\nFigure 1 - Spearman ({len(scores.dropna())} cap tai lieu x phuong phap, ket qua da hau xu ly)")
for (a, b), paper in paper_fig1.items():
    print(f"  {a:>3}-{b:<3}  ban: {rho.loc[a, b]:+.2f}   paper: {paper:+.2f}")
print(f"\nDa luu: {out_dir / 'bi_audit_selections.csv'}")