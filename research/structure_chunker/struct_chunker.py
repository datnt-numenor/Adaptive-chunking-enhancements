"""Chunker moi: gom block theo cau truc (structure-aware block packing), phien ban A - khong can embedding.

Y tuong (xem PROPOSAL_new_chunker_and_data.md):
  - Chi cat tai ranh gioi block cua parser (split_points)  -> khong cat giua block (BI)
  - Chon diem cat bang quy hoach dong (DP) toi thieu tong chi phi:
        ((so_token - target) / target)^2                     chunk gan kich thuoc muc tieu, <= max_tokens
      + phat_ngan   neu chunk < min_tokens                     (SC)
      + phat_tieu_de neu chunk ket thuc bang mot tieu de       (tieu de phai di cung noi dung ben duoi)
      + phat_dai_tu  neu chunk KE TIEP mo dau bang dai tu       (giu cap thuc the - dai tu, RC)
  - Block dai hon max_tokens de nguyen 1 chunk; buoc hau xu ly CUA PIPELINE GOC se tach (giong page/llm_regex).
Sau do chay DUNG hau xu ly cua paper: tach chunk qua dai (1100) -> gop chunk nho (100, tran 1150).

Lenh (chay tu thu muc goc repo GOC, .venv dang bat):
  chunk   : tao chunks/{raw,no_oversizing,small_merged}/chunks.parquet + tom tat SC/BI/RC (CPU, vai phut)
  metrics : tinh du 5 metric bang code goc (can Jina v3: GPU hoac JINA_API_KEY)
  compare : dat vao Table 3 cung metric cua cac phuong phap cu (doc chunking_metrics.parquet cua ban)
  python -X utf8 research\\struct_chunker\\struct_chunker.py chunk   --data-dir data\\clair --out runs\\struct_pack_t800 --target 800
  python -X utf8 research\\struct_chunker\\struct_chunker.py metrics --data-dir data\\clair --out runs\\struct_pack_t800 --device cuda
  python -X utf8 research\\struct_chunker\\struct_chunker.py compare --out runs\\struct_pack_t800 --baseline-metrics runs\\full_kaggle\\results\\chunking_metrics.parquet
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_chunking.chunking_utils import count_tokens
from adaptive_chunking.metrics import PERSONAL_PRONOUNS

TOKENS = partial(count_tokens, model="gpt-4o")          # giong replicate.py
SEPARATORS = [r"(?<=\n)#{1}\s", r"(?<=\n)#{2}\s", r"(?<=\n)#{3}\s", r"(?<=\n)#{4}\s", r"(?<=\n)#{5}\s",
              r"(?<=\n)#{6}\s", r"(?<=\n)\s*\(?[A-Za-z0-9]{1,4}[.)]\s+", r"(?<=\n)\s*[-*·•●▪◦‣▸▹○◯‒–—]\s+",
              r"\n{2,}", r"\n", r"[.!?]\s+", r",\s+"]   # chep tu replicate.py (dung cho buoc tach chunk qua dai)
ANAPHORS = set(PERSONAL_PRONOUNS) | {"this", "these", "those", "such", "the latter", "the former"}
ADAPTIVE_CANDIDATES = ["our_recurs_1100", "our_recurs_600", "page", "llm_regex"]
METRICS = ["size_compliance", "block_integrity", "intrachunk_cohesion", "document_contextual_coherence",
           "references_completeness"]
SHORT = {"size_compliance": "SC", "block_integrity": "BI", "intrachunk_cohesion": "ICC",
         "document_contextual_coherence": "DCC", "references_completeness": "RC"}


class StructPackSplitter:
    """split_text(full_text) -> list[str]; tra lai van ban tai lieu theo full_text de dung split_points cua parser."""

    def __init__(self, parsed_docs: dict[str, dict], target: int = 800, max_tokens: int = 1100,
                 min_tokens: int = 100, short_penalty: float = 4.0, heading_penalty: float = 3.0,
                 anaphor_penalty: float = 0.5, anaphor_window_words: int = 25):
        self.by_text = {doc["full_text"]: doc for doc in parsed_docs.values()}
        self.target, self.max_tokens, self.min_tokens = target, max_tokens, min_tokens
        self.short_penalty, self.heading_penalty, self.anaphor_penalty = short_penalty, heading_penalty, anaphor_penalty
        self.window = anaphor_window_words

    def blocks(self, doc: dict) -> list[str]:
        text = doc["full_text"]
        cuts = sorted({p for p in doc["split_points"] if 0 < p < len(text)})
        bounds = [0, *cuts, len(text)]
        return [text[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]

    def starts_with_anaphor(self, block: str) -> bool:
        words = re.findall(r"[A-Za-z']+", block)[: self.window]
        lowered = [w.lower() for w in words]
        return any(w in ANAPHORS for w in lowered) or "the latter" in " ".join(lowered) or "the former" in " ".join(lowered)

    @staticmethod
    def is_heading(block: str) -> bool:
        return block.lstrip().startswith("#")

    def split_text(self, text: str) -> list[str]:
        doc = self.by_text.get(text)
        if doc is None:
            raise KeyError("Khong tim thay tai lieu co full_text nay trong adi_parsed")
        return self.split_doc(doc)

    def split_doc(self, doc: dict) -> list[str]:
        blocks = self.blocks(doc)
        n = len(blocks)
        tokens = [TOKENS(b) for b in blocks]
        prefix = np.concatenate([[0], np.cumsum(tokens)])
        heading = [self.is_heading(b) for b in blocks]
        anaphor = [self.starts_with_anaphor(b) for b in blocks]
        best = [0.0] + [float("inf")] * n
        back = [0] * (n + 1)
        for j in range(1, n + 1):
            for i in range(j - 1, -1, -1):
                size = int(prefix[j] - prefix[i])
                if size > self.max_tokens and j - i > 1:
                    break                                  # them block nua chi lam chunk dai hon
                cost = ((size - self.target) / self.target) ** 2
                if size < self.min_tokens and not (i == 0 and j == n):
                    cost += self.short_penalty
                if j < n and heading[j - 1]:
                    cost += self.heading_penalty
                if j < n and anaphor[j]:
                    cost += self.anaphor_penalty
                if best[i] + cost < best[j]:
                    best[j], back[j] = best[i] + cost, i
        cuts, j = [], n
        while j > 0:
            cuts.append((back[j], j))
            j = back[j]
        chunks = ["".join(blocks[a:b]) for a, b in reversed(cuts)]
        assert "".join(chunks) == doc["full_text"], "chunk khong ghep lai dung van ban goc"
        return chunks


def load_docs(data_dir: Path) -> dict[str, dict]:
    docs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((data_dir / "adi_parsed").glob("*.json"))}
    if len(docs) != 33:
        print(f"CANH BAO: {len(docs)} tai lieu (paper: 33)")
    return docs


def quick_metrics(chunks_df: pd.DataFrame, docs: dict, data_dir: Path) -> pd.DataFrame:
    """SC, BI, RC bang dung ham cua repo goc (khong can embedding)."""
    from adaptive_chunking.metrics import (compute_block_integrity, compute_filtered_missing_ref_error,
                                           compute_size_compliance)
    mentions = {}
    for f in (data_dir / "mentions").glob("*.parquet"):
        m = pd.read_parquet(f)
        mentions[m["doc_name"].iloc[0]] = m["entity_pron_mentions"].iloc[0]
    rows = []
    for (doc_name, method), group in chunks_df.groupby(["doc_name", "method"]):
        chunks = group.sort_values("chunk_index")["chunk_text"].tolist()
        doc = docs[doc_name]
        pairs = mentions.get(doc_name)
        mre = compute_filtered_missing_ref_error(doc["full_text"], chunks, pairs) if pairs is not None and len(pairs) else None
        rows.append({"doc_name": doc_name, "method": method, "chunks": len(chunks),
                     "mean_tokens": float(group["chunk_len"].mean()), "max_tokens": int(group["chunk_len"].max()),
                     "SC": compute_size_compliance(chunks, max_tokens=1100, min_tokens=100),
                     "BI": compute_block_integrity(chunks, doc["split_points"], doc["full_text"], tolerance_chars=5),
                     "RC": None if mre is None else 1 - mre})
    return pd.DataFrame(rows)


def cmd_chunk(args) -> None:
    from adaptive_chunking.postprocessing import (merge_small_chunks_from_df, merge_small_chunks_to_neighbours,
                                                   split_oversized_chunks, split_oversized_chunks_from_df)
    from adaptive_chunking.split_documents import split_documents_from_dir
    from adaptive_chunking.splitters import RecursiveSplitter

    if args.out.exists():
        sys.exit(f"{args.out} da ton tai - chon thu muc khac (moi cau hinh mot thu muc)")
    method = f"struct_pack_t{args.target}"
    docs = load_docs(args.data_dir)
    splitter = StructPackSplitter(docs, target=args.target, max_tokens=args.max_tokens, min_tokens=args.min_tokens,
                                  short_penalty=args.short_penalty, heading_penalty=args.heading_penalty,
                                  anaphor_penalty=args.anaphor_penalty)
    started = time.time()
    raw = args.out / "chunks" / "raw"
    asyncio.run(split_documents_from_dir(parsed_docs_dir=args.data_dir / "adi_parsed",
                                         sync_splitters={method: splitter}, async_splitters={}, output_dir=raw,
                                         count_tokens_func=TOKENS, skip_non_english=False, replace_all_results=True))
    oversized = RecursiveSplitter(separators=SEPARATORS, chunk_size=1100, chunk_overlap=0, is_separator_regex=True,
                                  attach_separator_to="start", length_function=TOKENS, merging="to_chunk_size")
    split_oversized_chunks_from_df(
        parsed_docs_dir=args.data_dir / "adi_parsed", chunks_path=raw / "chunks.parquet",
        output_dir=args.out / "chunks" / "no_oversizing", methods_to_be_regularized={method},
        split_oversized_func=lambda c: split_oversized_chunks(c, oversized, TOKENS, max_chunk_tokens=1100),
        count_tokens_func=TOKENS, replace_all_results=True)
    merge_small_chunks_from_df(
        parsed_docs_dir=args.data_dir / "adi_parsed", chunks_path=args.out / "chunks" / "no_oversizing" / "chunks.parquet",
        output_dir=args.out / "chunks" / "small_merged", methods_to_be_regularized={method},
        merge_small_chunks_func=lambda c: merge_small_chunks_to_neighbours(c, TOKENS, min_limit=100, max_limit=1150,
                                                                          merge_to="next"),
        count_tokens_func=TOKENS, replace_all_results=True)
    final = pd.read_parquet(args.out / "chunks" / "small_merged" / "chunks.parquet")
    quick = quick_metrics(final, docs, args.data_dir)
    quick.to_csv(args.out / "quick_metrics_SC_BI_RC.csv", index=False)
    (args.out / "chunker_config.json").write_text(json.dumps({
        "method": method, "created_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "params": {k: getattr(args, k) for k in ("target", "max_tokens", "min_tokens", "short_penalty",
                                                 "heading_penalty", "anaphor_penalty")},
        "postprocessing": "paper: split oversized >1100 (RecursiveSplitter, SEPARATORS), merge small <100 (cap 1150, to next)",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seconds": round(time.time() - started, 1)}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{method}: {len(final)} chunk / {final.doc_name.nunique()} tai lieu; "
          f"token TB {final.chunk_len.mean():.0f}, lon nhat {final.chunk_len.max()}")
    print(f"SC TB {100 * quick.SC.mean():.1f} | BI TB {100 * quick.BI.mean():.1f} | "
          f"RC TB {100 * quick.RC.dropna().mean():.1f} ({quick.RC.notna().sum()} tai lieu co cap dai tu)")
    print(f"Da ghi {args.out}")


def cmd_metrics(args) -> None:
    from adaptive_chunking.compute_metrics import compute_metrics_per_origin
    from adaptive_chunking.paper.replicate import _make_embedder
    out = args.out / "results"
    if (out / "chunking_metrics.parquet").exists():
        print("Da co metric cho mot so tai lieu - harness goc se bo qua tai lieu da tinh (chay tiep duoc)")
    compute_metrics_per_origin(chunks_dir=args.out / "chunks" / "small_merged", mentions_dir=args.data_dir / "mentions",
                               parsed_docs_dir=args.data_dir / "adi_parsed",
                               models={"sentence_embedder": _make_embedder(args.device)}, output_dir=out,
                               batch_size=args.batch_size)
    print(f"Da ghi {out / 'chunking_metrics.parquet'}")


def cmd_compare(args) -> None:
    from scipy.stats import wilcoxon
    new = pd.read_parquet(args.out / "results" / "chunking_metrics.parquet")
    base = pd.read_parquet(args.baseline_metrics)
    both = pd.concat([base, new], ignore_index=True)
    both = both[both.metric_name.isin(METRICS)]
    wide = both.pivot_table(index=["doc_name", "chunking_method"], columns="metric_name", values="score",
                            aggfunc="first")[METRICS]
    final = wide.mean(axis=1, skipna=True)                   # trong so bang nhau, bo NaN (nhu find_best_method)
    new_method = new.chunking_method.iloc[0]
    docs = sorted(set(new.doc_name))
    per_doc = final.unstack("chunking_method").loc[docs]

    def adaptive(candidates):
        return per_doc[candidates].max(axis=1)
    per_doc["ADAPTIVE (4 ung vien)"] = adaptive(ADAPTIVE_CANDIDATES)
    per_doc["ADAPTIVE + moi"] = adaptive(ADAPTIVE_CANDIDATES + [new_method])
    table = wide.groupby(level="chunking_method").mean() * 100
    table["Final"] = final.groupby(level="chunking_method").mean() * 100
    table = table.rename(columns=SHORT).sort_values("Final", ascending=False)
    print("Table 3 (processed, trung binh theo tai lieu, %):")
    print(table.round(2).to_string())
    print(f"\nADAPTIVE (4 ung vien) Final = {100 * per_doc['ADAPTIVE (4 ung vien)'].mean():.2f} | "
          f"{new_method} = {100 * per_doc[new_method].mean():.2f} | ADAPTIVE + moi = {100 * per_doc['ADAPTIVE + moi'].mean():.2f}")
    rng = np.random.default_rng(2026)
    for ref in ["ADAPTIVE (4 ung vien)", "our_recurs_1100", "page"]:
        if ref not in per_doc:
            continue
        d = (per_doc[new_method] - per_doc[ref]).to_numpy() * 100
        boots = d[rng.integers(0, len(d), (10_000, len(d)))].mean(axis=1)
        p = wilcoxon(d, zero_method="pratt").pvalue if np.any(d != 0) else 1.0
        print(f"  {new_method} - {ref}: {d.mean():+.2f} diem, 95% CI [{np.quantile(boots, .025):+.2f}, "
              f"{np.quantile(boots, .975):+.2f}], thang/thua {int((d > 0).sum())}/{int((d < 0).sum())}, Wilcoxon p={p:.3g}")
    print("Luu y: Wilcoxon chua hieu chinh boi so sanh boi; 'ADAPTIVE + moi' >= moi ung vien theo cach xay dung.")
    table.round(4).to_csv(args.out / "table3_with_new.csv")
    per_doc.to_csv(args.out / "table3_final_per_doc.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chunk")
    c.add_argument("--data-dir", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--target", type=int, default=800)
    c.add_argument("--max-tokens", type=int, default=1100)
    c.add_argument("--min-tokens", type=int, default=100)
    c.add_argument("--short-penalty", type=float, default=4.0)
    c.add_argument("--heading-penalty", type=float, default=3.0)
    c.add_argument("--anaphor-penalty", type=float, default=0.5)
    m = sub.add_parser("metrics")
    m.add_argument("--data-dir", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--device", default="cpu")
    m.add_argument("--batch-size", type=int, default=32)
    k = sub.add_parser("compare")
    k.add_argument("--out", type=Path, required=True)
    k.add_argument("--baseline-metrics", type=Path, required=True)
    args = ap.parse_args()
    {"chunk": cmd_chunk, "metrics": cmd_metrics, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
