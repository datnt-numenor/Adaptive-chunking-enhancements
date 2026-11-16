"""Phan A: kiem tra tu dong bo 99 QA da freeze. CHI DOC du lieu.

Chay:  .venv\\Scripts\\python.exe -X utf8 runs/qa_new/review/check_qa_integrity.py [--overwrite]

Ghi duy nhat runs/qa_new/review/integrity_report.json (khong ghi de neu da ton tai,
tru khi co --overwrite). Khong sua bat ky file nao khac.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
QA_DIR = ROOT / "runs" / "qa_new"
FROZEN = QA_DIR / "qa_new_frozen.jsonl"
MANIFEST = QA_DIR / "qa_new_frozen_manifest.json"
REVIEW_CSV = QA_DIR / "qa_review.csv"
PROMPT = QA_DIR / "prompt.txt"
PARSED = ROOT / "data" / "clair" / "adi_parsed"
OUT_DIR = QA_DIR / "review"
OUT = OUT_DIR / "integrity_report.json"

PERSONA_RE = re.compile(r"(?<!\w)(ada|norma|sophia|technique|légal|sciences\s+sociales)(?!\w)", re.I)
VAGUE_RE = re.compile(r"(?<!\w)(this document|the report|the text|the paper|above)(?!\w)", re.I)
WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.U)
# Tach cau (heuristic): dau . ! ? roi khoang trang roi chu hoa/so/ngoac.
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"“(A-Z0-9])")
# Comment cua parser ADI danh dau header/footer/so trang.
PAGE_META_RE = re.compile(r"<!--\s*(PageHeader|PageFooter|PageNumber|PageBreak)[^>]*?-->", re.S)

# Heuristic ngon ngu: dem stopword tieng Anh vs tieng Phap.
EN_STOP = set("""the a an of to and in is are was were be been for on with that this which by as at from
or it its not can must may should under their they who what when where how have has had will would also
only any such than other into these those there if each""".split())
FR_STOP = set("""le la les des du un une et est sont dans pour sur avec que qui par au aux ne pas ce cette
ces être été il elle ils elles leur leurs se sa son ses ou mais donc dont où lors selon entre""".split())
FR_ACCENT_RE = re.compile(r"[àâçéèêëîïôûùüÿœæ]", re.I)

QUOTE_MIN, QUOTE_MAX = 40, 300
MAX_Q_WORDS = 30
JACCARD_WARN = 0.6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def count_occurrences(haystack: str, needle: str) -> list[int]:
    """Moi vi tri bat dau (ke ca chong lan) cua needle trong haystack."""
    out, i = [], haystack.find(needle)
    while i != -1:
        out.append(i)
        i = haystack.find(needle, i + 1)
    return out


def english_heuristic(text: str) -> dict:
    ws = words(text)
    en = sum(w in EN_STOP for w in ws)
    fr = sum(w in FR_STOP for w in ws)
    accents = FR_ACCENT_RE.findall(text)
    if en == 0 and fr == 0:
        label = "UNKNOWN"
    elif fr > en:
        label = "NOT_EN"
    else:
        label = "EN"
    return {"label": label, "en_stop": en, "fr_stop": fr, "fr_accent_chars": len(accents)}


def result(status: str, summary: str, **details) -> dict:
    return {"status": status, "summary": summary, **details}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="cho phep ghi de integrity_report.json")
    args = ap.parse_args()

    if OUT.exists() and not args.overwrite:
        print(f"[DUNG] {OUT} da ton tai. Chay lai voi --overwrite neu muon ghi de.")
        return 2

    sha_start = sha256(FROZEN)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in FROZEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    checks: dict[str, dict] = {}

    # ---------- 1. sha256 ----------
    ok1 = sha_start == manifest.get("sha256")
    checks["1_sha256"] = result(
        "PASS" if ok1 else "FAIL",
        f"sha256 file = {sha_start}; manifest = {manifest.get('sha256')}",
        sha256_file=sha_start, sha256_manifest=manifest.get("sha256"),
        info_prompt_sha256_file=sha256(PROMPT), info_prompt_sha256_manifest=manifest.get("prompt_sha256"),
    )
    if not ok1:
        print("[DUNG] sha256 cua qa_new_frozen.jsonl KHAC manifest. Khong tiep tuc.")
        print(f"  file={sha_start}\n  manifest={manifest.get('sha256')}")
        return 3

    # ---------- 2. so luong / trung lap ----------
    per_doc = Counter(r["doc_name"] for r in rows)
    id_dup = [k for k, v in Counter(r["qa_id"] for r in rows).items() if v > 1]
    q_dup = [k for k, v in Counter(r["question"] for r in rows).items() if v > 1]
    bad_doc_count = {d: n for d, n in per_doc.items() if n != 3}
    id_re = re.compile(r"^\d{2}-[123]$")
    bad_ids = [r["qa_id"] for r in rows if not id_re.match(r["qa_id"])]
    # moi tai lieu dung 1 tien to qa_id va du -1/-2/-3
    prefix_by_doc = defaultdict(set)
    suffix_by_doc = defaultdict(list)
    for r in rows:
        p, s = r["qa_id"].split("-") if "-" in r["qa_id"] else (r["qa_id"], "")
        prefix_by_doc[r["doc_name"]].add(p)
        suffix_by_doc[r["doc_name"]].append(s)
    bad_prefix = {d: sorted(p) for d, p in prefix_by_doc.items() if len(p) != 1}
    bad_suffix = {d: sorted(s) for d, s in suffix_by_doc.items() if sorted(s) != ["1", "2", "3"]}
    ok2 = (len(rows) == 99 and len(per_doc) == 33 and not bad_doc_count and not id_dup and not q_dup
           and not bad_ids and not bad_prefix and not bad_suffix
           and manifest.get("n_qa") == len(rows) and manifest.get("n_docs") == len(per_doc))
    checks["2_counts_duplicates"] = result(
        "PASS" if ok2 else "FAIL",
        f"{len(rows)} dong, {len(per_doc)} tai lieu; trung qa_id={len(id_dup)}; trung cau hoi={len(q_dup)}",
        n_rows=len(rows), n_docs=len(per_doc), docs_not_3=bad_doc_count, dup_qa_id=id_dup,
        dup_question=q_dup, bad_qa_id_format=bad_ids, doc_multi_prefix=bad_prefix, doc_bad_suffix=bad_suffix,
        manifest_n_qa=manifest.get("n_qa"), manifest_n_docs=manifest.get("n_docs"),
    )

    # ---------- 3. tai lieu + trang ----------
    docs, missing_docs, missing_pages, fulltext_mismatch = {}, [], [], []
    for d in per_doc:
        f = PARSED / f"{d}.json"
        if not f.exists():
            missing_docs.append(d)
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        keys = sorted(doc["pages"], key=int)
        offsets, pos = {}, 0
        for k in keys:
            offsets[int(k)] = pos
            pos += len(doc["pages"][k])
        if "".join(doc["pages"][k] for k in keys) != doc["full_text"]:
            fulltext_mismatch.append(d)
        doc["_offsets"] = offsets
        docs[d] = doc
    for r in rows:
        doc = docs.get(r["doc_name"])
        if doc is None:
            continue
        for ev in r["evidence"]:
            if str(ev["page"]) not in doc["pages"]:
                missing_pages.append({"qa_id": r["qa_id"], "page": ev["page"]})
    ok3 = not missing_docs and not missing_pages
    checks["3_docs_pages_exist"] = result(
        "PASS" if ok3 else "FAIL",
        f"thieu file tai lieu={len(missing_docs)}; trang khong ton tai={len(missing_pages)}",
        missing_docs=missing_docs, missing_pages=missing_pages,
        info_full_text_not_concat_of_pages=fulltext_mismatch,
    )

    # ---------- 4. quote khop dung 1 lan; do dai ----------
    quote_fail, quote_len_warn = [], []
    spans_by_qa: dict[str, list[dict]] = {}
    n_quotes = 0
    for r in rows:
        doc = docs.get(r["doc_name"])
        spans = []
        for i, ev in enumerate(r["evidence"]):
            n_quotes += 1
            q = ev["quote"]
            L = len(q)
            if not (QUOTE_MIN <= L <= QUOTE_MAX):
                quote_len_warn.append({"qa_id": r["qa_id"], "ev": i, "len": L, "quote": q[:80]})
            if doc is None or str(ev["page"]) not in doc["pages"]:
                continue
            page_text = doc["pages"][str(ev["page"])]
            occ = count_occurrences(page_text, q)
            if len(occ) != 1:
                quote_fail.append({"qa_id": r["qa_id"], "ev": i, "page": ev["page"], "n_occ": len(occ),
                                   "quote": q[:80]})
            if occ:
                local = occ[0]
                g = doc["_offsets"][int(ev["page"])] + local
                # quote co cham vao comment header/footer/so trang cua parser khong (ho tro B8)
                meta_hits = sorted({m.group(1) for m in PAGE_META_RE.finditer(page_text)
                                    if m.start() < local + L and local < m.end()})
                spans.append({"ev": i, "page": ev["page"], "start": g, "end": g + L, "page_meta": meta_hits})
        spans_by_qa[r["qa_id"]] = spans
    checks["4_quotes_exact_once"] = result(
        "PASS" if not quote_fail else "FAIL",
        f"{n_quotes} quote; khong khop dung 1 lan={len(quote_fail)}; "
        f"canh bao do dai ngoai [{QUOTE_MIN},{QUOTE_MAX}]={len(quote_len_warn)}",
        n_quotes=n_quotes, failures=quote_fail, warnings_length=quote_len_warn,
    )

    # ---------- 5. nhan persona / cum mo ho ----------
    persona_hits, vague_hits = [], []
    for r in rows:
        ph = [m.group(0) for m in PERSONA_RE.finditer(r["question"])]
        vh = [m.group(0) for m in VAGUE_RE.finditer(r["question"])]
        if ph:
            persona_hits.append({"qa_id": r["qa_id"], "hits": ph, "question": r["question"]})
        if vh:
            vague_hits.append({"qa_id": r["qa_id"], "hits": vh, "question": r["question"]})
    # Thong tin them: nhan persona trong cau tra loi (khong tinh FAIL)
    persona_in_answer = [{"qa_id": r["qa_id"], "hits": [m.group(0) for m in PERSONA_RE.finditer(r["answer"])]}
                         for r in rows if PERSONA_RE.search(r["answer"])]
    ok5 = not persona_hits and not vague_hits
    checks["5_persona_and_vague"] = result(
        "PASS" if ok5 else "FAIL",
        f"cau hoi chua nhan persona/linh vuc={len(persona_hits)}; chua cum mo ho={len(vague_hits)}",
        persona_hits=persona_hits, vague_hits=vague_hits, info_persona_in_answer=persona_in_answer,
    )

    # ---------- 6. do dai cau hoi / so cau tra loi ----------
    q_too_long, ans_warn, q_lens = [], [], []
    for r in rows:
        nq = len(r["question"].split())
        q_lens.append(nq)
        if nq > MAX_Q_WORDS:
            q_too_long.append({"qa_id": r["qa_id"], "n_words": nq, "question": r["question"]})
        sents = [s for s in SENT_SPLIT_RE.split(r["answer"].strip()) if s.strip()]
        if not (1 <= len(sents) <= 3):
            ans_warn.append({"qa_id": r["qa_id"], "n_sentences_heuristic": len(sents), "answer": r["answer"]})
    checks["6_lengths"] = result(
        "PASS" if not q_too_long else "FAIL",
        f"cau hoi > {MAX_Q_WORDS} tu={len(q_too_long)} (max={max(q_lens)}); "
        f"canh bao cau tra loi ngoai 1-3 cau={len(ans_warn)} (tach cau bang heuristic)",
        question_too_long=q_too_long, warnings_answer_sentences=ans_warn,
        question_words_max=max(q_lens), question_words_mean=round(sum(q_lens) / len(q_lens), 1),
    )

    # ---------- 7. ngon ngu cau tra loi (heuristic) ----------
    lang = {r["qa_id"]: english_heuristic(r["answer"]) for r in rows}
    q_lang = {r["qa_id"]: english_heuristic(r["question"]) for r in rows}
    not_en = [{"qa_id": k, **v} for k, v in lang.items() if v["label"] != "EN"]
    q_not_en = [{"qa_id": k, **v} for k, v in q_lang.items() if v["label"] != "EN"]
    accent_warn = [{"qa_id": k, "fr_accent_chars": v["fr_accent_chars"]} for k, v in lang.items()
                   if v["fr_accent_chars"] > 0]
    checks["7_answer_english_HEURISTIC"] = result(
        "PASS" if not not_en else "FAIL",
        f"HEURISTIC (dem stopword EN vs FR): cau tra loi khong phai EN={len(not_en)}; "
        f"cau hoi khong phai EN={len(q_not_en)}; cau tra loi co ky tu dau tieng Phap (canh bao)={len(accent_warn)}",
        method="Dem stopword tieng Anh vs tieng Phap; NOT_EN neu FR > EN. Chi la heuristic, khong phai bo nhan dien ngon ngu.",
        answers_not_en=not_en, questions_not_en=q_not_en, warnings_french_accents_in_answer=accent_warn,
    )

    # ---------- 8. cau hoi gan trung trong cung tai lieu ----------
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r["doc_name"]].append(r)
    near_dup, max_j = [], 0.0
    for d, rs in by_doc.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = set(words(rs[i]["question"])), set(words(rs[j]["question"]))
                jac = len(a & b) / max(len(a | b), 1)
                max_j = max(max_j, jac)
                if jac >= JACCARD_WARN:
                    near_dup.append({"qa_ids": [rs[i]["qa_id"], rs[j]["qa_id"]], "jaccard": round(jac, 3)})
    checks["8_near_duplicate_questions"] = result(
        "PASS" if not near_dup else "WARN",
        f"cap cau hoi Jaccard >= {JACCARD_WARN} trong cung tai lieu={len(near_dup)} (max={max_j:.3f})",
        warnings=near_dup, max_jaccard=round(max_j, 3),
    )

    # ---------- 9. vi tri evidence + split_points ----------
    review = {}
    with REVIEW_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            review[row["qa_id"]] = row
    per_qa, cross_block_disagree, unlocated = [], [], []
    for r in rows:
        doc = docs.get(r["doc_name"])
        spans = sorted(spans_by_qa.get(r["qa_id"], []), key=lambda s: s["start"])
        if doc is None or len(spans) != len(r["evidence"]):
            unlocated.append(r["qa_id"])
        sp = doc["split_points"] if doc else []
        block = lambda pos: sum(1 for p in sp if p <= pos)  # noqa: E731  (cung quy uoc voi validate_qa.py)
        block_ids = [block(s["start"]) for s in spans]
        quote_crosses_split = [any(s["start"] < p < s["end"] for p in sp) for s in spans]
        gaps = [spans[k + 1]["start"] - spans[k]["end"] for k in range(len(spans) - 1)]
        splits_between = [sum(1 for p in sp if spans[k]["end"] <= p <= spans[k + 1]["start"])
                          for k in range(len(spans) - 1)]
        span_chars = (max(s["end"] for s in spans) - min(s["start"] for s in spans)) if spans else 0
        cross_block = len(set(block_ids)) > 1 or any(quote_crosses_split)
        rec = {
            "qa_id": r["qa_id"], "doc_name": r["doc_name"], "n_evidence": len(r["evidence"]),
            "pages": [s["page"] for s in spans],
            "positions": [[s["start"], s["end"]] for s in spans],
            "block_ids": block_ids,
            "n_distinct_blocks": len(set(block_ids)),
            "quote_crosses_split": quote_crosses_split,
            "gap_chars_between_quotes": gaps,
            "split_points_between_quotes": splits_between,
            "split_points_between_total": sum(splits_between),
            "span_chars": span_chars,
            "cross_page": len({s["page"] for s in spans}) > 1,
            "cross_block": cross_block,
            "quote_touches_page_meta": sorted({m for s in spans for m in s["page_meta"]}),
            "question_words": len(r["question"].split()),
        }
        per_qa.append(rec)
        rv = review.get(r["qa_id"])
        if rv is not None and (rv["cross_block"] == "True") != cross_block:
            cross_block_disagree.append({"qa_id": r["qa_id"], "csv": rv["cross_block"], "recomputed": cross_block})
    multi = [p for p in per_qa if p["n_evidence"] >= 2]
    page_meta = [{"qa_id": p["qa_id"], "meta": p["quote_touches_page_meta"]} for p in per_qa
                 if set(p["quote_touches_page_meta"]) - {"PageBreak"}]
    checks["9_evidence_positions"] = result(
        "PASS" if not unlocated else "FAIL",
        f"QA co >=2 quote={len(multi)}; trong do qua >=1 split_point giua cac quote="
        f"{sum(p['split_points_between_total'] > 0 for p in multi)}; cross_block (tinh lai)="
        f"{sum(p['cross_block'] for p in per_qa)}/99; lech so voi qa_review.csv={len(cross_block_disagree)}; "
        f"quote cham comment header/footer/so trang={len(page_meta)}",
        unlocated=unlocated, cross_block_disagree_with_csv=cross_block_disagree,
        info_quote_touches_header_footer=page_meta,
    )

    sha_end = sha256(FROZEN)
    report = {
        "frozen_file": str(FROZEN.relative_to(ROOT)),
        "sha256_start": sha_start, "sha256_end": sha_end, "sha256_manifest": manifest.get("sha256"),
        "checks": checks,
        "per_qa": per_qa,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    for name, c in checks.items():
        print(f"[{c['status']:4s}] {name}: {c['summary']}")
    print("=" * 72)
    print(f"sha256 dau={sha_start}\nsha256 cuoi={sha_end}  {'(khop)' if sha_start == sha_end else '(KHAC!)'}")
    print(f"Da ghi {OUT.relative_to(ROOT)}")
    if sha_start != sha_end:
        print("[DUNG] sha256 thay doi trong luc chay!")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
