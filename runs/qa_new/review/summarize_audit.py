"""Phan C: tong hop qa_audit.jsonl -> qa_audit_summary.md + human_spot_check.csv. CHI DOC du lieu goc.

Chay: .venv\\Scripts\\python.exe -X utf8 runs/qa_new/review/summarize_audit.py [--overwrite]

- Moi so lieu trong summary deu do script nay tinh (khong go tay).
- Voi moi qa_id lay ban ghi co `revision` cao nhat (dong khong co truong revision = 1).
- Khong ghi de file output da ton tai neu khong co --overwrite.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
QA_DIR = ROOT / "runs" / "qa_new"
FROZEN = QA_DIR / "qa_new_frozen.jsonl"
MANIFEST = QA_DIR / "qa_new_frozen_manifest.json"
REVIEW_CSV = QA_DIR / "qa_review.csv"
AUDIT = QA_DIR / "review" / "qa_audit.jsonl"
OUT_MD = QA_DIR / "review" / "qa_audit_summary.md"
OUT_CSV = QA_DIR / "review" / "human_spot_check.csv"

DOMAINS = ["Ada", "Norma", "Sophia"]
POS = ["1", "2", "3"]
POS_NAME = {"1": "-1 (MULTI-PART)", "2": "-2 (STRUCTURE)", "3": "-3 (PARAPHRASED)"}
ORDERS = {
    "grounded": ["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"],
    "evidence_complete": [True, False],
    "missing_info_location": ["none", "adjacent", "same_page", "other_page", "not_in_document"],
    "self_contained": [True, False],
    "actual_type": ["MULTI-PART", "STRUCTURE", "PARAPHRASED", "SIMPLE"],
    "unique_answer": ["UNIQUE", "AMBIGUOUS"],
    "translation": ["OK", "ISSUE", "N/A"],
    "chunking_sensitivity": ["EASY", "MEDIUM", "HARD"],
    "forbidden_source": [True, False],
    "verdict": ["KEEP", "FIX", "DROP"],
    "doc_language": ["en", "fr"],
}
TITLES = {
    "grounded": "B1 `grounded`", "evidence_complete": "B2 `evidence_complete`",
    "missing_info_location": "`missing_info_location`", "self_contained": "B3 `self_contained`",
    "actual_type": "B4 `actual_type`", "unique_answer": "B5 `unique_answer`", "translation": "B6 `translation`",
    "chunking_sensitivity": "B7 `chunking_sensitivity` [Suy luận]", "forbidden_source": "B8 `forbidden_source`",
    "verdict": "`verdict`", "doc_language": "`doc_language` (heuristic)",
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fmt(v) -> str:
    return "true" if v is True else "false" if v is False else str(v)


def md_escape(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "–"


def first_sentence(text: str, limit: int = 260, min_len: int = 60) -> str:
    """Cau dau cua reasoning; bo qua diem cat qua som (vd. 'Art.', 'et al.') bang do dai toi thieu."""
    t = re.sub(r"^\[[^\]]+\]\s*", "", text or "")
    cut = None
    for m in re.finditer(r"(?<=[.!?])\s", t):
        if m.start() >= min_len and not re.search(r"\b(Art|al|e\.g|i\.e|No|p|fn)\.$", t[:m.start()]):
            cut = m.start()
            break
    s = t[:cut] if cut is not None and cut < limit else t[:limit]
    return s + ("…" if len(s) < len(t) else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    for p in (OUT_MD, OUT_CSV):
        if p.exists() and not args.overwrite:
            print(f"[DUNG] {p} da ton tai. Chay lai voi --overwrite neu muon ghi de.")
            return 2

    sha_start = sha256(FROZEN)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frozen = {r["qa_id"]: r for r in map(json.loads, FROZEN.read_text(encoding="utf-8").splitlines()) if r}
    review = {r["qa_id"]: r for r in csv.DictReader(REVIEW_CSV.open(encoding="utf-8", newline=""))}

    lines = [json.loads(l) for l in AUDIT.read_text(encoding="utf-8").splitlines() if l.strip()]
    latest: dict[str, dict] = {}
    for o in lines:
        o.setdefault("revision", 1)
        if o["qa_id"] not in latest or o["revision"] >= latest[o["qa_id"]]["revision"]:
            latest[o["qa_id"]] = o
    recs = [latest[k] for k in sorted(latest)]
    n = len(recs)
    missing_ids = sorted(set(frozen) - set(latest))
    revised = sorted(k for k, v in latest.items() if v["revision"] > 1)
    for r in recs:
        r["_pos"] = r["qa_id"].split("-")[1]
        r.setdefault("minor_issues", [])
        r.setdefault("missing_info_location", None)
        r.setdefault("doc_language", None)

    out: list[str] = []
    w = out.append
    w("# Tổng hợp kiểm tra ngữ nghĩa bộ 99 QA (Phần C)")
    w("")
    w("> File này do `runs/qa_new/review/summarize_audit.py` sinh tự động từ `qa_audit.jsonl`. "
      "Mọi con số đều do script tính. Với mỗi `qa_id`, script dùng bản ghi có `revision` cao nhất.")
    w("")
    w("## 0. Dữ liệu đầu vào")
    w("")
    w(f"- `qa_new_frozen.jsonl` sha256 lúc chạy: `{sha_start}`. Manifest: `{manifest.get('sha256')}`, "
      f"{'**khớp**' if sha_start == manifest.get('sha256') else '**KHÁC — cần kiểm tra!**'}.")
    w(f"- `qa_audit.jsonl`: {len(lines)} dòng, {n} `qa_id` khác nhau. Có {len(revised)} `qa_id` có bản sửa "
      f"(revision > 1): {', '.join(revised) if revised else 'không có'}.")
    w(f"- `qa_id` của file frozen chưa được chấm: {', '.join(missing_ids) if missing_ids else 'không có'}.")
    w("")

    # ---------- 1. bang dem nhan ----------
    w("## 1. Số lượng nhãn B1–B8 và verdict")
    w("")
    w("Cột `-1/-2/-3` là vị trí câu trong tài liệu, tức loại mà prompt kỳ vọng.")
    w("")
    by_dom = {d: [r for r in recs if r["domain"] == d] for d in DOMAINS}
    by_pos = {p: [r for r in recs if r["_pos"] == p] for p in POS}
    head = "| Nhãn | Tổng | " + " | ".join(f"{d} (n={len(by_dom[d])})" for d in DOMAINS) + " | " + \
           " | ".join(f"-{p} (n={len(by_pos[p])})" for p in POS) + " |"
    sep = "|---|---:|" + "---:|" * (len(DOMAINS) + len(POS))
    for field in ["grounded", "evidence_complete", "missing_info_location", "self_contained", "actual_type",
                  "unique_answer", "translation", "chunking_sensitivity", "forbidden_source", "verdict",
                  "doc_language"]:
        w(f"### {TITLES[field]}")
        w("")
        w(head)
        w(sep)
        vals = ORDERS[field] + sorted({r.get(field) for r in recs} - set(ORDERS[field]), key=str)
        for v in vals:
            row = [sum(r.get(field) == v for r in recs)]
            row += [sum(r.get(field) == v for r in by_dom[d]) for d in DOMAINS]
            row += [sum(r.get(field) == v for r in by_pos[p]) for p in POS]
            w(f"| {fmt(v)} | " + " | ".join(str(x) for x in row) + " |")
        w("")

    # fix_type
    fixes = [r for r in recs if r["verdict"] == "FIX"]
    ft_counter = Counter()
    for r in fixes:
        ft = r.get("fix_type")
        for t in ([ft] if isinstance(ft, str) else (ft or [])):
            ft_counter[t] += 1
    w("### `fix_type` (trong các câu FIX; một câu có thể có nhiều loại)")
    w("")
    w("| fix_type | Số câu FIX |")
    w("|---|---:|")
    for t in ["evidence_extend", "evidence_add", "answer_change", "question_change"]:
        w(f"| {t} | {ft_counter.get(t, 0)} |")
    only_ev = sum(1 for r in fixes if set(r.get("fix_type") or []) <= {"evidence_extend", "evidence_add"})
    w(f"| *(chỉ sửa evidence, giữ nguyên câu hỏi và câu trả lời)* | {only_ev} |")
    w("")

    # minor issues
    mi = Counter(t for r in recs for t in r["minor_issues"])
    w("### `minor_issues` (không ảnh hưởng verdict)")
    w("")
    w("| minor_issue | Số câu | qa_id |")
    w("|---|---:|---|")
    for t, c in sorted(mi.items()):
        ids = ", ".join(r["qa_id"] for r in recs if t in r["minor_issues"])
        w(f"| {t} | {c} | {ids} |")
    if not mi:
        w("| (không có) | 0 | |")
    w("")

    # expected vs actual
    w("### Loại kỳ vọng (vị trí) × loại thực tế (B4)")
    w("")
    acts = ORDERS["actual_type"]
    w("| expected \\ actual | " + " | ".join(acts) + " |")
    w("|---|" + "---:|" * len(acts))
    for p in POS:
        w(f"| {POS_NAME[p]} | " + " | ".join(str(sum(r['actual_type'] == a for r in by_pos[p])) for a in acts) + " |")
    agree = sum(r["actual_type"] == r["expected_type"] for r in recs)
    w("")
    w(f"Loại thực tế trùng loại kỳ vọng: {agree}/{n} ({pct(agree, n)}).")
    w("")

    # ---------- 2. chunking sensitivity ----------
    w("## 2. Độ nhạy với chunking (B7) và nguy cơ bão hoà")
    w("")
    sens = Counter(r["chunking_sensitivity"] for r in recs)
    n_easy = sens.get("EASY", 0)
    w(f"- Phân bố trên bộ frozen hiện tại: EASY={sens.get('EASY', 0)}, MEDIUM={sens.get('MEDIUM', 0)}, "
      f"HARD={sens.get('HARD', 0)}. **Tỉ lệ EASY = {n_easy}/{n} ({pct(n_easy, n)})**.")
    for d in DOMAINS:
        e = sum(r["chunking_sensitivity"] == "EASY" for r in by_dom[d])
        h = sum(r["chunking_sensitivity"] == "HARD" for r in by_dom[d])
        w(f"  - {d}: EASY {e}/{len(by_dom[d])} ({pct(e, len(by_dom[d]))}), HARD {h}/{len(by_dom[d])} ({pct(h, len(by_dom[d]))})")
    for p in POS:
        e = sum(r["chunking_sensitivity"] == "EASY" for r in by_pos[p])
        h = sum(r["chunking_sensitivity"] == "HARD" for r in by_pos[p])
        w(f"  - {POS_NAME[p]}: EASY {e}/{len(by_pos[p])} ({pct(e, len(by_pos[p]))}), HARD {h}/{len(by_pos[p])} ({pct(h, len(by_pos[p]))})")
    easy = [r for r in recs if r["chunking_sensitivity"] == "EASY"]
    easy_txt = ", ".join("{} ({})".format(r["qa_id"], r["verdict"]) for r in easy) or "không có"
    w(f"- Các câu EASY (đánh dấu để thống kê, không làm đổi verdict): {easy_txt}.")
    v2 = [r for r in recs if r["verdict"] != "DROP"]
    eff = Counter((r["chunking_sensitivity_after_fix"] if r["verdict"] == "FIX" else r["chunking_sensitivity"]) for r in v2)
    changed = [(r["qa_id"], r["chunking_sensitivity"], r["chunking_sensitivity_after_fix"]) for r in fixes
               if r.get("chunking_sensitivity_after_fix") != r["chunking_sensitivity"]]
    w(f"- Nếu áp dụng mọi FIX và bỏ mọi DROP (bộ v2 giả định, {len(v2)} câu): EASY={eff.get('EASY', 0)}, "
      f"MEDIUM={eff.get('MEDIUM', 0)}, HARD={eff.get('HARD', 0)}; tỉ lệ EASY = {pct(eff.get('EASY', 0), len(v2))}.")
    w(f"- Các câu FIX có B7 thay đổi sau khi sửa: "
      f"{', '.join(f'{q} ({a}→{b})' for q, a, b in changed) if changed else 'không có'}.")
    w("")

    # ---------- 3. bang FIX/DROP ----------
    w("## 3. Danh sách các câu FIX và DROP")
    w("")
    w("Cột 'Vấn đề' liệt kê các nhãn không đạt. Cột 'Lý do' là câu đầu của `reasoning` [Suy luận]. "
      "Toàn văn, cùng `evidence_for_verdict` [Bằng chứng] và `proposed_fix` đầy đủ, nằm trong `qa_audit.jsonl`.")
    w("")
    w("| qa_id | verdict | Vấn đề | fix_type | Đề xuất sửa | Lý do |")
    w("|---|---|---|---|---|---|")
    for r in recs:
        if r["verdict"] == "KEEP":
            continue
        issues = []
        if r["grounded"] != "SUPPORTED":
            issues.append(f"B1={r['grounded']}")
        if r["evidence_complete"] is False:
            issues.append(f"B2=false ({r.get('missing_info_location')})")
        if r["self_contained"] is False:
            issues.append("B3=false")
        if r["unique_answer"] != "UNIQUE":
            issues.append(f"B5={r['unique_answer']}")
        if r["translation"] == "ISSUE":
            issues.append("B6=ISSUE")
        if r["forbidden_source"]:
            issues.append("B8=true")
        if r["minor_issues"]:
            issues.append("minor: " + ", ".join(r["minor_issues"]))
        fx = r.get("proposed_fix") or {}
        src = frozen[r["qa_id"]]
        parts = []
        if fx:
            evs = fx.get("evidence", [])
            parts.append(f"evidence: {len(evs)} quote (trang {', '.join(str(e['page']) for e in evs)}; "
                         f"{', '.join(str(len(e['quote'])) for e in evs)} ký tự)")
            if fx.get("question") != src["question"]:
                parts.append(f"câu hỏi mới: “{fx['question']}”")
            if fx.get("answer") != src["answer"]:
                parts.append(f"câu trả lời mới: “{fx['answer']}”")
        ft = r.get("fix_type")
        ft = ", ".join(ft) if isinstance(ft, list) else (ft or "")
        w(f"| {r['qa_id']} | {r['verdict']} | {md_escape('; '.join(issues))} | {md_escape(ft)} | "
          f"{md_escape('; '.join(parts))} | {md_escape(first_sentence(r.get('reasoning', '')))} |")
    w("")
    w("Các câu chỉ có `minor_fix` (vẫn KEEP, đề xuất rút gọn câu hỏi xuống ≤ 30 từ): "
      + (", ".join(r["qa_id"] for r in recs if r["verdict"] == "KEEP" and r.get("minor_fix")) or "không có") + ".")
    w("")

    # ---------- 4. so sanh voi nhan tu dong ----------
    w("## 4. So sánh nhãn tự động (`qa_review.csv`) với đánh giá của Claude")
    w("")

    def crosstab(title, keyf, keys, colf, cols):
        w(f"### {title}")
        w("")
        w("| | " + " | ".join(str(c) for c in cols) + " | Tổng |")
        w("|---|" + "---:|" * (len(cols) + 1))
        for k in keys:
            sub = [r for r in recs if keyf(r) == k]
            w(f"| {k} | " + " | ".join(str(sum(colf(r) == c for r in sub)) for c in cols) + f" | {len(sub)} |")
        w("")

    rv = lambda r, k: review[r["qa_id"]][k]  # noqa: E731
    crosstab("`cross_block` (csv) × B7", lambda r: f"cross_block={rv(r, 'cross_block')}", ["cross_block=True", "cross_block=False"],
             lambda r: r["chunking_sensitivity"], ORDERS["chunking_sensitivity"])
    crosstab("`cross_page` (csv) × B7", lambda r: f"cross_page={rv(r, 'cross_page')}", ["cross_page=True", "cross_page=False"],
             lambda r: r["chunking_sensitivity"], ORDERS["chunking_sensitivity"])

    def lex_bin(r):
        x = float(rv(r, "lexical_overlap"))
        return "lex < 0.15" if x < 0.15 else ("0.15 ≤ lex < 0.35" if x < 0.35 else "lex ≥ 0.35")
    crosstab("`lexical_overlap` (csv) × B7", lex_bin, ["lex < 0.15", "0.15 ≤ lex < 0.35", "lex ≥ 0.35"],
             lambda r: r["chunking_sensitivity"], ORDERS["chunking_sensitivity"])
    crosstab("`has_table` (csv) × B4", lambda r: f"has_table={rv(r, 'has_table')}", ["has_table=True", "has_table=False"],
             lambda r: r["actual_type"], ORDERS["actual_type"])
    crosstab("`n_evidence` (csv) × B4", lambda r: f"n_evidence={rv(r, 'n_evidence')}", ["n_evidence=1", "n_evidence=2", "n_evidence=3"],
             lambda r: r["actual_type"], ORDERS["actual_type"])
    crosstab("`n_evidence` (csv) × B2", lambda r: f"n_evidence={rv(r, 'n_evidence')}", ["n_evidence=1", "n_evidence=2", "n_evidence=3"],
             lambda r: r["evidence_complete"], ORDERS["evidence_complete"])

    w("### Các trường hợp lệch đáng chú ý")
    w("")
    groups = [
        ("cross_block=True nhưng Claude chấm EASY", lambda r: rv(r, "cross_block") == "True" and r["chunking_sensitivity"] == "EASY"),
        ("cross_block=False nhưng Claude chấm HARD", lambda r: rv(r, "cross_block") == "False" and r["chunking_sensitivity"] == "HARD"),
        ("cross_page=True nhưng Claude không chấm HARD", lambda r: rv(r, "cross_page") == "True" and r["chunking_sensitivity"] != "HARD"),
        ("n_evidence=1 nhưng Claude chấm MULTI-PART", lambda r: rv(r, "n_evidence") == "1" and r["actual_type"] == "MULTI-PART"),
        ("has_table=True nhưng Claude không chấm STRUCTURE", lambda r: rv(r, "has_table") == "True" and r["actual_type"] != "STRUCTURE"),
        ("lexical_overlap ≥ 0.35 nhưng Claude chấm HARD", lambda r: float(rv(r, "lexical_overlap")) >= 0.35 and r["chunking_sensitivity"] == "HARD"),
    ]
    for title, cond in groups:
        sub = [r for r in recs if cond(r)]
        w(f"- **{title}**: {len(sub)} câu" + (":" if sub else "."))
        for r in sub:
            w(f"  - {r['qa_id']} (csv: cross_block={rv(r, 'cross_block')}, cross_page={rv(r, 'cross_page')}, "
              f"lex={rv(r, 'lexical_overlap')}; Claude: {r['actual_type']}/{r['chunking_sensitivity']}): "
              f"{md_escape(first_sentence(r.get('reasoning', ''), 220))}")
    w("")
    n_cb = sum(rv(r, "cross_block") == "True" for r in recs)
    n_cb_hard = sum(rv(r, "cross_block") == "True" and r["chunking_sensitivity"] == "HARD" for r in recs)
    cb_med = [r["qa_id"] for r in recs if rv(r, "cross_block") == "True" and r["chunking_sensitivity"] == "MEDIUM"]
    w(f"Tóm tắt: trong {n_cb} câu `cross_block=True`, Claude chấm HARD {n_cb_hard} câu ({pct(n_cb_hard, n_cb)}). "
      f"Các câu `cross_block=True` nhưng chấm MEDIUM: {', '.join(cb_med) if cb_med else 'không có'}.")
    w("")
    w("[Suy luận] Mức đồng thuận cao, nhưng **không phải kiểm chứng độc lập**: khi chấm B7, Claude đã dùng `cross_block`, "
      "`cross_page`, `lexical_overlap` và vị trí split_point ở Phần A làm đầu vào. `cross_block=True` mà chấm MEDIUM "
      "thường là các quote nằm sát nhau, chỉ bị PageBreak hoặc một tiêu đề ngắn chen giữa. `lexical_overlap` gần như "
      "không dự đoán được B7 (xem bảng), vì độ khó của MULTI-PART đến từ khoảng cách giữa các đoạn, chứ không từ việc ít trùng từ.")
    w("")

    # ---------- 5. khuyen nghi ----------
    n_keep = sum(r["verdict"] == "KEEP" for r in recs)
    n_fix = len(fixes)
    n_drop = sum(r["verdict"] == "DROP" for r in recs)
    per_doc_v2 = Counter(frozen[r["qa_id"]]["doc_name"] for r in v2)
    docs_short = sorted(d for d in {f["doc_name"] for f in frozen.values()} if per_doc_v2.get(d, 0) < 3)
    q_changed = [r["qa_id"] for r in fixes if (r.get("proposed_fix") or {}).get("question") != frozen[r["qa_id"]]["question"]]
    a_changed = [r["qa_id"] for r in fixes if (r.get("proposed_fix") or {}).get("answer") != frozen[r["qa_id"]]["answer"]]
    infeasible = [r["qa_id"] for r in recs if "evidence_fix_infeasible_3_quotes" in r["minor_issues"]]
    not_in_doc = [r["qa_id"] for r in recs if r.get("missing_info_location") == "not_in_document"]
    ambiguous = [r["qa_id"] for r in recs if r["unique_answer"] == "AMBIGUOUS"]
    minor_fix_ids = [r["qa_id"] for r in recs if r.get("minor_fix")]
    w("## 5. Khuyến nghị")
    w("")
    w(f"Kết quả: KEEP={n_keep}, FIX={n_fix}, DROP={n_drop} (trên {n} câu).")
    w("")
    w("**Chọn (b): tạo `qa_new_frozen_v2.jsonl` cùng manifest mới, áp dụng các FIX và bỏ các DROP.**")
    w("")
    w(f"- Số câu còn lại trong v2: **{len(v2)}**. Số tài liệu còn đủ 3 câu: "
      f"{len(per_doc_v2) - len(docs_short)}/{len({f['doc_name'] for f in frozen.values()})}"
      + (f"; tài liệu thiếu câu: {', '.join(docs_short)}." if docs_short else "; không có tài liệu nào thiếu câu."))
    w(f"- {only_ev}/{n_fix} câu FIX chỉ cần sửa evidence (mở rộng hoặc thêm quote nguyên văn đã kiểm khớp đúng 1 lần), "
      "giữ nguyên câu hỏi và câu trả lời. Cách sửa này chi phí thấp và quan trọng, vì harness dùng vị trí quote làm "
      "ground truth cho Recall/nDCG.")
    w(f"- Câu FIX có đổi câu hỏi: {', '.join(q_changed) if q_changed else 'không có'}. Câu FIX có đổi câu trả lời: "
      f"{', '.join(a_changed) if a_changed else 'không có'}.")
    w(f"  - Đổi vì câu trả lời dựa vào thông tin không có trong tài liệu (`not_in_document`): {', '.join(not_in_doc) or 'không có'}.")
    w(f"  - Đổi vì B5 AMBIGUOUS: {', '.join(ambiguous) or 'không có'}.")
    residual = [r["qa_id"] for r in recs if "residual_missing_after_fix" in r["minor_issues"]]
    w(f"  - Ngoại lệ `evidence_fix_infeasible_3_quotes`: {', '.join(infeasible) or 'không có'}. Các câu này cần nhiều hơn 3 quote "
      "(hoặc quote dài hơn 300 ký tự) mới phủ đủ câu trả lời. Theo quyết định của người dùng, chỉ sửa evidence (tối đa 3 quote, "
      "phủ được nhiều ý nhất), giữ nguyên câu hỏi và câu trả lời. Các câu vẫn còn thiếu ý sau khi sửa "
      f"(`residual_missing_after_fix`): {', '.join(residual) or 'không có'}. Ý còn thiếu được ghi trong `reasoning` của bản ghi revision 2.")
    w(f"- `minor_fix` (rút gọn câu hỏi xuống ≤ 30 từ, không ảnh hưởng verdict): {', '.join(minor_fix_ids) or 'không có'}. "
      "Bạn quyết định có áp dụng vào v2 hay không. Nếu áp dụng thì dùng `minor_fix`, vì bản này đã gộp cả evidence đã sửa.")
    w(f"- Tỉ lệ EASY dự kiến của v2: {pct(eff.get('EASY', 0), len(v2))}, so với {pct(n_easy, n)} ở bản frozen. "
      "[Suy luận] Tỉ lệ EASY thấp nên nguy cơ bão hoà do câu dễ là thấp. Tuy vậy, B7 mới chỉ là suy luận.")
    w("")
    w("Nhắc lại hai nguyên tắc:")
    w("")
    w("1. v2 là **file mới** (`qa_new_frozen_v2.jsonl` + manifest mới có sha256 riêng). **Không ghi đè** `qa_new_frozen.jsonl`.")
    w("2. Mọi thay đổi phải làm và freeze **trước khi** xem bất kỳ kết quả retrieval nào trên bộ QA này.")
    w("")

    # ---------- 6. han che ----------
    w("## 6. Hạn chế")
    w("")
    w("- **Người review (Claude) cùng họ model với model đã sinh QA** (generator trong manifest: "
      f"“{manifest.get('generator')}”). Vì vậy đánh giá có thể thiên về hướng chấm dễ, hoặc bỏ qua cùng loại lỗi mà "
      "generator mắc phải. Người dùng không chấm tay; thay vào đó dự kiến đối chiếu với một người chấm độc lập khác họ model "
      "(GPT-4.1, chạy bằng script riêng). `human_spot_check.csv` vẫn được giữ làm mẫu phân tầng tham khảo.")
    w("- **B7 `chunking_sensitivity` là suy luận**, chưa được kiểm chứng bằng retrieval thật. Các quy tắc đã dùng: "
      "hai đoạn xa nhau hoặc khác trang có ranh giới block → HARD; đại từ hoặc từ viết tắt có referent ở block khác → HARD; "
      "bảng nằm trọn trong một block → MEDIUM; một đoạn ngắn trùng nhiều từ với câu hỏi → EASY; "
      "quote liền nhau nhưng có split_point ở giữa → không chấm EASY. Các quy tắc này dùng chính các tín hiệu của "
      "validator (cross_block, cross_page, split_point), nên mức đồng thuận ở mục 4 không phải bằng chứng độc lập.")
    w("- Quy tắc chấm nghiêm (đã thống nhất): B2=false khi bất kỳ ý nào của câu trả lời, kể cả ý phụ, nằm ngoài quote. "
      "Vì vậy tỉ lệ FIX cao, nhưng phần lớn chỉ cần sửa evidence.")
    w("- Ngôn ngữ tài liệu (`doc_language`) và kiểm tra tiếng Anh ở Phần A đều dùng heuristic đếm stopword.")
    w("- B5 dựa trên tìm từ khoá trong `full_text`. Các đoạn diễn đạt khác nhưng cùng nghĩa có thể bị bỏ sót. "
      "Một số đoạn lặp nguyên văn ở phần tóm tắt (21-1, 21-2) không làm đáp án mơ hồ, nhưng có thể khiến Recall theo vị trí quote "
      "phạt oan retriever, nên đã được ghi vào `minor_issues`.")
    w("- Ràng buộc của prompt (≤ 3 quote, mỗi quote ≤ 300 ký tự) khiến một số câu không sửa được chỉ bằng evidence. Các câu này dùng ngoại lệ ở mục 5.")
    w("")

    # ---------- 7. spot check ----------
    rng = random.Random(2026)
    picked = []
    for d in DOMAINS:
        ids = sorted(r["qa_id"] for r in by_dom[d])
        picked += rng.sample(ids, 5)
    w("## 7. Mẫu chấm tay (`human_spot_check.csv`)")
    w("")
    w("Chọn bằng `random.Random(2026)`, phân tầng 5 câu cho mỗi domain theo thứ tự Ada → Norma → Sophia, "
      "dùng `rng.sample` trên danh sách `qa_id` đã sắp xếp của từng domain: " + ", ".join(picked) + ".")
    vc = Counter(latest[q]["verdict"] for q in picked)
    w(f"Verdict của Claude trong mẫu: KEEP={vc.get('KEEP', 0)}, FIX={vc.get('FIX', 0)}, DROP={vc.get('DROP', 0)}.")
    w("")
    sha_end = sha256(FROZEN)
    w("## 8. Kiểm tra tính toàn vẹn")
    w("")
    w(f"- sha256 `qa_new_frozen.jsonl` lúc đầu: `{sha_start}`")
    w(f"- sha256 `qa_new_frozen.jsonl` lúc cuối: `{sha_end}`, {'khớp' if sha_end == sha_start else '**KHÁC!**'}")
    w("")

    OUT_MD.write_text("\n".join(out) + "\n", encoding="utf-8")
    import io
    buf = io.StringIO(newline="")
    wr = csv.writer(buf)
    wr.writerow(["qa_id", "question", "answer", "quotes", "claude_verdict", "human_verdict", "human_notes"])
    for q in picked:
        src = frozen[q]
        quotes = " || ".join(f"p{e['page']}: {e['quote']}" for e in src["evidence"])
        wr.writerow([q, src["question"], src["answer"], quotes, latest[q]["verdict"], "", ""])
    data = buf.getvalue().encode("utf-8-sig")
    if OUT_CSV.exists() and OUT_CSV.read_bytes() == data:
        print(f"{OUT_CSV.relative_to(ROOT)} khong doi noi dung, bo qua ghi.")
    else:
        OUT_CSV.write_bytes(data)

    print(f"Da ghi {OUT_MD.relative_to(ROOT)} va {OUT_CSV.relative_to(ROOT)}")
    print(f"KEEP={n_keep} FIX={n_fix} DROP={n_drop}; EASY={n_easy}/{n}; v2 EASY={eff.get('EASY', 0)}/{len(v2)}")
    print(f"spot-check: {picked}")
    print(f"sha256 dau={sha_start}\nsha256 cuoi={sha_end} {'(khop)' if sha_start == sha_end else '(KHAC!)'}")
    return 0 if sha_start == sha_end else 3


if __name__ == "__main__":
    sys.exit(main())
