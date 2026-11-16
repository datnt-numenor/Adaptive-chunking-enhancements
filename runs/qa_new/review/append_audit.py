"""Phan B: noi them danh gia vao runs/qa_new/review/qa_audit.jsonl (APPEND-ONLY).

Dung:
  .venv\\Scripts\\python.exe -X utf8 runs/qa_new/review/append_audit.py <batch.json>
  .venv\\Scripts\\python.exe -X utf8 runs/qa_new/review/append_audit.py --revise <batch.json>

<batch.json> la danh sach ban ghi do Claude soan sau khi doc tung QA. Script:
- kiem tra nhan dung danh sach gia tri cho phep, du truong, nhat quan giua cac truong;
- tu dien `doc_name`, `domain`, `expected_type` tu file frozen va `doc_language`
  (heuristic stopword FR/EN tren full_text) — khong go tay;
- kiem tra moi quote trong `proposed_fix` / `minor_fix` xuat hien DUNG 1 lan tren trang, dai 40-300;
- kiem tra `fix_type` khop voi thay doi thuc te cua `proposed_fix` so voi ban frozen;
- mac dinh bo qua qa_id da co (tiep tuc khi bi ngat). Voi --revise: CHI nhan qa_id da co va
  noi them ban moi voi `revision` = revision cu + 1 (khong sua dong cu). Nguoi doc lay ban co
  revision cao nhat cho moi qa_id (dong khong co truong `revision` = revision 1).
- KHONG sua bat ky file nao khac. Co loi thi khong ghi gi.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "runs" / "qa_new" / "qa_new_frozen.jsonl"
PARSED = ROOT / "data" / "clair" / "adi_parsed"
OUT = ROOT / "runs" / "qa_new" / "review" / "qa_audit.jsonl"

SENS = {"EASY", "MEDIUM", "HARD"}
ENUMS = {
    "grounded": {"SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"},
    "actual_type": {"MULTI-PART", "STRUCTURE", "PARAPHRASED", "SIMPLE"},
    "unique_answer": {"UNIQUE", "AMBIGUOUS"},
    "translation": {"OK", "ISSUE", "N/A"},
    "chunking_sensitivity": SENS,
    "verdict": {"KEEP", "FIX", "DROP"},
    "missing_info_location": {"none", "adjacent", "same_page", "other_page", "not_in_document"},
}
FIX_TYPES = {"evidence_extend", "evidence_add", "answer_change", "question_change"}
BOOLS = ("evidence_complete", "self_contained", "forbidden_source")
MAX_QUOTES = 5
EXPECTED = {"1": "MULTI-PART", "2": "STRUCTURE", "3": "PARAPHRASED"}
ORDER = ["qa_id", "revision", "doc_name", "domain", "doc_language", "grounded", "evidence_complete",
         "missing_info_location", "self_contained", "expected_type", "actual_type", "unique_answer",
         "translation", "chunking_sensitivity", "forbidden_source", "verdict", "fix_type",
         "chunking_sensitivity_after_fix", "evidence_for_verdict", "reasoning", "proposed_fix",
         "minor_issues", "minor_fix"]

WORD_RE = re.compile(r"[^\W_]+", re.U)
EN_STOP = set("the of to and in is are for on with that this which by as be from or it not".split())
FR_STOP = set("le la les des du et est dans pour sur avec que qui par au aux une un ce cette sont".split())


def doc_language(full_text: str) -> str:
    """Heuristic: dem stopword FR vs EN tren toan van ban."""
    ws = [w.lower() for w in WORD_RE.findall(full_text)]
    fr = sum(w in FR_STOP for w in ws)
    en = sum(w in EN_STOP for w in ws)
    return "fr" if fr > en else "en"


def check_fix(fix, pages, label, errs, original_question=None):
    """original_question: neu cau hoi cua fix giu nguyen cau goc thi khong bat loi > 30 tu
    (do dai chi la minor_issue, ban rut gon nam o minor_fix)."""
    if fix is None:
        return
    for k in ("question", "answer", "evidence"):
        if k not in fix:
            errs.append(f"{label}: thieu '{k}'")
    ev = fix.get("evidence", [])
    # Quyet dinh nguoi dung (sau cham cheo GPT-4.1): cho phep toi da 5 quote; gioi han 3 quote
    # trong prompt.txt chi la huong dan khi sinh, harness khong gioi han.
    if not 1 <= len(ev) <= MAX_QUOTES:
        errs.append(f"{label}: can 1-{MAX_QUOTES} quote, co {len(ev)}")
    if len(fix.get("question", "").split()) > 30 and fix.get("question") != original_question:
        errs.append(f"{label}: cau hoi > 30 tu")
    for e in ev:
        text = pages.get(str(e.get("page")))
        if text is None:
            errs.append(f"{label}: trang {e.get('page')} khong ton tai")
            continue
        q = e.get("quote", "")
        n = text.count(q)
        if n != 1:
            errs.append(f"{label}: quote xuat hien {n} lan tren trang {e['page']}: {q[:60]!r}")
        if not 40 <= len(q) <= 300:
            errs.append(f"{label}: quote dai {len(q)} ky tu: {q[:60]!r}")


def main() -> int:
    args = sys.argv[1:]
    revise = "--revise" in args
    args = [a for a in args if a != "--revise"]
    batch = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    frozen = {r["qa_id"]: r for r in map(json.loads, FROZEN.read_text(encoding="utf-8").splitlines()) if r}
    rev_of: dict[str, int] = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                rev_of[o["qa_id"]] = max(rev_of.get(o["qa_id"], 0), o.get("revision", 1))

    out_lines, errs, skipped = [], [], []
    lang_cache: dict[str, str] = {}
    for rec in batch:
        qid = rec["qa_id"]
        if qid not in frozen:
            errs.append(f"{qid}: khong co trong file frozen")
            continue
        if revise:
            if qid not in rev_of:
                errs.append(f"{qid}: --revise nhung chua co ban ghi nao")
                continue
            rec["revision"] = rev_of[qid] + 1
        else:
            if qid in rev_of:
                skipped.append(qid)
                continue
            rec["revision"] = 1
        src = frozen[qid]
        doc = json.loads((PARSED / f"{src['doc_name']}.json").read_text(encoding="utf-8"))
        pages = doc["pages"]
        if src["doc_name"] not in lang_cache:
            lang_cache[src["doc_name"]] = doc_language(doc["full_text"])
        if "doc_language" in rec and rec["doc_language"] != lang_cache[src["doc_name"]]:
            errs.append(f"{qid}: doc_language={rec['doc_language']} khac heuristic={lang_cache[src['doc_name']]}")
        rec["doc_language"] = lang_cache[src["doc_name"]]
        rec["doc_name"] = src["doc_name"]
        rec["domain"] = src["doc_name"].split(" ", 1)[0]
        rec["expected_type"] = EXPECTED[qid.split("-")[1]]
        rec.setdefault("minor_issues", [])
        rec.setdefault("minor_fix", None)
        rec.setdefault("fix_type", None)
        rec.setdefault("chunking_sensitivity_after_fix", None)

        for k, allowed in ENUMS.items():
            if rec.get(k) not in allowed:
                errs.append(f"{qid}: {k}={rec.get(k)!r} khong hop le")
        for k in BOOLS:
            if not isinstance(rec.get(k), bool):
                errs.append(f"{qid}: {k} phai la bool")
        if not isinstance(rec["minor_issues"], list):
            errs.append(f"{qid}: minor_issues phai la list")
        if not str(rec.get("evidence_for_verdict", "")).startswith("[Bằng chứng]"):
            errs.append(f"{qid}: evidence_for_verdict phai bat dau bang [Bằng chứng]")
        if not str(rec.get("reasoning", "")).startswith("[Suy luận]"):
            errs.append(f"{qid}: reasoning phai bat dau bang [Suy luận]")

        # --- nhat quan giua cac truong ---
        v, fix, ft = rec.get("verdict"), rec.get("proposed_fix"), rec.get("fix_type")
        if rec.get("evidence_complete") is True and rec.get("missing_info_location") != "none":
            errs.append(f"{qid}: evidence_complete=true nhung missing_info_location != none")
        if rec.get("evidence_complete") is False and rec.get("missing_info_location") == "none":
            errs.append(f"{qid}: evidence_complete=false nhung missing_info_location = none")
        if rec.get("grounded") == "PARTIAL" and rec.get("evidence_complete") is not False:
            errs.append(f"{qid}: PARTIAL phai kem evidence_complete=false")
        if v == "FIX":
            if fix is None:
                errs.append(f"{qid}: FIX nhung proposed_fix = null")
            if rec.get("chunking_sensitivity_after_fix") not in SENS:
                errs.append(f"{qid}: FIX can chunking_sensitivity_after_fix hop le")
            fts = [ft] if isinstance(ft, str) else (ft or [])
            if not fts or any(t not in FIX_TYPES for t in fts):
                errs.append(f"{qid}: fix_type={ft!r} khong hop le")
            if fix is not None:
                q_changed = fix.get("question") != src["question"]
                a_changed = fix.get("answer") != src["answer"]
                e_changed = fix.get("evidence") != src["evidence"]
                if q_changed != ("question_change" in fts):
                    errs.append(f"{qid}: question_change khong khop thay doi thuc te ({q_changed})")
                if a_changed != ("answer_change" in fts):
                    errs.append(f"{qid}: answer_change khong khop thay doi thuc te ({a_changed})")
                if e_changed != bool({"evidence_extend", "evidence_add"} & set(fts)):
                    errs.append(f"{qid}: evidence_* khong khop thay doi thuc te ({e_changed})")
                # answer_change chi khi: not_in_document, dich sai, hoac (ngoai le co ghi ro) khong the
                # phu du y bang <= 3 quote x <= 300 ky tu theo prompt.txt -> minor_issues phai co
                # 'evidence_fix_infeasible_3_quotes'.
                if "answer_change" in fts and not (rec.get("missing_info_location") == "not_in_document"
                                                   or rec.get("translation") == "ISSUE"
                                                   or "evidence_fix_infeasible_3_quotes" in rec["minor_issues"]):
                    errs.append(f"{qid}: answer_change chi dung khi not_in_document (hoac dich sai, "
                                f"hoac minor_issues co 'evidence_fix_infeasible_3_quotes')")
        else:
            if fix is not None or ft is not None or rec.get("chunking_sensitivity_after_fix") is not None:
                errs.append(f"{qid}: {v} thi proposed_fix/fix_type/chunking_sensitivity_after_fix phai null")
        nq = len(src["question"].split())
        if nq > 30:
            tag = f"length_{nq}_words"
            if tag not in rec["minor_issues"]:
                errs.append(f"{qid}: cau hoi {nq} tu, thieu minor_issues '{tag}'")
            if rec.get("minor_fix") is None:
                errs.append(f"{qid}: cau hoi {nq} tu, thieu minor_fix rut gon")
        check_fix(fix, pages, f"{qid}.proposed_fix", errs, original_question=src["question"])
        check_fix(rec.get("minor_fix"), pages, f"{qid}.minor_fix", errs)  # minor_fix luon phai <= 30 tu
        out_lines.append(json.dumps({k: rec.get(k) for k in ORDER}, ensure_ascii=False))

    if errs:
        print("[LOI] Khong ghi gi ca:")
        for e in errs:
            print("  -", e)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8", newline="\n") as fh:
        for line in out_lines:
            fh.write(line + "\n")
    n_ids = len(set(rev_of) | {json.loads(l)["qa_id"] for l in out_lines})
    print(f"Da noi {len(out_lines)} dong ({'revise' if revise else 'moi'}); bo qua (da co) {len(skipped)}: "
          f"{skipped}; so qa_id trong file = {n_ids}; ngon ngu: {lang_cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
