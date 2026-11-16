"""Dung ban DE XUAT v2 (chua freeze) tu qa_audit.jsonl -> runs/qa_new/review/candidate_v2.jsonl.

Chay: .venv\\Scripts\\python.exe -X utf8 runs/qa_new/review/build_candidate_v2.py [--overwrite]

Quy tac chon cho moi qa_id (ban ghi audit co revision cao nhat):
  minor_fix (neu co, da gop evidence da sua)  >  proposed_fix (neu co)  >  ban frozen v1.
Giu nguyen qa_id, doc_name, thu tu dong va schema v1 {qa_id, doc_name, question, answer, evidence[{page, quote}]}.
Kiem tra: quote xuat hien DUNG 1 lan tren trang, dai 40-300, 1-5 quote/cau, cau hoi <= 30 tu,
khong chua nhan persona / cum mo ho (giong Phan A). Co loi thi khong ghi file.
KHONG sua qa_new_frozen.jsonl (kiem sha256 truoc/sau).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
QA_DIR = ROOT / "runs" / "qa_new"
FROZEN = QA_DIR / "qa_new_frozen.jsonl"
MANIFEST = QA_DIR / "qa_new_frozen_manifest.json"
AUDIT = QA_DIR / "review" / "qa_audit.jsonl"
PARSED = ROOT / "data" / "clair" / "adi_parsed"
OUT = QA_DIR / "review" / "candidate_v2.jsonl"

PERSONA_RE = re.compile(r"(?<!\w)(ada|norma|sophia|technique|légal|sciences\s+sociales)(?!\w)", re.I)
VAGUE_RE = re.compile(r"(?<!\w)(this document|the report|the text|the paper|above)(?!\w)", re.I)
# Bao nham da duoc nguoi dung xac nhan o Phan A (khong tinh loi).
KNOWN_FALSE_POSITIVES = {("09-3", "technique"), ("15-1", "the report")}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if OUT.exists() and "--overwrite" not in sys.argv:
        print(f"[DUNG] {OUT} da ton tai. Chay lai voi --overwrite neu muon ghi de.")
        return 2
    sha_start = sha(FROZEN)
    if sha_start != json.loads(MANIFEST.read_text(encoding="utf-8"))["sha256"]:
        print("[DUNG] sha256 file frozen khac manifest.")
        return 3

    frozen = [json.loads(l) for l in FROZEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    latest: dict[str, dict] = {}
    for l in AUDIT.read_text(encoding="utf-8").splitlines():
        if l.strip():
            o = json.loads(l)
            o.setdefault("revision", 1)
            if o["qa_id"] not in latest or o["revision"] >= latest[o["qa_id"]]["revision"]:
                latest[o["qa_id"]] = o

    errs, rows, source = [], [], Counter()
    page_cache: dict[str, dict] = {}
    for src in frozen:
        qid = src["qa_id"]
        rec = latest.get(qid)
        if rec is None:
            errs.append(f"{qid}: chua co ban ghi audit")
            continue
        if rec.get("minor_fix"):
            fx, how = rec["minor_fix"], "minor_fix"
        elif rec.get("proposed_fix"):
            fx, how = rec["proposed_fix"], "proposed_fix"
        else:
            fx, how = src, "frozen"
        source[how] += 1
        row = {"qa_id": qid, "doc_name": src["doc_name"], "question": fx["question"], "answer": fx["answer"],
               "evidence": [{"page": int(e["page"]), "quote": e["quote"]} for e in fx["evidence"]]}
        if src["doc_name"] not in page_cache:
            page_cache[src["doc_name"]] = json.loads((PARSED / f"{src['doc_name']}.json").read_text(encoding="utf-8"))["pages"]
        pages = page_cache[src["doc_name"]]
        # Quyet dinh nguoi dung: toi da 5 quote (gioi han 3 cua prompt.txt chi ap dung khi sinh).
        if not 1 <= len(row["evidence"]) <= 5:
            errs.append(f"{qid}: {len(row['evidence'])} quote (can 1-5)")
        for e in row["evidence"]:
            text = pages.get(str(e["page"]))
            n = text.count(e["quote"]) if text is not None else -1
            if n != 1:
                errs.append(f"{qid}: quote xuat hien {n} lan tren trang {e['page']}: {e['quote'][:60]!r}")
            if not 40 <= len(e["quote"]) <= 300:
                errs.append(f"{qid}: quote dai {len(e['quote'])} ky tu")
        nw = len(row["question"].split())
        if nw > 30:
            errs.append(f"{qid}: cau hoi {nw} tu")
        for m in list(PERSONA_RE.finditer(row["question"])) + list(VAGUE_RE.finditer(row["question"])):
            if (qid, m.group(0).lower()) not in KNOWN_FALSE_POSITIVES:
                errs.append(f"{qid}: cau hoi chua '{m.group(0)}'")
        rows.append((row, how, rec))

    if len(rows) != 99 or len({r["qa_id"] for r, _, _ in rows}) != 99:
        errs.append(f"so dong = {len(rows)} (can 99, khong trung)")
    if errs:
        print("[LOI] Khong ghi file:")
        for e in errs:
            print("  -", e)
        return 1

    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for row, _, _ in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_q = sum(r["question"] != s["question"] for (r, _, _), s in zip(rows, frozen))
    n_a = sum(r["answer"] != s["answer"] for (r, _, _), s in zip(rows, frozen))
    n_e = sum(r["evidence"] != s["evidence"] for (r, _, _), s in zip(rows, frozen))
    n_unchanged = sum(r == s for (r, _, _), s in zip(rows, frozen))
    n_quotes = sum(len(r["evidence"]) for r, _, _ in rows)
    resid = [r["qa_id"] for r, _, rec in rows if "residual_missing_after_fix" in rec.get("minor_issues", [])]
    sha_end = sha(FROZEN)
    print(f"Da ghi {OUT.relative_to(ROOT)}: 99 dong, {n_quotes} quote (tat ca khop dung 1 lan, 40-300 ky tu)")
    print(f"Nguon: {dict(source)}")
    print(f"So voi v1: doi cau hoi={n_q}, doi cau tra loi={n_a}, doi evidence={n_e}, giu nguyen hoan toan={n_unchanged}")
    print(f"Con thieu y sau khi sua (residual_missing_after_fix): {resid}")
    print(f"candidate_v2 sha256 = {sha(OUT)}")
    print(f"frozen sha256 dau={sha_start}\nfrozen sha256 cuoi={sha_end} {'(khop)' if sha_start == sha_end else '(KHAC!)'}")
    return 0 if sha_start == sha_end else 3


if __name__ == "__main__":
    sys.exit(main())
