"""Freeze bo QA v2 tu runs/qa_new/review/candidate_v2.jsonl - chi chay khi moi dieu kien deu dat.

Dieu kien (sai mot cai la dung, khong ghi gi):
  - v1 khop manifest; candidate_v2 dung schema v1, 99 cau, 33 tai lieu x 3, khong trung qa_id/cau hoi,
    qa_id trung khop v1, moi quote xuat hien DUNG 1 LAN tren trang ghi kem, khong co nhan persona.
  - Lan cham cheo GPT-4.1 moi nhat (cross_review_manifest.json) da cham DUNG ban candidate_v2 hien tai
    (cung sha256), status complete.
  - Moi cau doi chu question/answer so voi v1, va moi cau GPT-4.1 khong cham KEEP tren v2, deu co
    quyet dinh "approve" trong runs/qa_new/review/adjudication_decisions.json:
        {"reviewer": "ten nguoi duyet", "decisions": {"03-1": "approve", "23-1": "approve", ...}}
    Cau bi "reject" -> dung lai: nho Claude Code sua candidate_v2 roi cham cheo lai.
Ghi: runs/qa_new/qa_new_frozen_v2.jsonl, runs/qa_new/qa_new_frozen_v2_manifest.json,
     runs/qa_new/review/v2_changelog.csv. Tu choi neu file v2 da ton tai. Khong sua v1.
Chay tu thu muc goc repo:  python -X utf8 runs\\qa_new\\review\\build_v2.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

QA = Path("runs/qa_new")
REVIEW = QA / "review"
DATA = Path("data/clair/adi_parsed")
KEYS = ["qa_id", "doc_name", "question", "answer", "evidence"]
PERSONA = re.compile(r"\b(Ada|Norma|Sophia)\b|sciences sociales|légal")
REVIEWER = "Claude (audit) + GPT-4.1 gpt-4.1-2025-04-14 (cross-review, blind) + human approval of text changes"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    errors, warnings = [], []
    v1_path, v1_manifest_path = QA / "qa_new_frozen.jsonl", QA / "qa_new_frozen_manifest.json"
    out, out_manifest = QA / "qa_new_frozen_v2.jsonl", QA / "qa_new_frozen_v2_manifest.json"
    if out.exists() or out_manifest.exists():
        sys.exit(f"{out} hoac manifest v2 da ton tai - khong ghi de ban da freeze")
    v1_sha = sha(v1_path)
    v1_manifest = json.loads(v1_manifest_path.read_text(encoding="utf-8"))
    if v1_sha != v1_manifest["sha256"]:
        sys.exit("v1 khong khop manifest - dung lai")
    v1 = {r["qa_id"]: r for r in read_jsonl(v1_path)}

    cand_path = REVIEW / "candidate_v2.jsonl"
    cand_sha = sha(cand_path)
    rows = read_jsonl(cand_path)

    # 1. Hinh thuc
    ids = [r.get("qa_id") for r in rows]
    per_doc = Counter(r.get("doc_name") for r in rows)
    if len(rows) != 99 or len(set(ids)) != 99 or set(ids) != set(v1):
        errors.append(f"can 99 qa_id duy nhat trung voi v1 (co {len(rows)} dong, {len(set(ids))} id)")
    if len(per_doc) != 33 or set(per_doc.values()) != {3}:
        errors.append("can 33 tai lieu x 3 cau")
    if len({r.get("question") for r in rows}) != len(rows):
        errors.append("co cau hoi trung nhau")
    docs: dict[str, dict] = {}
    for r in rows:
        q = r.get("qa_id")
        if list(r) != KEYS:
            errors.append(f"{q}: schema {list(r)} khac v1 {KEYS}")
            continue
        if q in v1 and r["doc_name"] != v1[q]["doc_name"]:
            errors.append(f"{q}: doc_name khac v1")
        if PERSONA.search(r["question"]) or PERSONA.search(r["answer"]):
            errors.append(f"{q}: con nhan persona trong question/answer")
        if len(r["question"].split()) > 30:
            warnings.append(f"{q}: cau hoi {len(r['question'].split())} tu (> 30)")
        if not r["evidence"]:
            errors.append(f"{q}: khong co evidence")
        name = r["doc_name"]
        if name not in docs:
            path = DATA / f"{name}.json"
            if not path.exists():
                errors.append(f"{q}: khong co tai lieu {path}")
                continue
            docs[name] = json.loads(path.read_text(encoding="utf-8"))
        for ev in r["evidence"]:
            page = docs[name]["pages"].get(str(ev.get("page")))
            if page is None:
                errors.append(f"{q}: trang {ev.get('page')} khong ton tai")
            elif page.count(ev.get("quote", "")) != 1:
                errors.append(f"{q}: quote xuat hien {page.count(ev.get('quote', ''))} lan tren trang {ev['page']}")
            if len(ev.get("quote", "")) > 300:
                warnings.append(f"{q}: quote dai {len(ev['quote'])} ky tu (> 300)")
        quotes = [(ev.get("page"), ev.get("quote")) for ev in r["evidence"]]
        if len(set(quotes)) != len(quotes):
            errors.append(f"{q}: co quote trung lap trong cung mot cau")
        if len(r["evidence"]) > 5:
            warnings.append(f"{q}: {len(r['evidence'])} quote (> 5)")

    # 2. Cham cheo GPT-4.1 phai la cua dung ban candidate nay
    cr_manifest_path = REVIEW / "cross_review_manifest.json"
    if not cr_manifest_path.exists() or not (REVIEW / "cross_review_gpt.jsonl").exists():
        print("KHONG FREEZE: chua co ket qua cham cheo GPT-4.1 - chay cross_review_qa.py --targets v2 --allow-paid-run")
        sys.exit(1)
    cr = json.loads(cr_manifest_path.read_text(encoding="utf-8"))
    if cr.get("inputs", {}).get("v2", {}).get("sha256") != cand_sha:
        errors.append("lan cham cheo moi nhat KHONG cham ban candidate_v2 hien tai - chay lai "
                      "cross_review_qa.py --targets v2 --allow-paid-run")
    if cr.get("mode") != "full" or cr.get("invalid") != 0:
        errors.append("lan cham cheo moi nhat chua phai full hoac con phan hoi khong hop le")
    gpt_v2 = {r["qa_id"]: r for r in read_jsonl(REVIEW / "cross_review_gpt.jsonl") if r["target"] == "v2"}
    if set(gpt_v2) != set(ids):
        errors.append(f"GPT-4.1 moi cham {len(gpt_v2)}/99 cau cua v2")

    # 3. Cau can nguoi duyet
    changes, need = [], {}
    for r in rows:
        q, old = r["qa_id"], v1.get(r["qa_id"], {})
        for field in ("question", "answer", "evidence"):
            if r.get(field) != old.get(field):
                changes.append({"qa_id": q, "field": field,
                                "old": json.dumps(old.get(field), ensure_ascii=False) if field == "evidence" else old.get(field),
                                "new": json.dumps(r[field], ensure_ascii=False) if field == "evidence" else r[field]})
                if field != "evidence":
                    need.setdefault(q, []).append(f"{field}_changed")
        if q in gpt_v2 and gpt_v2[q]["verdict"] != "KEEP":
            need.setdefault(q, []).append(f"gpt_v2_{gpt_v2[q]['verdict']}")
    decisions_path = REVIEW / "adjudication_decisions.json"
    decisions = {}
    if need:
        if not decisions_path.exists():
            errors.append(f"thieu {decisions_path} cho {len(need)} cau can duyet: {sorted(need)}")
        else:
            payload = json.loads(decisions_path.read_text(encoding="utf-8"))
            decisions = payload.get("decisions", {})
            if not payload.get("reviewer"):
                errors.append("adjudication_decisions.json thieu truong reviewer")
            for q in sorted(need):
                if decisions.get(q) != "approve":
                    errors.append(f"{q} ({', '.join(need[q])}): quyet dinh = {decisions.get(q)!r}, can 'approve'")

    for w in warnings:
        print("CANH BAO:", w)
    if errors:
        print("\nKHONG FREEZE - con loi:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    # 4. Ghi v2
    rows.sort(key=lambda r: r["qa_id"])
    with out.open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps({k: r[k] for k in KEYS}, ensure_ascii=False) + "\n")
    changed_ids = {c["qa_id"] for c in changes}
    by_field = Counter(c["field"] for c in changes)
    manifest = {
        "version": "v2", "sha256": sha(out), "n_qa": len(rows), "n_docs": len(per_doc),
        "generator": v1_manifest.get("generator"), "prompt_sha256": v1_manifest.get("prompt_sha256"),
        "parent_sha256": v1_sha, "candidate_sha256": cand_sha,
        "audit_sha256": sha(REVIEW / "qa_audit.jsonl"),
        "cross_review_manifest_sha256": sha(cr_manifest_path), "cross_review_cost_usd": cr.get("cost_usd"),
        "decisions_sha256": sha(decisions_path) if decisions_path.exists() else None,
        "reviewer": REVIEWER, "build_script_sha256": sha(Path(__file__)),
        "changes": {"items_changed": len(changed_ids),
                    "evidence_only": len({c["qa_id"] for c in changes} - {c["qa_id"] for c in changes if c["field"] != "evidence"}),
                    "question_changed": by_field["question"], "answer_changed": by_field["answer"],
                    "evidence_changed": by_field["evidence"]},
        "warnings": warnings, "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with (REVIEW / "v2_changelog.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qa_id", "field", "old", "new"])
        writer.writeheader()
        writer.writerows(changes)
    if sha(v1_path) != v1_sha:
        sys.exit("CANH BAO: v1 da thay doi trong luc chay!")
    print(f"Da freeze {out}\n  sha256 {manifest['sha256']}\n  thay doi: {manifest['changes']}")
    print(f"v1 khong doi: {v1_sha}")


if __name__ == "__main__":
    main()
