"""Kiem tra va gan nhan QA sinh tu doan chat doc lap, roi (tuy chon) freeze.

Moi tai lieu: dan JSON tra ve tu chat vao runs/qa_new/raw/<so>.json (cung so voi file export).
Kiem tra: dung schema {"qas": [3 x {question, answer, evidence: [{page, quote}]}]},
quote phai xuat hien DUNG 1 LAN tren trang (tu sua neu chi khac khoang trang/dau nhay), cau hoi tu du nghia.
Tu sua JSON neu chat/Notepad doi dau ngoac kep thang thanh dau cong.
Gan nhan do kho: so bang chung, khac trang, khac block (theo split_points), co bang, do trung tu.
Cach dung:  python -X utf8 runs\validate_qa.py            (chi kiem tra)
            python -X utf8 runs\validate_qa.py --freeze "<ten model/chat da dung>"
"""
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

DATA = Path("data/clair/adi_parsed")
BASE = Path("runs/qa_new")
VAGUE = re.compile(r"\b(this|the) (document|report|paper|text|section)\b|\babove\b", re.I)
STOP = set("the a an of to in and or for on by with is are was were be what which how why does do did "
           "that this it its as at from under about into than who whom whose when where".split())


def words(text):
    return {w for w in re.findall(r"[a-zà-ỹ0-9]+", text.lower()) if w not in STOP and len(w) > 2}


# Chat thuong "lam dep" dau nhay/gach noi (' -> ’, - -> –). Khi so khop coi cac bien the la mot.
VARIANTS = ["'’‘`´ʼ", '"“”«»„', "-‐‑–—"]


def _char(ch):
    for group in VARIANTS:
        if ch in group:
            return "[" + re.escape(group) + "]"
    return re.escape(ch)


def locate(page_text, quote):
    """Tra ve (vi tri, quote_chinh_xac, ghi_chu) hoac (None, None, loi)."""
    hits = [m.start() for m in re.finditer(re.escape(quote), page_text)]
    if len(hits) == 1:
        return hits[0], quote, ""
    pattern = r"\s+".join("".join(_char(c) for c in part) for part in quote.split())
    loose = list(re.finditer(pattern, page_text))
    if len(loose) == 1:
        return loose[0].start(), loose[0].group(0), "da sua dau nhay/khoang trang theo van ban goc"
    return None, None, f"quote xuat hien {len(hits) or len(loose)} lan"


def load_json(path):
    """Doc JSON tu chat. Neu dau " cau truc bi doi thanh “ ”, sua lai (giu nguyen “ ” trong noi dung)."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", path.read_text(encoding="utf-8-sig").strip())
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        if "“" not in text and "”" not in text:
            raise
    keys = "qas|question|answer|evidence|page|quote"
    fixed = re.sub(rf"[“”]({keys})[“”](\s*:)", r'"\1"\2', text)                          # ten truong
    fixed = re.sub(r'("(?:question|answer|quote)"\s*:\s*)[“”]', r'\1"', fixed)             # mo gia tri
    fixed = re.sub(rf'[“”](\s*,\s*"(?:{keys})")', r'"\1', fixed)                          # dong truoc truong ke tiep
    fixed = re.sub(r"[“”](\s*[}\]])", r'"\1', fixed)                                       # dong truoc } ]
    return json.loads(fixed), True


index = {row["file"].replace(".txt", ""): row for row in csv.DictReader((BASE / "export_index.csv").open(encoding="utf-8"))}
rows, problems = [], []
for raw in sorted((BASE / "raw").glob("*.json")):
    key = raw.stem
    meta = index.get(key)
    if meta is None:
        problems.append(f"{raw.name}: khong co trong export_index.csv")
        continue
    doc = json.loads((DATA / f"{meta['doc_name']}.json").read_text(encoding="utf-8"))
    offsets, pos = {}, 0
    for n in sorted(doc["pages"], key=int):
        offsets[int(n)] = pos
        pos += len(doc["pages"][n])
    try:
        payload, repaired = load_json(raw)
        qas = payload["qas"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        problems.append(f"{raw.name}: JSON khong hop le ({error})")
        continue
    if repaired:
        print(f"  {raw.name}: da sua dau ngoac kep cong thanh dau \" trong cau truc JSON")
    if len(qas) != 3:
        problems.append(f"{raw.name}: co {len(qas)} cau hoi, can 3")
    for j, qa in enumerate(qas, start=1):
        qa_id = f"{key}-{j}"
        spans, notes, ok = [], [], True
        for ev in qa.get("evidence", []):
            page = int(ev.get("page", -1))
            if str(page) not in doc["pages"]:
                problems.append(f"{qa_id}: trang {page} khong ton tai")
                ok = False
                continue
            start, exact, note = locate(doc["pages"][str(page)], str(ev.get("quote", "")))
            if start is None:
                problems.append(f"{qa_id}: trang {page}: {note}: {str(ev.get('quote', ''))[:60]!r}")
                ok = False
                continue
            ev["quote"] = exact
            if note:
                notes.append(note)
            s = offsets[page] + start
            spans.append((s, s + len(exact), page))
        if not qa.get("evidence"):
            problems.append(f"{qa_id}: thieu evidence")
            ok = False
        if VAGUE.search(qa.get("question", "")):
            notes.append("cau hoi mo ho (this document/above...)")
        blocks = {sum(1 for p in doc["split_points"] if p <= s) for s, _, _ in spans}
        crosses = any(any(s < p < e for p in doc["split_points"]) for s, e, _ in spans)
        evidence_words = set().union(*[words(ev["quote"]) for ev in qa.get("evidence", [])]) if qa.get("evidence") else set()
        q_words = words(qa.get("question", ""))
        rows.append({"qa_id": qa_id, "doc_name": meta["doc_name"], "ok": ok,
                     "n_evidence": len(spans), "cross_page": len({p for _, _, p in spans}) > 1,
                     "cross_block": len(blocks) > 1 or crosses,
                     "has_table": any("|" in ev["quote"] or "<t" in ev["quote"].lower() for ev in qa.get("evidence", [])),
                     "lexical_overlap": round(len(q_words & evidence_words) / max(len(q_words), 1), 2),
                     "notes": "; ".join(notes), "question": qa.get("question", ""),
                     "answer": qa.get("answer", ""), "evidence": qa.get("evidence", [])})

ok_rows = [r for r in rows if r["ok"]]
n = max(len(ok_rows), 1)
print(f"{len(rows)} cau hoi tu {len({r['doc_name'] for r in rows})} tai lieu; hop le: {len(ok_rows)}")
for tag in ("cross_page", "cross_block", "has_table"):
    print(f"  {tag:12} {sum(r[tag] for r in ok_rows):3d} ({sum(r[tag] for r in ok_rows) / n:.0%})")
print(f"  >=2 bang chung {sum(r['n_evidence'] > 1 for r in ok_rows):3d}; do trung tu TB {sum(r['lexical_overlap'] for r in ok_rows) / n:.2f} (cang thap cang kho cho BM25)")
for r in rows:
    if r["notes"]:
        print(f"  ghi chu {r['qa_id']}: {r['notes']}")
if problems:
    print("\nLOI CAN SUA:\n  " + "\n  ".join(problems))
with (BASE / "qa_review.csv").open("w", newline="", encoding="utf-8") as f:
    fields = [k for k in rows[0] if k != "evidence"] if rows else []
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

if "--freeze" in sys.argv:
    if problems:
        sys.exit("\nKhong freeze: con loi o tren.")
    generator = sys.argv[sys.argv.index("--freeze") + 1]
    frozen = BASE / "qa_new_frozen.jsonl"
    if frozen.exists():
        sys.exit(f"{frozen} da ton tai - khong ghi de bo QA da freeze")
    with frozen.open("w", encoding="utf-8") as f:
        for r in ok_rows:
            f.write(json.dumps({k: r[k] for k in ("qa_id", "doc_name", "question", "answer", "evidence")},
                               ensure_ascii=False) + "\n")
    prompt = BASE / "prompt.txt"
    manifest = {"sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(), "n_qa": len(ok_rows),
                "n_docs": len({r["doc_name"] for r in ok_rows}), "generator": generator,
                "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest() if prompt.exists() else None,
                "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (BASE / "qa_new_frozen_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDa freeze {len(ok_rows)} QA -> {frozen}\nSHA256 {manifest['sha256']}")
