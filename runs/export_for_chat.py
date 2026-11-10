"""Xuat moi tai lieu CLAIR thanh file text (co danh dau trang) de dan vao mot doan chat doc lap sinh QA.

Tai lieu dai duoc cat thanh mot cua so cac trang lien tiep (<= MAX_CHARS ky tu), vi tri bat dau chon
ngau nhien voi seed co dinh va ghi lai trong qa_new/export_index.csv de tai lap duoc.
Cach dung: python -X utf8 runs\export_for_chat.py
"""
import csv
import json
import random
from pathlib import Path

DATA = Path("data/clair/adi_parsed")
OUT = Path("runs/qa_new/export")
MAX_CHARS = 45_000   # ~11k token: vua 1 tin nhan chat, gan voi 10k token paper dung de sinh QA
SEED = 2026

OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(SEED)
rows = []
for i, path in enumerate(sorted(DATA.glob("*.json")), start=1):
    doc = json.loads(path.read_text(encoding="utf-8"))
    numbers = sorted(doc["pages"], key=int)
    total = sum(len(doc["pages"][n]) for n in numbers)
    start = 0
    if total > MAX_CHARS:   # chon trang bat dau sao cho cua so con du cho
        candidates = [k for k in range(len(numbers))
                      if sum(len(doc["pages"][n]) for n in numbers[k:]) >= MAX_CHARS * 0.8]
        start = rng.choice(candidates)
    chosen, size = [], 0
    for n in numbers[start:]:
        if chosen and size + len(doc["pages"][n]) > MAX_CHARS:
            break
        chosen.append(n)
        size += len(doc["pages"][n])
    body = "".join(f"\n<<<PAGE {n}>>>\n{doc['pages'][n]}" for n in chosen)
    name = f"{i:02d}.txt"
    # Chi ghi ten tai lieu, bo nhan bo du lieu (Ada/Norma/Sophia) vi nhan nay khong co trong van ban
    (OUT / name).write_text(f"DOCUMENT: {path.stem.split('_', 1)[-1]}\n{body}", encoding="utf-8")
    rows.append({"file": name, "doc_name": path.stem, "pages": f"{chosen[0]}-{chosen[-1]}",
                 "n_pages_total": len(numbers), "chars": size, "full_doc": total <= MAX_CHARS})
with (OUT.parent / "export_index.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Da xuat {len(rows)} file vao {OUT}; chi muc: {OUT.parent / 'export_index.csv'}")
