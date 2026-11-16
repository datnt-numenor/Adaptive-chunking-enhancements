"""Buoc 3: kham pha du lieu CLAIR va doi chieu voi Table 1 cua paper."""
import json
from pathlib import Path

import tiktoken

DATA = Path("data/clair/adi_parsed")
DOMAINS = {"Ada (technique)": "Tech", "Norma (légal)": "Legal", "Sophia (sciences sociales)": "Social"}
PAPER_TABLE1 = {  # docs, mean, max, min, total tokens (paper Table 1)
    "Tech": (9, 5257, 18605, 1475, 47313),
    "Legal": (16, 30895, 118424, 523, 494320),
    "Social": (8, 79862, 391660, 8013, 638896),
}
PAGE_BREAK = "<!-- PageBreak -->"
TOLERANCE = 5  # cung nguong tau cua metric Block Integrity

enc = tiktoken.get_encoding("o200k_base")
stats = {name: [] for name in PAPER_TABLE1}
pages_total = matched_total = 0

files = sorted(DATA.glob("*.json"))
print(f"So tai lieu: {len(files)}")
for path in files:
    doc = json.loads(path.read_text(encoding="utf-8"))
    domain = DOMAINS[path.stem.split("_", 1)[0]]
    stats[domain].append(len(enc.encode(doc["full_text"])))

    # Moi ranh gioi trang co trung voi mot ranh gioi block (split_points) khong?
    text, splits = doc["full_text"], doc["split_points"]
    start = 0
    while (pos := text.find(PAGE_BREAK, start)) != -1:
        pages_total += 1
        matched_total += any(abs(s - pos) <= TOLERANCE for s in splits)
        start = pos + 1

print(f"\n{'Domain':8}{'docs':>6}{'mean':>9}{'max':>9}{'min':>8}{'total':>10}   | paper: docs/mean/max/min/total")
grand = 0
for name, tokens in stats.items():
    grand += sum(tokens)
    row = (len(tokens), round(sum(tokens) / len(tokens)), max(tokens), min(tokens), sum(tokens))
    print(f"{name:8}{row[0]:>6}{row[1]:>9}{row[2]:>9}{row[3]:>8}{row[4]:>10}   | {PAPER_TABLE1[name]}")
print(f"Tong token: {grand:,} (paper: ~1.18 trieu)")
print(f"\nRanh gioi trang trung split_points (+-{TOLERANCE} ky tu): {matched_total}/{pages_total}")