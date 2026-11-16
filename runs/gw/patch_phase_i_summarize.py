"""Va phase_i_summarize.py: dinh nghia cau tu choi phai giong buoc cham va code goc.

phase_i_judge.py:261 va src/adaptive_chunking/paper/rag_eval.py:30 (repo goc, ea87ce8) coi la tu choi khi cau tra loi
CHUA chuoi "I don't know based on the provided context." -> khong cham Correctness.
phase_i_summarize.py:106 lai doi BANG DUNG chuoi do -> bao loi "abstention and correctness presence disagree"
khi model viet chuoi tu choi kem them chu. Ban va doi `==` thanh `in` (giong buoc cham). Khong doi diem nao.
Chay MOT lan tu thu muc goc repo nhom:  python -X utf8 patch_phase_i_summarize.py
Ghi lai research/week3/LOCAL_PATCH_phase_i_summarize.json (sha256 truoc/sau).
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

TARGET = Path("research/week3/phase_i_summarize.py")
RECORD = Path("research/week3/LOCAL_PATCH_phase_i_summarize.json")
OLD = '            is_abstention = row.get("generated_output") == ABSTAIN\n'
NEW = ('            is_abstention = ABSTAIN in (row.get("generated_output") or "")'
       '  # LOCAL PATCH: "chua" nhu phase_i_judge.py va rag_eval.py\n')


def main() -> None:
    if not TARGET.is_file():
        sys.exit(f"Khong thay {TARGET} - chay tu thu muc goc repo nhom")
    raw = TARGET.read_bytes()
    text = raw.decode("utf-8").replace("\r\n", "\n")
    if NEW in text:
        sys.exit("Da va tu truoc - khong lam gi")
    if text.count(OLD) != 1:
        sys.exit("Khong tim thay dung 1 cho can va - phien ban harness khac, dung lai va bao Claude")
    before = hashlib.sha256(raw).hexdigest()
    text = text.replace(OLD, NEW)
    compile(text, str(TARGET), "exec")
    crlf = b"\r\n" in raw
    TARGET.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    after = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    RECORD.write_text(json.dumps({
        "time": datetime.now(timezone.utc).isoformat(), "file": "research/week3/phase_i_summarize.py",
        "sha256_before": before, "sha256_after": after,
        "bug": "summarize treated abstention as exact equality while judge/rag_eval use substring containment",
        "change": "is_abstention uses `ABSTAIN in generated_output`, matching phase_i_judge.py:261 and rag_eval.py:30"},
        indent=2), encoding="utf-8")
    print(f"Da va {TARGET}: {before[:12]}... -> {after[:12]}...; ghi {RECORD}")


if __name__ == "__main__":
    main()
