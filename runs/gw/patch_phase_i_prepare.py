"""Va loi phase_i_table5.py prepare tren thu muc dau ra moi (NameError: 'old').

Dong 220 (commit 21edf03): `previous_output_cap or old.get(...) if requests_path.exists() else None`
-> requests vua duoc ghi nen exists() luon dung, ma `old` chi duoc gan khi thu muc da co plan cu.
Ban va: khoi tao old = None va chi doc old khi co. Hanh vi voi thu muc da co plan cu giu nguyen.
Chay MOT lan tu thu muc goc repo nhom:  python -X utf8 patch_phase_i_prepare.py
Ghi lai research/week3/LOCAL_PATCH_phase_i_prepare.json (sha256 truoc/sau).
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

TARGET = Path("research/week3/phase_i_table5.py")
RECORD = Path("research/week3/LOCAL_PATCH_phase_i_prepare.json")
EDITS = [
    ('    previous_output_cap = None\n    if requests_path.exists():\n',
     '    previous_output_cap = None\n'
     '    old: dict[str, Any] | None = None  # LOCAL PATCH: tranh NameError khi thu muc dau ra moi\n'
     '    if requests_path.exists():\n'),
    ('"previous_output_cap": previous_output_cap or old.get("previous_output_cap") if requests_path.exists() else None,',
     '"previous_output_cap": previous_output_cap or (old.get("previous_output_cap") if old else None),  # LOCAL PATCH'),
]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    if not TARGET.is_file():
        sys.exit(f"Khong thay {TARGET} - chay tu thu muc goc repo nhom")
    text = TARGET.read_text(encoding="utf-8")
    if all(new in text for _, new in EDITS):
        sys.exit("Da va tu truoc - khong lam gi")
    for old, _ in EDITS:
        if text.count(old) != 1:
            sys.exit("Khong tim thay dung 1 cho can va - phien ban harness khac, dung lai va bao Claude")
    before = hashlib.sha256(TARGET.read_bytes()).hexdigest()   # sha cua file that, nhu harness tinh
    for old, new in EDITS:
        text = text.replace(old, new)
    compile(text, str(TARGET), "exec")
    crlf = b"\r\n" in TARGET.read_bytes()            # giu nguyen kieu xuong dong (Windows/git autocrlf)
    TARGET.write_text(text, encoding="utf-8", newline="\r\n" if crlf else "\n")
    after = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    RECORD.write_text(json.dumps({
        "time": datetime.now(timezone.utc).isoformat(), "file": str(TARGET).replace("\\", "/"),
        "sha256_before": before, "sha256_after": after,
        "bug": "prepare() raised NameError on a fresh output directory (operator precedence on line 220)",
        "change": "initialise old=None; read old only when an earlier plan exists"}, indent=2), encoding="utf-8")
    print(f"Da va {TARGET}: {before[:12]}... -> {after[:12]}...; ghi {RECORD}")


if __name__ == "__main__":
    main()
