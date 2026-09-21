"""Read-only integrity audit of a downloaded nonsemantic export."""
import io
import json
import sys
import zipfile
from pathlib import Path
import pandas as pd

from reproduction_runner import NONSEMANTIC_METHODS, STAGES, _validate_chunk_frame

with zipfile.ZipFile(Path(sys.argv[1])) as archive:
    assert archive.testzip() is None, "ZIP CRC error"
    report = {}
    for stage in STAGES:
        paths = [
            name
            for name in archive.namelist()
            if name.endswith(f"/chunks/{stage}/chunks.parquet")
            and "-checkpoints/" not in name
        ]
        assert len(paths) == 1, (stage, paths)
        frame = pd.read_parquet(io.BytesIO(archive.read(paths[0])))
        _validate_chunk_frame(frame, stage, allow_empty_page_chunks=stage != "small_merged")
        assert frame.doc_name.nunique() == 11
        for name, group in frame.groupby("doc_name"):
            assert set(group.method) == NONSEMANTIC_METHODS, name
        report[stage] = dict(documents=11, methods=7, rows=len(frame),
                             empty_page_chunks=int(frame.chunk_text.fillna("").eq("").sum()))
    print(json.dumps(dict(status="ZIP_AUDIT_OK", stages=report), indent=2))
