#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/phase-g-full/adaptive-chunking
LOG=/workspace/adaptive-retrieval-benchmark.json

cd "$ROOT"
source .venv/bin/activate
export HF_HOME=/workspace/model-cache/huggingface
export TORCH_HOME=/workspace/model-cache/torch

started_epoch=$(date +%s)
python -X utf8 research/week3/phase_g_runner.py retrieve \
  --prepared-dir research/week3/artifacts/prepared \
  --qa-dir research/week3/artifacts/qa-full \
  --index-dir research/week3/artifacts/index-full \
  --output-dir research/week3/artifacts/retrieval-full \
  --environment-file /workspace/phase-g-full-environment.txt \
  --model-cache-dir /workspace/model-cache \
  --device cuda:0 \
  --batch-size 1 \
  --systems adaptive
finished_epoch=$(date +%s)

python - "$LOG" "$started_epoch" "$finished_epoch" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
started = int(sys.argv[2])
finished = int(sys.argv[3])
path.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "stage": "adaptive-retrieval-benchmark",
            "status": "complete",
            "started_epoch": started,
            "finished_epoch": finished,
            "elapsed_seconds": finished - started,
            "systems": ["adaptive"],
            "qa_count": 99,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

touch /workspace/PHASE_G_ADAPTIVE_RETRIEVAL_COMPLETE
