#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/phase-g-full/adaptive-chunking
TIMING=/workspace/phase-g-full-timing.json

cd "$ROOT"
source .venv/bin/activate
export HF_HOME=/workspace/model-cache/huggingface
export TORCH_HOME=/workspace/model-cache/torch

run_started=$(date +%s)
index_started=$(date +%s)
python -X utf8 research/week3/phase_g_runner.py index \
  --prepared-dir research/week3/artifacts/prepared \
  --qa-dir research/week3/artifacts/qa-full \
  --output-dir research/week3/artifacts/index-full \
  --environment-file /workspace/phase-g-full-environment.txt \
  --smoke-evaluation-manifest research/week3/artifacts/phase-g-smoke-results/evaluation-smoke/retrieval_evaluation.json \
  --model-cache-dir /workspace/model-cache \
  --device cuda:0 \
  --batch-size 1 \
  --systems all
index_finished=$(date +%s)

retrieval_started=$(date +%s)
python -X utf8 research/week3/phase_g_runner.py retrieve \
  --prepared-dir research/week3/artifacts/prepared \
  --qa-dir research/week3/artifacts/qa-full \
  --index-dir research/week3/artifacts/index-full \
  --output-dir research/week3/artifacts/retrieval-full \
  --environment-file /workspace/phase-g-full-environment.txt \
  --model-cache-dir /workspace/model-cache \
  --device cuda:0 \
  --batch-size 1 \
  --systems all
retrieval_finished=$(date +%s)

evaluation_started=$(date +%s)
python -X utf8 research/week3/phase_g_runner.py evaluate-retrieval \
  --prepared-dir research/week3/artifacts/prepared \
  --qa-dir research/week3/artifacts/qa-full \
  --retrieval-dir research/week3/artifacts/retrieval-full \
  --output-dir research/week3/artifacts/evaluation-full \
  --systems all
evaluation_finished=$(date +%s)

python - "$TIMING" "$run_started" "$index_started" "$index_finished" \
  "$retrieval_started" "$retrieval_finished" "$evaluation_started" \
  "$evaluation_finished" <<'PY'
import json
import sys
from pathlib import Path

(
    path,
    run_started,
    index_started,
    index_finished,
    retrieval_started,
    retrieval_finished,
    evaluation_started,
    evaluation_finished,
) = sys.argv[1:]
values = [
    int(run_started),
    int(index_started),
    int(index_finished),
    int(retrieval_started),
    int(retrieval_finished),
    int(evaluation_started),
    int(evaluation_finished),
]
payload = {
    "schema_version": 1,
    "status": "complete",
    "run_started_epoch": values[0],
    "run_finished_epoch": values[-1],
    "elapsed_seconds": values[-1] - values[0],
    "stages": {
        "index_all_resume": values[2] - values[1],
        "retrieve_all_resume": values[4] - values[3],
        "evaluate_all": values[6] - values[5],
    },
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

cd /workspace
tar -czf phase-g-full-results.tar.gz \
  phase-g-full/adaptive-chunking/research/week3/artifacts/index-full \
  phase-g-full/adaptive-chunking/research/week3/artifacts/retrieval-full \
  phase-g-full/adaptive-chunking/research/week3/artifacts/evaluation-full \
  phase-g-full-environment.txt \
  adaptive-index-benchmark.json \
  adaptive-retrieval-benchmark.json \
  phase-g-full-timing.json \
  phase-g-full-setup.log \
  phase-g-adaptive-retrieval.log \
  phase-g-full-continue.log
sha256sum phase-g-full-results.tar.gz > phase-g-full-results.tar.gz.sha256
touch /workspace/PHASE_G_FULL_COMPLETE
