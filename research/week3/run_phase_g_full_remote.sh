#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
BUNDLE_DIR="$WORKSPACE/phase-g-full"
REPO_DIR="$BUNDLE_DIR/adaptive-chunking"
LOG_DIR="$WORKSPACE/phase-g-full-logs"
MODEL_CACHE="$WORKSPACE/model-cache"

mkdir -p "$LOG_DIR" "$MODEL_CACHE"
cd "$WORKSPACE"
sha256sum -c runpod_phase_g_full.zip.sha256
rm -rf "$BUNDLE_DIR"
mkdir -p "$REPO_DIR"
python3.11 -m zipfile -e runpod_phase_g_full.zip "$REPO_DIR"
cd "$REPO_DIR"

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
python -m pip install -r research/week3/requirements-gpu.txt
MAX_JOBS=4 python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip freeze > "$WORKSPACE/phase-g-full-environment.txt"

export HF_HOME="$MODEL_CACHE/huggingface"
export TORCH_HOME="$MODEL_CACHE/torch"
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import flash_attn
import torch

assert torch.cuda.get_device_name(0) == "NVIDIA RTX A5000"
assert torch.cuda.get_device_capability(0) >= (8, 0)
assert torch.cuda.is_bf16_supported()
assert flash_attn.__version__ == "2.7.4.post1"
print({
    "gpu": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "bf16": torch.cuda.is_bf16_supported(),
    "torch": torch.__version__,
    "flash_attn": flash_attn.__version__,
})
PY

start_epoch=$(date +%s)
python -X utf8 research/week3/phase_g_runner.py index \
  --prepared-dir research/week3/artifacts/prepared \
  --qa-dir research/week3/artifacts/qa-full \
  --output-dir research/week3/artifacts/index-full \
  --environment-file "$WORKSPACE/phase-g-full-environment.txt" \
  --smoke-evaluation-manifest research/week3/artifacts/phase-g-smoke-results/evaluation-smoke/retrieval_evaluation.json \
  --model-cache-dir "$MODEL_CACHE" \
  --device cuda:0 --batch-size 1 \
  --systems adaptive 2>&1 | tee "$LOG_DIR/adaptive-index.log"
end_epoch=$(date +%s)

python - "$start_epoch" "$end_epoch" <<'PY'
import json
from pathlib import Path
import sys

start, end = map(int, sys.argv[1:])
manifest = json.loads(
    Path("research/week3/artifacts/index-full/index_manifest.json").read_text()
)
payload = {
    "schema_version": 1,
    "stage": "adaptive-index-benchmark",
    "status": manifest["status"],
    "started_epoch": start,
    "finished_epoch": end,
    "elapsed_seconds": end - start,
    "chunks": manifest["systems"]["adaptive"]["chunks"],
    "qa_count": manifest["qa_count"],
    "gpu": manifest["gpu"],
}
Path("/workspace/adaptive-index-benchmark.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
PY

touch "$WORKSPACE/PHASE_G_ADAPTIVE_BENCHMARK_COMPLETE"
