#!/usr/bin/env bash
# Retrieval bo test clair_ext (438 cau) voi dense Qwen3-Embedding-4B (dung model/revision cua harness A5000),
# + kiem tra tren CLAIR so voi ket qua A5000 chinh thuc. Chay tren pod >=1 GPU (>=24 GB).
#   mkdir -p /workspace/retr && cd /workspace/retr && python3 -c "import zipfile; zipfile.ZipFile('../retrieval_bundle.zip').extractall('.')"
#   nohup bash run_retrieval_ext_remote.sh all > all.log 2>&1 &      ;  tail -f all.log   (xong: "XONG TAT CA")
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO"
export HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1 AC_HARNESS="$REPO/harness"
M=/workspace/models; QWEN4="$M/Qwen3-Embedding-4B@5cf2132"; SNOW="$M/snowflake-arctic-embed-l-v2.0@ac6544c"
PR="research/structure_chunker/retrieval/proxy_retrieval.py"
act() { source "$REPO/.venv/bin/activate"; }

setup() {
  nvidia-smi --query-gpu=index,name,memory.total --format=csv
  [ -d .venv ] || python3 -m venv .venv
  act
  pip install -q --upgrade pip
  pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  pip install -q -e .
  pip install -q haystack-ai==2.31.0 transformers==4.57.6 sentence-transformers==5.1.2 pandas==3.0.6 numpy==2.4.6 \
      tiktoken==0.14.0 huggingface-hub==0.36.2 tokenizers==0.22.2 safetensors==0.8.0
  python - "$QWEN4" "$SNOW" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-Embedding-4B", revision="5cf2132abc99cad020ac570b19d031efec650f2b", local_dir=sys.argv[1])
snapshot_download("Snowflake/snowflake-arctic-embed-l-v2.0", revision="ac6544c8a46e00af67e330e85a9028c66b8cfd9a", local_dir=sys.argv[2])
print("models ok")
PY
  pip freeze > environment.txt
  echo "SETUP XONG"
}

ext_part() {  # $1 = gpu, $2 = systems, $3 = out
  CUDA_VISIBLE_DEVICES=$1 HF_HUB_OFFLINE=1 python -u -X utf8 $PR --prepared runs/ext_prepared_v1/prepared \
     --qa runs/qa_ext_v1/qa_frozen.jsonl --data-dir data/clair_ext_set --cache-tag ext_ --systems "$2" --reprs plain,doc \
     --dense-model "$QWEN4" --rerank-model "$SNOW" --out "$3"
}
clair_check() {  # $1 = gpu
  CUDA_VISIBLE_DEVICES=$1 HF_HUB_OFFLINE=1 python -u -X utf8 $PR --prepared runs/table5_repr_v1/prepared \
     --qa runs/table5_v2/qa-full/qa_frozen.jsonl --data-dir data/clair --systems adaptive,sdp_a_t600 --reprs plain,doc \
     --dense-model "$QWEN4" --rerank-model "$SNOW" --out out/clair_check_4b
  python compare_with_a5000.py out/clair_check_4b a5000_per_query_metrics.jsonl | tee out/clair_check_vs_a5000.txt
}

run() {
  act; mkdir -p out
  local n; n=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  if [ "$n" -ge 2 ]; then
    ( ext_part 0 adaptive,raw__page,sdp_b out/ext_4b_part0 > out/part0.log 2>&1 ) &
    ( ext_part 1 raw__langch_recurs_default,sdp_a_t600 out/ext_4b_part1 > out/part1.log 2>&1; clair_check 1 > out/clair.log 2>&1 ) &
    wait
  else
    ext_part 0 adaptive,raw__page,sdp_b,raw__langch_recurs_default,sdp_a_t600 out/ext_4b_part0 > out/part0.log 2>&1
    clair_check 0 > out/clair.log 2>&1
  fi
  grep -h "@\|Error\|Traceback" out/*.log | tail -30
  cat out/clair_check_vs_a5000.txt
}

pack() {
  cp environment.txt out/ 2>/dev/null || true
  nvidia-smi --query-gpu=name --format=csv,noheader > out/gpus.txt
  tar -czf /workspace/retrieval_results.tar.gz --exclude='_emb_cache' out
  sha256sum /workspace/retrieval_results.tar.gz | tee /workspace/retrieval_results.tar.gz.sha256
}

case "${1:-all}" in
  setup) setup ;; run) run ;; pack) pack ;;
  all) setup; run; pack; echo "XONG TAT CA $(date -u)" ;;
  *) echo "dung: bash $0 [setup|run|pack|all]"; exit 1 ;;
esac
