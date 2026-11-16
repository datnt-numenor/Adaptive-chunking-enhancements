#!/usr/bin/env bash
# 5 metric cua paper (jina-v3, revision ghim nhu Kaggle) cho bo test clair_ext, chay tren MOI GPU cua pod.
#   mkdir -p /workspace/metrics && cd /workspace/metrics && python3 -c "import zipfile; zipfile.ZipFile('../metrics_bundle.zip').extractall('.')"
#   (KHONG dung `unzip`: no lam hong ten file co dau, vd "Norma (légal)_...")
#   nohup bash run_metrics_remote.sh all > all.log 2>&1 &
#   tail -f all.log            # xem tien do; xong khi thay "XONG TAT CA"
# Cac buoc rieng: setup | validate | run | pack
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
export HF_HOME="${HF_HOME:-/workspace/model-cache/huggingface}" HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1
OUT=metrics_out
JINA_REV=ab036b023d30b4d1138c4c3bfa9f0c445ab455d6
CODE_REV=bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3

act() { source "$REPO/.venv/bin/activate"; }

setup() {
  nvidia-smi --query-gpu=index,name,memory.total --format=csv
  [ -d .venv ] || python3 -m venv .venv
  act
  pip install -q --upgrade pip
  pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  pip install -q -e .
  pip install -q transformers==4.57.6 sentence-transformers==5.1.2 einops==0.8.2 pandas==3.0.6 numpy==2.4.6 \
      pyarrow==25.0.1 tiktoken==0.14.0 huggingface-hub==0.36.2 tokenizers==0.22.2 safetensors==0.8.0
  # jina-v3 custom code tai tokenizer + xlm-roberta-flash-implementation tu nhanh `main`, bo qua revision.
  # Giong lan chay Kaggle/laptop: tai dung commit ghim roi tro refs/main vao commit do, sau do chay offline.
  python - "$JINA_REV" "$CODE_REV" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
for repo, rev in [("jinaai/jina-embeddings-v3", sys.argv[1]), ("jinaai/xlm-roberta-flash-implementation", sys.argv[2])]:
    snap = Path(snapshot_download(repo, revision=rev))
    ref = snap.parents[1] / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(rev)
    print(repo, "->", rev, snap)
PY
  pip freeze > environment.txt
  echo "SETUP XONG"
}

validate() {
  act
  export HF_HUB_OFFLINE=1
  python -X utf8 research/structure_chunker/score_full.py validate --docs 2 --methods our_recurs_1100 page \
      --out runs/sdp/_validate 2>&1 | tail -12 | tee validate.log
}

run() {
  act
  export HF_HUB_OFFLINE=1
  [ -f "$OUT/plan.json" ] || python -X utf8 research/data_expansion/score_shard.py plan \
      --run ext_v1=runs/ext_v1 --run sdp_a=runs/sdp_ext/A_target600 --run sdp_b=runs/sdp_ext/B_default \
      --data-dir data/clair_ext_set --skip skip.json --out-root "$OUT"
  echo "Tien do: tail -f $REPO/$OUT/worker*.log"
  python -X utf8 research/data_expansion/score_shard.py launch --out-root "$OUT"
  grep -h "DONE\|Error\|Traceback" "$OUT"/worker*.log || true
}

pack() {
  cp environment.txt validate.log "$OUT"/ 2>/dev/null || true
  nvidia-smi --query-gpu=name --format=csv,noheader > "$OUT/gpus.txt"
  tar -czf /workspace/metrics_results.tar.gz "$OUT"
  sha256sum /workspace/metrics_results.tar.gz | tee /workspace/metrics_results.tar.gz.sha256
  echo "Tai ve: /workspace/metrics_results.tar.gz (+ .sha256)"
}

case "${1:-all}" in
  setup) setup ;;
  validate) validate ;;
  run) run ;;
  pack) pack ;;
  all) setup; validate; run; pack; echo "XONG TAT CA $(date -u)" ;;
  *) echo "dung: bash $0 [setup|validate|run|pack|all]"; exit 1 ;;
esac
