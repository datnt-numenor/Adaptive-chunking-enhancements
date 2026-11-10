#!/usr/bin/env bash
# Retrieval cho chunker SDP tren RTX A5000 (hoac A6000), cung cau hinh voi run_table5_v2_remote.sh.
# Khac biet duy nhat: goi harness qua sdp_launcher.py (them ten he thong SDP vao ALL_SYSTEMS trong bo nho,
# khong sua file harness), va danh sach he thong doc tu table5_sdp/systems.txt.
#   bash run_sdp_retrieval_remote.sh check   # kiem tra GPU
#   bash run_sdp_retrieval_remote.sh setup   # cai moi truong (1 lan)
#   bash run_sdp_retrieval_remote.sh smoke   # 6 cau / 2 tai lieu
#   bash run_sdp_retrieval_remote.sh full    # 99 cau / 33 tai lieu
#   bash run_sdp_retrieval_remote.sh pack    # nen ket qua -> /workspace/table5_sdp_results.tar.gz
#   bash run_sdp_retrieval_remote.sh all     # tat ca cac buoc tren, mot lenh (dung khi muon chay tu dong)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T5="$REPO/table5_sdp"
WORKSPACE="${WORKSPACE:-/workspace}"
MODEL_CACHE="$WORKSPACE/model-cache"
SYSTEMS="$(tr -d ' \r\n' < "$T5/systems.txt")"
export SDP_SYSTEMS="$(tr -d ' \r\n' < "$T5/sdp_systems.txt")"
ENV_FILE="$T5/environment.txt"
LOGS="$T5/logs"
mkdir -p "$LOGS" "$MODEL_CACHE"
cd "$REPO"

export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HOME="$MODEL_CACHE/huggingface" TORCH_HOME="$MODEL_CACHE/torch" TOKENIZERS_PARALLELISM=false

activate() {
  [ -f "$REPO/.venv/bin/activate" ] || { echo "Chua setup: chay 'bash $0 setup' truoc"; exit 1; }
  # shellcheck disable=SC1091
  source "$REPO/.venv/bin/activate"
}

refuse_existing() {
  for d in "$@"; do
    if [ -e "$d" ]; then echo "$d da ton tai - khong ghi de."; exit 1; fi
  done
}

step() {
  local name="$1"; shift
  local start; start=$(date +%s)
  echo "=== $name ($(date -u +%H:%M:%S) UTC)"
  "$@" 2>&1 | tee "$LOGS/$name.log"
  local end; end=$(date +%s)
  echo "{\"stage\": \"$name\", \"started_epoch\": $start, \"elapsed_seconds\": $((end - start))}" >> "$T5/timings.jsonl"
  echo "=== $name xong sau $((end - start)) giay"
}

runner() { python -X utf8 sdp_launcher.py "$@"; }

summary() {
  python - "$1" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"\nKET QUA ({m['mode']}, {m['qa_count']} cau) - metric chinh: recall@10")
cols = ["recall@10", "recall@5", "recall@1", "hit@10", "mrr@10", "ndcg@10"]
print(f"{'he thong':30}" + "".join(f"{c:>11}" for c in cols))
for name, values in m["summary"].items():
    print(f"{name:30}" + "".join(f"{values.get(c, float('nan')):>11.3f}" for c in cols))
PY
}

check_gpu() {
  local name
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  case "$name" in
    *A5000*) echo "GPU: $name - dung cau hinh cua nhom." ;;
    *"RTX A6000"*)
      echo "GPU: $name - va 1 dong kiem tra GPU trong ban copy harness tren may nay (co ghi lai)."
      python3 - "$REPO/research/week3/phase_g_runner.py" "$T5/harness_patch.json" "$name" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
path, record, gpu = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
old = '    if "A5000" not in name:\n        raise PhaseGError(f"Expected an RTX A5000, found {name}")'
new = ('    if "A5000" not in name and "RTX A6000" not in name:  # LOCAL PATCH: cho phep RTX A6000 (Ampere sm_86)\n'
       '        raise PhaseGError(f"Expected an RTX A5000 or RTX A6000, found {name}")')
text = path.read_text(encoding="utf-8")
if new in text:
    print("Harness da duoc va tu truoc."); sys.exit(0)
if text.count(old) != 1:
    sys.exit("Khong tim thay dung 1 cho kiem tra A5000 - dung lai")
before = hashlib.sha256(path.read_bytes()).hexdigest()
path.write_text(text.replace(old, new), encoding="utf-8")
after = hashlib.sha256(path.read_bytes()).hexdigest()
record.write_text(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "gpu": gpu,
    "file": "research/week3/phase_g_runner.py", "sha256_before": before, "sha256_after": after,
    "change": "GPU preflight accepts RTX A6000 in addition to RTX A5000"}, indent=2))
print(f"Da va harness: {before[:12]}... -> {after[:12]}...")
PY
      ;;
    *) echo "GPU '$name' khong duoc ho tro: can RTX A5000 hoac RTX A6000."; exit 1 ;;
  esac
}

case "${1:-}" in
  check) check_gpu ;;
  setup)
    check_gpu
    command -v python3.11 >/dev/null || { echo "Khong co python3.11 - chon template PyTorch co Python 3.11"; exit 1; }
    [ -d "$REPO/.venv" ] || python3.11 -m venv "$REPO/.venv"
    activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    python -m pip install -e .
    python -m pip install -r research/week3/requirements-gpu.txt
    MAX_JOBS=4 python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
    python -m pip freeze > "$ENV_FILE"
    echo "SETUP XONG. Tiep: bash $0 smoke"
    ;;
  smoke)
    activate
    refuse_existing "$T5/index-smoke" "$T5/retrieval-smoke" "$T5/evaluation-smoke"
    common=(--prepared-dir "$T5/prepared" --qa-dir "$T5/qa-smoke" --systems "$SYSTEMS")
    gpu=(--device cuda:0 --batch-size 1 --model-cache-dir "$MODEL_CACHE" --environment-file "$ENV_FILE")
    step index-smoke runner index "${common[@]}" "${gpu[@]}" --output-dir "$T5/index-smoke"
    step retrieve-smoke runner retrieve "${common[@]}" "${gpu[@]}" --index-dir "$T5/index-smoke" --output-dir "$T5/retrieval-smoke"
    step evaluate-smoke runner evaluate-retrieval "${common[@]}" --retrieval-dir "$T5/retrieval-smoke" --output-dir "$T5/evaluation-smoke"
    summary "$T5/evaluation-smoke/retrieval_evaluation.json"
    echo "Gui bang tren cho Claude xem truoc khi chay: bash $0 full"
    ;;
  full)
    activate
    [ -f "$T5/evaluation-smoke/retrieval_evaluation.json" ] || { echo "Chua co smoke"; exit 1; }
    refuse_existing "$T5/index-full" "$T5/retrieval-full" "$T5/evaluation-full"
    common=(--prepared-dir "$T5/prepared" --qa-dir "$T5/qa-full" --systems "$SYSTEMS")
    gpu=(--device cuda:0 --batch-size 1 --model-cache-dir "$MODEL_CACHE" --environment-file "$ENV_FILE")
    step index-full runner index "${common[@]}" "${gpu[@]}" --output-dir "$T5/index-full" \
      --smoke-evaluation-manifest "$T5/evaluation-smoke/retrieval_evaluation.json"
    step retrieve-full runner retrieve "${common[@]}" "${gpu[@]}" --index-dir "$T5/index-full" --output-dir "$T5/retrieval-full"
    step evaluate-full runner evaluate-retrieval "${common[@]}" --retrieval-dir "$T5/retrieval-full" --output-dir "$T5/evaluation-full"
    summary "$T5/evaluation-full/retrieval_evaluation.json"
    echo "Tiep: bash $0 pack"
    ;;
  pack)
    out="$WORKSPACE/table5_sdp_results.tar.gz"
    refuse_existing "$out"
    cd "$REPO"
    items=(bundle_manifest.json table5_sdp/environment.txt table5_sdp/timings.jsonl table5_sdp/logs table5_sdp/systems.txt)
    [ -e table5_sdp/harness_patch.json ] && items+=(table5_sdp/harness_patch.json)
    for d in retrieval-smoke evaluation-smoke retrieval-full evaluation-full; do
      [ -e "table5_sdp/$d" ] && items+=("table5_sdp/$d")
    done
    for f in table5_sdp/index-*/index_manifest.json; do [ -e "$f" ] && items+=("$f"); done
    tar czf "$out" "${items[@]}"
    (cd "$WORKSPACE" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256")
    ls -lh "$out"; cat "$out.sha256"
    echo "Tai 2 file nay ve may, roi TAT VA XOA POD."
    ;;
  all)  # chay het mot mach, khong can cho: setup (neu chua) -> smoke -> full -> pack
    [ -f "$REPO/.venv/bin/activate" ] || bash "$0" setup
    [ -f "$T5/evaluation-smoke/retrieval_evaluation.json" ] || bash "$0" smoke
    [ -f "$T5/evaluation-full/retrieval_evaluation.json" ] || bash "$0" full
    bash "$0" pack
    echo "XONG TAT CA. Tai /workspace/table5_sdp_results.tar.gz + .sha256 ve, roi XOA POD."
    ;;
  *) echo "Cach dung: bash $0 {check|setup|smoke|full|pack|all}"; exit 1 ;;
esac
