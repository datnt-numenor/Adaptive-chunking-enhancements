#!/usr/bin/env bash
# Free GPU chain for the test set (Git Bash): mentions -> SDP-B -> 5 metrics for every run. Resumable.
set -uo pipefail
export PYTHONUTF8=1 HF_HUB_OFFLINE=1
D=data/clair_ext_set; PY=".venv/Scripts/python.exe -u -X utf8"
echo "=== mentions $(date)"; /d/Project/.venv-coref/Scripts/python.exe research/data_expansion/mentions_ext.py --parsed $D/adi_parsed --out $D/mentions --device cuda:0 2>&1 | grep -E "^\[mentions\]|Error|Traceback"
echo "=== sdp_b $(date)"; [ -e runs/sdp_ext/B_default/config.json ] || $PY research/structure_chunker/run_sdp.py --data-dir $D --out runs/sdp_ext/B_default --method sdp_b --embed 2>&1 | tail -2
for R in runs/ext_v1 runs/sdp_ext/A_target600 runs/sdp_ext/B_default; do
  echo "=== metrics $R $(date)"; $PY research/structure_chunker/score_full.py run $R --data-dir $D 2>&1 | grep -E "seconds|Error|Traceback" | tail -3
done
echo "GPU CHAIN DONE $(date)"
