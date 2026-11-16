#!/usr/bin/env bash
# Waits for the QA chain and the GPU chain, then: prepare systems -> proxy retrieval -> statistics. Git Bash:
#   bash research/data_expansion/run_final_ext.sh
set -uo pipefail
export PYTHONUTF8=1 HF_HUB_OFFLINE=1
PY=".venv/Scripts/python.exe -u -X utf8"; D=data/clair_ext_set; QA=runs/qa_ext_v1/qa_frozen.jsonl
until [ -f "$QA" ] && grep -q "GPU CHAIN DONE" runs/ext_gpu_chain.log; do sleep 60; done
echo "=== prepare $(date)"
[ -d runs/ext_prepared_v1 ] || $PY research/data_expansion/prepare_ext.py --data-dir $D --chunks runs/ext_v1 \
   --sdp sdp_a_t600=runs/sdp_ext/A_target600 --sdp sdp_b=runs/sdp_ext/B_default --out runs/ext_prepared_v1
SYS=adaptive,raw__langch_recurs_default,raw__page,sdp_a_t600,sdp_b,processed__our_recurs_1100,processed__page,processed__our_recurs_600,processed__llm_regex
echo "=== proxy retrieval $(date)"
[ -d runs/proxy_ext/r1 ] || $PY research/structure_chunker/retrieval/proxy_retrieval.py --prepared runs/ext_prepared_v1/prepared \
   --qa $QA --data-dir $D --cache-tag ext_ --systems $SYS --reprs plain,doc --out runs/proxy_ext/r1 2>&1 | grep -E "@|Saved|Error|Traceback"
echo "=== stats $(date)"
$PY research/structure_chunker/retrieval/analyze_sdp_retrieval.py --run runs/proxy_ext/r1 --qa $QA \
   --candidates sdp_a_t600@doc,sdp_b@doc,adaptive@doc --references adaptive@plain,raw__langch_recurs_default@plain,raw__page@plain \
   --out runs/proxy_ext/analysis_vs_plain 2>&1 | tail -30
$PY research/structure_chunker/retrieval/analyze_sdp_retrieval.py --run runs/proxy_ext/r1 --qa $QA \
   --candidates sdp_a_t600@doc,sdp_b@doc --references adaptive@doc,raw__langch_recurs_default@doc,raw__page@doc \
   --out runs/proxy_ext/analysis_same_repr 2>&1 | tail -30
echo "FINAL EXT DONE $(date)"
