#!/usr/bin/env bash
# One Table-5 job end to end (prepare -> generate -> export -> judge -> summarize), resumable, via t5_multi.py.
# Retries only on the harness's own "TPM rate limit ... wait and resume" error; any other error stops the job.
#   bash run_t5_job.sh OUT SYSTEMS N_QA QA RETR_DIR EVAL ENV MODEL BUDGET
set -uo pipefail
OUT=$1; export T5_SYSTEMS=$2 T5_QA=$3; QA=$4; RETR=$5; EVAL=$6; ENVF=$7; MODEL=$8; BUDGET=$9
G=/d/Project/Adaptive-chunking-enhancements; A=/d/Project/adaptive-chunking
PY="$A/.venv/Scripts/python.exe -u -X utf8"; PYJ="$G/.venv-judge/Scripts/python.exe -u -X utf8"
L="$A/research/table5_ext/t5_multi.py --model $MODEL --price-in 0.15 --price-out 0.60"
export PYTHONUTF8=1
cd "$G"
retry() {  # retry on rate-limit only
  for i in $(seq 1 30); do
    out=$("$@" 2>&1); rc=$?; echo "$out" | tail -4
    [ $rc -eq 0 ] && return 0
    echo "$out" | grep -q "rate limit\|RateLimit\|wait and resume" || { echo "STOP (non rate-limit error)"; return 1; }
    echo "rate limited, sleep 90 (try $i)"; sleep 90
  done; return 1
}
echo "=== $OUT prepare $(date)"
[ -f "$OUT/generation_plan.json" ] || $PY $L table5 prepare --qa "$QA" --retrieval-dir "$RETR" --evaluation "$EVAL" --output-dir "$OUT" | tail -2
grep -o '"requests": [0-9]*\|"generation_upper_usd": [0-9.]*' "$OUT/generation_plan.json"
echo "=== generate $(date)"; retry $PY $L --env "$ENVF" table5 generate --output-dir "$OUT" --allow-paid-run --budget-usd "$BUDGET" --delay-seconds 0.5 || exit 1
echo "=== export $(date)"; $PY $L table5 export --output-dir "$OUT" | tail -2 || exit 1
echo "=== judge $(date)"; retry $PYJ $L --env "$ENVF" judge --output-dir "$OUT" --budget-usd "$BUDGET" --allow-paid-run --delay-seconds 2 || exit 1
echo "=== summarize $(date)"; $PY $L summarize --input-dir "$OUT/judge-full" --output-dir "$OUT/summary" --expected-queries "$T5_QA" | tail -3
echo "JOB DONE $OUT $(date)"
