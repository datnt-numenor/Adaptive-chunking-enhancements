#!/usr/bin/env bash
# QA chain for the test set (paid steps via gateway, all cached + budget-capped). Git Bash:
#   bash research/data_expansion/run_qa_chain.sh runs/qa_ext_v1
set -euo pipefail
W="$1"; P=data/clair_ext_set/adi_parsed; ENV=D:/Project/keys/gateway.env; PY=".venv/Scripts/python.exe -X utf8"
G=research/data_expansion/gen_qa.py
$PY $G generate --work "$W" --parsed $P --model gpt-4.1-2025-04-14 --env $ENV --price-in 2 --price-out 8 --budget-usd 10 --allow-paid-run --workers 6
$PY $G validate --work "$W" --parsed $P | head -3
$PY $G review --work "$W" --parsed $P --model claude-sonnet-4-5 --env $ENV --price-in 3 --price-out 15 --allow-paid-run --workers 6
$PY $G review --work "$W" --parsed $P --model gemini-2.5-pro --env $ENV --price-in 1.25 --price-out 10 --allow-paid-run --workers 6
$PY $G freeze --work "$W" --parsed $P --review-dirs "$W/review_claude-sonnet-4-5" "$W/review_gemini-2.5-pro"
echo "QA CHAIN DONE"
