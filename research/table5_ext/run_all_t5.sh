#!/usr/bin/env bash
# Viec 3 (CLAIR, gateway), viec 4 (bo test, gateway, 1 job/he thong), viec 5 (CLAIR, API chinh thuc) - song song.
A=D:/Project/adaptive-chunking; J="bash $A/research/table5_ext/run_t5_job.sh"; M=gpt-4o-mini-2024-07-18
GW=D:/Project/keys/gateway.env; OF=D:/Project/keys/official.env; LOG=$A/runs/t5_logs; mkdir -p $LOG
CQA=$A/runs/table5_v2/qa-full/qa_frozen.jsonl; CR=$A/runs/table5_repr_results/table5_sdp
EI=$A/runs/t5_ext_inputs_v1
$J $A/runs/t5_clair_sdp_gw_4omini adaptive,raw__langch_recurs_default,raw__page,adaptive__doc,sdp_a_t600,sdp_a_t600__doc,sdp_b 99 \
   $CQA $CR/retrieval-full $CR/evaluation-full/retrieval_evaluation.json $GW $M 3 > $LOG/clair_gw.log 2>&1 &
for S in adaptive@plain sdp_a_t600@plain sdp_b@plain raw__langch_recurs_default@plain; do
  N=${S%@plain}
  $J $A/runs/t5_ext_gw_4omini_$N $S 438 $EI/qa_t5.jsonl $EI/retrieval-full $EI/retrieval_evaluation.json $GW $M 2 > $LOG/ext_$N.log 2>&1 &
  sleep 20
done
$J $A/runs/t5_clair_sdp_official_4omini adaptive,adaptive__doc,sdp_a_t600__doc 99 \
   $CQA $CR/retrieval-full $CR/evaluation-full/retrieval_evaluation.json $OF $M 2.5 > $LOG/clair_official.log 2>&1 &
wait
grep -h "JOB DONE\|STOP" $LOG/*.log
echo "ALL T5 DONE $(date)"
