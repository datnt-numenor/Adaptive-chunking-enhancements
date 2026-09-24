# Phase G — protocol-faithful RAG retrieval benchmark

This phase consumes the validated Week 2 artifacts. It does not rerun chunking,
semantic splitting, or intrinsic metrics. The workflow is deliberately gated:

`prepare -> 2-document QA -> human review -> 2-document GPU smoke -> 33-document QA -> human review -> full GPU run`

Answer generation and LLM-as-judge evaluation are not part of Phase G.

## Reproducibility pins

- Source artifacts: upstream commit `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d`.
- QA model: `gpt-4.1-2025-04-14`, no SDK retries.
- Embedding model: `Qwen/Qwen3-Embedding-4B` at
  `5cf2132abc99cad020ac570b19d031efec650f2b`.
- Reranker model: `Snowflake/snowflake-arctic-embed-l-v2.0` at
  `ac6544c8a46e00af67e330e85a9028c66b8cfd9a`.
- GPU environment: `haystack-ai==2.31.0`,
  `sentence-transformers==5.1.2`, `transformers==4.57.6`, and
  `torch==2.6.0+cu124`. Haystack 3.x removes the imported embedding
  components, while SentenceTransformers 3.x is incompatible with Haystack
  2.31's ranker API.
- Fold seed: `2026`; folds are grouped by document and stratified by domain.
- Retrieval: BM25 top-50 + dense top-50, deduplicate/merge, reranker top-10.
  Snowflake Arctic Embed is used as the bi-encoder described by its model
  card: query embeddings use `prompt_name="query"`, document embeddings use no
  query prompt, and candidates are ordered by cosine similarity. It must not
  be loaded as a cross-encoder classifier because that creates an untrained
  classification head.
- QA budget ceiling: USD 5. RunPod ceiling: USD 3.

All commands below are run from the repository root with the project virtual
environment. Generated artifacts live under `research/week3/artifacts/` and are
ignored by Git.

## 1. Prepare (already completed locally)

```powershell
& .\.venv\Scripts\python.exe -X utf8 research\week3\phase_g_runner.py prepare `
  --final-dir research\week2\artifacts\final_table3 `
  --data-dir data\clair `
  --output-dir research\week3\artifacts\prepared
```

Expected: 33 documents and 10 systems. `raw__page` reports four explicitly
excluded empty chunks; these are the four accepted empty intermediate chunks
already documented in Week 2.

## 2. Generate the paid two-document QA smoke set

First ensure `OPENAI_API_KEY` is loaded from the repository-root `.env` or the
current environment. Never paste it into a command or artifact. The free
estimate has already been written to
`research/week3/artifacts/qa-smoke/qa_smoke_estimate.json`.

This call sends up to 10,000 tokens from each selected CLAIR document to
OpenAI. Obtain explicit approval for that data transfer before running it.

```powershell
& .\.venv\Scripts\python.exe -X utf8 research\week3\phase_g_runner.py generate-qa `
  --prepared-dir research\week3\artifacts\prepared `
  --data-dir data\clair `
  --output-dir research\week3\artifacts\qa-smoke `
  --smoke-docs 2 `
  --budget-usd 5 `
  --allow-paid-run
```

The runner caches each request by model, prompt, schema, and source hash. It
stops after the first error and does not retry automatically.

## 3. Review and freeze the smoke QA

Open `qa-smoke/qa_review.csv`. For each of six rows set `status` to:

- `accept`: keep the generated QA;
- `edit`: fill `edited_question`, `edited_answer`, and
  `edited_evidence_json`;
- `reject`: do not freeze; regenerate a replacement deliberately.

For an edit, evidence is a JSON array such as:

```json
[{"page": 3, "quote": "An exact quote copied from the source."}]
```

Freeze only after all six rows are accepted or validly edited:

```powershell
& .\.venv\Scripts\python.exe -X utf8 research\week3\phase_g_runner.py freeze-qa `
  --prepared-dir research\week3\artifacts\prepared `
  --data-dir data\clair `
  --qa-dir research\week3\artifacts\qa-smoke `
  --expected-documents 2
```

For an auditable machine-readable approval, pass
`--review-decisions <decisions.json>` instead of editing the CSV in place.

Exact quote, page, source offsets, QA IDs, and QA content hashes are validated.

## 4. Run the two-document GPU smoke

Use a fresh RunPod A5000 development Pod with PyTorch 2.6/CUDA 12.4,
FlashAttention 2, bf16 support, and a hard terminate-after alarm. Copy the
repository plus `prepared/` and `qa-smoke/` to `/workspace`. Do not copy `.env`
or API caches. On the Pod:

Create and verify the secret-free transfer bundle locally before starting a
billable Pod:

```powershell
& .\.venv\Scripts\python.exe -X utf8 `
  research\week3\prepare_runpod_smoke_bundle.py
Get-FileHash -Algorithm SHA256 `
  research\week3\artifacts\runpod_phase_g_smoke.zip
```

The script includes only package source, the Phase G runner, prepared system
files, and frozen smoke QA. It deliberately excludes `.env`, API caches, QA
candidates, and secrets. Upload both the ZIP and its `.sha256` sidecar, then
verify the checksum before extraction.

On the Pod:

```bash
cd /workspace/adaptive-chunking
source .venv/bin/activate
python -m pip install -r research/week3/requirements-gpu.txt
python -m pip freeze > /workspace/phase-g-environment.txt
python -X utf8 research/week3/phase_g_runner.py index \
  --prepared-dir research/week3/artifacts/prepared \
  --qa-dir research/week3/artifacts/qa-smoke \
  --output-dir research/week3/artifacts/index-smoke \
  --environment-file /workspace/phase-g-environment.txt \
  --device cuda:0 --batch-size 1

python -X utf8 research/week3/phase_g_runner.py retrieve \
  --prepared-dir research/week3/artifacts/prepared \
  --qa-dir research/week3/artifacts/qa-smoke \
  --index-dir research/week3/artifacts/index-smoke \
  --output-dir research/week3/artifacts/retrieval-smoke \
  --environment-file /workspace/phase-g-environment.txt \
  --device cuda:0 --batch-size 1

python -X utf8 research/week3/phase_g_runner.py evaluate-retrieval \
  --prepared-dir research/week3/artifacts/prepared \
  --qa-dir research/week3/artifacts/qa-smoke \
  --retrieval-dir research/week3/artifacts/retrieval-smoke \
  --output-dir research/week3/artifacts/evaluation-smoke
```

Download all manifests/results, verify their checksums, then terminate the Pod.
The full paid/API and GPU gates reject an invalid smoke evaluation manifest.

## 5. Generate, review, and freeze all 99 QA

The free worst-case estimate is stored at
`qa-full/qa_full_estimate.json`. It is USD 1.377244 at the recorded official
GPT-4.1 rates, below the USD 5 hard cap.

```powershell
& .\.venv\Scripts\python.exe -X utf8 research\week3\phase_g_runner.py generate-qa `
  --prepared-dir research\week3\artifacts\prepared `
  --data-dir data\clair `
  --output-dir research\week3\artifacts\qa-full `
  --budget-usd 5 `
  --request-delay-seconds 30 `
  --smoke-manifest research\week3\artifacts\qa-smoke\qa_smoke_manifest.json `
  --smoke-evaluation-manifest research\week3\artifacts\evaluation-smoke\retrieval_evaluation.json `
  --allow-paid-run
```

Review all 99 rows as above, then:

```powershell
& .\.venv\Scripts\python.exe -X utf8 research\week3\phase_g_runner.py freeze-qa `
  --prepared-dir research\week3\artifacts\prepared `
  --data-dir data\clair `
  --qa-dir research\week3\artifacts\qa-full `
  --expected-documents 33
```

## 6. Full GPU run

Repeat the three GPU commands using `qa-full`, `index-full`,
`retrieval-full`, and `evaluation-full`. The full `index` command additionally
requires:

```text
--smoke-evaluation-manifest research/week3/artifacts/evaluation-smoke/retrieval_evaluation.json
```

Run the index/retrieval workload on a small benchmark first, project total
runtime, and proceed only while the projected RunPod cost remains below USD 3.
Download and checksum the results before terminating the Pod.

The final evaluation emits:

- `per_query_metrics.jsonl`;
- `retrieval_summary.csv`;
- `retrieval_evaluation.json` with paper-protocol, all-fixed,
  cross-validated best-fixed, and per-query oracle results.

Metrics are Hit@1/3/5/10, Recall@1/3/5/10, MRR@10, and graded nDCG@10,
computed only from source-span overlap in the correct document.

## 7. Phase H statistical analysis and BI sensitivity

Phase H is local-only. It uses documents, rather than individual QA, as the
statistical unit and does not call an API or require a GPU:

```powershell
& .\.venv\Scripts\python.exe -X utf8 research\week3\phase_h_analysis.py `
  --evaluation-dir research\week3\artifacts\evaluation-full `
  --prepared-dir research\week3\artifacts\prepared `
  --metrics-path research\week2\artifacts\final_table3\results\chunking_metrics.parquet `
  --data-dir data\clair `
  --adaptive-selections research\week3\artifacts\prepared\adaptive_selections.json `
  --output-dir research\week3\artifacts\phase-h-analysis `
  --bootstrap-samples 10000 `
  --seed 2026
```

The analysis emits document-bootstrap confidence intervals, paired two-sided
Wilcoxon tests with Holm correction within each metric family, fold/domain
tables, and a Block Integrity sensitivity analysis. The page-neutral
sensitivity sets the BI weight to zero and equally weights the other four
intrinsic metrics. This is deliberately a conservative ablation: it does not
claim that a new independent structural gold standard exists.

`page_neutral_replay_*` is a counterfactual selector diagnostic assembled from
the fixed-index results. It is not an exact mixed-index retrieval rerun because
the distractor documents in each source result used one fixed chunker. Exact
downstream claims require building and retrieving one new BI-neutral mixed
index.

## 8. Phase I: paper Table 5 answer generation

The paper Table 5 comparison uses only Adaptive and the two raw baselines
(`langch_recurs_default`, `page`). It uses the already-frozen 99 QA and the
already-complete top-10 retrieval results, so indexing and retrieval are not
repeated. Prepare and audit the paid requests locally:

```powershell
& .\.venv\Scripts\python.exe -X utf8 research\week3\phase_i_table5.py prepare `
  --qa research\week3\artifacts\qa-full\qa_frozen.jsonl `
  --retrieval-dir research\week3\artifacts\retrieval-full `
  --evaluation research\week3\artifacts\evaluation-full\retrieval_evaluation.json `
  --output-dir research\week3\artifacts\phase-i-table5
```

The prepared manifest pins the frozen QA and retrieval hashes, the exact
upstream answer prompt, GPT-4.1 snapshot, price, token estimates, and a
1024-token output limit (raised from 512 after one smoke response was cut).
The limit is an explicit cost-control deviation from
the upstream generator, which does not set one. The cached generation stage
requires both `--allow-paid-run` and `--budget-usd`; it makes one request at a
time with no SDK retries and records a durable attempt before each call. A
pending attempt must be inspected before retrying. The two-document smoke
and the subsequent full 297-request run have completed under the user's USD 10
combined ceiling. The conservative generation-journal liability is USD
4.805448.

```powershell
& .\.venv\Scripts\python.exe -X utf8 research\week3\phase_i_table5.py generate `
  --output-dir research\week3\artifacts\phase-i-table5 `
  --smoke --budget-usd 0.50 --allow-paid-run --delay-seconds 25
```

After smoke answers are cached, `phase_i_table5.py export --smoke --output-dir
research\week3\artifacts\phase-i-table5` writes the original `rag_eval.py`
generation-result shape for the paper's DeepEval judge. The smoke export has
6 QA per system. Its 18 complete responses cost USD 0.346414 per API usage;
one cut 512-token response remains reserved at USD 0.025754, bringing the
conservative journal liability to USD 0.372168. The full export contains 99
unique QA per system and ten contexts per QA.

The paid judge is complete for all 297 system-query rows with the pinned
`gpt-4.1-2025-04-14` snapshot and DeepEval 3.5.9. The judge liability is USD
4.821514 and the conservative combined generation-plus-judge liability is USD
9.626962, with zero pending requests. One TPM rejection was recorded as
nonbillable and the run resumed from its durable cache at a slower pace.

The code-compatible Table 5 aggregation is:

| Metric | Adaptive | Raw LangChain recursive default | Raw page |
|---|---:|---:|---:|
| Retrieval Completeness | 98.99 ± 10.05 | 99.49 ± 5.03 | 100.00 ± 0.00 |
| Answer Correctness (answered only) | 94.32 ± 6.67 | 94.65 ± 6.42 | 94.25 ± 7.07 |
| Final Score (upstream flat mean) | 96.67 | 97.07 | 97.14 |
| Answered queries | 98/99 | 99/99 | 98/99 |
| Coverage-adjusted correctness | 93.37 | 94.65 | 93.30 |

These are **not a numerical reproduction of the paper's published Table 5**.
The original 99 QA are unavailable; this run uses the evidence-checked Phase G
QA and the corrected raw-baseline retrieval artifacts. The near-saturated
Retrieval Completeness scores and substantially higher answer coverage differ
materially from the paper's 67.7/58.1/59.1 completeness and 65/49/49 answered
counts. The result is therefore a controlled code/protocol rerun, not evidence
that the published values were independently recovered.

`phase_i_summarize.py` validates the 3×99 shape, identical query-ID sets,
metric errors, abstention handling, and score bounds. Five DeepEval scores were
`1.0000000000000002`; the raw evidence remains unchanged and only the derived
summary clips this machine-precision roundoff to 1.0. The validated outputs are
`judge-full/summary/table5_summary.json` and `table5_summary.csv`. The full
repository suite passes with `87 passed, 1 skipped`; the separate DeepEval
offline check also passes with zero provider calls.

The offline smoke review is in `SMOKE_REVIEW.md`. To validate the original
judge inputs and obtain a **non-binding** token/cost illustration without
loading DeepEval or sending requests:

```powershell
& .\.venv\Scripts\python.exe -X utf8 research\week3\phase_i_judge_preflight.py `
  --input-dir research\week3\artifacts\phase-i-table5\judge-input-smoke `
  --generation-plan research\week3\artifacts\phase-i-table5\generation_plan.json `
  --original-judge src\adaptive_chunking\paper\rag_eval.py `
  --output research\week3\artifacts\phase-i-table5\judge_preflight_smoke.json
```

The original `rag_eval.py` leaves judge output length and DeepEval's internal
G-Eval prompt unbounded. The preflight figure is not a hard budget. An isolated
Windows/Python 3.12 environment with the direct dependencies in
`requirements-judge.txt` passed an offline compatibility test for DeepEval
3.5.9. To repeat it without credentials or provider calls:

```powershell
& research\week3\artifacts\judge-venv\Scripts\python.exe -X utf8 `
  research\week3\verify_phase_i_judge_offline.py
```

The script disables DeepEval telemetry/dotenv loading before import, loads the
original judge source unchanged, and scores synthetic cases with a fake model.
`phase_i_judge.py` uses the same metric definitions with pinned DeepEval 3.5.9,
but replaces provider execution with serial, no-SDK-retry calls, a 512-token
output cap, durable per-call cache/journal, hash-checked resume, and a combined
generation-plus-judge budget guard. Both the offline compatibility path and
the authorized paid provider path have completed.
