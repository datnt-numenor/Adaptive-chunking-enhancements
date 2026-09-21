# Week 2 reproduction checklist

The checklist separates an exact paper reproduction from the fair benchmark that the group will use for later contributions.

## A. Freeze provenance before running

- [x] Clone official upstream repository.
- [x] Record commit `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d`.
- [x] Download arXiv `2603.25333v1` and record its SHA-256.
- [x] Inventory bundled parsed data and mention artifacts.
- [x] Compute SHA-256 per parsed document in `artifacts/dataset_bi_audit.json`.
- [ ] Decide whether to reproduce both the paper-date commit `21fec93` and current fixed commit `ea87ce8`.
- [ ] Record OS, CPU, RAM, verified GPU name/VRAM, driver, CUDA runtime, and free disk outside the managed shell.
- [ ] Create a full lock for every Python dependency; do not rely only on lower bounds in `pyproject.toml`.
- [ ] Record exact Hugging Face revisions for Jina v3, Qwen3-Embedding-0.6B/4B, and Snowflake reranker.
- [ ] Store a read-only experiment manifest containing commit, data hashes, model revisions, dependency lock hash, seed, and command.

## B. Validate the repository cheaply

- [x] Create Python 3.12 `.venv` audit profile.
- [x] Run tests: 54 passed, 1 skipped.
- [x] Verify CLI help with UTF-8 mode.
- [x] Run core source pipeline on three small documents with page, recursive-600, and recursive-1100.
- [x] Confirm Parquet writing, chunk ordering, size metrics, BI computation, and page metadata.
- [x] Audit raw-page BI on all 33 documents.
- [x] Run Jina metrics smoke: 3 documents, 3 methods, 10 metrics, 90 rows.
- [ ] Install Tier B/full paper dependencies only after GPU/disk verification.
- [ ] Download and record the Stanza English resource version.
- [ ] Run a one-document full CLI probe with `--skip-llm-regex --skip-semantic`.
- [ ] Run a three-document semantic probe on Kaggle/RTX and record peak VRAM/time.
- [ ] Run one LLM-regex request only after exposing/pinning the paper's GPT-5 model choice.

## C. Reproduce intrinsic results

- [ ] Preserve raw chunks separately from post-processed chunks.
- [ ] Run all eight raw chunkers with fixed configs.
- [ ] Apply oversized splitting only to the intended paper methods.
- [ ] Apply tiny-chunk merging only to the intended paper methods.
- [ ] Compute raw metrics for dagger methods and processed metrics for star methods.
- [ ] Restrict Adaptive candidates to the four star methods before selection.
- [ ] Recreate Tables 1--4 and Figure 1 from computed results, not embedded constants.
- [ ] Implement the reported paired Wilcoxon tests; the current repository does not contain them.
- [ ] Save per-document metric rows so Table 3 and selection frequencies are auditable.
- [ ] Compare current commit against paper-date commit for RC and method selection changes.

## D. Reproduce RAG results without contaminating the fair benchmark

- [ ] Fix one QA file before comparing systems; never regenerate it inside each system run.
- [ ] Add evidence text/span/page and source document ID to every QA item.
- [ ] Hash and mark the frozen QA file read-only.
- [ ] Use raw page and raw LangChain-default chunks for the paper Table 5 baselines.
- [ ] Use only the four intended methods for the paper Adaptive index.
- [ ] Verify dense similarity and embedding normalization explicitly.
- [ ] Cache document embeddings, retrieval outputs, prompts, raw API responses, and model snapshots.
- [ ] Add an API call/token/cost ceiling and resume keys based on stable `query_id`, not question text.
- [ ] Reproduce paper metrics: Retrieval Completeness, selective Answer Correctness, and answered count.
- [ ] Add fair metrics: evidence Recall@k, MRR, nDCG, coverage, overall utility, and latency/cost.
- [ ] Run paired/clustered inference at document level with 95% confidence intervals.

## E. Leakage and selector protocol for the group's contribution

- [ ] Assign immutable `document_id` before QA generation or feature extraction.
- [ ] Freeze document-level train/development/test split.
- [ ] Fit weights/scalers/calibrators only on training documents.
- [ ] Use GroupKFold grouped by `document_id` inside training/development.
- [ ] Choose metrics, router thresholds, and context variants without reading final-test scores.
- [ ] Run final test once and retain all configurations, including negative results.
- [ ] Report selector accuracy, regret to oracle, best fixed, equal weight, random, and oracle.

## F. Decision gate for Query-Aware Multi-Index Selection

- [ ] QA set frozen and hashed.
- [ ] Four candidate indexes built from identical source documents.
- [ ] Fixed and Adaptive baselines complete end to end.
- [ ] Document/query mapping tests pass.
- [ ] At least 35% of API budget remains.

If any box fails, redirect week 7 to robustness and error analysis as agreed; do not weaken the final-test protocol.

## G. Low-cost Tables 1--3 execution status

- [x] Implement the seven-command reproduction helper outside `src/`.
- [x] Add deterministic, count-constrained sharding and duplicate-key checks.
- [x] Create three local Kaggle data shards with 11 documents each.
- [x] Copy precomputed mention artifacts and record document hashes/commit.
- [x] Package Kaggle-safe ASCII aliases and verify lossless restoration of all 103 files.
- [x] Add tests for shard completeness, merge preservation, semantic config, and per-document metric independence.
- [ ] Run seven non-semantic methods for shards 00--02 on Kaggle.
- [ ] Merge and validate 33 documents x 7 methods locally.
- [ ] Benchmark exact semantic chunking on the rented A5000.
- [ ] Complete exact semantic within the USD 3 gate, or run the all-document SDPA fallback.
- [ ] Create and run six processed plus six raw metric shards on Kaggle.
- [ ] Merge 2,640 processed and 1,650 raw metric rows.
- [ ] Generate and archive Tables 1--3 console output and deltas.
