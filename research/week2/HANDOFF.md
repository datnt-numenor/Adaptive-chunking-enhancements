# Week 2 handoff — Adaptive Chunking reproduction

Updated: 2026-09-18 (Asia/Saigon)

## Read first

Before continuing, read repository-root `AGENTS.md` and `LLM.md`. Preserve the
upstream implementation under `src/`; reproduction helpers belong under
`research/week2/`.

For research findings, always separate:

1. **Paper says** — supported by a paper section, page, figure, or table.
2. **Code actually does** — supported by a source path, symbol, command, log,
   test, or artifact.
3. **Analyst inference** — interpretation or hypothesis based on the evidence.

If evidence is absent, write: **Không tìm thấy bằng chứng trong paper/code.**

## Reproduction identity and constraints

- Upstream repository: `ekimetrics/adaptive-chunking`.
- Pinned commit: `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d`.
- Dataset: CLAIR, 33 parsed documents with precomputed mentions.
- API budget: USD 60–80 total; rented-GPU budget for exact semantic chunking:
  at most USD 3.
- OpenAI is used only for the original `llm_regex` chunker in this stage.
- Do not run Tables 4–5 as part of the current reproduction task.
- Do not rerun a paid shard unless its checkpoint/cache state has been audited.

## Completed work

### Kaggle non-semantic runs

- Kaggle notebook: `datnguyenksnb/reproduction-adaptive-chunking`.
- Safety/cache notebook published as version 20.
- Shards `00`, `01`, and `02` completed; each contains 11 documents and these
  seven methods:
  - `page`
  - `sentence`
  - `langch_recurs_default`
  - `langch_recurs_1100`
  - `our_recurs_1100`
  - `our_recurs_600`
  - `llm_regex`
- The three ZIP exports were downloaded to `C:\Users\Admin\Downloads` and
  passed CRC and structural validation.
- Keep those ZIP files as recovery copies. Do not publish them or their API
  caches as a public Kaggle/GitHub artifact.

The current runner executes documents sequentially, saves a per-document
checkpoint, and uses `cached_openai_replicate.py` to atomically cache a
successful Chat Completions response. Identical requests reuse the cache.
Automatic SDK retries are disabled so an uncertain failed request is inspected
before an application-level retry.

### Local extraction and merge

The ZIPs have already been extracted under:

```text
research/week2/artifacts/nonsemantic_shards/
```

The three shards have already been merged under:

```text
research/week2/artifacts/combined_chunks/
```

Do not extract or merge them again unless recovering from corruption.

Validated combined output:

| Stage | Rows | Documents | Methods | Intermediate empty page chunks |
|---|---:|---:|---:|---:|
| `raw` | 19,548 | 33 | 7 | 4 |
| `no_oversizing` | 20,333 | 33 | 7 | 4 |
| `small_merged` | 17,582 | 33 | 7 | 0 |

Validation report:

```text
research/week2/artifacts/combined_chunks/validation_report.json
```

Merge manifest:

```text
research/week2/artifacts/combined_chunks/chunks_merge_manifest.json
```

The four empty chunks are `page` chunks from one source document at the
intermediate `raw` and `no_oversizing` stages. They are absent from
`small_merged`; this is accepted by the stage-aware validator. There are no
validation warnings in the combined result.

### Relevant checks and helper changes

- `tests/test_reproduction_runner.py`: 20 targeted tests passed locally.
- `research/week2/audit_chunk_zip.py` now ignores per-document checkpoint
  parquet files and validates only the merged result inside each export.
- `research/week2/cached_openai_replicate.py` provides request-hash response
  caching without editing the upstream `src/` implementation.
- `research/week2/reproduction_runner.py` requires
  `--allow-fresh-paid-run` when neither a valid checkpoint nor cache exists.
- Generated Kaggle notebook source is under
  `research/week2/artifacts/kaggle_recovery_corrected/`.
- The semantic runner now records all 33 source hashes before execution and
  refuses `--resume` if the source commit, model/configuration, or dataset
  identity changed. It also requires an immutable 40-character Hugging Face
  model revision. This prevents exact and fallback rows or model snapshots from
  being mixed.
- A validated, secret-free RunPod upload bundle is prepared at
  `research/week2/artifacts/runpod_exact_semantic_input.zip`, with its SHA256
  sidecar beside it. Rebuild it with
  `research/week2/prepare_runpod_semantic_bundle.py` after changing the runner
  or RunPod guide.

### Exact semantic RunPod run

- The exact semantic run completed for all 33 documents on one NVIDIA RTX
  A5000 using `flash-attn==2.7.4.post1`, bfloat16, batch size 16, and the pinned
  upstream commit.
- Model: `Qwen/Qwen3-Embedding-0.6B`; immutable revision:
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- The three-document benchmark completed in 62.66 seconds and projected 0.0505
  hours for all 33 documents. The completed manifest records 235.42 seconds of
  document processing.
- `pyarrow==25.0.1` was added because the upstream pipeline writes Parquet but
  its default install did not provide a Parquet engine. No source behavior or
  semantic configuration was changed.
- The downloaded archive SHA256 is
  `2113b61e8f92443de6e36557f124730a5660a502679706e09b376afaefb318b1`.
- The local artifact is under
  `research/week2/artifacts/semantic_output/`. Validation passed without
  warnings: raw 1,705 rows, no-oversizing 2,342 rows, and small-merged 1,891
  rows; every stage has 33 documents and only the `semantic` method.
- The RunPod pod was terminated after the local archive and Parquet framing
  were verified. The final pod list was empty. Its 2,538-second uptime implies
  about USD 0.19 at USD 0.27/hour; the billing API had not settled a record at
  the final check.

### Eight-method merge and metric shards

- `combined_chunks/` and `semantic_output/` were merged into
  `research/week2/artifacts/full_chunks/` without rerunning any chunker.
- The full profile validator passed without warnings for 33 documents and all
  eight methods at every stage:
  - raw: 21,253 rows;
  - no-oversizing: 22,675 rows;
  - small-merged: 19,473 rows.
- The four accepted empty `page` chunks remain only in raw and no-oversizing;
  small-merged has none.
- Six deterministic metric shards were created under
  `research/week2/artifacts/metric_shards/`. Direct Parquet checks confirmed
  that every document occurs in exactly one shard, every shard contains all
  eight methods, and the per-stage row totals reproduce `full_chunks/` exactly.
- Shard 00 has six documents and 20,022 total rows because it contains the
  indivisible 391,660-token source document. Shards 01–05 contain five or six
  documents and 8,466–8,906 total rows each.

## Phase E/F completion

- All 12 Kaggle metric outputs are downloaded and validated: processed shards
  00–05 and raw shards 00–05. Do **not** rerun them.
- Processed shard 05 is the already validated pilot output. It has 400 rows for
  five documents, eight methods, and ten metrics; its eight null
  `references_completeness` scores are expected.
- Raw shard 04 version 1 failed on a T4 because a 10,113-token semantic chunk
  exhausted VRAM at batch size 4. Version 2 completed at batch size 1. Raw
  shards 00–03 were also run at batch size 1 after inspecting their longest
  semantic chunks. Raw shard 05 had already completed at batch size 4.
- The final merge is under `research/week2/artifacts/final_table3/`. Full-profile
  validation passed for 33 documents: processed metrics have 2,640 rows and
  raw metrics have 1,650 rows. The merged chunk counts remain 21,253 raw,
  22,675 no-oversizing, and 19,473 small-merged.
- Tables 1 and 2, Figure 1, and the Table 3 reproduction completed locally.
  The successful console record is
  `research/week2/artifacts/final_table3/analysis_table3.log`, ending with
  `All methods computed locally.` Table 3 matches the paper closely; the
  largest displayed delta is -1.30 percentage points for raw sentence
  chunking.
- The evidence-grounded human-readable report is
  `research/week2/RESULTS.md`; it separates paper claims, code behavior, and
  analyst inference and records the remaining protocol limitations.
- `src/adaptive_chunking/paper/analysis.py` now assigns correlation diagonals
  through pandas rather than mutating the read-only `DataFrame.values` view,
  for compatibility with the installed pandas/NumPy versions.
- Final verification: `63 passed, 1 skipped`.

## Current boundary

Week 2 reproduction through Phase F is complete. Do **not** rerun RunPod,
Kaggle chunkers, Kaggle metric shards, merges, or Table 3 unless recovering
from verified corruption. Any next work should begin by reviewing the final
artifacts and deciding how the reproduced results will be reported.

## Prompt for the next chat

```text
Read AGENTS.md, LLM.md, and research/week2/HANDOFF.md completely. Week 2 Phase
F is complete and validated. Do not rerun RunPod, Kaggle chunkers, metric
shards, or merges. Review research/week2/artifacts/final_table3 and continue
only with reporting or explicitly requested follow-up analysis.
```
