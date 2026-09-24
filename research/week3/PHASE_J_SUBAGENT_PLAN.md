# Phase J subagent plan — exact BI-neutral mixed-index experiment

## Objective and completion criteria

Build and evaluate exactly one new `adaptive_bi_neutral` mixed index to test
whether excluding Block Integrity improves downstream retrieval. Reuse the
frozen 99-QA set and all locked Phase G settings. Do not rerun chunking,
intrinsic metrics, QA generation, existing indexes, or Phase I LLM judging.

Phase J is complete only when:

- the BI-neutral selections cover all 33 documents and match the locked Phase H
  selections exactly;
- a two-document GPU smoke passes before the full run;
- the full index and retrieval contain all 99 QA with top-10 results;
- document-level statistical comparisons and a human-readable report exist;
- downloaded artifacts pass checksums before the Pod is terminated;
- the full local test suite passes and the root agent performs the only commit
  and push.

## Locked experimental configuration

Subagents must not change these values. Any proposed change must be reported to
the root agent instead of applied.

| Setting | Locked value |
|---|---|
| Upstream source commit | `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d` |
| Adaptive candidates | `our_recurs_1100`, `our_recurs_600`, `page`, `llm_regex` |
| BI-neutral metrics | `size_compliance`, `intrachunk_cohesion`, `document_contextual_coherence`, `references_completeness` |
| Metric weights | `0.25` each; `block_integrity=0` |
| QA | Existing frozen 99 QA, 3 per document |
| Fold seed | `2026`, grouped by document |
| Embedding model | `Qwen/Qwen3-Embedding-4B` |
| Embedding revision | `5cf2132abc99cad020ac570b19d031efec650f2b` |
| Retrieval/reranker model | `Snowflake/snowflake-arctic-embed-l-v2.0` |
| Retrieval/reranker revision | `ac6544c8a46e00af67e330e85a9028c66b8cfd9a` |
| GPU execution | RTX A5000, CUDA, bfloat16, FlashAttention 2 |
| Retrieval | Existing Phase G BM25/dense/merge/top-10 protocol |
| Statistical bootstrap | 10,000 samples, seed `2026`, document-level units |

No subagent may call GPT-4.1 or another paid API, provision a Pod, change model
or dependency revisions, commit, push, or modify ignored baseline artifacts.

## Agent model policy and execution waves

Use `fork_turns: "none"` with the prompts below so the long parent conversation
is not copied into every agent. Each agent must read the listed repository files
itself. Reuse an existing agent with `followup_task` instead of spawning a new
agent for corrections.

Run at most two subagents concurrently:

1. Wave 1: `bi_neutral_selector` and `bi_neutral_runpod`.
2. Wave 2: `bi_neutral_analysis` while the root agent integrates Wave 1.
3. Wave 3: one short `gpt-6-astra` review only after all local tests pass.
4. The root agent alone performs GPU provisioning, result download, termination,
   final integration, Git commit, and push.

| Agent | Model | Reasoning | Purpose |
|---|---|---|---|
| `bi_neutral_selector` | `gpt-6-sol` | `high` | Selector and exact mixed prepared system |
| `bi_neutral_runpod` | `gpt-6-sol` | `medium` | Secret-free bundle and remote scripts |
| `bi_neutral_analysis` | `gpt-6-sol` | `high` | Statistics and report generation |
| Final reviewer | `gpt-6-astra` | `high` | One bounded correctness/reproducibility review |

## File ownership and artifact contract

Agents share one worktree. They must not edit files outside their ownership.

### `bi_neutral_selector`

Owns only:

- `research/week3/phase_g_runner.py`;
- `tests/test_phase_g_runner.py`.

Responsibilities:

- add system ID `adaptive_bi_neutral` without changing existing systems;
- reproduce `artifacts/phase-h-analysis/page_neutral_selections.json` exactly;
- materialize a JSONL mixed system covering all 33 documents;
- record selection definition, source hashes, chunk count, document count, and
  system hash in the prepare manifest;
- allow `--systems adaptive_bi_neutral` for index, retrieve, and evaluation;
- add tests for exact selection equality, candidate restriction, coverage,
  unique chunk identity, manifest hashes, and unchanged existing behavior.

Expected prepared artifact contract:

```text
artifacts/phase-j-bi-neutral/prepared/
  prepare_manifest.json
  adaptive_bi_neutral_selections.json
  systems/adaptive_bi_neutral.jsonl
```

### `bi_neutral_runpod`

Owns only new files whose names start with:

```text
research/week3/prepare_phase_j_
research/week3/run_phase_j_
tests/test_prepare_phase_j_
```

Responsibilities:

- create a minimal secret-free bundle containing the Phase G code, frozen QA,
  smoke gate, and BI-neutral prepared system;
- exclude `.env`, credentials, API caches, previous indexes, and unrelated
  artifacts;
- emit archive SHA256 and a per-file manifest;
- provide separate two-document smoke and full-run remote commands;
- archive index, retrieval, evaluation, environment, and timing outputs;
- validate locally without provisioning any remote resource.

Expected remote result contract:

```text
phase-j-results/
  index/index_manifest.json
  retrieval/retrieval_manifest.json
  retrieval/adaptive_bi_neutral.jsonl
  evaluation/retrieval_evaluation.json
  environment.txt
  timing.json
```

### `bi_neutral_analysis`

Owns only:

- `research/week3/phase_j_analysis.py`;
- `tests/test_phase_j_analysis.py`.

Responsibilities:

- accept the existing Phase G evaluation/folds plus the new BI-neutral result;
- compare against original Adaptive, raw page, raw LangChain recursive default,
  best-fixed cross-validation, and oracle;
- aggregate the three QA per document before inferential statistics;
- calculate Hit@1/3/5/10, Recall@1/3/5/10, MRR@10, and nDCG@10;
- run 10,000-sample document bootstrap confidence intervals, paired two-sided
  Wilcoxon tests, and Holm correction within each metric family;
- produce CSV/JSON outputs and `SUMMARY.md` separating paper claims, artifact
  evidence, and analyst inference;
- initially validate against synthetic fixtures with known expected results.

Expected analysis contract:

```text
artifacts/phase-j-bi-neutral/analysis/
  manifest.json
  document_metrics.csv
  paired_comparisons.csv
  system_summary.csv
  SUMMARY.md
```

## Spawn prompts

### Selector prompt

```text
Read AGENTS.md, LLM.md, research/week3/HANDOFF.md and
research/week3/PHASE_J_SUBAGENT_PLAN.md completely.

You are bi_neutral_selector. Implement only the assignment and file ownership
defined for bi_neutral_selector. Preserve all locked experimental settings.
Run the relevant local tests and report changed files, commands, results, and
remaining risks. Do not use GPU or paid APIs, provision infrastructure, commit,
push, or edit another agent's files.
```

### RunPod prompt

```text
Read AGENTS.md, LLM.md, research/week3/HANDOFF.md and
research/week3/PHASE_J_SUBAGENT_PLAN.md completely.

You are bi_neutral_runpod. Implement only the bundle and remote-run assignment
and obey its file ownership. Preserve all locked settings and exclude every
secret or prior large artifact. Validate locally only. Do not edit
phase_g_runner.py, provision a Pod, run paid services, commit, or push.
```

### Analysis prompt

```text
Read AGENTS.md, LLM.md, research/week3/HANDOFF.md and
research/week3/PHASE_J_SUBAGENT_PLAN.md completely.

You are bi_neutral_analysis. Implement only the local statistical-analysis
assignment and its tests. Use document-level units and synthetic fixtures until
the exact GPU result exists. Do not edit Phase G or RunPod files, call APIs,
use GPU resources, commit, or push.
```

### Final reviewer prompt

```text
Read AGENTS.md, LLM.md, research/week3/HANDOFF.md,
research/week3/PHASE_J_SUBAGENT_PLAN.md, and the complete staged diff.

Perform one read-only correctness and reproducibility review. Check file
ownership, locked model revisions, QA/fold identity, data leakage, manifest and
checksum coverage, statistical units, cost guards, secret exclusion, and tests.
Report only evidence-backed findings with file and line references. Do not edit,
run paid services, provision infrastructure, commit, or push.
```

## Root-agent integration gates

The root agent must enforce these gates in order:

1. Review each agent's changed-file list and reject ownership violations.
2. Reproduce BI-neutral selections locally and compare all 33 entries with the
   locked Phase H artifact.
3. Run targeted tests, then the full repository test suite.
4. Build and inspect the secret-free bundle and verify every checksum.
5. Run the two-document A5000 smoke and validate its manifest.
6. Project full runtime and proceed only while estimated infrastructure cost is
   below USD 3; no OpenAI API call is authorized.
7. Run the full single-system index/retrieval, download the archive, verify its
   checksum and contents, then terminate the Pod.
8. Run Phase J analysis locally and complete the bounded Astra review.
9. Update `HANDOFF.md`, rerun tests and secret scan, then create one root-owned
   commit and push only after local/remote SHA verification.

If any gate fails, retain resumable artifacts, record the failure, and stop
before the next paid or destructive action. Do not silently fall back to a
different GPU, model, revision, QA set, retrieval configuration, or statistic.
