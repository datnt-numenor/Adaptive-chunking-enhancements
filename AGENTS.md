# AGENTS.md

## Scope

These instructions apply to the entire repository. They guide AI coding agents
working on the group research project based on the paper "Adaptive Chunking:
Optimizing Chunking-Method Selection for RAG".

Read `LLM.md` for the upstream repository architecture and commands. Treat it as
technical context, not as evidence that the paper and implementation are
consistent.

## Research objective

The group has three goals:

1. Understand the paper and its mechanism.
2. Reproduce and audit the published results.
3. Propose focused improvements to the original method.

Do not expand the project into building an entirely new RAG system. Prefer
small, testable changes to individual components.

The agreed research process is:

> understand the mechanism -> build trustworthy comparisons -> identify
> weaknesses through experiments -> improve one component at a time -> verify
> on data not used during design

## Evidence discipline

For every audit finding or research conclusion, clearly separate:

1. **Paper says**: claims explicitly supported by the paper, including the
   relevant section, table, figure, or page when available.
2. **Code actually does**: behavior supported by a source path, symbol, command,
   test, log, or generated artifact.
3. **Analyst inference**: interpretation, hypothesis, risk, or proposed
   explanation derived from the evidence.

Never present an inference as a claim from the paper or code. If evidence is
missing, state exactly:

> Không tìm thấy bằng chứng trong paper/code.

When the paper, README, and source disagree, preserve all versions in the
report and identify the discrepancy. Do not silently choose one as correct.

## Reproducibility requirements

- Pin and record the repository commit, Python version, dependency versions,
  dataset version, model revision, device, operating system, and command line.
- Record every configuration, random seed, prompt, API model name/version, and
  evaluation definition.
- Fix the QA set before optimizing a selector or representation.
- Split training, validation, and test data by document. Never allow chunks or
  questions from the same document to cross these partitions.
- Use `GroupKFold` with document ID as the group for learned selectors.
- Cache embeddings and API responses. Never silently replace an existing cache
  created with a different configuration.
- Give experiment outputs unique, descriptive paths and retain their metadata.
- Compare results only when their datasets, splits, retrieval settings, and
  evaluation definitions are compatible.
- Use document-level confidence intervals or statistical tests for final
  comparisons. Report both effect sizes and uncertainty.
- Record failed and interrupted runs; do not report partial output as a
  completed experiment.

## Mandatory baselines and audits

Before claiming an improvement, implement and evaluate:

- every required fixed-chunker baseline;
- the original equal-weight adaptive selector;
- the best fixed chunker selected without test-set information;
- an oracle selector used only as an upper bound;
- document-grouped train/validation/test partitions;
- a Block Integrity audit, including an original and a page-neutral variant;
- paper--README--code consistency checks;
- retrieval metrics and downstream answer-quality evaluation.

The key audit questions are:

- Does the selector improve downstream RAG, or only proxy metrics?
- Is there leakage across documents or evaluation stages?
- Does Block Integrity encode page or dataset-structure bias?
- Does improved retrieval produce better answers from the LLM?

## Planned contributions

### 1. Reliability-Aware Downstream-Calibrated Selector

This is the main contribution. Use downstream retrieval or QA outcomes to
calibrate a lightweight selector from features such as Relative Cohesion,
Inter-Chunk Cohesion, Document Coverage, Block Integrity, and Size Consistency.

Compare equal-weight, learned-weight, best-fixed, and oracle selection. Report
selector accuracy, regret relative to the oracle, Recall@k, MRR, nDCG, and
answer quality. Include the Block Integrity page-neutral ablation.

### 2. Context-Enriched Chunk Representation

This contribution is committed and should include at least:

- V0: embed the original chunk and give the original chunk to the LLM;
- V1: embed `title + chunk`, but give only the original chunk to the LLM;
- V2: use `title + chunk` for both embedding and LLM context.

Track retrieval quality, answer quality, context length, latency, and token/API
cost. Keep the chunk boundaries unchanged so that representation effects are
not confused with chunking effects.

### 3. Query-Aware Multi-Index Selection

This is a stretch goal. Implement it only if, by the end of week 4:

- the QA set is frozen;
- four indexes are stable;
- baseline evaluation is complete; and
- at least 35% of the API budget remains.

Otherwise, spend this time on error analysis, robustness tests, and completing
the first two contributions.

## Evaluation and cost controls

- Use local or Kaggle compute for preprocessing, embeddings, retrieval,
  reranking, and lightweight models when practical.
- Do not attempt to fine-tune a large LLM on the laptop RTX 3060.
- The total API budget is USD 60--80. Do not start a potentially expensive run
  without estimating its number of documents, configurations, requests, and
  tokens.
- Run cheap retrieval evaluation first. Send only approximately 5--6 strongest
  configurations to LLM-based evaluation.
- Start new pipelines with a small smoke-test subset before a full run.
- Cache all paid API responses and make retries resumable.
- Never enable automatic billing or auto-recharge on behalf of the user.

## Secrets and local configuration

- Load secrets from the repository-root `.env` file or the platform's secret
  manager.
- Never print, log, commit, or copy secret values into reports, notebooks,
  prompts, screenshots, test fixtures, or generated artifacts.
- Do not place secrets inside `.venv`, `AGENTS.md`, `LLM.md`, source files, or
  command history.
- Relevant variables may include `JINA_API_KEY`, `OPENAI_API_KEY`,
  `GROQ_API_KEY`, `ADI_ENDPOINT`, and `ADI_KEY`; only require a variable when
  the selected pipeline actually uses its service.

## Working practices for AI agents

- Inspect and diagnose before changing upstream source code.
- Preserve user changes and unrelated work in a dirty worktree.
- Keep research additions isolated where practical, for example under
  `research/`, rather than rewriting the upstream implementation prematurely.
- Make the smallest change that tests the current hypothesis.
- Do not update libraries or model revisions silently; explain the expected
  reproducibility impact first.
- Do not fabricate missing files, results, citations, commands, metrics, or
  paper claims.
- Do not claim success from a command that was not run or an artifact that was
  not inspected.
- For a requested implementation, run targeted tests or smoke checks and report
  the exact validation performed.
- If GPU access is unavailable, prefer an API, Kaggle GPU, or a CPU-compatible
  small model, and disclose that this changes runtime or comparability.

## Research records

For each important experiment, retain:

- objective and hypothesis;
- dataset and document split identifiers;
- exact command and configuration;
- code commit and environment snapshot;
- input/output artifact paths;
- start/end time, device, latency, and API/token cost;
- retrieval and answer-quality metrics;
- uncertainty estimates;
- errors, exclusions, and deviations from the intended protocol;
- conclusion labeled as evidence or inference.

Prefer machine-readable JSON, JSONL, CSV, or Parquet for raw results, with a
short Markdown summary for humans. Tables and slides must be generated from or
checked against the saved raw results.

## Team ownership

- Member 1: dataset, preprocessing, chunking methods, and indexes.
- Member 2: selector, contextual chunk representation, and query-aware router.
- Member 3: evaluation, statistical analysis, logging, result tables, and
  documentation.

All members review failures, validate important results, and contribute to the
discussion. File ownership does not remove the need for cross-review.
