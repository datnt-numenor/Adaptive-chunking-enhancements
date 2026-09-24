# Baseline conclusion after the original-paper experiments

## Scope and completion state

The reproducibility baseline is now closed for the experiments that can be
reconstructed from the public repository and available data:

- Week 2 reproduced the eight-method chunking and intrinsic-metric pipeline for
  all 33 CLAIR documents. The final Table 3 merge contains 2,640 processed and
  1,650 raw metric rows and passed the full-profile validator.
- Phase G evaluated 10 retrieval systems on one frozen set of 99 evidence-backed
  QA. It produced 990 system-query rows and 9,900 ranked top-10 results.
- Phase H completed document-level bootstrap intervals, paired Wilcoxon tests,
  Holm correction, and the Block Integrity page-neutral sensitivity analysis.
- Phase I generated and judged answers for Adaptive plus the two raw Table 5
  baselines. All 297 system-query rows are complete and validated.

The raw Phase I artifacts remain local and ignored because they are about
12.7 MB and contain repeated retrieved contexts. Their byte lengths and SHA256
checksums are committed in `PHASE_I_RESULT_MANIFEST.json`.

## Paper says

- Published Table 3 reports Adaptive as the best overall intrinsic-metric
  method and reports paired Wilcoxon significance.
- Published Table 5 reports Retrieval Completeness of 67.7/58.1/59.1,
  selective Answer Correctness of 78.0/70.1/73.3, and answered-query counts of
  65/49/49 for Adaptive, LangChain recursive, and page splitting.
- The paper reports a Wilcoxon result for Table 5 Retrieval Completeness, but
  does not provide the original 99 QA or enough detail to reconstruct the
  exact pairing protocol.

## Code and artifacts show

- The reconstructed Table 3 is close to the publication. The largest displayed
  difference recorded in the Week 2 audit is -1.30 percentage points for raw
  sentence chunking.
- The corrected retrieval benchmark does not show a statistically reliable
  Adaptive advantage. For document-level nDCG@10, Adaptive minus raw page is
  +0.0251 with a 95% bootstrap interval of [-0.0197, +0.0725] and Holm-adjusted
  p=1. Adaptive minus raw LangChain recursive default is -0.0277 with interval
  [-0.0786, +0.0238] and Holm-adjusted p=1.
- Block Integrity is strongly aligned with page structure: 1,730 of 1,732 page
  boundaries coincide with parser split points within five characters. Removing
  BI changes 9 of 33 selector decisions and reduces page selections from 15 to
  9. The counterfactual replay improves nDCG@10 by +0.0151, but its interval
  [-0.0109, +0.0460] includes zero and it is not an exact mixed-index rerun.
- The completed Phase I controlled rerun produced:

| Metric | Adaptive | Raw LangChain recursive | Raw page |
|---|---:|---:|---:|
| Retrieval Completeness | 98.99 | 99.49 | 100.00 |
| Correctness, answered only | 94.32 | 94.65 | 94.25 |
| Answered queries | 98/99 | 99/99 | 98/99 |
| Upstream flat-mean final score | 96.67 | 97.07 | 97.14 |

- All three systems use the same 99 QA IDs. No judge metric has an error and
  no request remains pending. Conservative answer-generation plus judging
  liability is USD 9.626962 under the USD 10 ceiling.

## Analyst inference

- Table 3 is reproducible closely enough to use as the intrinsic-quality
  baseline, but that does not establish downstream superiority.
- Phase I is a completed code/protocol rerun, not a numerical reproduction of
  published Table 5. The original QA are unavailable, while this run uses the
  evidence-reviewed Phase G QA and corrected raw retrieval baselines.
- Retrieval Completeness is nearly saturated on the new QA and therefore has
  little power to distinguish these systems. The tiny final-score ordering
  must not be described as a meaningful win without an effect-size and
  uncertainty analysis designed for these answer-quality scores.
- The strongest evidence-backed weakness in the original selector is the
  dependence of Block Integrity on page/parser boundaries. The next focused
  experiment should be one exact BI-neutral mixed-index retrieval run. That
  test can determine whether the positive counterfactual replay survives when
  the candidate chunks compete in one real index.

## Frozen baseline boundary

Do not rerun the completed Kaggle shards, semantic chunking, QA generation,
Phase G indexes, or Phase I LLM judging unless an artifact fails its committed
checksum. New work should use this state as the baseline and write to new,
descriptive output directories.
