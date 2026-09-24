# Phase I Table 5 smoke review (offline)

Date: 2026-09-23. Scope: 2 documents, 6 frozen QA, 3 paper-protocol systems,
18 generated answers. This is a human surface review, **not** the paper's
LLM-judge score or a full source-level fact check. No API request was made for
this review.

## Paper says

Section 2.4 and Table 5 use GPT-4.1 for answers and compare Adaptive against
raw LangChain recursive default and raw page. Retrieval Completeness is judged
on the 0/1/2 scale and Answer Correctness with G-Eval. The original paper's QA
set is not available here; this reproduction uses the frozen reviewed QA.

## Code and artifact observations

- `phase-i-table5/judge-input-smoke/` contains six matched QA per system, all
  answers nonempty, and no exact abstention string. The export manifest pins
  each answer file's SHA256.
- Each answer addresses the main question in a surface comparison with the
  frozen reference answer. This is **not** a correctness pass: generated
  answers often add claims outside the reference, and those claims need a
  source-aware judge or human check.
- In `per_query_metrics.jsonl`, raw LangChain recursive default for
  `doc-0e35d2296535d358::q3` has source-span `Hit@10 = 0` and
  `Recall@10 = 0`. However, its retrieved chunks 0, 1, 2, 6, 7, and 9
  include discussion of random sampling; chunk 1 explicitly discusses
  discrimination, efficiency, and new types of fraud. Thus a zero overlap
  with the *frozen evidence offsets* does not prove the retrieved context
  lacks answer-relevant text.
- For Adaptive on `doc-6e712798758b54ab::q2`, only 2/10 retrieved chunks
  are from the target AI Act document. Its answer cites Documents 0, 1, and
  2, including an unrelated Sophia document in that range. The core answer
  is present, but the citation mixture needs source-level checking.
- Answers for `doc-0e35d2296535d358::q2` add examples and adjacent concerns
  beyond the reference answer. Answers for
  `doc-6e712798758b54ab::q3` additionally discuss the separate sensitive-
  attribute prohibition. Extra detail is not automatically an error, but
  neither is it validated merely by matching the reference's main point.

| QA suffix | Adaptive | Raw LC default | Raw page | Main review caution |
|---|---|---|---|---|
| Sophia q1, democratic control | answers | answers | answers | Extensive claims beyond the reference |
| Sophia q2, generative AI debate | answers | answers | answers | Additional examples and policy claims |
| Sophia q3, random sampling | answers | answers | answers | Raw LC source-span Hit@10 is zero despite relevant retrieved text |
| AI Act q1, objective | answers | answers | answers | Reference has a Member State restriction detail not emphasized in outputs |
| AI Act q2, exclusions | answers | answers | answers | Adaptive mixes in a non-target document citation |
| AI Act q3, biometrics | answers | answers | answers | Additional prohibition discussion exceeds the narrow reference |

## Analyst inference and next gate

The smoke supports proceeding to a **separately budgeted** judge smoke after
pinning and testing a compatible DeepEval version. It does not establish that
Adaptive is better, that all answers are fully grounded, or that Table 5 has
been reproduced. The source-span discrepancy is an audit target, not a
confirmed retrieval bug.

`phase_i_judge_preflight.py` read the original prompt and G-Eval steps without
importing DeepEval or calling an API. It validated all 18 inputs and estimated
18 Retrieval Completeness plus 18 Correctness evaluations. The exact
completeness prompts total 143,589 tokens; reference/generated-answer payloads
plus G-Eval steps total 8,093 tokens **before** DeepEval's internal template.
At an illustrative 256 output tokens and 256 chat-framing tokens per call,
GPT-4.1 pricing yields USD 0.395524. This is **not a spending cap**: the
original judge does not cap output and the installed environment has no
DeepEval. A paid judge run needs a pinned, dry-tested implementation and a
new explicit budget.
