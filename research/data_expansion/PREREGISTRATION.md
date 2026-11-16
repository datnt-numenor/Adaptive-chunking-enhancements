# Pre-registration: independent test set `clair_ext` (written 2026-11-09, before any QA on the new documents exists)

## Why
33 CLAIR documents give a minimum detectable effect of about 0.066 on recall@10. Every design choice so far
was made on CLAIR, so CLAIR is the development set. The new documents are the test set and are run **once**.

## Fixed before seeing test results
- **Documents:** official AI-governance publications that can be downloaded with a plain script
  (NIST, GAO, OMB, UK DSIT, PDPC, OECD, G7, ENISA, EDPB, Council of Europe, Datatilsynet, CNIL,
  Défenseur des droits, Commission IA), plus arXiv cs.CY papers.
  - arXiv papers come from fixed API queries (`candidates_arxiv_v0.tsv`): published 2023-01-01 or later,
    ≥ 10 pages when the arXiv comment states it.
  - Inclusion after parsing: 2,000–80,000 tokens. Download and parse failures are logged and excluded.
- **Parser:** Docling 2.149.0, OCR off (`parse_docling.py`). This is a different parser from CLAIR's Azure DI,
  so BI is not comparable across the two sets; the new set is reported separately.
- **QA:** prompt `runs/qa_new/prompt.txt` adapted to 6 QA per document (2 per type); excerpt rule as for QA v1.
  - Automatic quote validation.
  - LLM review: a QA is kept only if every reviewer says keep.
  - The set is frozen (sha256) before any retrieval run.
- **Systems:**
  1. Adaptive: the paper's 4 candidates (page, our_recurs_1100, our_recurs_600, llm_regex), equal weights,
     selection by the 5 metrics.
  2. LangChain recursive default (raw).
  3. page (raw).
  4. **SDP-A, target 600** (`run_sdp.py --set target_tokens=600`). It was chosen by 5-fold CV on CLAIR using
     proxy recall@10 (`runs/proxy/cv_r123_doc`: target 600 in 4 of 5 folds, target 700 in the other).
  5. SDP-B with jina (`--embed`, defaults), which won on CLAIR Table 3.
- **Representations:** `plain` (the harness default) and `doc` ("Document: <title>" prefix), applied identically to
  every system.
- **Primary metric:** recall@10 (user decision, 2026-11-09). Secondary metrics: nDCG@10, recall@budget,
  and the Table 3 intrinsic score.
- **Hypotheses:**
  - H1: SDP-A t600 ≥ Adaptive on recall@10 (doc representation).
  - H2: the `doc` prefix improves recall@10 for every system.
  - H3: SDP-B ≥ Adaptive on Table 3.
- **Statistics:** document-level bootstrap (10,000 resamples, seed 2026), Wilcoxon (Pratt), Holm correction
  over the comparisons listed in the hypotheses.

## What is not allowed
Changing systems, parameters or the QA set after seeing test retrieval or Table 3 results. Any later exploration
is reported as exploratory.
