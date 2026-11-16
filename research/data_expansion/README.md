# Independent test set `clair_ext`

The goal is more documents, to get enough statistical power: CLAIR's 33 documents give an MDE of about 0.066 on
recall@10. CLAIR stays the development set; this set is the test set. The pre-registration is in
`PREREGISTRATION.md`.

| Step | Script | Environment | Cost |
|---|---|---|---|
| 1. List PDFs, download | `candidates_*.tsv`, `download.py` (plain UA, 3 s between arXiv requests) | `.venv` | free |
| 2. Parse | `parse_docling.py` (Docling 2.149, OCR off, offsets computed correctly) | `D:\Project\.venv-docling` | free (GPU) |
| 3. Inclusion rule, doc_id | `build_set.py` (2k–80k tokens) → `data\clair_ext_set\` | `.venv` | free |
| 4. Chunking (4 Adaptive candidates + LC default + page) | `chunk_ext.py` (llm_regex = gpt-4o, cached) | `.venv` | **paid** (llm_regex) |
| 5. Coreference mentions (RC) | replicate `--steps mentions` | `.venv-coref` or Linux | free (GPU) |
| 6. Intrinsic metrics | `..\structure_chunker\score_full.py run <run> --data-dir data\clair_ext_set` | `.venv` | free (GPU) |
| 7. QA | `gen_qa.py export → generate → validate → review → freeze` | `.venv` | **paid** |
| 8. Retrieval | proxy on the laptop, then the A5000 for the final numbers | | free / RunPod |

Bug found in the original repo [Code]: `DoclingParser.convert_raw_results_to_markdown`
(`src/adaptive_chunking/parsing.py`) appends `<!-- PageBreak -->` after `split_points` and `titles` have been
computed. Every offset from page 2 onwards is therefore shifted. `parse_docling.py` writes page breaks inline
instead.
