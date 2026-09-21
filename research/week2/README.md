# Week 2 reproduction workspace

Status as of 2026-09-11:

- Official repository cloned and pinned.
- Paper v1 downloaded and hashed.
- Lightweight Python 3.12 audit environment created.
- Unit tests pass: 54 passed, 1 skipped.
- Three-document CPU/core smoke test passes without API or GPU.
- Full 33-document dataset and raw-page BI audit completed.
- Low-cost Tables 1--3 runner and cloud hand-off guide implemented.
- Non-semantic LLM-regex runs are serialized per document with 65-second TPM
  spacing and durable resume checkpoints after a shard-01 429 audit.
- Recovery tooling and private Kaggle zero-API integration audit are described in
  `recovery_audit_2026_09_11.md`; paid continuation requires existing checkpoints.

## Artifacts

- `environment_snapshot.md`: host/runtime/dependency and dataset snapshot.
- `reproduce_checklist.md`: staged reproduction checklist and decision gates.
- `paper_code_audit.md`: evidence-separated paper/code/analysis findings.
- `KAGGLE_REPRODUCTION_GUIDE.md`: copy-paste Kaggle/RunPod workflow with cost and validation gates.
- `reproduction_runner.py`: resumable non-semantic and semantic execution,
  sharding, merging, metrics, and validation CLI.
- `prepare_kaggle_upload.py`: packages document files under Kaggle-safe ASCII aliases.
- `restore_kaggle_upload.py`: restores exact upstream filenames inside `/kaggle/working`.
- `run_core_smoke.py`: reproducible no-API core pipeline smoke test.
- `audit_dataset_and_bi.py`: dataset hashes/invariants and raw-page BI diagnostic.
- `artifacts/core_smoke/smoke_summary.json`: smoke metrics.
- `artifacts/dataset_bi_audit.json`: per-document checksums and BI audit.
- `artifacts/jina_metrics_smoke/`: successful 3-document, 90-row Jina smoke result.
- `artifacts/data_shards/`: three local-only 11-document Kaggle upload shards.

## Re-run commands

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -rs
.\.venv\Scripts\python.exe -X utf8 research/week2/run_core_smoke.py --documents 3
.\.venv\Scripts\python.exe -X utf8 research/week2/audit_dataset_and_bi.py
.\.venv\Scripts\python.exe -X utf8 research/week2/reproduction_runner.py --help
.\.venv\Scripts\python.exe -X utf8 -m adaptive_chunking.paper.replicate --help
```

The helper prepares and validates a Tables 1--3 reproduction. The actual Kaggle and
RunPod jobs are not executed automatically from this local workspace. Tables 4--5
remain out of scope for the low-cost plan.
