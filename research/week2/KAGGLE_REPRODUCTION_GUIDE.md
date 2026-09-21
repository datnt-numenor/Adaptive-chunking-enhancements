# Low-cost Tables 1--3 reproduction guide

This guide runs the upstream repository at commit
`ea87ce8e1a97888f3f179e7f1359ff7f43fb179d` without modifying its algorithms.
It uses Kaggle for the free workloads and an Ampere GPU rental only for the
upstream semantic configuration.

## What this run can claim

- **Paper nói gì:** Table 3 compares raw and post-processed outputs using five
  intrinsic metrics.
- **Code thực sự làm gì:** the pinned CLI builds eight chunking methods, uses
  `gpt-4o` for LLM-regex, and prints published Table 3 values stored as source
  constants beside locally computed values.
- **Suy luận của người phân tích:** this is a code-level reproduction. It is not an exact
  paper reproduction when the paper and code use different LLMs or when an
  unpinned dependency has changed.

If evidence is unavailable, record: **Không tìm thấy bằng chứng trong paper/code**.

## Cost and safety limits

- OpenAI: use a project-specific key, record usage before and after each shard,
  and set a USD 10 project alert/limit where available. The source calls
  `gpt-4o`; do not silently substitute another model.
- Jina: use the free API only for the three-document local smoke test. Full
  metrics must not have `JINA_API_KEY` in the Kaggle environment.
- GPU rental: use an on-demand RTX A5000 at the displayed rate only if it is at
  or below the agreed budget. Stop at 9--10 hours and USD 3 total.
- Never paste a secret into a notebook cell, terminal command, log, Git commit,
  screenshot, or uploaded dataset.

Official references:

- [Kaggle notebooks and GPU sessions](https://www.kaggle.com/docs/notebooks)
- [Kaggle GPU efficiency and quota](https://www.kaggle.com/docs/efficient-gpu-usage)
- [FlashAttention GPU requirements](https://github.com/Dao-AILab/flash-attention)
- [RunPod pricing](https://www.runpod.io/pricing)
- [RunPod secret handling](https://docs.runpod.io/pods/templates/secrets)
- [OpenAI GPT-4o model and pricing](https://developers.openai.com/api/docs/models/gpt-4o)

## Artifact layout

All local artifacts stay below:

```text
research/week2/artifacts/
├── data_shards/
├── nonsemantic_shards/
├── combined_chunks/
├── semantic_output/
├── full_chunks/
├── metric_shards/
├── processed_metric_shards/
├── raw_metric_shards/
└── final_table3/
```

Use a new directory name if a run configuration changes. Do not merge outputs
from different commits, models, semantic backends, or dependency environments.

## Phase A -- local preparation

Run every local command from `D:\Research\adaptive-chunking`.

### A1. Confirm the source and environment

```powershell
git rev-parse HEAD
git status --short
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip freeze
```

Expected commit:

```text
ea87ce8e1a97888f3f179e7f1359ff7f43fb179d
```

Do not discard the existing untracked research files.

### A2. Run the local Jina metrics smoke test

```powershell
.\.venv\Scripts\python.exe -X utf8 -c "from dotenv import load_dotenv; load_dotenv(); from adaptive_chunking.jina_embedder import JinaEmbedder; from adaptive_chunking.compute_metrics import compute_metrics_per_origin; compute_metrics_per_origin(chunks_dir='research/week2/artifacts/core_smoke', mentions_dir='data/clair/mentions', parsed_docs_dir='data/clair/adi_parsed', models={'sentence_embedder': JinaEmbedder()}, output_dir='research/week2/artifacts/jina_metrics_smoke', batch_size=32)"
```

Check it:

```powershell
.\.venv\Scripts\python.exe -X utf8 -c "import pandas as pd; p='research/week2/artifacts/jina_metrics_smoke/chunking_metrics.parquet'; d=pd.read_parquet(p); print('rows=',len(d)); print('docs=',d.doc_name.nunique()); print('methods=',sorted(d.chunking_method.unique())); print('metrics=',sorted(d.metric_name.unique()))"
```

Expected: 3 documents, 3 methods, 10 recorded metric/statistic names, and 90
rows. A rerun is resumable and should skip completed documents.

### A3. Create three balanced input shards

```powershell
.\.venv\Scripts\python.exe -X utf8 research/week2/reproduction_runner.py `
  make-data-shards `
  --data-dir data/clair `
  --output-dir research/week2/artifacts/data_shards `
  --shards 3
```

Inspect `data_shards/shards_manifest.json`. It must say 33 documents and 3
shards. Each document must occur once. Upload the prebuilt
`artifacts/kaggle_week2_upload_kaggle_safe.zip` to one **private** Kaggle
Dataset; never include `.env`.

The source document names contain characters that Kaggle rejects in uploaded
archive entries (`\`, `&`, and ASCII apostrophes). The safe ZIP therefore uses
ASCII aliases and includes `filename_mapping.json` plus
`restore_kaggle_upload.py`. Do not use either of the older upload ZIPs.

The generated cloud hand-off directories are intentionally ignored by the
repository-local `artifacts/.gitignore`; they remain on disk for upload but
cannot be committed accidentally.

## Phase B -- seven non-semantic chunkers on Kaggle

Run one data shard per saved Kaggle notebook version. Turn on Internet. A GPU is
optional for this phase, but `cuda:0` lets Stanza use it if available.

### B1. Load the OpenAI secret without printing it

Create a Kaggle secret named `OPENAI_API_KEY`, attach it to the private
notebook, then run:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["OPENAI_API_KEY"] = UserSecretsClient().get_secret("OPENAI_API_KEY")
print("OPENAI key loaded:", bool(os.environ.get("OPENAI_API_KEY")))
```

Only the Boolean result may be printed.

### B2. Clone and pin the repository

```bash
!git clone https://github.com/ekimetrics/adaptive-chunking.git /kaggle/working/adaptive-chunking
%cd /kaggle/working/adaptive-chunking
!git checkout --detach ea87ce8e1a97888f3f179e7f1359ff7f43fb179d
!git rev-parse HEAD
!mkdir -p research/week2
!cp /kaggle/input/adaptive-chunking-shards/reproduction_runner.py research/week2/
```

Adjust the private Dataset path in the `cp` command if Kaggle mounted it under
a different slug.

### B3. Install only the runtime needed by this phase

Try the official environment first:

```bash
!python -m pip install -e ".[paper,test]"
!python -m pip install httpx langchain-community
```

If the official extra fails only in unused `docling` or `maverick-coref`
dependencies, restart the Kaggle session and use the targeted runtime:

```bash
!python -m pip install -e .
!python -m pip install pytest pytest-asyncio stanza nltk "langchain>=0.3.21" \
    langchain-experimental langchain-community openai matplotlib seaborn tabulate httpx
```

Record the official install error in the audit. Do not run parsing or mention
extraction; parsed JSON and precomputed mentions already exist.

### B4. Run one shard

Set `SHARD_INDEX` to `00`, `01`, or `02`. Adjust `DATASET_ROOT` to the actual
private Kaggle Dataset mount shown in the right sidebar.

```python
from pathlib import Path
import subprocess

SHARD_INDEX = "00"
DATASET_ROOT = Path("/kaggle/input/adaptive-chunking-shards")
RESTORED_ROOT = Path("/kaggle/working/data_shards")

subprocess.run(
    [
        "python", str(DATASET_ROOT / "restore_kaggle_upload.py"),
        "--input-dir", str(DATASET_ROOT / "data_shards_safe"),
        "--mapping", str(DATASET_ROOT / "filename_mapping.json"),
        "--output-dir", str(RESTORED_ROOT),
    ],
    check=True,
)

SHARD_DIR = RESTORED_ROOT / f"shard-{SHARD_INDEX}"
OUTPUT_DIR = Path("/kaggle/working") / f"nonsemantic-shard-{SHARD_INDEX}"
assert SHARD_DIR.is_dir(), SHARD_DIR
```

Run through the rate-limited helper. It invokes the unchanged upstream command
for one document at a time, waits between requests, saves a checkpoint after
each document, and merges the 11 outputs only after all have passed validation.

```python
import subprocess

command = [
    "python", "-X", "utf8",
    "research/week2/reproduction_runner.py",
    "run-nonsemantic-shard",
    "--data-dir", str(SHARD_DIR),
    "--output-dir", str(OUTPUT_DIR),
    "--device", "cuda:0",
    "--resume",
    "--min-interval-seconds", "65",
    "--initial-cooldown-seconds", "70",
]
subprocess.run(command, check=True)
```

The cooldown is an execution-control deviation needed for accounts with a
30,000 TPM limit. The model (`gpt-4o`), prompt, temperature, chunking methods,
and post-processing are unchanged. Inspect the underlying error before retrying:
waiting helps a TPM 429, but does not fix validation or environment failures.
The helper reuses validated document checkpoints and complete merged output.
If partial parquet exists, it refuses another potentially paid request until
that output has been inspected/recovered. Preserve the current session or export
the checkpoint directory before ending it; `--resume` cannot recover missing files.
Intermediate empty `page` chunks are preserved in raw/no_oversizing; empty chunks
remain invalid for other methods or for small_merged.

### B5. Validate and export the shard

```python
import pandas as pd

for stage in ["raw", "no_oversizing", "small_merged"]:
    path = OUTPUT_DIR / "chunks" / stage / "chunks.parquet"
    frame = pd.read_parquet(path)
    print(stage, frame.doc_name.nunique(), sorted(frame.method.unique()))
    assert frame.doc_name.nunique() == 11
    assert set(frame.method.unique()) == {
        "page", "sentence", "langch_recurs_default", "langch_recurs_1100",
        "our_recurs_1100", "our_recurs_600", "llm_regex",
    }
```

Save the notebook version, then download the output directory and log to:

```text
research/week2/artifacts/nonsemantic_shards/shard-XX/
```

Check OpenAI Usage. Do not rerun a successful shard. The helper protects
completed per-document checkpoints during a failed or interrupted run, while
the original all-document command has no response cache.

### B6. Merge all three shards locally

```powershell
.\.venv\Scripts\python.exe -X utf8 research/week2/reproduction_runner.py `
  merge-chunks `
  --inputs research/week2/artifacts/nonsemantic_shards `
  --output-dir research/week2/artifacts/combined_chunks

.\.venv\Scripts\python.exe -X utf8 research/week2/reproduction_runner.py `
  validate `
  --output-dir research/week2/artifacts/combined_chunks `
  --expected-docs 33 `
  --profile nonsemantic
```

## Phase C -- exact semantic chunker on a short RunPod rental

Use an on-demand Linux RTX A5000. Confirm the displayed hourly rate before
starting. Do not use spot/interruptible capacity. Keep all artifacts below
`/workspace`. RunPod listed the RTX A5000 Secure Cloud rate as USD 0.27/hour on
2026-09-12, but the deployment screen is authoritative. At that rate, nine
hours of runner time costs USD 2.43; installation, download, validation, and
storage still consume budget. Set a pod-lifetime alarm before deployment and
terminate before the total reaches USD 3.

Prepare and upload the checked, secret-free input bundle from the local clone:

```powershell
.\.venv\Scripts\python.exe -X utf8 research/week2/prepare_runpod_semantic_bundle.py `
  --output-zip research/week2/artifacts/runpod_exact_semantic_input.zip
```

Upload both the ZIP and its `.sha256` sidecar. The bundle contains only the
runner, this guide, and the 33 `adi_parsed` JSON files; it contains no API keys
or caches.

### C1. Install the pinned runtime

```bash
cd /workspace
sha256sum -c runpod_exact_semantic_input.zip.sha256
unzip -q runpod_exact_semantic_input.zip -d semantic-input

git clone https://github.com/ekimetrics/adaptive-chunking.git
cd adaptive-chunking
git checkout --detach ea87ce8e1a97888f3f179e7f1359ff7f43fb179d
test "$(git rev-parse HEAD)" = "ea87ce8e1a97888f3f179e7f1359ff7f43fb179d"

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
python -m pip install stanza nltk "langchain>=0.3.21" \
  langchain-experimental langchain-community matplotlib packaging psutil ninja
MAX_JOBS=4 python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip freeze > /workspace/semantic-environment.txt
MODEL_REVISION=$(python -c "from huggingface_hub import HfApi; print(HfApi().model_info('Qwen/Qwen3-Embedding-0.6B').sha)")
test "${#MODEL_REVISION}" = "40"
printf '%s\n' "$MODEL_REVISION" > /workspace/semantic-model-revision.txt
```

Install the audited runner into the pinned clone:

```bash
mkdir -p research/week2
cp /workspace/semantic-input/research/week2/reproduction_runner.py \
  research/week2/reproduction_runner.py
```

### C2. Enforce the GPU gate

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print('bf16=',torch.cuda.is_bf16_supported())"
python -c "import flash_attn; print('FLASH_ATTN_OK')"
```

Continue only when compute capability is at least 8.0 and `bf16=True`.

### C3. Benchmark representative documents

```bash
python -X utf8 research/week2/reproduction_runner.py \
  run-semantic-only \
  --data-dir /workspace/semantic-input/data/clair \
  --output-dir /workspace/semantic-output \
  --device cuda:0 \
  --model-revision "$(cat /workspace/semantic-model-revision.txt)" \
  --benchmark-documents 3 \
  --max-runtime-hours 2
```

Inspect `/workspace/semantic-output/semantic_run_manifest.json`. Continue only
if `estimated_total_hours_from_session_mean` is at most 9 hours.

### C4. Resume the remaining documents

```bash
python -X utf8 research/week2/reproduction_runner.py \
  run-semantic-only \
  --data-dir /workspace/semantic-input/data/clair \
  --output-dir /workspace/semantic-output \
  --device cuda:0 \
  --model-revision "$(cat /workspace/semantic-model-revision.txt)" \
  --resume \
  --max-runtime-hours 7
```

The helper uses previous per-document timings and stops before starting a new
document when its conservative time estimate would cross the requested cap.
It also refuses a resume when the model, attention implementation, dtype,
batch size, source commit, or any input document hash differs. The two commands
still have separate runtime caps and do not include setup time. Monitor the
RunPod billing clock; the USD 3 budget is authoritative.

### C5. Validate and download before terminating

```bash
python -X utf8 research/week2/reproduction_runner.py \
  validate \
  --output-dir /workspace/semantic-output \
  --expected-docs 33 \
  --profile semantic

cd /workspace
tar -czf semantic-output.tar.gz semantic-output
sha256sum semantic-output.tar.gz > semantic-output.tar.gz.sha256
```

Download both files. Verify the hash locally, open the archive, then terminate
the Pod. Extract it to `research/week2/artifacts/semantic_output`.

### C6. Budget fallback

If exact semantic cannot finish within USD 3, preserve the partial exact
artifact. Do not mix exact and fallback rows. A fallback run must process all 33
documents with:

```bash
--attention-implementation sdpa --dtype float16 --batch-size 2
```

Any final table using that artifact must be labeled **approximate
reproduction**.

## Phase D -- build the eight-method corpus

Merge the seven-method and semantic outputs:

```powershell
.\.venv\Scripts\python.exe -X utf8 research/week2/reproduction_runner.py `
  merge-chunks `
  --inputs `
    research/week2/artifacts/combined_chunks `
    research/week2/artifacts/semantic_output `
  --output-dir research/week2/artifacts/full_chunks

.\.venv\Scripts\python.exe -X utf8 research/week2/reproduction_runner.py `
  validate `
  --output-dir research/week2/artifacts/full_chunks `
  --expected-docs 33 `
  --profile full
```

Create six metric shards:

```powershell
.\.venv\Scripts\python.exe -X utf8 research/week2/reproduction_runner.py `
  make-metric-shards `
  --chunks-dir research/week2/artifacts/full_chunks `
  --output-dir research/week2/artifacts/metric_shards `
  --shards 6
```

Upload `metric_shards`, the original `data/clair/adi_parsed` and
`data/clair/mentions`, and `reproduction_runner.py` to a private Kaggle Dataset.
Do not upload `.env`.

## Phase E -- metrics on free Kaggle sessions

Use one metric shard per run. Enable GPU and Internet. Do not attach the Jina or
OpenAI secrets to these notebooks.

Clone the pinned repository, install `-e .`, and copy the helper into
`research/week2/` using `mkdir -p research/week2` plus the adjusted private
Dataset path. Save `python -m pip freeze` and the GPU name beside each shard.
Then run processed metrics:

```bash
!python -X utf8 research/week2/reproduction_runner.py \
  run-metrics-shard \
  --kind processed \
  --chunks-parquet /kaggle/input/ADJUST_PATH/metric_shards/shard-00/chunks/small_merged/chunks.parquet \
  --data-dir /kaggle/input/ADJUST_PATH/data/clair \
  --output-dir /kaggle/working/processed-shard-00 \
  --device cuda:0 \
  --batch-size 32
```

Run raw metrics separately:

```bash
!python -X utf8 research/week2/reproduction_runner.py \
  run-metrics-shard \
  --kind raw \
  --chunks-parquet /kaggle/input/ADJUST_PATH/metric_shards/shard-00/chunks/raw/chunks.parquet \
  --data-dir /kaggle/input/ADJUST_PATH/data/clair \
  --output-dir /kaggle/working/raw-shard-00 \
  --device cuda:0 \
  --batch-size 32
```

Repeat `00` through `05`. The helper refuses to run when `JINA_API_KEY` is set
unless `--allow-jina-api` is explicitly supplied. Do not supply it for the full
run.

After every saved version, download results to:

```text
research/week2/artifacts/processed_metric_shards/shard-XX/
research/week2/artifacts/raw_metric_shards/shard-XX/
```

The upstream metrics function saves after every document, so a partial shard is
resumable if its output directory is restored in the next session.

## Phase F -- merge and generate Tables 1--3 locally

Install only the small analysis dependencies if they are absent:

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib seaborn tabulate
```

Merge metrics and copy chunks into the CLI-compatible output layout:

```powershell
.\.venv\Scripts\python.exe -X utf8 research/week2/reproduction_runner.py `
  merge-metrics `
  --processed-inputs research/week2/artifacts/processed_metric_shards `
  --raw-inputs research/week2/artifacts/raw_metric_shards `
  --chunks-dir research/week2/artifacts/full_chunks `
  --output-dir research/week2/artifacts/final_table3

.\.venv\Scripts\python.exe -X utf8 research/week2/reproduction_runner.py `
  validate `
  --output-dir research/week2/artifacts/final_table3 `
  --expected-docs 33 `
  --profile full
```

Expected final counts:

- processed metrics: `33 x 8 x 10 = 2,640` rows;
- raw metrics: `33 x 5 x 10 = 1,650` rows;
- nine displayed Table 3 rows because page appears in raw and post-processed
  forms.

Generate the tables and preserve the console output:

```powershell
$env:MPLBACKEND = "Agg"
.\.venv\Scripts\python.exe -X utf8 -m adaptive_chunking.paper.replicate `
  --data-dir data/clair `
  --output-dir research/week2/artifacts/final_table3 `
  --steps analysis table3 `
  --device cpu 2>&1 | Tee-Object `
  research/week2/artifacts/final_table3/analysis_table3.log
```

Completion requires the message:

```text
All methods computed locally.
```

Report every metric delta and null References Completeness value. Do not remove
nulls or failed documents silently.
