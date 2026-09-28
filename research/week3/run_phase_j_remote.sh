#!/usr/bin/env bash
# Run explicitly as: bash research/week3/run_phase_j_remote.sh smoke
# After inspecting the smoke archive: bash research/week3/run_phase_j_remote.sh full
set -euo pipefail

MODE=${1:-}
if [[ "$MODE" != smoke && "$MODE" != full ]]; then
  echo "Usage: bash research/week3/run_phase_j_remote.sh {smoke|full}" >&2
  exit 2
fi

WORKSPACE=/workspace
BUNDLE="$WORKSPACE/runpod_phase_j.zip"
BUNDLE_MANIFEST="$WORKSPACE/runpod_phase_j.zip.manifest.json"
REPO="$WORKSPACE/phase-j/adaptive-chunking"
PREPARED="$REPO/research/week3/artifacts/phase-j-bi-neutral/prepared"
QA_FULL="$REPO/research/week3/artifacts/qa-full"
QA_SMOKE="$WORKSPACE/phase-j-qa-smoke"
RESULT="$WORKSPACE/phase-j-results"
if [[ "$MODE" == smoke ]]; then RESULT="$WORKSPACE/phase-j-results-smoke"; fi
SYSTEM=adaptive_bi_neutral
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=5

# The archive and per-file manifest are checked before any code is executed.
cd "$WORKSPACE"
sha256sum -c runpod_phase_j.zip.sha256
python3.11 - "$BUNDLE" "$BUNDLE_MANIFEST" <<'PY'
import hashlib, json, sys, zipfile
from pathlib import PurePosixPath
archive_path, manifest_path = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
with zipfile.ZipFile(archive_path) as archive:
    actual = set(archive.namelist())
    expected = {entry["path"] for entry in manifest["files"]}
    assert actual == expected, "Archive file set differs from manifest"
    for entry in manifest["files"]:
        name = entry["path"]
        path = PurePosixPath(name)
        assert not path.is_absolute() and ".." not in path.parts, name
        data = archive.read(name)
        assert len(data) == entry["bytes"], name
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], name
PY

if [[ "$MODE" == smoke ]]; then
  if [[ -e "$REPO" || -e "$RESULT" || -e "$QA_SMOKE" ]]; then
    echo "Phase J smoke paths already exist; inspect or move them before retrying" >&2
    exit 1
  fi
  mkdir -p "$REPO" "$RESULT" "$QA_SMOKE" "$WORKSPACE/model-cache"
  python3.11 -m zipfile -e "$BUNDLE" "$REPO"
  cd "$REPO"
  python3.11 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
  python -m pip install -e .
  python -m pip install -r research/week3/requirements-gpu.txt
  MAX_JOBS=4 python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
  python -m pip freeze > "$RESULT/environment.txt"

  python - "$QA_FULL" "$QA_SMOKE" "$REPO/research/week3/artifacts/prepared/prepare_manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
full_dir, smoke_dir, prepared_manifest = map(Path, sys.argv[1:])
full_manifest = json.loads((full_dir / "qa_frozen_manifest.json").read_text(encoding="utf-8"))
full_path = full_dir / "qa_frozen.jsonl"
assert hashlib.sha256(full_path.read_bytes()).hexdigest() == full_manifest["qa_sha256"]
assert full_manifest["qa_count"] == 99 and full_manifest["documents"] == 33
prepared = json.loads(prepared_manifest.read_text(encoding="utf-8"))
ids = {row["doc_id"] for row in prepared["smoke_selection"]["documents"]}
assert len(ids) == 2
rows = [json.loads(line) for line in full_path.read_text(encoding="utf-8").splitlines()]
smoke = [row for row in rows if row["doc_id"] in ids]
assert len(smoke) == 6 and {row["doc_id"] for row in smoke} == ids
payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in smoke)
smoke_dir.joinpath("qa_frozen.jsonl").write_text(payload, encoding="utf-8")
smoke_dir.joinpath("qa_frozen_manifest.json").write_text(json.dumps({
    "schema_version": 1, "status": "complete", "mode": "smoke",
    "qa_count": 6, "documents": 2,
    "qa_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    "source_qa_sha256": full_manifest["qa_sha256"],
    "smoke_doc_ids": sorted(ids),
}, indent=2) + "\n", encoding="utf-8")
PY
else
  if [[ ! -f "$WORKSPACE/phase-j-results-smoke/evaluation/retrieval_evaluation.json" ]]; then
    echo "Completed Phase J smoke evaluation is required for full mode" >&2
    exit 1
  fi
  if [[ -e "$RESULT" ]]; then
    echo "Phase J full result path already exists; inspect or move it before retrying" >&2
    exit 1
  fi
  mkdir -p "$RESULT"
  cp "$WORKSPACE/phase-j-results-smoke/environment.txt" "$RESULT/environment.txt"
  cd "$REPO"
  source .venv/bin/activate
  python - "$WORKSPACE/phase-j-results-smoke/evaluation/retrieval_evaluation.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["status"] == "complete" and manifest["mode"] == "smoke"
assert manifest["qa_count"] == 6 and manifest["systems"] == ["adaptive_bi_neutral"]
qa = Path("/workspace/phase-j-qa-smoke/qa_frozen.jsonl")
assert manifest["input_qa_sha256"] == hashlib.sha256(qa.read_bytes()).hexdigest()
PY
fi

python - "$REPO" "$BUNDLE_MANIFEST" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, manifest_path = Path(sys.argv[1]), Path(sys.argv[2])
for record in json.loads(manifest_path.read_text(encoding="utf-8"))["files"]:
    path = root / record["path"]
    assert path.is_file(), record["path"]
    assert path.stat().st_size == record["bytes"], record["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], record["path"]
PY

export HF_HOME="$WORKSPACE/model-cache/huggingface"
export TORCH_HOME="$WORKSPACE/model-cache/torch"
export TOKENIZERS_PARALLELISM=false
python - <<'PY'
import flash_attn, torch
assert torch.cuda.get_device_name(0) == "NVIDIA RTX A5000"
assert torch.cuda.get_device_capability(0) >= (8, 0)
assert torch.cuda.is_bf16_supported()
assert torch.__version__.startswith("2.6.0")
assert flash_attn.__version__ == "2.7.4.post1"
PY

QA="$QA_FULL"
if [[ "$MODE" == smoke ]]; then QA="$QA_SMOKE"; fi
started=$(date +%s)
index_started=$(date +%s)
INDEX_ARGS=()
if [[ "$MODE" == full ]]; then
  INDEX_ARGS=(--smoke-evaluation-manifest "$WORKSPACE/phase-j-results-smoke/evaluation/retrieval_evaluation.json")
fi
python -X utf8 research/week3/phase_g_runner.py index \
  --prepared-dir "$PREPARED" --qa-dir "$QA" --output-dir "$RESULT/index" \
  --environment-file "$RESULT/environment.txt" --model-cache-dir "$WORKSPACE/model-cache" \
  --device cuda:0 --batch-size 1 --systems "$SYSTEM" "${INDEX_ARGS[@]}"
index_finished=$(date +%s)

python -X utf8 research/week3/phase_g_runner.py retrieve \
  --prepared-dir "$PREPARED" --qa-dir "$QA" --index-dir "$RESULT/index" \
  --output-dir "$RESULT/retrieval" --environment-file "$RESULT/environment.txt" \
  --model-cache-dir "$WORKSPACE/model-cache" --device cuda:0 --batch-size 1 \
  --systems "$SYSTEM"
retrieve_finished=$(date +%s)

python -X utf8 research/week3/phase_g_runner.py evaluate-retrieval \
  --prepared-dir "$PREPARED" --qa-dir "$QA" --retrieval-dir "$RESULT/retrieval" \
  --output-dir "$RESULT/evaluation" --systems "$SYSTEM"
evaluate_finished=$(date +%s)

python - "$RESULT" "$MODE" "$started" "$index_finished" "$retrieve_finished" "$evaluate_finished" <<'PY'
import json, sys
from pathlib import Path
root, mode = Path(sys.argv[1]), sys.argv[2]
start, index_end, retrieve_end, eval_end = map(int, sys.argv[3:])
expected = 6 if mode == "smoke" else 99
index = json.loads((root / "index/index_manifest.json").read_text(encoding="utf-8"))
retrieval = json.loads((root / "retrieval/retrieval_manifest.json").read_text(encoding="utf-8"))
evaluation = json.loads((root / "evaluation/retrieval_evaluation.json").read_text(encoding="utf-8"))
system = "adaptive_bi_neutral"
assert all(item["status"] == "complete" for item in (index, retrieval, evaluation))
assert index["qa_count"] == expected and retrieval["systems"][system]["queries"] == expected
assert evaluation["qa_count"] == expected and evaluation["systems"] == [system]
rows = (root / f"retrieval/{system}.jsonl").read_text(encoding="utf-8").splitlines()
assert len(rows) == expected
for row in rows:
    result = json.loads(row)
    assert result["system_id"] == system and len(result["results"]) == 10
root.joinpath("timing.json").write_text(json.dumps({
    "schema_version": 1, "status": "complete", "mode": mode,
    "run_started_epoch": start, "run_finished_epoch": eval_end,
    "elapsed_seconds": eval_end - start,
    "stages_seconds": {
        "index": index_end - start,
        "retrieve": retrieve_end - index_end,
        "evaluate": eval_end - retrieve_end,
    },
}, indent=2) + "\n", encoding="utf-8")
PY

cd "$WORKSPACE"
tar -czf "$(basename "$RESULT").tar.gz" "$(basename "$RESULT")"
sha256sum "$(basename "$RESULT").tar.gz" > "$(basename "$RESULT").tar.gz.sha256"
echo "Completed $MODE. Inspect and download $(basename "$RESULT").tar.gz and its checksum."
