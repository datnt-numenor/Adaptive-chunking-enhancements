"""Prepare a secret-free Kaggle dataset and isolated metric-run notebooks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path


PINNED_COMMIT = "ea87ce8e1a97888f3f179e7f1359ff7f43fb179d"
MODEL_REVISION = "ab036b023d30b4d1138c4c3bfa9f0c445ab455d6"
MODEL_CODE_REVISION = "bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3"
SENTENCE_TRANSFORMERS_VERSION = "3.1.0"
TRANSFORMERS_VERSION = "4.41.2"
KAGGLE_ACCELERATOR = "NvidiaTeslaT4"
KINDS = ("processed", "raw")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _reset_generated_directory(path: Path) -> None:
    if path.exists():
        resolved = path.resolve()
        if path.name not in {"kaggle_metric_dataset", "kaggle_metric_kernels"}:
            raise ValueError(f"Refusing to replace unexpected directory: {resolved}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _notebook_source(kind: str, shard: int, dataset_slug: str) -> str:
    output_name = f"{kind}-shard-{shard:02d}"
    stage = "small_merged" if kind == "processed" else "raw"
    expected_methods = 8 if kind == "processed" else 5
    return f'''import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

KIND = {kind!r}
SHARD = {shard}
PINNED_COMMIT = {PINNED_COMMIT!r}
MODEL_REVISION = {MODEL_REVISION!r}
MODEL_CODE_REVISION = {MODEL_CODE_REVISION!r}
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/tmp/adaptive-week2-metrics")
REPO = WORK_ROOT / "adaptive-chunking"
VENV_PYTHON = Path(sys.executable)
OUTPUT = Path("/kaggle/working/{output_name}")
LOG = Path("/kaggle/working/{output_name}.log")

for secret_name in ("JINA_API_KEY", "OPENAI_API_KEY"):
    assert not os.environ.get(secret_name), f"{{secret_name}} must be unset"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

gpu_environment = {{
    "kind": KIND,
    "shard": SHARD,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "model_revision": MODEL_REVISION,
    "model_code_revision": MODEL_CODE_REVISION,
    "metric_batch_size": 4,
}}
if torch.cuda.is_available():
    gpu_environment["gpu_name"] = torch.cuda.get_device_name(0)
    gpu_environment["gpu_capability"] = list(torch.cuda.get_device_capability(0))
Path("/kaggle/working/{output_name}-environment.json").write_text(
    json.dumps(gpu_environment, indent=2), encoding="utf-8"
)
assert torch.cuda.is_available(), gpu_environment

metric_candidates = [
    path.parent
    for path in INPUT_ROOT.rglob("upload_manifest.json")
    if (path.parent / "reproduction_runner.py").is_file()
]
source_candidates = [
    path.parent
    for path in INPUT_ROOT.rglob("filename_mapping.json")
    if (path.parent / "data_shards_safe").is_dir()
]
assert len(metric_candidates) == 1, metric_candidates
assert len(source_candidates) == 1, source_candidates
METRIC_INPUT = metric_candidates[0]
SOURCE_INPUT = source_candidates[0]
assert not OUTPUT.exists(), OUTPUT

subprocess.run(
    ["git", "clone", "https://github.com/ekimetrics/adaptive-chunking.git", str(REPO)],
    check=True,
)
subprocess.run(["git", "checkout", "--detach", PINNED_COMMIT], cwd=REPO, check=True)
actual_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
).strip()
assert actual_commit == PINNED_COMMIT, actual_commit

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        str(REPO),
        "pyarrow",
        "einops",
        "sentence-transformers=={SENTENCE_TRANSFORMERS_VERSION}",
        "transformers=={TRANSFORMERS_VERSION}",
    ],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "peft"],
    check=True,
)
assert importlib.util.find_spec("peft") is None

helper_dir = REPO / "research" / "week2"
helper_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(METRIC_INPUT / "reproduction_runner.py", helper_dir / "reproduction_runner.py")

restored = WORK_ROOT / "restored-data-shards"
subprocess.run(
    [
        sys.executable,
        str(METRIC_INPUT / "restore_kaggle_upload.py"),
        "--input-dir", str(SOURCE_INPUT / "data_shards_safe"),
        "--mapping", str(SOURCE_INPUT / "filename_mapping.json"),
        "--output-dir", str(restored),
    ],
    check=True,
)

data_dir = WORK_ROOT / "data" / "clair"
for child in ("adi_parsed", "mentions"):
    destination = data_dir / child
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(restored.glob(f"shard-*/{{child}}/*")):
        target = destination / source.name
        assert not target.exists(), target
        shutil.copy2(source, target)

assert len(list((data_dir / "adi_parsed").glob("*.json"))) == 33
assert len(list((data_dir / "mentions").glob("*.parquet"))) == 33

chunks = (
    METRIC_INPUT
    / "metric_shards"
    / f"shard-{{SHARD:02d}}"
    / "chunks"
    / {stage!r}
    / "chunks.parquet"
)
assert chunks.is_file(), chunks

environment = {{
    "kind": KIND,
    "shard": SHARD,
    "source_commit": actual_commit,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "model_revision": MODEL_REVISION,
    "model_code_revision": MODEL_CODE_REVISION,
    "sentence_transformers_version": "{SENTENCE_TRANSFORMERS_VERSION}",
    "transformers_version": "{TRANSFORMERS_VERSION}",
    "peft_installed": importlib.util.find_spec("peft") is not None,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
    "chunks_sha256": hashlib.sha256(chunks.read_bytes()).hexdigest(),
}}
try:
    environment["nvidia_smi"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
except Exception as error:
    environment["nvidia_smi_error"] = repr(error)
Path("/kaggle/working/{output_name}-environment.json").write_text(
    json.dumps(environment, indent=2), encoding="utf-8"
)

command = [
    sys.executable,
    "-X", "utf8",
    str(helper_dir / "reproduction_runner.py"),
    "run-metrics-shard",
    "--kind", KIND,
    "--chunks-parquet", str(chunks),
    "--data-dir", str(data_dir),
    "--output-dir", str(OUTPUT),
    "--device", "cuda:0",
    "--batch-size", "4",
]
with LOG.open("w", encoding="utf-8") as log_handle:
    result = subprocess.run(
        command,
        cwd=REPO,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
if result.returncode != 0:
    print(LOG.read_text(encoding="utf-8", errors="replace")[-12000:])
    raise RuntimeError(f"metric runner failed with exit code {{result.returncode}}")

validation_script = """
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

kind, output_arg, expected_methods_arg, model_revision, code_revision = sys.argv[1:]
output = Path(output_arg)
subdir = "results" if kind == "processed" else "results_raw"
metrics_path = output / subdir / "chunking_metrics.parquet"
manifest_path = output / f"{{kind}}_metrics_manifest.json"
metrics = pd.read_parquet(metrics_path)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected_docs = manifest["document_count"]
expected_methods = int(expected_methods_arg)
assert metrics.doc_name.nunique() == expected_docs
assert metrics.chunking_method.nunique() == expected_methods
assert metrics.metric_name.nunique() == 10
assert len(metrics) == expected_docs * expected_methods * 10
assert manifest["embedding_backend"] == "sentence_transformers_local"
assert manifest["embedding_model_revision"] == model_revision
assert manifest["embedding_code_revision"] == code_revision
validation = {{
    "status": "ok",
    "kind": kind,
    "documents": expected_docs,
    "methods": sorted(metrics.chunking_method.unique().tolist()),
    "metric_names": sorted(metrics.metric_name.unique().tolist()),
    "rows": len(metrics),
    "null_scores": int(metrics.score.isna().sum()),
    "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
}}
(output / "validation.json").write_text(
    json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
)
"""
subprocess.run(
    [
        sys.executable,
        "-X", "utf8",
        "-c", validation_script,
        KIND,
        str(OUTPUT),
        str({expected_methods}),
        MODEL_REVISION,
        MODEL_CODE_REVISION,
    ],
    check=True,
)
validation = json.loads((OUTPUT / "validation.json").read_text(encoding="utf-8"))
validation["shard"] = SHARD
(OUTPUT / "validation.json").write_text(
    json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
)
subprocess.run(
    [sys.executable, "-m", "pip", "freeze"],
    stdout=(OUTPUT / "pip-freeze.txt").open("w", encoding="utf-8"),
    check=True,
)
print(json.dumps(validation, ensure_ascii=False, indent=2))
print("METRIC_SHARD_COMPLETE")
'''


def _make_notebook(kind: str, shard: int, dataset_slug: str) -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "run-description",
                "metadata": {},
                "source": [
                    f"# Adaptive Chunking metrics: {kind} shard {shard:02d}\n",
                    "Private, zero-API run using the pinned local embedding model revision.\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "metric-run",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _notebook_source(kind, shard, dataset_slug).splitlines(True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def prepare(
    root: Path,
    metric_shards: Path,
    runner: Path,
    restore_script: Path,
    base_kernel_metadata: Path,
    username: str,
    dataset_slug: str,
) -> dict:
    dataset_dir = root / "kaggle_metric_dataset"
    kernels_dir = root / "kaggle_metric_kernels"
    _reset_generated_directory(dataset_dir)
    _reset_generated_directory(kernels_dir)

    shutil.copytree(metric_shards, dataset_dir / "metric_shards")
    shutil.copy2(runner, dataset_dir / "reproduction_runner.py")
    shutil.copy2(restore_script, dataset_dir / "restore_kaggle_upload.py")
    _write_json(
        dataset_dir / "dataset-metadata.json",
        {
            "id": f"{username}/{dataset_slug}",
            "title": dataset_slug,
            "licenses": [{"name": "unknown"}],
        },
    )

    files = sorted(
        path for path in dataset_dir.rglob("*") if path.is_file()
    )
    forbidden = [
        str(path.relative_to(dataset_dir))
        for path in files
        if path.name.lower() in {".env", "kaggle.json"}
        or "secret" in path.name.lower()
    ]
    if forbidden:
        raise ValueError(f"Forbidden files in upload staging: {forbidden}")
    upload_manifest = {
        "schema_version": 1,
        "source_commit": PINNED_COMMIT,
        "embedding_model": "jinaai/jina-embeddings-v3",
        "embedding_model_revision": MODEL_REVISION,
        "embedding_code_revision": MODEL_CODE_REVISION,
        "sentence_transformers_version": SENTENCE_TRANSFORMERS_VERSION,
        "transformers_version": TRANSFORMERS_VERSION,
        "files": [
            {
                "path": path.relative_to(dataset_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
            if path.name != "upload_manifest.json"
        ],
    }
    _write_json(dataset_dir / "upload_manifest.json", upload_manifest)

    base_metadata = json.loads(base_kernel_metadata.read_text(encoding="utf-8"))
    base_metadata.pop("id_no", None)
    kernel_count = 0
    for kind in KINDS:
        for shard in range(6):
            slug = f"adaptive-metrics-{kind}-shard-{shard:02d}"
            directory = kernels_dir / f"{kind}-shard-{shard:02d}"
            directory.mkdir(parents=True)
            notebook = _make_notebook(kind, shard, dataset_slug)
            code = "".join(notebook["cells"][1]["source"])
            compile(code, f"{slug}.py", "exec")
            _write_json(directory / "notebook.ipynb", notebook)
            metadata = dict(base_metadata)
            metadata.pop("docker_image", None)
            metadata.update(
                {
                    "id": f"{username}/{slug}",
                    "title": slug,
                    "code_file": "notebook.ipynb",
                    "is_private": True,
                    "enable_gpu": True,
                    "enable_tpu": False,
                    "enable_internet": True,
                    "dataset_sources": [
                        f"{username}/{dataset_slug}",
                        f"{username}/adaptive-chunking-shards",
                    ],
                    "kernel_sources": [],
                    "competition_sources": [],
                    "model_sources": [],
                    "machine_shape": KAGGLE_ACCELERATOR,
                }
            )
            _write_json(directory / "kernel-metadata.json", metadata)
            kernel_count += 1

    return {
        "dataset_dir": str(dataset_dir),
        "dataset_file_count": len(files) + 1,
        "dataset_bytes": sum(path.stat().st_size for path in dataset_dir.rglob("*") if path.is_file()),
        "kernels_dir": str(kernels_dir),
        "kernel_count": kernel_count,
        "embedding_model_revision": MODEL_REVISION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metric-shards", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--restore-script", type=Path, required=True)
    parser.add_argument("--base-kernel-metadata", type=Path, required=True)
    parser.add_argument("--username", default="datnguyenksnb")
    parser.add_argument("--dataset-slug", default="adaptive-chunking-metric-shards")
    args = parser.parse_args()
    result = prepare(
        args.root,
        args.metric_shards,
        args.runner,
        args.restore_script,
        args.base_kernel_metadata,
        args.username,
        args.dataset_slug,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
