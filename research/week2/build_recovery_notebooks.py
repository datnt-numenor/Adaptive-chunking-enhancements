"""Build a corrected notebook and a private zero-API integration audit."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research/week2/artifacts"
nb = json.loads((BASE / "kaggle_cli_notebook/reproduction-adaptive-chunking.ipynb").read_text(encoding="utf-8"))
for cell in nb["cells"]:
    if isinstance(cell["source"], list):
        cell["source"] = "".join(cell["source"])
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

runner = (ROOT / "research/week2/reproduction_runner.py").read_text(encoding="utf-8")
cache_wrapper = (ROOT / "research/week2/cached_openai_replicate.py").read_text(encoding="utf-8")
tests = (ROOT / "tests/test_reproduction_runner.py").read_text(encoding="utf-8")
bootstrap = nb["cells"][1]
old = '''shutil.copy2(
    DATASET_ROOT / "reproduction_runner.py",
    helper_dir / "reproduction_runner.py",
)'''
assert old in bootstrap["source"]
bootstrap["source"] = bootstrap["source"].replace(old,
    '(helper_dir / "reproduction_runner.py").write_text(' + repr(runner) + ', encoding="utf-8")\n'
    '(helper_dir / "cached_openai_replicate.py").write_text(' + repr(cache_wrapper) + ', encoding="utf-8")')
nb["cells"][3]["source"] = nb["cells"][3]["source"].replace(
    'environment_info.get("cuda")', 'environment_info["environment"].get("cuda")')

paid = nb["cells"][5]
paid["source"] = paid["source"].replace(
    'RUN_PAID_SHARD = False\nSHARD_INDEX = "SET_ME"',
    'RUN_PAID_SHARD = False\nSHARD_INDEX = "SET_ME"\nALLOW_FRESH_PAID_RUN = False'
)
paid["source"] = paid["source"].replace(
    '    assert shard_dir.is_dir(),',
    '    assert helper.with_name("cached_openai_replicate.py").is_file(), "Missing API cache helper; run Cell 1 again."\n    assert shard_dir.is_dir(),',
)
command_anchor = '''        "--initial-cooldown-seconds", "70",
    ]'''
assert command_anchor in paid["source"]
paid["source"] = paid["source"].replace(command_anchor, '''        "--initial-cooldown-seconds", "70",
    ]
    if ALLOW_FRESH_PAID_RUN:
        command.append("--allow-fresh-paid-run")''')
paid["source"] = paid["source"].replace(
    '        "checkpoint_dir": str(',
    '        "fresh_paid_run_acknowledged": ALLOW_FRESH_PAID_RUN,\n        "checkpoint_dir": str('
)
paid["source"] = paid["source"].replace(
    'Shard tạm dừng. Chờ ít nhất 70 giây rồi chạy lại cùng Cell 3; ',
    'Shard tạm dừng. Giữ output và đọc nguyên nhân trong log trước khi thử lại; ')

export = nb["cells"][7]
old = '''        if "chunk_text" in frame.columns:
            assert frame["chunk_text"].fillna("").str.strip().ne("").all()'''
new = '''        empty = frame["chunk_text"].isna() | frame["chunk_text"].astype(str).eq("")
        invalid = empty & ((stage == "small_merged") | frame[method_column].ne("page"))
        assert not invalid.any(), (stage, "unexpected empty chunks")
        assert not frame.duplicated(["doc_name", method_column, "chunk_index"]).any()
        for name, group in frame.groupby("doc_name"):
            assert set(group[method_column]) == expected_methods, (stage, name)
        print("Intermediate blank page chunks:", int(empty.sum()))'''
assert old in export["source"]
export["source"] = export["source"].replace(old, new)
export["source"] = export["source"].replace('    manifest_path, manifest = successful[-1]',
    '    manifest_path, manifest = max(successful, key=lambda item: item[0].stat().st_mtime)')
export["source"] = export["source"].replace('    if bundle_dir.exists():\n        shutil.rmtree(bundle_dir)',
    '    if bundle_dir.exists():\n        import uuid\n        bundle_dir.rename(bundle_dir.with_name(bundle_dir.name + "-previous-" + uuid.uuid4().hex[:8]))')
export["source"] = export["source"].replace(
    '    shutil.copytree(OUTPUT_DIR, bundle_dir / OUTPUT_DIR.name)',
    '''    shutil.copytree(OUTPUT_DIR, bundle_dir / OUTPUT_DIR.name)
    checkpoint_dir = OUTPUT_DIR.with_name(OUTPUT_DIR.name + "-checkpoints")
    if checkpoint_dir.is_dir():
        shutil.copytree(checkpoint_dir, bundle_dir / checkpoint_dir.name)'''
)
export["source"] = export["source"].replace('    import os\n    from IPython.display',
    '    import hashlib, zipfile\n    with zipfile.ZipFile(archive_path) as archive:\n        assert archive.testzip() is None\n    print("SHA256:", hashlib.sha256(archive_path.read_bytes()).hexdigest())\n    import os\n    from IPython.display')

def save(notebook, directory, slug, title):
    directory.mkdir(parents=True, exist_ok=True)
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(cell["source"], f"cell-{i}", "exec")
    (directory / "notebook.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = json.loads((BASE / "kaggle_cli_notebook/kernel-metadata.json").read_text(encoding="utf-8"))
    meta.pop("id_no", None)
    meta.update(id="datnguyenksnb/" + slug, title=title, code_file="notebook.ipynb",
                enable_gpu=False, machine_shape="Cpu")
    (directory / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

save(nb, BASE / "kaggle_recovery_corrected", "reproduction-adaptive-chunking", "reproduction-adaptive-chunking")

audit = json.loads(json.dumps(nb))
audit["cells"][0]["source"] = "# Week 2 recovery audit — NO PAID API CALLS\nSynthetic fixtures are only software tests; real outputs contain six free methods."
audit["cells"][5]["source"] = '''# NO API: test checkpoint recovery and process all 33 documents using six free methods.
import os, subprocess, sys, json, shutil
from pathlib import Path
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("JINA_API_KEY", None)
repo = Path("/tmp/adaptive-week2/adaptive-chunking")
python = Path("/tmp/adaptive-week2/venv/bin/python")
test_path = repo / "tests/test_reproduction_runner.py"
test_path.parent.mkdir(exist_ok=True)
test_path.write_text(TEST_SOURCE, encoding="utf-8")
subprocess.run([str(python), "-m", "pytest", str(test_path), "-q", "-p", "no:cacheprovider"], cwd=repo, check=True)
env = os.environ.copy()
env.update(USE_TF="0", USE_FLAX="0", TRANSFORMERS_NO_TF="1", TOKENIZERS_PARALLELISM="false",
           STANZA_RESOURCES_DIR="/tmp/adaptive-week2/nlp_data/stanza", NLTK_DATA="/tmp/adaptive-week2/nlp_data/nltk")
import pandas as pd
expected = {"page", "sentence", "langch_recurs_default", "langch_recurs_1100", "our_recurs_1100", "our_recurs_600"}
report = []
for shard in ["00", "01", "02"]:
    data = Path("/tmp/adaptive-week2/data_shards") / ("shard-" + shard)
    output = Path("/kaggle/working/free-audit-shard-" + shard)
    with Path(str(output) + ".log").open("w") as log:
        result = subprocess.run([str(python), "-m", "adaptive_chunking.paper.replicate", "--data-dir", str(data), "--output-dir", str(output), "--steps", "chunking", "--device", "cpu", "--skip-semantic", "--skip-llm-regex"], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, str(output) + ".log"
    for stage in ["raw", "no_oversizing", "small_merged"]:
        frame = pd.read_parquet(output / "chunks" / stage / "chunks.parquet")
        assert frame.doc_name.nunique() == 11
        assert not frame.duplicated(["doc_name", "method", "chunk_index"]).any()
        for name, group in frame.groupby("doc_name"):
            assert set(group.method) == expected, (shard, stage, name)
        empty = frame.chunk_text.isna() | frame.chunk_text.astype(str).eq("")
        invalid = empty & ((stage == "small_merged") | frame.method.ne("page"))
        assert not invalid.any(), (shard, stage, "empty chunk")
        report.append(dict(shard=shard, stage=stage, documents=11, rows=len(frame), empty_page_chunks=int(empty.sum())))
    print("FREE_SHARD_AUDIT_OK", shard, flush=True)
Path("/kaggle/working/free-audit-report.json").write_text(json.dumps(report, indent=2))
print("ALL_33_DOCUMENTS_FREE_AUDIT_OK", flush=True)
'''.replace('TEST_SOURCE', repr(tests))

# Exercise the exact corrected export cell against clearly labeled synthetic fixtures.
audit["cells"][7]["source"] = '''import pandas as pd, json
from pathlib import Path
work = Path("/kaggle/working")
methods = ["page", "sentence", "langch_recurs_default", "langch_recurs_1100", "our_recurs_1100", "our_recurs_600", "llm_regex"]
for stage in ["raw", "no_oversizing", "small_merged"]:
    target = work / "nonsemantic-shard-SYNTHETIC-TEST/chunks" / stage
    target.mkdir(parents=True, exist_ok=True)
    rows = [dict(doc_name=f"TEST-{i}", method=m, chunk_index=0, chunk_text="synthetic", chunk_len=1, chunk_pages=[1], titles_context="test") for i in range(11) for m in methods]
    if stage != "small_merged":
        rows[0]["chunk_text"] = ""
    pd.DataFrame(rows).to_parquet(target / "chunks.parquet")
(work / "nonsemantic-shard-SYNTHETIC-TEST-run.json").write_text(json.dumps(dict(status="success", shard_id="SYNTHETIC-TEST", synthetic=True)))
''' + export["source"] + '\nprint("EXPORT_CELL_TEST_OK_SYNTHETIC_ONLY")\n'
save(audit, BASE / "kaggle_recovery_audit", "week2-recovery-audit-no-api", "Week2 recovery audit no API")
print("Built corrected notebook and zero-API audit notebook")
