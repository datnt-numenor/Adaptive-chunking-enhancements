"""Tests for the low-cost reproduction helpers.

Heavy GPU/model code is intentionally not imported by these tests.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


RUNNER_PATH = (
    Path(__file__).parents[1] / "research" / "week2" / "reproduction_runner.py"
)
SPEC = importlib.util.spec_from_file_location("week2_reproduction_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _chunk_frame(doc_name: str, method: str = "page") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "doc_name": doc_name,
                "method": method,
                "type": "sync",
                "chunk_index": 0,
                "chunk_text": f"text for {doc_name}",
                "chunk_pages": [1],
                "titles_context": "title",
                "chunk_len": 5,
            }
        ]
    )


def _performance_frame(doc_name: str, method: str = "page") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "doc_name": doc_name,
                "method": method,
                "method_type": "sync",
                "time": 0.1,
            }
        ]
    )


def _write_chunk_root(root: Path, doc_name: str, method: str = "page") -> None:
    for stage in runner.STAGES:
        stage_dir = root / "chunks" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        _chunk_frame(doc_name, method).to_parquet(stage_dir / "chunks.parquet")
        _performance_frame(doc_name, method).to_parquet(
            stage_dir / "performances.parquet"
        )


def test_semantic_defaults_match_upstream_configuration():
    assert runner.SEMANTIC_MODEL == "Qwen/Qwen3-Embedding-0.6B"
    assert runner.SEMANTIC_CONFIG == {
        "attention_implementation": "flash_attention_2",
        "dtype": "bfloat16",
        "batch_size": 16,
        "breakpoint_threshold_type": "gradient",
    }
    args = runner.build_parser().parse_args(
        [
            "run-semantic-only",
            "--data-dir",
            "data",
            "--output-dir",
            "out",
            "--model-revision",
            "a" * 40,
        ]
    )
    assert args.attention_implementation == "flash_attention_2"
    assert args.dtype == "bfloat16"
    assert args.batch_size == 16

    upstream = (
        Path(__file__).parents[1]
        / "src"
        / "adaptive_chunking"
        / "paper"
        / "replicate.py"
    ).read_text(encoding="utf-8")
    for expected_source_fragment in [
        'model_name="Qwen/Qwen3-Embedding-0.6B"',
        '"attn_implementation": "flash_attention_2"',
        '"torch_dtype": torch.bfloat16',
        'encode_kwargs={"batch_size": 16}',
        'breakpoint_threshold_type="gradient"',
    ]:
        assert expected_source_fragment in upstream


def test_semantic_resume_rejects_configuration_mixing(tmp_path: Path):
    output = tmp_path / "semantic-output"
    raw = output / "chunks" / "raw"
    raw.mkdir(parents=True)
    _chunk_frame("doc-a", "semantic").to_parquet(raw / "chunks.parquet")
    source_documents = [
        {"doc_name": "doc-a", "tokens_o200k_base": 10, "sha256": "abc"}
    ]
    exact = {
        "source_commit": runner.PINNED_COMMIT,
        "model": runner.SEMANTIC_MODEL,
        "model_revision": "a" * 40,
        "attention_implementation": "flash_attention_2",
        "dtype": "bfloat16",
        "batch_size": 16,
        "breakpoint_threshold_type": "gradient",
    }
    (output / "semantic_run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "configuration": exact,
                "source_documents": source_documents,
            }
        ),
        encoding="utf-8",
    )
    fallback = {**exact, "attention_implementation": "sdpa", "dtype": "float16"}

    with pytest.raises(runner.ReproductionError, match="configuration mismatch"):
        runner._load_semantic_resume_manifest(
            output,
            resume=True,
            configuration=fallback,
            source_documents=source_documents,
        )


def test_semantic_resume_rejects_unverifiable_partial_artifacts(tmp_path: Path):
    output = tmp_path / "semantic-output"
    raw = output / "chunks" / "raw"
    raw.mkdir(parents=True)
    _chunk_frame("doc-a", "semantic").to_parquet(raw / "chunks.parquet")

    with pytest.raises(runner.ReproductionError, match="without semantic_run_manifest"):
        runner._load_semantic_resume_manifest(
            output,
            resume=True,
            configuration={},
            source_documents=[],
        )


def test_semantic_resume_rejects_dataset_mixing(tmp_path: Path):
    output = tmp_path / "semantic-output"
    output.mkdir()
    configuration = {key: "same" for key in runner.SEMANTIC_IDENTITY_KEYS}
    (output / "semantic_run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "configuration": configuration,
                "source_documents": [{"doc_name": "doc-a", "sha256": "old"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(runner.ReproductionError, match="dataset identity mismatch"):
        runner._load_semantic_resume_manifest(
            output,
            resume=True,
            configuration=configuration,
            source_documents=[{"doc_name": "doc-a", "sha256": "new"}],
        )


def test_semantic_run_requires_immutable_model_revision(tmp_path: Path):
    with pytest.raises(runner.ReproductionError, match="40-character"):
        runner.run_semantic_only(
            data_dir=tmp_path / "missing",
            output_dir=tmp_path / "output",
            device="cuda:0",
            model_revision="main",
            benchmark_documents=3,
            resume=False,
            attention_implementation="flash_attention_2",
            dtype_name="bfloat16",
            batch_size=16,
            max_runtime_hours=2,
        )


def test_balanced_assignments_are_deterministic_complete_and_unique():
    items = [(f"doc-{index}", weight) for index, weight in enumerate(range(1, 10))]
    first = runner.balanced_assignments(items, 3)
    second = runner.balanced_assignments(list(reversed(items)), 3)
    assert first == second
    flattened = [name for shard in first for name, _ in shard]
    assert sorted(flattened) == sorted(name for name, _ in items)
    assert len(flattened) == len(set(flattened))
    assert [len(shard) for shard in first] == [3, 3, 3]
    totals = [sum(weight for _, weight in shard) for shard in first]
    assert max(totals) - min(totals) <= max(weight for _, weight in items)


def test_make_data_shards_copies_every_document_once(tmp_path: Path):
    data_dir = tmp_path / "data"
    parsed_dir = data_dir / "adi_parsed"
    mentions_dir = data_dir / "mentions"
    parsed_dir.mkdir(parents=True)
    mentions_dir.mkdir()
    names = ["alpha", "beta", "gamma", "delta"]
    for index, name in enumerate(names, start=1):
        (parsed_dir / f"{name}.json").write_text(
            json.dumps({"full_text": (name + " ") * index}), encoding="utf-8"
        )
        (mentions_dir / f"{name}.json").write_text("{}", encoding="utf-8")

    output_dir = tmp_path / "shards"
    manifest = runner.make_data_shards(data_dir, output_dir, 2)

    assigned = [
        document["doc_name"]
        for shard in manifest["shards"]
        for document in shard["documents"]
    ]
    assert sorted(assigned) == sorted(names)
    assert len(assigned) == len(set(assigned))
    for shard in manifest["shards"]:
        shard_dir = Path(shard["directory"])
        assert len(list((shard_dir / "adi_parsed").glob("*.json"))) == shard[
            "document_count"
        ]
        assert len(list((shard_dir / "mentions").glob("*.json"))) == shard[
            "document_count"
        ]


def test_balanced_assignments_limit_six_shards_to_five_or_six_documents():
    items = [(f"doc-{index:02d}", 1000 - index) for index in range(33)]

    assignments = runner.balanced_assignments(items, 6)

    assert sorted(len(shard) for shard in assignments) == [5, 5, 5, 6, 6, 6]


def test_merge_chunks_preserves_rows_and_metadata(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_chunk_root(first, "doc-a")
    _write_chunk_root(second, "doc-b")

    output = tmp_path / "merged"
    manifest = runner.merge_chunks([first, second], output)

    assert {record["document_count"] for record in manifest["stages"]} == {2}
    for stage in runner.STAGES:
        merged = pd.read_parquet(output / "chunks" / stage / "chunks.parquet")
        assert sorted(merged["doc_name"].tolist()) == ["doc-a", "doc-b"]
        assert merged.set_index("doc_name").loc["doc-a", "titles_context"] == "title"
        assert merged.set_index("doc_name").loc["doc-b", "chunk_pages"] == [1]


def test_merge_chunks_rejects_duplicate_keys(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_chunk_root(first, "same-doc")
    _write_chunk_root(second, "same-doc")

    with pytest.raises(runner.ReproductionError, match="Duplicate keys"):
        runner.merge_chunks([first, second], tmp_path / "merged")


def test_empty_page_chunks_are_allowed_only_in_intermediate_stages():
    page_frame = _chunk_frame("doc-a", "page")
    page_frame.loc[0, "chunk_text"] = ""

    runner._validate_chunk_frame(
        page_frame,
        "raw chunks",
        allow_empty_page_chunks=True,
    )
    with pytest.raises(runner.ReproductionError, match="empty chunks"):
        runner._validate_chunk_frame(page_frame, "small_merged chunks")

    sentence_frame = page_frame.copy()
    sentence_frame.loc[0, "method"] = "sentence"
    with pytest.raises(runner.ReproductionError, match="sentence"):
        runner._validate_chunk_frame(
            sentence_frame,
            "raw chunks",
            allow_empty_page_chunks=True,
        )


def test_nonsemantic_runner_checkpoints_resumes_and_merges(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    data_dir = tmp_path / "data"
    parsed_dir = data_dir / "adi_parsed"
    parsed_dir.mkdir(parents=True)
    for name in ["alpha", "beta", "gamma"]:
        (parsed_dir / f"{name}.json").write_text(
            json.dumps({"full_text": name}), encoding="utf-8"
        )

    calls = []

    def fake_run_document(*, data_dir, output_dir, device, log_path, cache_dir):
        document = next((data_dir / "adi_parsed").glob("*.json"))
        calls.append(document.stem)
        for stage in runner.STAGES:
            stage_dir = output_dir / "chunks" / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            chunks = pd.concat(
                [
                    _chunk_frame(document.stem, method)
                    for method in runner.NONSEMANTIC_METHODS
                ],
                ignore_index=True,
            )
            chunks["chunk_index"] = range(len(chunks))
            if document.stem == "beta" and stage in {"raw", "no_oversizing"}:
                page_row = chunks.index[chunks["method"].eq("page")][0]
                chunks.loc[page_row, "chunk_text"] = ""
            chunks.to_parquet(stage_dir / "chunks.parquet")
            performances = pd.concat(
                [
                    _performance_frame(document.stem, method)
                    for method in runner.NONSEMANTIC_METHODS
                ],
                ignore_index=True,
            )
            performances.to_parquet(stage_dir / "performances.parquet")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok", encoding="utf-8")
        return 0

    output = tmp_path / "nonsemantic-shard-01"
    first = runner.run_nonsemantic_shard(
        data_dir,
        output,
        resume=True,
        min_interval_seconds=0,
        initial_cooldown_seconds=0,
        run_document=fake_run_document,
        allow_fresh_paid_run=True,
    )
    assert first["status"] == "success"
    assert calls == ["alpha", "beta", "gamma"]
    output.rename(tmp_path / "first-merged-output")

    second = runner.run_nonsemantic_shard(
        data_dir,
        output,
        resume=True,
        min_interval_seconds=0,
        initial_cooldown_seconds=0,
        run_document=fake_run_document,
    )
    assert second["status"] == "success"
    # The valid intermediate blank page must be recognized as a completed
    # checkpoint, so the paid document runner is never called a second time.
    assert calls == ["alpha", "beta", "gamma"]
    third = runner.run_nonsemantic_shard(
        data_dir, output, resume=True, run_document=fake_run_document,
        initial_cooldown_seconds=0, min_interval_seconds=0,
    )
    assert third["reused_existing_output"] is True
    assert calls == ["alpha", "beta", "gamma"]
    for stage in runner.STAGES:
        frame = pd.read_parquet(output / "chunks" / stage / "chunks.parquet")
        assert set(frame["doc_name"].unique()) == {"alpha", "beta", "gamma"}
        assert set(frame["method"].unique()) == runner.NONSEMANTIC_METHODS
        empty = frame["chunk_text"].astype(str).eq("")
        if stage in {"raw", "no_oversizing"}:
            assert set(frame.loc[empty, "method"]) == {"page"}
        else:
            assert not empty.any()


def test_partial_checkpoint_refuses_to_repeat_paid_call(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    data = tmp_path / "data" / "adi_parsed"
    data.mkdir(parents=True)
    (data / "doc-a.json").write_text('{}', encoding="utf-8")
    output = tmp_path / "shard"
    part = tmp_path / "shard-checkpoints" / "parts" / "doc-00" / "chunks" / "raw"
    part.mkdir(parents=True)
    _chunk_frame("doc-a", "llm_regex").to_parquet(part / "chunks.parquet")
    def forbidden_call(**kwargs):
        pytest.fail("Paid runner must not be called for ambiguous partial output")
    with pytest.raises(runner.ReproductionError, match="Refusing to repeat"):
        runner.run_nonsemantic_shard(
            data.parent, output, resume=True, run_document=forbidden_call,
            initial_cooldown_seconds=0, min_interval_seconds=0,
        )
    assert (part / "chunks.parquet").is_file()


def test_fresh_paid_run_requires_explicit_acknowledgement(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    parsed = tmp_path / "data" / "adi_parsed"
    parsed.mkdir(parents=True)
    (parsed / "doc-a.json").write_text('{"full_text":"x"}', encoding="utf-8")
    with pytest.raises(runner.ReproductionError, match="allow-fresh-paid-run"):
        runner.run_nonsemantic_shard(
            parsed.parent,
            tmp_path / "shard",
            resume=True,
            initial_cooldown_seconds=0,
            min_interval_seconds=0,
        )


def test_openai_response_cache_reuses_identical_request(monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace
    from research.week2 import cached_openai_replicate as cache_module

    calls = []
    response_payload = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "regex", "role": "assistant"},
                }
            ],
            "created": 1,
            "model": "gpt-4o",
            "object": "chat.completion",
        }

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload
            message = SimpleNamespace(content=payload["choices"][0]["message"]["content"])
            self.choices = [SimpleNamespace(message=message)]

        def model_dump(self, mode="json"):
            assert mode == "json"
            return self.payload

    class FakeChatCompletion:
        @classmethod
        def model_validate(cls, payload):
            return FakeResponse(payload)

    response = FakeResponse(response_payload)

    class FakeCompletions:
        async def create(self, *args, **kwargs):
            calls.append(kwargs)
            return response

    class FakeClient:
        def __init__(self, *args, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    fake_openai = SimpleNamespace(AsyncOpenAI=FakeClient)
    cache_module.install_cache(tmp_path, fake_openai, FakeChatCompletion)
    client = fake_openai.AsyncOpenAI()
    request = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "same prompt"}],
        "temperature": 0,
    }
    first = asyncio.run(client.chat.completions.create(**request))
    second = asyncio.run(client.chat.completions.create(**request))

    assert first.choices[0].message.content == "regex"
    assert second.choices[0].message.content == "regex"
    assert len(calls) == 1
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    record = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert record["request"]["parameters"] == request
    assert "api_key" not in cache_files[0].read_text(encoding="utf-8")


def test_make_metric_shards_uses_same_document_partition_for_all_stages(
    tmp_path: Path,
):
    chunks_root = tmp_path / "full"
    for stage in runner.STAGES:
        stage_dir = chunks_root / "chunks" / stage
        stage_dir.mkdir(parents=True)
        pd.concat(
            [_chunk_frame("doc-a"), _chunk_frame("doc-b")], ignore_index=True
        ).to_parquet(stage_dir / "chunks.parquet")

    output = tmp_path / "metric-shards"
    manifest = runner.make_metric_shards(chunks_root, output, 2)

    assigned = [name for shard in manifest["shards"] for name in shard["documents"]]
    assert sorted(assigned) == ["doc-a", "doc-b"]
    for shard in manifest["shards"]:
        expected = set(shard["documents"])
        shard_dir = output / f"shard-{shard['shard']:02d}" / "chunks"
        for stage in runner.STAGES:
            frame = pd.read_parquet(shard_dir / stage / "chunks.parquet")
            assert set(frame["doc_name"].unique()) == expected


def test_validate_nonsemantic_chunk_artifacts(tmp_path: Path):
    root = tmp_path / "nonsemantic"
    for stage in runner.STAGES:
        stage_dir = root / "chunks" / stage
        stage_dir.mkdir(parents=True)
        frame = pd.concat(
            [_chunk_frame("doc-a", method) for method in runner.NONSEMANTIC_METHODS],
            ignore_index=True,
        )
        frame["chunk_index"] = range(len(frame))
        frame.to_parquet(stage_dir / "chunks.parquet")

    report = runner.validate_artifacts(root, expected_docs=1, profile="nonsemantic")

    assert report["chunks"]["raw"]["documents"] == 1
    assert set(report["chunks"]["raw"]["methods"]) == runner.NONSEMANTIC_METHODS
    assert (root / "validation_report.json").is_file()


def test_run_metrics_refuses_implicit_jina_usage(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("JINA_API_KEY", "not-a-real-secret")
    with pytest.raises(runner.ReproductionError, match="JINA_API_KEY is set"):
        runner.run_metrics_shard(
            kind="processed",
            chunks_parquet=tmp_path / "missing.parquet",
            data_dir=tmp_path,
            output_dir=tmp_path / "out",
            device="cuda:0",
            batch_size=32,
            allow_jina_api=False,
        )


def test_merge_metrics_preserves_scores_and_rows(tmp_path: Path):
    processed_roots = []
    raw_roots = []
    expected_scores = {}
    for index, doc_name in enumerate(["doc-a", "doc-b"]):
        processed_root = tmp_path / f"processed-{index}" / "results"
        raw_root = tmp_path / f"raw-{index}" / "results_raw"
        processed_root.mkdir(parents=True)
        raw_root.mkdir(parents=True)
        score = 0.25 + index
        expected_scores[doc_name] = score
        metric_frame = pd.DataFrame(
            [
                {
                    "doc_name": doc_name,
                    "chunking_method": "page",
                    "metric_name": "block_integrity",
                    "score": score,
                }
            ]
        )
        perf_frame = pd.DataFrame(
            [{"doc_name": doc_name, "metric": "block_integrity", "time": 0.1}]
        )
        metric_frame.to_parquet(processed_root / "chunking_metrics.parquet")
        perf_frame.to_parquet(processed_root / "metrics_performance.parquet")
        metric_frame.to_parquet(raw_root / "chunking_metrics.parquet")
        perf_frame.to_parquet(raw_root / "metrics_performance.parquet")
        processed_roots.append(processed_root.parent)
        raw_roots.append(raw_root.parent)

    output = tmp_path / "final"
    manifest = runner.merge_metrics(processed_roots, raw_roots, output, None)

    assert manifest["processed"]["rows"] == 2
    assert manifest["raw"]["rows"] == 2
    for directory in ["results", "results_raw"]:
        merged = pd.read_parquet(output / directory / "chunking_metrics.parquet")
        actual_scores = merged.set_index("doc_name")["score"].to_dict()
        assert actual_scores == expected_scores


def test_metrics_for_a_document_are_independent_of_other_shard_documents(
    tmp_path: Path,
):
    from adaptive_chunking.compute_metrics import compute_metrics_per_origin

    class DeterministicEmbedder:
        def encode(self, texts, **_kwargs):
            rows = []
            for value in texts:
                vector = np.array(
                    [len(value), sum(map(ord, value)) % 997, value.count(" ") + 1],
                    dtype=float,
                )
                rows.append(vector / np.linalg.norm(vector))
            return np.vstack(rows)

    parsed_dir = tmp_path / "parsed"
    mentions_dir = tmp_path / "mentions"
    parsed_dir.mkdir()
    mentions_dir.mkdir()
    chunks = []
    for doc_name, parts in {
        "doc-a": ["Alpha first sentence. ", "Alpha second sentence."],
        "doc-b": ["Beta first sentence. ", "Beta second sentence."],
    }.items():
        full_text = "".join(parts)
        first_boundary = len(parts[0])
        (parsed_dir / f"{doc_name}.json").write_text(
            json.dumps(
                {
                    "full_text": full_text,
                    "split_points": [0, first_boundary, len(full_text)],
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [{"doc_name": doc_name, "entity_pron_mentions": []}]
        ).to_parquet(mentions_dir / f"{doc_name}.parquet")
        for chunk_index, chunk_text in enumerate(parts):
            chunks.append(
                {
                    "doc_name": doc_name,
                    "method": "page",
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "chunk_len": len(chunk_text.split()),
                }
            )

    all_chunks = pd.DataFrame(chunks)
    single_input = tmp_path / "single-input"
    shard_input = tmp_path / "shard-input"
    single_input.mkdir()
    shard_input.mkdir()
    all_chunks[all_chunks.doc_name == "doc-a"].to_parquet(
        single_input / "chunks.parquet"
    )
    all_chunks.to_parquet(shard_input / "chunks.parquet")

    common = {
        "mentions_dir": mentions_dir,
        "parsed_docs_dir": parsed_dir,
        "models": {"sentence_embedder": DeterministicEmbedder()},
        "batch_size": 32,
    }
    compute_metrics_per_origin(
        chunks_dir=single_input, output_dir=tmp_path / "single-output", **common
    )
    compute_metrics_per_origin(
        chunks_dir=shard_input, output_dir=tmp_path / "shard-output", **common
    )

    single = pd.read_parquet(tmp_path / "single-output/chunking_metrics.parquet")
    sharded = pd.read_parquet(tmp_path / "shard-output/chunking_metrics.parquet")
    sharded = sharded[sharded.doc_name == "doc-a"]
    key = ["doc_name", "chunking_method", "metric_name"]
    single = single.sort_values(key).reset_index(drop=True)
    sharded = sharded.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(single[key], sharded[key])
    np.testing.assert_allclose(
        single["score"].to_numpy(),
        sharded["score"].to_numpy(),
        rtol=0,
        atol=1e-6,
        equal_nan=True,
    )
