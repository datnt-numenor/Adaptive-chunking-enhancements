"""Run a zero-API, CPU-only smoke test of the repository's core pipeline.

This intentionally exercises only the page splitter and the two recursive
candidate splitters. It does not claim to reproduce Tables 1--5.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from adaptive_chunking.metrics import compute_block_integrity, compute_size_compliance
from adaptive_chunking.paper.replicate import SEPARATORS, count_tokens_func
from adaptive_chunking.split_documents import split_documents_from_dir
from adaptive_chunking.splitters import RecursiveSplitter


def _smallest_documents(parsed_docs_dir: Path, count: int) -> list[tuple[Path, int]]:
    candidates: list[tuple[int, Path]] = []
    for path in parsed_docs_dir.glob("*.json"):
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        candidates.append((count_tokens_func(document["full_text"]), path))
    if not candidates:
        raise FileNotFoundError(f"No parsed JSON documents found in {parsed_docs_dir}")
    return [(path, token_count) for token_count, path in sorted(candidates)[:count]]


async def run(data_dir: Path, output_dir: Path, document_count: int) -> dict:
    parsed_docs_dir = data_dir / "adi_parsed"
    sources = _smallest_documents(parsed_docs_dir, document_count)

    splitters = {
        "page": None,
        "our_recurs_600": RecursiveSplitter(
            separators=SEPARATORS,
            chunk_size=600,
            chunk_overlap=0,
            is_separator_regex=True,
            attach_separator_to="start",
            length_function=count_tokens_func,
            merging="to_chunk_size",
            merging_order="forward",
        ),
        "our_recurs_1100": RecursiveSplitter(
            separators=SEPARATORS,
            chunk_size=1100,
            chunk_overlap=0,
            is_separator_regex=True,
            attach_separator_to="start",
            length_function=count_tokens_func,
            merging="to_chunk_size",
            merging_order="forward",
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="adaptive_chunking_smoke_") as temp_dir:
        one_doc_dir = Path(temp_dir) / "adi_parsed"
        one_doc_dir.mkdir()
        for source_path, _ in sources:
            shutil.copy2(source_path, one_doc_dir / source_path.name)
        await split_documents_from_dir(
            parsed_docs_dir=one_doc_dir,
            sync_splitters=splitters,
            async_splitters={},
            output_dir=output_dir,
            count_tokens_func=count_tokens_func,
            skip_non_english=False,
            replace_all_results=True,
        )

    chunks_df = pd.read_parquet(output_dir / "chunks.parquet")

    documents: dict[str, dict] = {}
    token_counts = {path.stem: tokens for path, tokens in sources}
    source_paths = {path.stem: path for path, _ in sources}
    for doc_name, doc_rows in chunks_df.groupby("doc_name"):
        with source_paths[doc_name].open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        methods: dict[str, dict] = {}
        for method, rows in doc_rows.groupby("method"):
            ordered = rows.sort_values("chunk_index")
            chunks = ordered["chunk_text"].tolist()
            methods[method] = {
                "num_chunks": len(chunks),
                "min_tokens": int(ordered["chunk_len"].min()),
                "mean_tokens": float(ordered["chunk_len"].mean()),
                "max_tokens": int(ordered["chunk_len"].max()),
                "size_compliance_100_1100": compute_size_compliance(
                    chunks, min_tokens=100, max_tokens=1100
                ),
                "block_integrity": compute_block_integrity(
                    chunks=chunks,
                    doc_split_points=source["split_points"],
                    full_text=source["full_text"],
                    tolerance_chars=5,
                ),
                "all_chunks_have_page_metadata": bool(
                    ordered["chunk_pages"].map(lambda value: len(value) > 0).all()
                ),
            }
        documents[doc_name] = {
            "source_tokens_o200k_base": token_counts[doc_name],
            "methods": methods,
        }

    result = {
        "scope": "core smoke test; not a paper reproduction",
        "document_selection": f"{len(sources)} smallest documents by o200k_base token count",
        "documents": documents,
    }
    result_path = output_dir / "smoke_summary.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/clair"))
    parser.add_argument("--documents", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/week2/artifacts/core_smoke"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(run(args.data_dir, args.output_dir, args.documents)), indent=2
        )
    )


if __name__ == "__main__":
    main()
