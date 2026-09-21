"""Audit the bundled CLAIR data and the raw-page Block Integrity signal."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from adaptive_chunking.metrics import compute_block_integrity
from adaptive_chunking.paper.replicate import count_tokens_func


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(data_dir: Path, output_path: Path) -> dict:
    parsed_dir = data_dir / "adi_parsed"
    mentions_dir = data_dir / "mentions"
    parsed_paths = sorted(parsed_dir.glob("*.json"))
    mention_json_stems = {path.stem for path in mentions_dir.glob("*.json")}
    mention_parquet_stems = {path.stem for path in mentions_dir.glob("*.parquet")}
    parsed_stems = {path.stem for path in parsed_paths}

    domain_counts: Counter[str] = Counter()
    domain_tokens: Counter[str] = Counter()
    documents: list[dict] = []
    schema_errors: list[dict] = []
    raw_page_bi_scores: list[float] = []
    page_boundary_total = 0
    page_boundaries_matching_parser_splits = 0

    required_keys = {"pages", "full_text", "split_points", "titles"}
    for path in parsed_paths:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        missing_keys = sorted(required_keys - document.keys())
        pages = list(document.get("pages", {}).values())
        full_text = document.get("full_text", "")
        split_points = document.get("split_points", [])
        domain = path.stem.split("_", 1)[0]
        tokens = count_tokens_func(full_text)

        if missing_keys:
            schema_errors.append({"file": path.name, "missing_keys": missing_keys})
        if "".join(pages) != full_text:
            schema_errors.append({"file": path.name, "error": "pages_do_not_concat_to_full_text"})
        if split_points != sorted(set(split_points)):
            schema_errors.append({"file": path.name, "error": "split_points_not_strictly_sorted_unique"})
        if any(point < 0 or point > len(full_text) for point in split_points):
            schema_errors.append({"file": path.name, "error": "split_point_out_of_bounds"})

        bi = compute_block_integrity(
            chunks=pages,
            doc_split_points=split_points,
            full_text=full_text,
            tolerance_chars=5,
        )
        if bi is not None:
            raw_page_bi_scores.append(float(bi))

        cumulative = 0
        page_boundaries: list[int] = []
        for page in pages[:-1]:
            cumulative += len(page)
            page_boundaries.append(cumulative)
        matched = sum(
            1
            for boundary in page_boundaries
            if any(abs(boundary - split_point) <= 5 for split_point in split_points)
        )
        page_boundary_total += len(page_boundaries)
        page_boundaries_matching_parser_splits += matched

        domain_counts[domain] += 1
        domain_tokens[domain] += tokens
        documents.append(
            {
                "document_id": path.stem,
                "sha256": _sha256(path),
                "tokens_o200k_base": tokens,
                "pages": len(pages),
                "parser_split_points": len(split_points),
                "raw_page_block_integrity": bi,
                "page_boundaries": len(page_boundaries),
                "page_boundaries_matching_parser_splits_within_5_chars": matched,
            }
        )

    result = {
        "dataset": {
            "parsed_documents": len(parsed_paths),
            "total_tokens_o200k_base": sum(domain_tokens.values()),
            "documents_per_domain": dict(domain_counts),
            "tokens_per_domain": dict(domain_tokens),
            "schema_errors": schema_errors,
            "mention_json_count": len(mention_json_stems),
            "mention_parquet_count": len(mention_parquet_stems),
            "parsed_without_mention_json": sorted(parsed_stems - mention_json_stems),
            "parsed_without_mention_parquet": sorted(parsed_stems - mention_parquet_stems),
            "mention_json_without_parsed_document": sorted(mention_json_stems - parsed_stems),
            "mention_parquet_without_parsed_document": sorted(mention_parquet_stems - parsed_stems),
        },
        "block_integrity_audit": {
            "definition": "repository compute_block_integrity on raw page chunks",
            "documents_scored": len(raw_page_bi_scores),
            "documents_with_bi_1_0": sum(score == 1.0 for score in raw_page_bi_scores),
            "min_bi": min(raw_page_bi_scores) if raw_page_bi_scores else None,
            "mean_bi": (
                sum(raw_page_bi_scores) / len(raw_page_bi_scores)
                if raw_page_bi_scores
                else None
            ),
            "page_boundaries_total": page_boundary_total,
            "page_boundaries_matching_parser_splits_within_5_chars": page_boundaries_matching_parser_splits,
            "page_boundary_match_rate": (
                page_boundaries_matching_parser_splits / page_boundary_total
                if page_boundary_total
                else None
            ),
        },
        "documents": documents,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/clair"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/week2/artifacts/dataset_bi_audit.json"),
    )
    args = parser.parse_args()
    result = run(args.data_dir, args.output)
    print(json.dumps({key: result[key] for key in ("dataset", "block_integrity_audit")}, indent=2))


if __name__ == "__main__":
    main()
