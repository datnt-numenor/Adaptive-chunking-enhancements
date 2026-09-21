"""Prepare an auditable Codex-assisted review proposal for the full QA set.

This script never freezes QA. It accepts already exact evidence provisionally,
repairs invalid evidence with exact source paragraphs when confidence is high,
and emits a report for final human approval.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import unicodedata


PUNCT_TRANSLATION = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "‐": "-",
        "‑": "-",
        "…": "...",
        "\u00a0": " ",
    }
)
STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "and",
    "are",
    "because",
    "been",
    "being",
    "between",
    "but",
    "can",
    "could",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "into",
    "its",
    "may",
    "more",
    "not",
    "that",
    "the",
    "their",
    "these",
    "they",
    "this",
    "through",
    "under",
    "was",
    "were",
    "what",
    "when",
    "which",
    "while",
    "with",
    "would",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(PUNCT_TRANSLATION).lower()
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", normalize(text))
        if token not in STOPWORDS
    }


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
        spans.append((match.start(), match.end()))
    return spans


def evidence_text(candidate: dict) -> str:
    evidence = candidate.get("evidence") or candidate.get("original_evidence") or []
    return "\n".join(item["quote"] for item in evidence)


def answer_support_coverage(candidate: dict, quote: str | None = None) -> float:
    answer_terms = tokens(candidate["reference_answer"])
    if not answer_terms:
        return 0.0
    evidence_terms = tokens(quote if quote is not None else evidence_text(candidate))
    return len(answer_terms & evidence_terms) / len(answer_terms)


def best_exact_source_window(original_item: dict, source: dict) -> dict:
    original = original_item["quote"]
    original_norm = normalize(original)
    claimed_pages = [str(original_item["page"])]
    best: dict | None = None
    original_terms = tokens(original)

    def scan_pages(pages: list[str]) -> None:
        nonlocal best
        for page in pages:
            page_text = source["pages"][page]
            spans = paragraph_spans(page_text)
            for start_index in range(len(spans)):
                for width in range(1, min(4, len(spans) - start_index + 1)):
                    start = spans[start_index][0]
                    end = spans[start_index + width - 1][1]
                    quote = page_text[start:end].strip()
                    quote_norm = normalize(quote)
                    if not quote_norm:
                        continue
                    similarity = SequenceMatcher(
                        None, original_norm, quote_norm, autojunk=False
                    ).ratio()
                    overlap = (
                        len(original_terms & tokens(quote)) / len(original_terms)
                        if original_terms
                        else 0.0
                    )
                    score = 0.75 * similarity + 0.25 * overlap
                    row = {
                        "page": int(page),
                        "quote": quote,
                        "similarity": similarity,
                        "token_overlap": overlap,
                        "score": score,
                        "claimed_page": page in claimed_pages,
                    }
                    if best is None or row["score"] > best["score"]:
                        best = row

    claimed_existing = [page for page in dict.fromkeys(claimed_pages) if page in source["pages"]]
    scan_pages(claimed_existing)
    if best is None or best["score"] < 0.55:
        scan_pages([page for page in source["pages"] if page not in claimed_existing])
    if best is None:
        raise RuntimeError("No source paragraph found for evidence item")
    return best


def repair_evidence(candidate: dict, source: dict) -> dict:
    """Repair each generated quote independently and preserve multi-quote support."""
    matches = [
        best_exact_source_window(item, source)
        for item in candidate["original_evidence"][:3]
    ]
    evidence: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for match in matches:
        key = (match["page"], match["quote"])
        if key not in seen:
            evidence.append({"page": match["page"], "quote": match["quote"]})
            seen.add(key)
    return {
        "evidence": evidence,
        "matches": matches,
        "minimum_score": min(match["score"] for match in matches),
        "mean_score": sum(match["score"] for match in matches) / len(matches),
    }


def duplicate_flags(candidates: list[dict]) -> dict[str, list[str]]:
    flags = {candidate["qa_id"]: [] for candidate in candidates}
    by_document: dict[str, list[dict]] = {}
    for candidate in candidates:
        by_document.setdefault(candidate["doc_id"], []).append(candidate)
    for rows in by_document.values():
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                ratio = SequenceMatcher(
                    None,
                    normalize(left["question"]),
                    normalize(right["question"]),
                    autojunk=False,
                ).ratio()
                if ratio >= 0.8:
                    message = f"possible duplicate question ({ratio:.3f})"
                    flags[left["qa_id"]].append(f"{message}: {right['qa_id']}")
                    flags[right["qa_id"]].append(f"{message}: {left['qa_id']}")
    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--overrides", type=Path)
    args = parser.parse_args()

    candidates = read_jsonl(args.qa_dir / "qa_candidates.jsonl")
    documents = {
        row["doc_id"]: row
        for row in read_jsonl(args.prepared_dir / "documents.jsonl")
    }
    duplicates = duplicate_flags(candidates)
    overrides = (
        json.loads(args.overrides.read_text(encoding="utf-8"))
        if args.overrides is not None
        else {}
    )
    decisions: list[dict] = []
    report_rows: list[dict] = []

    for candidate in candidates:
        flags = list(duplicates[candidate["qa_id"]])
        support = answer_support_coverage(candidate)
        if support < 0.35:
            flags.append(f"low lexical answer/evidence coverage ({support:.3f})")

        decision = {
            "qa_id": candidate["qa_id"],
            "status": "accept",
            "notes": "Codex-assisted proposal; requires final human approval.",
        }
        repair = None
        if candidate["validation_status"] != "valid":
            document = documents[candidate["doc_id"]]
            source_path = args.repo_root / document["source_path"]
            source = json.loads(source_path.read_text(encoding="utf-8"))
            repair = repair_evidence(candidate, source)
            repaired_quote = "\n".join(item["quote"] for item in repair["evidence"])
            repaired_support = answer_support_coverage(candidate, repaired_quote)
            if repair["minimum_score"] < 0.55 or repaired_support < 0.35:
                flags.append(
                    "repair requires manual inspection "
                    f"(match={repair['minimum_score']:.3f}, support={repaired_support:.3f})"
                )
            decision.update(
                {
                    "status": "edit",
                    "edited_question": candidate["question"],
                    "edited_answer": candidate["reference_answer"],
                    "edited_evidence_json": json.dumps(
                        repair["evidence"],
                        ensure_ascii=False,
                    ),
                    "notes": (
                        "Codex-assisted exact-source evidence repair; "
                        f"minimum_match={repair['minimum_score']:.3f}; "
                        "requires final human approval."
                    ),
                }
            )
        override = overrides.get(candidate["qa_id"])
        if override is not None:
            override_status = override.get("status", "edit")
            if override_status == "accept":
                if candidate["validation_status"] != "valid":
                    raise ValueError(
                        f"Cannot accept invalid evidence: {candidate['qa_id']}"
                    )
                decision = {
                    "qa_id": candidate["qa_id"],
                    "status": "accept",
                    "notes": override["notes"],
                }
            elif override_status == "edit":
                evidence = override.get("evidence")
                if evidence is None:
                    evidence = (
                        [
                            {"page": item["page"], "quote": item["quote"]}
                            for item in candidate["evidence"]
                        ]
                        if candidate["validation_status"] == "valid"
                        else repair["evidence"]
                    )
                decision = {
                    "qa_id": candidate["qa_id"],
                    "status": "edit",
                    "edited_question": override.get(
                        "question", candidate["question"]
                    ),
                    "edited_answer": override.get(
                        "answer", candidate["reference_answer"]
                    ),
                    "edited_evidence_json": json.dumps(
                        evidence, ensure_ascii=False
                    ),
                    "notes": override["notes"],
                }
            else:
                raise ValueError(
                    f"Unsupported override status for {candidate['qa_id']}: "
                    f"{override_status}"
                )
        decisions.append(decision)
        report_rows.append(
            {
                "qa_id": candidate["qa_id"],
                "doc_name": candidate["doc_name"],
                "question": candidate["question"],
                "reference_answer": candidate["reference_answer"],
                "original_status": candidate["validation_status"],
                "support_coverage": support,
                "flags": flags,
                "repair": repair,
                "manual_resolution": override,
            }
        )

    proposal = {
        "schema_version": 1,
        "reviewer": "codex-assisted-proposal",
        "reviewed_at": None,
        "requires_final_human_approval": True,
        "decisions": decisions,
    }
    (args.qa_dir / "qa_review_proposed.json").write_text(
        json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.qa_dir / "qa_review_audit.json").write_text(
        json.dumps(report_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "candidates": len(candidates),
        "proposed_accept": sum(row["status"] == "accept" for row in decisions),
        "proposed_edit": sum(row["status"] == "edit" for row in decisions),
        "flagged": sum(bool(row["flags"]) for row in report_rows),
        "unresolved_flagged": sum(
            bool(row["flags"]) and row["manual_resolution"] is None
            for row in report_rows
        ),
        "low_confidence_repairs": sum(
            row["repair"] is not None
            and (
                row["repair"]["minimum_score"] < 0.55
                or answer_support_coverage(
                    next(c for c in candidates if c["qa_id"] == row["qa_id"]),
                    "\n".join(item["quote"] for item in row["repair"]["evidence"]),
                )
                < 0.35
            )
            for row in report_rows
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
