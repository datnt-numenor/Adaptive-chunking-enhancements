"""Structure-aware DP chunker (SDP).

Chunks are cut only at parser block boundaries (``split_points``), and the set of
cut points is chosen by dynamic programming over a per-chunk score.

Version A (this module's default) uses structural signals only, no embeddings:

* every chunk costs ``tau`` -> the DP prefers few, large chunks (good for DCC, RC);
* a cut right before a Markdown heading earns a bonus (good for ICC: chunks
  follow sections);
* a cut before a block that continues the previous one (lower-case start, i.e.
  a paragraph broken by a page break) or opens with a pronoun is penalised;
* page artefact blocks (``<!-- PageBreak -->``, page header/footer comments)
  should not start a chunk (penalised, so they stay with the preceding content);
* chunk size is hard-capped at ``max_tokens``; chunks under ``min_tokens`` get a
  large penalty (soft, so the DP is always feasible).

Version B plugs an embedding term in through ``extra_chunk_score``.

Blocks longer than ``max_tokens`` are pre-split into pieces (the only place the
chunker can break a parser block).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s")
_LIST_RE = re.compile(r"^(?:[-*+•●▪◦‣▸▹○]\s|\(?[A-Za-z0-9]{1,4}[.)]\s)")
_PRONOUN_STARTS = {
    "it", "its", "this", "these", "those", "they", "them", "their", "such",
    "he", "she", "his", "her", "that", "which", "there",
}


@dataclass(frozen=True)
class Unit:
    """A contiguous span of the document that the chunker never cuts inside."""

    start: int
    end: int
    tokens: int
    is_artifact: bool = False      # only page-layout comments / whitespace
    heading_level: int = 0         # 1..6 if the content starts with a Markdown heading
    starts_lower: bool = False     # continuation of the previous paragraph
    starts_pronoun: bool = False
    starts_list: bool = False
    ends_colon: bool = False       # e.g. "the following:" introducing a list
    internal: bool = False         # piece of a pre-split oversized block


@dataclass
class SDPConfig:
    max_tokens: int = 1100
    min_tokens: int = 100
    tau: float = 1.0                                  # cost per chunk
    heading_bonus: Sequence[float] = (0.6, 0.5, 0.4, 0.3, 0.3, 0.3)
    continuation_penalty: float = 3.0
    pronoun_penalty: float = 0.5
    colon_penalty: float = 0.5
    list_penalty: float = 0.3                         # cut between two list items
    small_penalty: float = 5.0                        # per chunk under min_tokens
    artifact_penalty: float = 2.0                     # chunk starting with a page header/footer/break
    target_tokens: int | None = None                  # if set, penalise |size - target| / target
    target_weight: float = 1.0
    max_units_per_chunk: int = 2000
    extra_chunk_score: Callable[[int, int], float] | None = field(default=None, repr=False)


def _content_head(text: str) -> str:
    return _COMMENT_RE.sub("", text).strip()


def _describe(text: str, start: int, end: int, tokens: int, internal: bool = False) -> Unit:
    head = _content_head(text)
    if not head:
        return Unit(start, end, tokens, is_artifact=True, internal=internal)
    m = _HEADING_RE.match(head)
    first_word = re.match(r"[A-Za-z]+", head)
    first_alpha = next((c for c in head if c.isalpha()), "")
    return Unit(
        start=start,
        end=end,
        tokens=tokens,
        heading_level=len(m.group(1)) if m else 0,
        starts_lower=bool(not m and first_alpha and first_alpha.islower() and head[0].isalpha()),
        starts_pronoun=bool(first_word and first_word.group(0).lower() in _PRONOUN_STARTS),
        starts_list=bool(_LIST_RE.match(head)),
        ends_colon=head.rstrip().endswith(":"),
        internal=internal,
    )


def build_units(
    full_text: str,
    split_points: Sequence[int],
    count_tokens: Callable[[str], int],
    max_tokens: int,
    oversize_splitter: Callable[[str], list[str]] | None = None,
) -> list[Unit]:
    """Turn parser blocks into units; pre-split blocks over ``max_tokens``."""
    bounds = [0, *sorted({p for p in split_points if 0 < p < len(full_text)}), len(full_text)]
    units: list[Unit] = []
    for a, b in zip(bounds, bounds[1:]):
        text = full_text[a:b]
        tokens = count_tokens(text)
        if tokens <= max_tokens or oversize_splitter is None:
            units.append(_describe(text, a, b, tokens))
            continue
        pos = a
        for piece in oversize_splitter(text):
            idx = full_text.find(piece, pos, b)
            if idx == -1 or not piece:
                continue
            if idx > pos:  # glue any skipped whitespace to the previous piece
                if units and units[-1].end == pos:
                    last = units.pop()
                    units.append(_describe(full_text[last.start:idx], last.start, idx,
                                           count_tokens(full_text[last.start:idx]), True))
                else:
                    piece = full_text[pos:idx] + piece
                    idx = pos
            end = idx + len(piece)
            units.append(_describe(full_text[idx:end], idx, end, count_tokens(full_text[idx:end]), True))
            pos = end
        if pos < b:  # trailing remainder
            if units and units[-1].end == pos:
                last = units.pop()
                units.append(_describe(full_text[last.start:b], last.start, b,
                                       count_tokens(full_text[last.start:b]), True))
            else:
                units.append(_describe(full_text[pos:b], pos, b, count_tokens(full_text[pos:b]), True))
    return units


def boundary_score(units: Sequence[Unit], j: int, cfg: SDPConfig) -> float | None:
    """Score of cutting right before unit ``j``; ``None`` if the cut is not allowed."""
    u = units[j]
    if u.is_artifact:
        return -cfg.artifact_penalty
    if u.heading_level:
        return cfg.heading_bonus[u.heading_level - 1]
    score = 0.0
    if u.starts_lower:
        score -= cfg.continuation_penalty
    if u.starts_pronoun:
        score -= cfg.pronoun_penalty
    prev = next((units[k] for k in range(j - 1, -1, -1) if not units[k].is_artifact), None)
    if prev is not None:
        if prev.ends_colon:
            score -= cfg.colon_penalty
        if prev.starts_list and u.starts_list:
            score -= cfg.list_penalty
    return score


def dp_segment(units: Sequence[Unit], cfg: SDPConfig) -> list[int]:
    """Return chunk start indices (into ``units``), always beginning with 0."""
    n = len(units)
    if n == 0:
        return []
    prefix = [0]
    for u in units:
        prefix.append(prefix[-1] + u.tokens)
    allowed = [0.0] + [boundary_score(units, j, cfg) for j in range(1, n)]
    allowed.append(0.0)  # end of document is always a valid end

    neg_inf = float("-inf")
    best = [neg_inf] * (n + 1)
    back = [-1] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        if allowed[j] is None:
            continue
        i = j - 1
        while i >= 0 and j - i <= cfg.max_units_per_chunk:
            size = prefix[j] - prefix[i]
            if size > cfg.max_tokens and i < j - 1:
                break
            if allowed[i] is not None and best[i] > neg_inf:
                score = best[i] - cfg.tau + (allowed[i] if i > 0 else 0.0)
                if size < cfg.min_tokens:
                    score -= cfg.small_penalty
                if cfg.target_tokens:
                    score -= cfg.target_weight * abs(size - cfg.target_tokens) / cfg.target_tokens
                if cfg.extra_chunk_score is not None:
                    score += cfg.extra_chunk_score(i, j)
                if score > best[j]:
                    best[j], back[j] = score, i
            i -= 1
    if back[n] == -1:
        raise RuntimeError("DP found no feasible segmentation")
    starts, j = [], n
    while j > 0:
        j = back[j]
        starts.append(j)
    return starts[::-1]


def sdp_chunk(
    full_text: str,
    split_points: Sequence[int],
    count_tokens: Callable[[str], int],
    cfg: SDPConfig | None = None,
    oversize_splitter: Callable[[str], list[str]] | None = None,
    units: Sequence[Unit] | None = None,
) -> list[str]:
    """Chunk ``full_text``; the concatenation of the returned chunks equals ``full_text``."""
    cfg = cfg or SDPConfig()
    if not full_text:
        return []
    if units is None:
        units = build_units(full_text, split_points, count_tokens, cfg.max_tokens, oversize_splitter)
    starts = dp_segment(units, cfg)
    offsets = [units[s].start for s in starts] + [len(full_text)]
    offsets[0] = 0
    return [full_text[a:b] for a, b in zip(offsets, offsets[1:]) if b > a]
