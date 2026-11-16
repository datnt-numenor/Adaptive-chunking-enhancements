import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sdp_chunker import SDPConfig, build_units, dp_segment, sdp_chunk  # noqa: E402


def words(text: str) -> int:
    return len(text.split())


def make_doc(blocks: list[str]) -> tuple[str, list[int]]:
    text, splits = "", []
    for b in blocks:
        if text:
            splits.append(len(text))
        text += b
    return text, splits


def para(n: int, word: str = "lorem") -> str:
    return " ".join([word] * n) + "\n\n"


def test_concatenation_equals_text_and_cuts_only_at_split_points():
    blocks = [f"# S{i}\n\n" + para(30) for i in range(10)]
    text, splits = make_doc(blocks)
    cfg = SDPConfig(max_tokens=100, min_tokens=20)
    chunks = sdp_chunk(text, splits, words, cfg)
    assert "".join(chunks) == text
    pos, starts = 0, []
    for c in chunks:
        starts.append(pos)
        pos += len(c)
    assert set(starts[1:]) <= set(splits)
    assert all(words(c) <= 100 for c in chunks)


def test_prefers_cutting_at_headings():
    # 4 blocks of 40 words; heading on block 2 -> best 2-chunk split is [0,1] | [2,3]
    blocks = [para(40), para(40), "# H\n\n" + para(39), para(40)]
    text, splits = make_doc(blocks)
    cfg = SDPConfig(max_tokens=100, min_tokens=10)
    chunks = sdp_chunk(text, splits, words, cfg)
    assert len(chunks) == 2
    assert chunks[1].startswith("# H")


def test_never_cuts_before_continuation_when_avoidable():
    blocks = [para(40, "Alpha"), "continued " + para(39), para(40, "Gamma"), para(40, "Delta")]
    text, splits = make_doc(blocks)
    chunks = sdp_chunk(text, splits, words, SDPConfig(max_tokens=100, min_tokens=10))
    assert not any(c.startswith("continued") for c in chunks)


def test_artifacts_never_start_a_chunk():
    blocks = [para(60), "<!-- PageBreak -->\n\n", "<!-- PageHeader: X -->\n\n", para(60, "Next")]
    text, splits = make_doc(blocks)
    chunks = sdp_chunk(text, splits, words, SDPConfig(max_tokens=100, min_tokens=10))
    assert len(chunks) == 2
    assert chunks[1].startswith("Next")
    assert "PageHeader" in chunks[0]


def test_oversized_block_is_presplit():
    big = para(250)
    text, splits = make_doc([para(10), big, para(10)])

    def splitter(t: str) -> list[str]:
        w = t.split(" ")
        return [" ".join(w[:120]) + " ", " ".join(w[120:])]

    units = build_units(text, splits, words, 200, splitter)
    assert "".join(text[u.start:u.end] for u in units) == text
    assert any(u.internal for u in units)
    chunks = sdp_chunk(text, splits, words, SDPConfig(max_tokens=200, min_tokens=5), splitter)
    assert "".join(chunks) == text
    assert all(words(c) <= 200 for c in chunks)


def test_small_doc_is_single_chunk():
    text, splits = make_doc([para(5), para(5)])
    assert sdp_chunk(text, splits, words, SDPConfig()) == [text]
    assert sdp_chunk("", [], words) == []


def test_target_tokens_shrinks_chunks():
    blocks = [para(20, "Lorem") for _ in range(20)]  # 400 words; capitalised so cuts are not continuations
    text, splits = make_doc(blocks)
    big = sdp_chunk(text, splits, words, SDPConfig(max_tokens=400, min_tokens=10))
    small = sdp_chunk(text, splits, words, SDPConfig(max_tokens=400, min_tokens=10, tau=0.1,
                                                     target_tokens=100, target_weight=2.0))
    assert len(big) == 1
    assert len(small) == 4 and all(words(c) == 100 for c in small)


def test_extra_score_hook_is_used():
    blocks = [para(30) for _ in range(4)]
    text, splits = make_doc(blocks)
    units = build_units(text, splits, words, 1000)
    # reward exactly the chunk made of units [0, 2) so the DP splits there
    cfg = SDPConfig(max_tokens=1000, min_tokens=1, extra_chunk_score=lambda i, j: 5.0 if (i, j) == (0, 2) else 0.0)
    assert dp_segment(units, cfg) == [0, 2]
