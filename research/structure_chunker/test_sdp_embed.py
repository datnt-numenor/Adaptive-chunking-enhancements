import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sdp_chunker import SDPConfig, Unit  # noqa: E402
from sdp_embed import EmbedScorer, dinkelbach_segment  # noqa: E402


def units_of(n: int, tokens: int = 50) -> list[Unit]:
    return [Unit(start=k * 10, end=(k + 1) * 10, tokens=tokens) for k in range(n)]


def topic_embeddings(labels: list[int], dim: int = 16, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.eye(dim)[: max(labels) + 1]
    e = centers[labels] + noise * rng.standard_normal((len(labels), dim))
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def test_icc_proxy_is_one_for_identical_units():
    u = units_of(3)
    e = np.tile(np.eye(4)[0], (3, 1))
    icc, dcc = EmbedScorer(u, e).parts(0, 3)
    assert abs(icc - 1) < 1e-9 and abs(dcc - 1) < 1e-9


def test_cuts_at_topic_change():
    labels = [0] * 6 + [1] * 6  # 12 units x 50 tokens, topic switch at unit 6
    u = units_of(len(labels))
    scorer = EmbedScorer(u, topic_embeddings(labels), window_tokens=300)
    cfg = SDPConfig(max_tokens=400, min_tokens=100)
    starts, hist = dinkelbach_segment(u, scorer, cfg, struct_weight=0.0)
    assert 6 in starts
    assert all(b - a <= 8 for a, b in zip(starts, starts[1:] + [len(u)]))  # 8 x 50 = 400 tokens cap
    assert len(hist) >= 1


def test_mean_objective_converges():
    labels = [0, 0, 1, 1, 2, 2, 0, 0]
    u = units_of(len(labels))
    scorer = EmbedScorer(u, topic_embeddings(labels, seed=1), window_tokens=200)
    _, hist = dinkelbach_segment(u, scorer, SDPConfig(max_tokens=300, min_tokens=50), struct_weight=0.0)
    assert hist[-1] >= hist[0] - 1e-9
