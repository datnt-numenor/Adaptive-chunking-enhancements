"""SDP version B: embedding-aware chunk score + Dinkelbach DP (maximise the *mean* chunk score).

Each unit (parser block) is embedded once. For a candidate chunk = units[i:j]:

* chunk embedding proxy  c = normalise(sum_k w_k e_k)        (w_k = unit tokens)
* ICC proxy              mean_k <e_k, c>                      (the metric: mean cos(block, chunk);
                                                               chunks with one block are skipped by the
                                                               metric, so they get a neutral value)
* DCC proxy              <c, normalise(sum of w_k e_k over a ~3000-token window centred on the chunk)>

ICC and DCC are averages over chunks, so the target is  max_seg mean_C s(C).  Dinkelbach's method solves
it with repeated DP runs on  sum_C (s(C) - rho), updating rho to the achieved mean until it stops moving.
Structural terms of version A (heading bonus, continuation / artefact penalties, min-size penalty) are kept
as additive per-chunk terms scaled by ``struct_weight``.
"""
from __future__ import annotations

import bisect
from dataclasses import replace
from typing import Sequence

import numpy as np

from sdp_chunker import SDPConfig, Unit, boundary_score, dp_segment


class EmbedScorer:
    def __init__(self, units: Sequence[Unit], emb: np.ndarray, window_tokens: int = 3000,
                 icc_weight: float = 1.0, dcc_weight: float = 1.0, single_unit_icc: float | None = None):
        self.units = units
        self.E = emb.astype(np.float64)
        w = np.array([max(u.tokens, 1) for u in units], dtype=np.float64)
        self.tok_prefix = np.concatenate([[0.0], np.cumsum(w)])
        self.sumE = np.vstack([np.zeros(emb.shape[1]), np.cumsum(self.E, axis=0)])
        self.sumWE = np.vstack([np.zeros(emb.shape[1]), np.cumsum(self.E * w[:, None], axis=0)])
        self.window = window_tokens
        self.a, self.b = icc_weight, dcc_weight
        self.single = single_unit_icc
        self.cache: dict[tuple[int, int], tuple[float, float]] = {}

    def _unit_at_token(self, t: float) -> int:
        return min(max(bisect.bisect_right(self.tok_prefix, t) - 1, 0), len(self.units))

    def parts(self, i: int, j: int) -> tuple[float, float]:
        key = (i, j)
        if key in self.cache:
            return self.cache[key]
        s = self.sumWE[j] - self.sumWE[i]
        norm = np.linalg.norm(s) or 1.0
        c = s / norm
        m = j - i
        if m >= 2:
            icc = float((self.sumE[j] - self.sumE[i]) @ c) / m
        else:
            icc = self.single if self.single is not None else float("nan")
        mid = (self.tok_prefix[i] + self.tok_prefix[j]) / 2
        lo = self._unit_at_token(mid - self.window / 2)
        hi = max(self._unit_at_token(mid + self.window / 2) + 1, j)
        lo = min(lo, i)
        hi = min(hi, len(self.units))
        ws = self.sumWE[hi] - self.sumWE[lo]
        dcc = float(ws @ c) / (np.linalg.norm(ws) or 1.0)
        self.cache[key] = (icc, dcc)
        return icc, dcc

    def score(self, i: int, j: int, icc_fill: float) -> float:
        icc, dcc = self.parts(i, j)
        if np.isnan(icc):
            icc = icc_fill
        return self.a * icc + self.b * dcc


def dinkelbach_segment(units: Sequence[Unit], scorer: EmbedScorer, cfg: SDPConfig,
                       struct_weight: float = 0.1, max_iter: int = 12, tol: float = 1e-6) -> tuple[list[int], list[float]]:
    """Maximise mean_C [scorer(C)] + struct_weight * structural terms. Returns (starts, rho history)."""
    rho = 0.0
    history = []
    starts = [0]
    icc_fill = 0.7  # neutral value for one-unit chunks; set once from the first solution, then frozen
    prefix = [0]
    for u in units:
        prefix.append(prefix[-1] + u.tokens)

    def objective(local_cfg: SDPConfig, bounds: list[tuple[int, int]], fill: float) -> float:
        """Mean over chunks of the exact per-chunk DP objective (embedding + structural terms)."""
        total = 0.0
        for i, j in bounds:
            v = scorer.score(i, j, fill) - local_cfg.tau
            if i > 0:
                v += boundary_score(units, i, local_cfg) or 0.0
            if prefix[j] - prefix[i] < local_cfg.min_tokens:
                v -= local_cfg.small_penalty
            total += v
        return total / len(bounds)

    for it in range(max_iter):
        local = replace(
            cfg,
            tau=struct_weight * cfg.tau,
            heading_bonus=tuple(struct_weight * h for h in cfg.heading_bonus),
            continuation_penalty=struct_weight * cfg.continuation_penalty,
            pronoun_penalty=struct_weight * cfg.pronoun_penalty,
            colon_penalty=struct_weight * cfg.colon_penalty,
            list_penalty=struct_weight * cfg.list_penalty,
            artifact_penalty=struct_weight * cfg.artifact_penalty,
            small_penalty=cfg.small_penalty,  # keep the size floor strong
            extra_chunk_score=lambda i, j, r=rho, f=icc_fill: scorer.score(i, j, f) - r,
        )
        starts = dp_segment(units, local)
        bounds = list(zip(starts, starts[1:] + [len(units)]))
        if it == 0:
            iccs = [x for x in (scorer.parts(i, j)[0] for i, j in bounds) if not np.isnan(x)]
            if iccs:
                icc_fill = float(np.mean(iccs))
        new_rho = objective(local, bounds, icc_fill)
        history.append(new_rho)
        if it > 0 and abs(new_rho - rho) < tol:
            break
        rho = new_rho
    return starts, history


def embed_units(text: str, units: Sequence[Unit], model, batch_size: int = 8) -> np.ndarray:
    pieces = [text[u.start:u.end] for u in units]
    return model.encode(pieces, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
