"""EXPLORATORY proxy sweep of SDP-B settings (CPU only, cached jina block embeddings). NOT a result:
ICC and DCC below are proxies built from block embeddings; only SC / BI / RC are the real metrics.

Steps
1. Calibrate: for the runs that already have real metrics, compute the proxies on their chunks and regress
   (real - real_B_default) on (proxy - proxy_B_default) per document  ->  how far a proxy gain can be trusted.
2. Sweep: re-run the Dinkelbach DP with other weights (ICC / DCC / structure / an explicit reference-pair
   penalty), post-process like the paper, score SC / BI / RC for real and ICC / DCC by proxy.
3. Save the chunks of every setting under <out>/<name>/chunks/small_merged so score_full.py can score any of them.

Usage:  python -X utf8 research/structure_chunker/headroom/proxy_sweep.py --out runs/sdp_sweep_v1 [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from adaptive_chunking.chunking_utils import count_tokens  # noqa: E402
from adaptive_chunking.metrics import (  # noqa: E402
    compute_block_integrity, compute_filtered_missing_ref_error, compute_size_compliance)
from adaptive_chunking.paper.replicate import SEPARATORS  # noqa: E402
from adaptive_chunking.postprocessing import merge_small_chunks_to_neighbours  # noqa: E402
from adaptive_chunking.splitters import RecursiveSplitter  # noqa: E402
from run_sdp import records_for  # noqa: E402
from sdp_chunker import SDPConfig, build_units  # noqa: E402
from sdp_embed import EmbedScorer, dinkelbach_segment  # noqa: E402

DATA = Path("data/clair")
EMB = Path("runs/sdp/_unit_emb/jina_max1100")
CAL_RUNS = ["A_default", "A_target600", "A_target800", "A_target1000", "B_default", "B_qwen"]
REF = "B_default"

SWEEP = [
    {"name": "base"},                                   # must reproduce runs/sdp/B_default exactly
    {"name": "dcc1.5", "dcc": 1.5},
    {"name": "dcc2", "dcc": 2.0},
    {"name": "dcc3", "dcc": 3.0},
    {"name": "icc1.5", "icc": 1.5},
    {"name": "icc2", "icc": 2.0},
    {"name": "sw0", "sw": 0.0},
    {"name": "sw0.3", "sw": 0.3},
    {"name": "rc1", "rc": 1.0},
    {"name": "rc3", "rc": 3.0},
    {"name": "rc10", "rc": 10.0},
    {"name": "dcc1.5_rc3", "dcc": 1.5, "rc": 3.0},
    {"name": "dcc2_rc3", "dcc": 2.0, "rc": 3.0},
    {"name": "icc1.5_rc3", "icc": 1.5, "rc": 3.0},
]

# round 2: combinations around what round 1 (runs/sdp_sweep_v1) pointed to
SWEEP2 = [
    {"name": "base"},
    {"name": "sw0", "sw": 0.0},
    {"name": "sw0.05", "sw": 0.05},
    {"name": "sw0_rc1", "sw": 0.0, "rc": 1.0},
    {"name": "sw0_rc3", "sw": 0.0, "rc": 3.0},
    {"name": "sw0.05_rc1", "sw": 0.05, "rc": 1.0},
    {"name": "sw0_dcc1.5_rc1", "sw": 0.0, "dcc": 1.5, "rc": 1.0},
    {"name": "sw0_icc1.5_rc1", "sw": 0.0, "icc": 1.5, "rc": 1.0},
    {"name": "sw0_dcc1.5", "sw": 0.0, "dcc": 1.5},
    {"name": "sw0_icc1.5", "sw": 0.0, "icc": 1.5},
    {"name": "rc0.3", "rc": 0.3},
    {"name": "sw0_rc0.3", "sw": 0.0, "rc": 0.3},
]

# round 3: the settings singled out on CLAIR, fixed before looking at another corpus
SWEEP3 = [
    {"name": "base"},
    {"name": "sw0", "sw": 0.0},
    {"name": "rc1", "rc": 1.0},
    {"name": "sw0_rc1", "sw": 0.0, "rc": 1.0},
]

# round 4: which structural term holds ICC down? (each drops one term of the default config; rc = 1 throughout)
Z6 = [0.0] * 6
SWEEP4 = [
    {"name": "base"},
    {"name": "rc1", "rc": 1.0},
    {"name": "sw0_rc1", "sw": 0.0, "rc": 1.0},
    {"name": "notau_rc1", "rc": 1.0, "cfg": {"tau": 0.0}},
    {"name": "nohead_rc1", "rc": 1.0, "cfg": {"heading_bonus": Z6}},
    {"name": "noart_rc1", "rc": 1.0, "cfg": {"artifact_penalty": 0.0}},
    {"name": "nocont_rc1", "rc": 1.0, "cfg": {"continuation_penalty": 0.0}},
    {"name": "nominor_rc1", "rc": 1.0, "cfg": {"pronoun_penalty": 0.0, "colon_penalty": 0.0, "list_penalty": 0.0}},
    # keep only the two terms that stop ugly cuts (mid-paragraph, page-artefact chunk starts)
    {"name": "keepcut_rc1", "rc": 1.0, "cfg": {"tau": 0.0, "heading_bonus": Z6, "pronoun_penalty": 0.0,
                                                "colon_penalty": 0.0, "list_penalty": 0.0}},
    {"name": "keepcont_rc1", "rc": 1.0, "cfg": {"tau": 0.0, "heading_bonus": Z6, "pronoun_penalty": 0.0,
                                                 "colon_penalty": 0.0, "list_penalty": 0.0, "artifact_penalty": 0.0}},
]

_DOCS = None


class RCScorer(EmbedScorer):
    """EmbedScorer plus a penalty on the chunk's opening cut for the entity-pronoun pairs that cut separates."""

    def set_rc(self, cross: np.ndarray, lam: float) -> None:
        self.cross, self.lam = cross, lam

    def score(self, i: int, j: int, icc_fill: float) -> float:
        return super().score(i, j, icc_fill) - self.lam * self.cross[i]


def load_docs(data: Path | None = None):
    global _DOCS, DATA
    if data is not None and Path(data) != DATA:
        DATA, _DOCS = Path(data), None
    if _DOCS is not None:
        return _DOCS
    cfg = SDPConfig()
    oversize = RecursiveSplitter(separators=SEPARATORS, chunk_size=cfg.max_tokens, chunk_overlap=0,
                                 is_separator_regex=True, attach_separator_to="start",
                                 length_function=count_tokens, merging="to_chunk_size")
    mentions = {}
    for pq in (DATA / "mentions").glob("*.parquet"):
        m = pd.read_parquet(pq)
        mentions[m["doc_name"].iloc[0]] = m["entity_pron_mentions"].iloc[0]
    docs = {}
    for path in sorted((DATA / "adi_parsed").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        text = doc["full_text"]
        units = build_units(text, doc["split_points"], count_tokens, cfg.max_tokens, oversize.split_text)
        emb = np.load(EMB / f"{path.stem}.npy")
        assert len(emb) == len(units), path.stem
        pairs = mentions.get(path.stem)
        pairs = pairs if pairs is not None and len(pairs) else None
        cross = np.zeros(len(units) + 1)
        if pairs is not None:
            starts = np.array([u.start for u in units])
            for a, b in pairs:
                lo, hi = (a, b) if a[0] <= b[0] else (b, a)
                # cuts at units[k].start with lo.start < cut <= hi.end separate the pair
                k0, k1 = np.searchsorted(starts, lo[0], side="right"), np.searchsorted(starts, hi[1], side="right")
                cross[k0:k1] += 1
            n0 = max(sum(u.tokens for u in units) / 550.0, 1.0)      # expected number of chunks
            cross = cross * n0 / len(pairs)                          # ~ RC loss expressed per chunk
        docs[path.stem] = {"doc": doc, "text": text, "units": units, "emb": emb, "pairs": pairs, "cross": cross}
    _DOCS = docs
    return docs


def unit_bounds(units, chunks: list[str]) -> list[tuple[int, int]] | None:
    start_of = {u.start: k for k, u in enumerate(units)}
    idx, pos = [], 0
    for c in chunks:
        if pos not in start_of:
            return None
        idx.append(start_of[pos])
        pos += len(c)
    return list(zip(idx, idx[1:] + [len(units)]))


def proxies(scorer: EmbedScorer, bounds: list[tuple[int, int]]) -> dict:
    """ICC proxy; DCC proxy with the DP's centred window and with the metric's own window construction."""
    iccs = [scorer.parts(i, j)[0] for i, j in bounds if j - i >= 2]
    dcc_c = [scorer.parts(i, j)[1] for i, j in bounds]
    tok, swe = scorer.tok_prefix, scorer.sumWE

    def vec(i, j):
        v = swe[j] - swe[i]
        return v / (np.linalg.norm(v) or 1.0)

    cemb = [vec(i, j) for i, j in bounds]
    sims, last_end, n = [], None, len(bounds)
    for i in range(n):
        total, j = 0.0, i
        while j < n:
            t = tok[bounds[j][1]] - tok[bounds[j][0]]
            if j > i and total + t > 3000:
                break
            total += t
            j += 1
        members = list(range(i, j))
        if len(members) > 1 and (last_end is None or members[-1] != last_end):
            w = vec(bounds[i][0], bounds[j - 1][1])
            sims += [float(w @ cemb[k]) for k in members]
            last_end = members[-1]
    return {"icc_p": float(np.mean(iccs)) if iccs else np.nan, "dcc_c": float(np.mean(dcc_c)),
            "dcc_s": float(np.mean(sims)) if sims else np.nan}


def cheap(d: dict, merged: list[str]) -> dict:
    rc_err = compute_filtered_missing_ref_error(d["text"], merged, d["pairs"]) if d["pairs"] is not None else None
    return {"SC": compute_size_compliance(merged, max_tokens=1100, min_tokens=100),
            "BI": compute_block_integrity(merged, d["doc"]["split_points"], d["text"], tolerance_chars=5),
            "RC": np.nan if rc_err is None else 1 - rc_err,
            "n": len(merged), "tok": sum(count_tokens(c) for c in merged) / len(merged)}


def run_setting(s: dict) -> tuple[str, list[dict], list[dict]]:
    docs = load_docs(s.get("_data"))
    rows, recs = [], []
    for name, d in docs.items():
        scorer = RCScorer(d["units"], d["emb"], 3000, s.get("icc", 1.0), s.get("dcc", 1.0))
        scorer.set_rc(d["cross"], s.get("rc", 0.0))
        starts, _ = dinkelbach_segment(d["units"], scorer, SDPConfig(**s.get("cfg", {})), struct_weight=s.get("sw", 0.1))
        offs = [0] + [d["units"][k].start for k in starts[1:]] + [len(d["text"])]
        chunks = [d["text"][a:b] for a, b in zip(offs, offs[1:]) if b > a]
        merged = merge_small_chunks_to_neighbours(chunks, count_tokens, min_limit=100, max_limit=1150, merge_to="next")
        assert "".join(merged) == d["text"], name
        plain = EmbedScorer(d["units"], d["emb"], 3000)
        rows.append({"setting": s["name"], "doc_name": name, **cheap(d, merged), **proxies(plain, unit_bounds(d["units"], merged))})
        recs += records_for(name, f"sdp_b_{s['name']}", "small_merged", merged, d["doc"])
    return s["name"], rows, recs


def calibrate(cal: dict[str, tuple[str, str]]) -> pd.DataFrame:
    docs = load_docs()
    rows = []
    for run, (chunks_pq, metrics_pq) in cal.items():
        ch = pd.read_parquet(chunks_pq).sort_values(["doc_name", "chunk_index"])
        met = pd.read_parquet(metrics_pq)
        real = met.pivot_table(index="doc_name", columns="metric_name", values="score")
        for name, g in ch.groupby("doc_name"):
            if name not in docs or name not in real.index:
                continue
            d = docs[name]
            b = unit_bounds(d["units"], list(g.chunk_text))
            if b is None:
                continue
            rows.append({"setting": run, "doc_name": name, **proxies(EmbedScorer(d["units"], d["emb"], 3000), b),
                         "icc_real": real.loc[name, "intrachunk_cohesion"],
                         "dcc_real": real.loc[name, "document_contextual_coherence"],
                         "SC": real.loc[name, "size_compliance"], "BI": real.loc[name, "block_integrity"],
                         "RC": real.loc[name].get("references_completeness", np.nan)})
    return pd.DataFrame(rows)


def boot(d: np.ndarray, n: int = 10000, seed: int = 2026):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--round", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--cal", nargs="*", default=None, help="name=chunks.parquet=metrics.parquet (default: CLAIR runs)")
    ap.add_argument("--ref-chunks", default="runs/sdp/B_default/chunks/small_merged/chunks.parquet")
    ap.add_argument("--beta", nargs=2, type=float, default=None, metavar=("ICC", "DCC"),
                    help="use these proxy->real slopes (e.g. the ones fitted on CLAIR) instead of fitting them here")
    args = ap.parse_args()
    global SWEEP
    SWEEP = {1: SWEEP, 2: SWEEP2, 3: SWEEP3, 4: SWEEP4}[args.round]
    SWEEP = [{**x, "_data": str(args.data_dir)} for x in SWEEP]
    load_docs(args.data_dir)
    cal_spec = ({r: (f"runs/sdp/{r}/chunks/small_merged/chunks.parquet", f"runs/sdp/{r}/metrics/chunking_metrics.parquet")
                 for r in CAL_RUNS} if args.cal is None else
                {a.split("=")[0]: (a.split("=")[1], a.split("=")[2]) for a in args.cal})
    if args.out.exists():
        raise SystemExit(f"{args.out} exists - choose a new folder")
    args.out.mkdir(parents=True)
    lines = []

    def say(*p):
        line = " ".join(str(x) for x in p)
        print(line, flush=True)
        lines.append(line)

    t0 = time.time()
    cal = calibrate(cal_spec)
    cal.to_csv(args.out / "calibration_per_doc.csv", index=False)
    ref = cal[cal.setting == REF].set_index("doc_name")
    say("=== 1. Proxy calibration on runs with real metrics (points, mean over documents) ===")
    say(f"{'run':14}{'n':>4}{'ICC real':>10}{'ICC proxy':>11}{'DCC real':>10}{'DCC centred':>13}{'DCC struct':>12}")
    for run, g in cal.groupby("setting", sort=False):
        say(f"{run:14}{len(g):4d}{g.icc_real.mean() * 100:10.2f}{g.icc_p.mean() * 100:11.2f}{g.dcc_real.mean() * 100:10.2f}"
            f"{g.dcc_c.mean() * 100:13.2f}{g.dcc_s.mean() * 100:12.2f}")
    others = cal[cal.setting != REF].set_index("doc_name")
    dd = pd.DataFrame({"setting": others.setting,
                       "d_icc_real": others.icc_real - ref.icc_real.reindex(others.index).to_numpy(),
                       "d_icc_p": others.icc_p - ref.icc_p.reindex(others.index).to_numpy(),
                       "d_dcc_real": others.dcc_real - ref.dcc_real.reindex(others.index).to_numpy(),
                       "d_dcc_s": others.dcc_s - ref.dcc_s.reindex(others.index).to_numpy(),
                       "d_dcc_c": others.dcc_c - ref.dcc_c.reindex(others.index).to_numpy()}).dropna()
    beta = {}
    for real, prox in (("d_icc_real", "d_icc_p"), ("d_dcc_real", "d_dcc_s"), ("d_dcc_real", "d_dcc_c")):
        x, y = dd[prox].to_numpy(), dd[real].to_numpy()
        b = float(x @ y / (x @ x))
        r = float(np.corrcoef(x, y)[0, 1])
        beta[prox] = b
        say(f"   {real} ~ beta * {prox}:  beta = {b:.2f}, Pearson r = {r:.2f}, n = {len(dd)} (document differences vs {REF})")
    if args.beta:
        beta["d_icc_p"], beta["d_dcc_s"] = args.beta
        say(f"   slopes used for the predictions (given on the command line): ICC {args.beta[0]}, DCC {args.beta[1]}")
    # the same relation restricted to B_qwen, the only calibration run that is close to B_default
    near = dd[dd.setting == "B_qwen"]
    for real, prox in ((("d_icc_real", "d_icc_p"), ("d_dcc_real", "d_dcc_s")) if len(near) else ()):
        say(f"   near-B check (B_qwen only, n={len(near)}): r({real},{prox}) = {np.corrcoef(near[prox], near[real])[0, 1]:.2f}; "
            f"mean real {near[real].mean() * 100:+.2f} vs proxy {near[prox].mean() * 100:+.2f} pts")

    say(f"\n=== 2. Sweep ({len(SWEEP)} settings, {args.workers} workers) ===")
    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, rows, recs in ex.map(run_setting, SWEEP):
            all_rows += rows
            d = args.out / name / "chunks" / "small_merged"
            d.mkdir(parents=True)
            pd.DataFrame(recs).to_parquet(d / "chunks.parquet")
            print(f"  done {name} ({time.time() - t0:.0f}s)", flush=True)
    sw = pd.DataFrame(all_rows)

    # sanity: 'base' must be the B_default chunking
    b_chunks = pd.read_parquet(args.ref_chunks).sort_values(["doc_name", "chunk_index"])
    mine = pd.read_parquet(args.out / "base" / "chunks" / "small_merged" / "chunks.parquet").sort_values(["doc_name", "chunk_index"])
    same = list(b_chunks.chunk_text) == list(mine.chunk_text)
    say(f"sanity: setting 'base' reproduces {args.ref_chunks} exactly: {same}")

    base = sw[sw.setting == "base"].set_index("doc_name")
    base = base[base.index.isin(ref.index)]
    realB = ref.reindex(base.index)
    out_rows = []
    for name, g in sw.groupby("setting", sort=False):
        g = g.set_index("doc_name").reindex(base.index)
        icc_pred = realB.icc_real + beta["d_icc_p"] * (g.icc_p - base.icc_p)
        dcc_pred = realB.dcc_real + beta["d_dcc_s"] * (g.dcc_s - base.dcc_s)
        five = pd.DataFrame({"SC": g.SC, "BI": g.BI, "RC": g.RC, "ICC": icc_pred, "DCC": dcc_pred}) * 100
        five0 = pd.DataFrame({"SC": base.SC, "BI": base.BI, "RC": base.RC, "ICC": realB.icc_real, "DCC": realB.dcc_real}) * 100
        g = g.assign(ICC_pred=five.ICC, DCC_pred=five.DCC, mean5_pred=five.mean(axis=1), d_mean5=five.mean(axis=1) - five0.mean(axis=1),
                     setting=name)
        out_rows.append(g.reset_index())
    res = pd.concat(out_rows, ignore_index=True)
    res.to_csv(args.out / "sweep_per_doc.csv", index=False)
    say(f"{'setting':14}{'tok':>6}{'SC':>8}{'BI':>8}{'RC':>8}{'ICC~':>8}{'DCC~':>8}{'mean5~':>9}   d_mean5 vs base [95% CI]   wins")
    for name, g in res.groupby("setting", sort=False):
        d = g.d_mean5.to_numpy()
        lo, hi = boot(d)
        say(f"{name:14}{g.tok.mean():6.0f}{g.SC.mean() * 100:8.2f}{g.BI.mean() * 100:8.2f}{g.RC.mean() * 100:8.2f}"
            f"{g.ICC_pred.mean():8.2f}{g.DCC_pred.mean():8.2f}{g.mean5_pred.mean():9.2f}   {d.mean():+.2f} [{lo:+.2f},{hi:+.2f}]   "
            f"{int((d > 1e-9).sum())}/{len(d)}")
    best = res.loc[res.groupby("doc_name").mean5_pred.idxmax()]
    d = best.d_mean5.to_numpy()
    lo, hi = boot(d)
    say(f"{'per-doc best':14}{best.tok.mean():6.0f}{best.SC.mean() * 100:8.2f}{best.BI.mean() * 100:8.2f}{best.RC.mean() * 100:8.2f}"
        f"{best.ICC_pred.mean():8.2f}{best.DCC_pred.mean():8.2f}{best.mean5_pred.mean():9.2f}   {d.mean():+.2f} [{lo:+.2f},{hi:+.2f}]   "
        f"(selected on the proxy itself: optimistic)")
    say("(~ = predicted from the proxy; SC/BI/RC are real. Exploratory, every document of the corpus, no folds: not a reportable result.)")
    (args.out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    (args.out / "manifest.json").write_text(json.dumps({"sweep": SWEEP, "calibration_runs": cal_spec, "reference": REF,
                                                         "data_dir": str(args.data_dir),
                                                         "beta": beta, "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
