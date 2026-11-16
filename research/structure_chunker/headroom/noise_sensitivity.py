"""How much do the SDP-B cuts depend on tiny embedding differences (another GPU / batch order / dtype)?

Block embeddings are perturbed with relative noise sigma (cos to the original ~ 1 - sigma^2/2), the DP is re-run,
and the result is compared with the unperturbed run: share of identical cut points, and the change of SC/BI/RC (real)
and of the ICC/DCC proxies (evaluated with the ORIGINAL embeddings, as a fixed scorer would). CPU only.

Usage: python -X utf8 research/structure_chunker/headroom/noise_sensitivity.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy_sweep as ps
from sdp_chunker import SDPConfig
from sdp_embed import EmbedScorer, dinkelbach_segment
from adaptive_chunking.chunking_utils import count_tokens
from adaptive_chunking.postprocessing import merge_small_chunks_to_neighbours

docs = ps.load_docs()
SETTINGS = {"base (SDP-B)": {}, "sw0_rc1": {"sw": 0.0, "rc": 1.0}}


def run(d, s, emb):
    sc = ps.RCScorer(d["units"], emb, 3000, 1.0, 1.0)
    sc.set_rc(d["cross"], s.get("rc", 0.0))
    starts, _ = dinkelbach_segment(d["units"], sc, SDPConfig(), struct_weight=s.get("sw", 0.1))
    offs = [0] + [d["units"][k].start for k in starts[1:]] + [len(d["text"])]
    chunks = [d["text"][a:b] for a, b in zip(offs, offs[1:]) if b > a]
    merged = merge_small_chunks_to_neighbours(chunks, count_tokens, min_limit=100, max_limit=1150, merge_to="next")
    b = ps.unit_bounds(d["units"], merged)
    pr = ps.proxies(EmbedScorer(d["units"], d["emb"], 3000), b)
    ch = ps.cheap(d, merged)
    vals = [ch["SC"], ch["BI"], ch["RC"], pr["icc_p"], pr["dcc_s"]]
    return {i for i, _ in b[1:]}, float(np.nanmean(vals)) * 100, pr["icc_p"] * 100, pr["dcc_s"] * 100


print(f"{'setting':14}{'sigma':>8}{'cos':>11}{'same cuts':>11}{'docs identical':>16}{'d mean5 (pts)':>15}{'max |d| doc':>13}{'d ICC~':>8}{'d DCC~':>8}")
for name, s in SETTINGS.items():
    ref = {k: run(d, s, d["emb"]) for k, d in docs.items()}
    for sigma in (1e-3, 3e-3, 1e-2):
        rng = np.random.default_rng(2026)
        same = tot = ident = 0
        dm, di, dd = [], [], []
        for k, d in docs.items():
            e = d["emb"].astype(np.float64)
            e = e + sigma * rng.standard_normal(e.shape) / np.sqrt(e.shape[1])
            e /= np.linalg.norm(e, axis=1, keepdims=True)
            cuts, m5, icc, dcc = run(d, s, e)
            same += len(cuts & ref[k][0]); tot += len(cuts | ref[k][0]); ident += cuts == ref[k][0]
            dm.append(m5 - ref[k][1]); di.append(icc - ref[k][2]); dd.append(dcc - ref[k][3])
        print(f"{name:14}{sigma:8.0e}{1 - sigma**2 / 2:11.6f}{same / tot:11.1%}{ident:11d}/{len(docs)}{np.mean(dm):15.3f}"
              f"{np.max(np.abs(dm)):13.3f}{np.nanmean(di):8.3f}{np.nanmean(dd):8.3f}")
