"""Gain over Adaptive per metric, set against the '±' of the paper's Table 3 (= st. dev. across documents).

For every system: mean ± sd over documents (as in the paper's Table 3), the gain over Adaptive, whether the gain is
larger than Adaptive's ± for that metric, the largest gain that is still possible (100 - Adaptive), and the paired
per-document bootstrap CI of the gain.

Real rows use real metrics. Rows marked (proxy) take SC / BI / RC from the real metric code and ICC / DCC from the
proxy prediction of proxy_sweep.py - they are predictions, not results.

Usage:  python -X utf8 research/structure_chunker/headroom/gain_vs_sd.py --out runs/headroom_gain_vs_sd_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_headroom import CAND, FIVE, boot, oracle, wide  # noqa: E402

M = ["RC", "ICC", "DCC", "BI", "SC", "mean5"]
LINES: list[str] = []


def say(*p) -> None:
    line = " ".join(str(x) for x in p)
    print(line)
    LINES.append(line)


def proxy_frame(csv: str, setting: str) -> pd.DataFrame | None:
    if not Path(csv).exists():
        return None
    d = pd.read_csv(csv)
    d = d[d.setting == setting].set_index("doc_name")
    out = pd.DataFrame({"RC": d.RC * 100, "ICC": d.ICC_pred, "DCC": d.DCC_pred, "BI": d.BI * 100, "SC": d.SC * 100})
    out["mean5"] = out[["RC", "ICC", "DCC", "BI", "SC"]].mean(axis=1)
    return out


def block(name: str, ada: pd.DataFrame, systems: dict[str, pd.DataFrame]) -> list[dict]:
    say(f"\n=== {name}: {len(ada)} documents. Cells: mean ± sd over documents ===")
    say(f"{'system':28}" + "".join(f"{m:>14}" for m in M))
    say(f"{'Adaptive (4 cand.)':28}" + "".join(f"{ada[m].mean():8.2f} ±{ada[m].std():4.1f}" for m in M))
    say(f"{'  room left (100 - mean)':28}" + "".join(f"{100 - ada[m].mean():14.2f}" for m in M))
    rows = []
    for label, t in systems.items():
        t = t.reindex(ada.index)
        say(f"{label:28}" + "".join(f"{t[m].mean():8.2f} ±{t[m].std():4.1f}" for m in M))
        cells = []
        for m in M:
            pair = pd.concat([t[m], ada[m]], axis=1).dropna()
            d = (pair.iloc[:, 0] - pair.iloc[:, 1]).to_numpy()
            lo, hi = boot(d)
            sd = ada[m].std()
            rows.append({"corpus": name, "system": label, "metric": m, "adaptive_mean": ada[m].mean(), "adaptive_sd": sd,
                         "system_mean": t[m].mean(), "gain": d.mean(), "ci_low": lo, "ci_high": hi,
                         "gain_over_sd": d.mean() / sd if sd else np.nan, "exceeds_sd": bool(d.mean() > sd),
                         "max_possible_gain": 100 - ada[m].mean(), "sd_of_paired_diff": d.std(ddof=1),
                         "wins": int((d > 0).sum()), "n": len(d)})
            cells.append(f"{d.mean():+7.2f} {'YES' if d.mean() > sd else 'no ':>3}  ")
        say(f"{'  gain | gain > Adaptive ±?':28}" + "".join(f"{c:>14}" for c in cells))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists - choose a new folder")

    # ---- CLAIR ----
    base = wide(Path("runs/full_kaggle/results/chunking_metrics.parquet"))
    ada = oracle(base, CAND)
    sysc: dict[str, pd.DataFrame] = {"SDP-B (real)": wide(Path("runs/sdp/B_default/metrics/chunking_metrics.parquet")).droplevel(0)}
    for label, path in (("SDP-B rc1 (real)", "runs/sdp_sweep_v1/rc1/metrics/chunking_metrics.parquet"),
                        ("SDP-B sw0_rc1 (real)", "runs/sdp_sweep_v2/sw0_rc1/metrics/chunking_metrics.parquet")):
        if Path(path).exists():
            t = wide(Path(path)).droplevel(0)
            # a finished run has environment.json; single-chunk documents legitimately have no ICC / DCC
            if (Path(path).parent / "environment.json").exists() and len(t) == len(ada):
                sysc[label] = t
            else:
                say(f"(skipped {label}: run not finished, {len(t)} of {len(ada)} documents so far)")
    for label, csv, s in (("SDP-B rc1 (proxy)", "runs/sdp_sweep_v1/sweep_per_doc.csv", "rc1"),
                          ("SDP-B sw0_rc1 (proxy)", "runs/sdp_sweep_v2/sweep_per_doc.csv", "sw0_rc1")):
        p = proxy_frame(csv, s)
        if p is not None:
            sysc[label] = p
    rows = block("CLAIR", ada, sysc)

    # ---- clair_ext ----
    ext = wide(Path("runs/ext_metrics_merged_v1/ext_v1/chunking_metrics.parquet"))
    ada_e = oracle(ext, CAND)
    syse = {"SDP-B (real)": wide(Path("runs/ext_metrics_merged_v1/sdp_b/chunking_metrics.parquet")).droplevel(0)}
    for label, s in (("SDP-B rc1 (proxy)", "rc1"), ("SDP-B sw0_rc1 (proxy)", "sw0_rc1")):
        p = proxy_frame("runs/sdp_sweep_ext_v1/sweep_per_doc.csv", s)
        if p is not None:
            syse[label] = p
    rows += block("clair_ext", ada_e, syse)

    args.out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(args.out / "gain_vs_sd.csv", index=False)
    (args.out / "report.txt").write_text("\n".join(LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
