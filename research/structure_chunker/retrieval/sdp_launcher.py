"""Run the group harness (research/week3/phase_g_runner.py) with extra system names, WITHOUT editing it.

The harness only accepts system ids listed in ALL_SYSTEMS. This launcher appends the names given in
SDP_SYSTEMS (comma-separated environment variable) to that tuple in memory, then calls the harness main().
New systems are NOT added to FIXED_SYSTEMS, so the harness's best_fixed_cv / oracle rows keep their meaning.

Usage (on the GPU machine, from the bundle root):
  SDP_SYSTEMS=sdp_cv,sdp_b python -X utf8 sdp_launcher.py index --prepared-dir ... --systems adaptive,raw__page,sdp_cv
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "research" / "week3"))
import phase_g_runner as pg  # noqa: E402

extra = tuple(s.strip() for s in os.environ.get("SDP_SYSTEMS", "").split(",") if s.strip())
clash = set(extra) & set(pg.ALL_SYSTEMS)
if clash:
    raise SystemExit(f"SDP_SYSTEMS clash with harness systems: {sorted(clash)}")
pg.ALL_SYSTEMS = (*pg.ALL_SYSTEMS, *extra)
print(f"[sdp_launcher] harness sha256 {pg._harness_sha256()[:12]}..., extra systems: {list(extra)}", file=sys.stderr)

if __name__ == "__main__":
    raise SystemExit(pg.main(sys.argv[1:]))
