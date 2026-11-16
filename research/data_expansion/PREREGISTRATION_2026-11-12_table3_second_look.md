# Pre-registration: second look at `clair_ext` for the optimised SDP-B (written 2026-11-12, before any ICC / DCC of the new settings on the test set exists)

Addendum to `PREREGISTRATION.md` (2026-11-09), which is left unchanged. The first look at the test set tested H1–H3.
This is a **second use of the same 129 documents** and must be reported as such.

## What is being tested
The Table 3 score (per-document mean of SC, BI, ICC, DCC, RC) of new SDP-B settings found on CLAIR after the first look.

## Fixed before the run
- **Chunks.** Produced on 2026-11-12 with `research/structure_chunker/headroom/proxy_sweep.py --round 4
  --data-dir data\clair_ext_set` → `runs/sdp_sweep_ext_v4/<setting>/chunks/small_merged/chunks.parquet`. sha256 (first 16 hex
  digits; full values in `runs/gpu_metrics_ext_v2/chunks_sha256.json`):
  - `sw0_rc1` `644a531e5dc2e001`, `rc1` `e701df4a1ce9556d`, `keepcont_rc1` `55f227b1a0b0a03b`, `noart_rc1` `1569a559428cf848`.
- **Settings** (all keep SDP-B's ICC : DCC weights 1 : 1, jina-v3 block embeddings, the paper's post-processing):
  - `sw0_rc1`: `struct_weight = 0`, entity–pronoun pair penalty λ = 1. **This is the setting already reported on CLAIR
    (+1.50 over Adaptive) and it is the only primary system.**
  - `rc1`: pair penalty only. `keepcont_rc1`: pair penalty, structural terms off except the continuation penalty.
    `noart_rc1`: pair penalty, only the page-artefact penalty off.
- **References.** Adaptive (4 candidates) and SDP-B scores of the first look, `runs/ext_metrics_merged_v1`; not recomputed.
- **Scoring.** `run_metrics_remote_v2.sh` on a pod with ≥ 24 GB per GPU: original `compute_metrics_per_origin`, jina-v3 revision
  `ab036b0`, same package versions as the first look. The `validate` step must agree with the Kaggle numbers to within 1e-3
  (0–1 scale); otherwise the run is not used.
- **Statistics.** `research/data_expansion/analyze_second_look.py` (written and dry-run on stand-in data before the pod
  run): unit = document, paired differences, 10,000-sample bootstrap (seed 2026), Wilcoxon (Pratt).

## Hypotheses
- **H4 (primary, tested alone):** `sw0_rc1` > Adaptive on the per-document 5-metric mean.
- **H5:** `sw0_rc1` > SDP-B.
- **Secondary:** `rc1`, `keepcont_rc1`, `noart_rc1`, each against Adaptive and against SDP-B.
  H5 and the six secondary comparisons are Holm-corrected together.
- Reported next to H4: the gain per metric and whether it exceeds Adaptive's standard deviation across documents
  (the "±" of the paper's Table 3).

## Predictions recorded before the run
From the proxy (`runs/headroom_predict_ext_v1/report.txt`), with the DCC error measured on CLAIR (−0.67):
`sw0_rc1` − Adaptive = **+1.84** (plausible range +1.70 to +1.97); `sw0_rc1` − SDP-B = +0.88; ICC ≈ 78.6, DCC ≈ 85.7.

## What is already known about the test set (not blind)
- SC, BI and RC of `rc1` and `sw0_rc1` on the test set are real numbers computed on CPU on 2026-11-12
  (RC 98.45 / 98.47 against 94.63 for SDP-B and 96.23 for Adaptive), together with proxy ICC / DCC
  (`runs/sdp_sweep_ext_v1`). The real ICC and DCC are unknown.
- For `keepcont_rc1` and `noart_rc1` the chunks were generated but their test-set scores (`runs/sdp_sweep_ext_v4/report.txt`)
  were not looked at when this file was written.
- The two changes (`rc1`, `sw0`) were designed by exploring all 33 CLAIR documents; CLAIR 5-fold selection
  (`runs/sdp/tune_with_sw0_rc1_v1`) picks `sw0_rc1` in 5 of 5 folds.

## What is not allowed
Changing the primary system, the settings or the statistics after seeing the pod results. If a secondary setting scores
higher than `sw0_rc1` on the test set, it is reported as secondary; it does not replace the primary result.

## Also in the pod bundle (not part of the hypotheses)
CLAIR scoring of `sw0_rc1` (cross-machine check against the laptop run), `keepcont_rc1` and `noart_rc1`.
