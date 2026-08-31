# Score-Calibrated XYZ-Sigma Promotion (0015)
**Created:** 2026-08-28
**Last updated:** 2026-08-28 (evening, ~18:30)
**Status:** Score-8 full run + plots complete; score-6 unfinished (resumable); score-9 full run was queued

## The question

Promote 0014's `alpha * monopole(x,y,z,sigma) * Omega[q]` model without
starting another model lineage. The open question: are maximized template
scores near the selection boundary spikes or noise? Captured fraction cannot
answer that — its denominator includes irreducible noise energy across the
whole local channel-by-time waveform.

## Job outcomes (2026-08-28)

**Score-8 chain — complete.** The first full job `16513755` was cancelled by
the USR1 signal trap after 2h02m at chunk 1094/1958 (all passes healthy —
not a pursuit failure). Resubmission `16517915` resumed from ~1094 with
`--resume` and finished in 1h36m52s; dependent plot job `16517916` wrote the
eight-panel suite to `residuals/out/0014_xyzsig_full_score8/`. Output run:
`residuals/runs/dataset1_p1/0014_xyzsig_full_score8/`.

**Score-6 chain — incomplete.** Full job `16514699` was cancelled by the
USR1 trap after 2h27m at chunk 772/1958, and its auto-requeue never
materialized; its plot job `16517890` died with the unsatisfiable
`afterok` dependency. To finish: resubmit
`residuals/src/preprocessing/run_0014_xyzsig_full_score6.sbatch` (`--resume`
picks up at ~772), then submit plots against the new job ID.

**Score-9.** Full run `16529272` was queued (proposal and fitted-projection
thresholds 9, four residual passes, 2,048-event batches, one-second chunks,
USR1 checkpoint/requeue), output `residuals/runs/dataset1_p1/0014_xyzsig_full_score9/`.
Its CUDA smoke `16529119` was submitted against the standard 10-second
window with audit retention extended to scores 9 and 10. Same plot rule: no
`afterok` on requeueable runs — submit the suite manually after completion.

Smoke evidence behind the thresholds: score-8 smoke `16503279` (2m56s,
202,654 events vs 356,315 at corrected score 6), audit showing fitted scores
≥ 8.00029 on chunk 0 and zero cross-pass events within 0.5 ms/48 µm; eight
standard plots under `out/0014_xyzsig_smoke_16503279/`.

## Decisions

1. Finish score-6 with `--resume`, then compare score 8 vs 6 on counts by
   pass, pass-wise energy drops, fitted-score distributions, sigma/rho
   boundary mass, runtime, and plots.
2. Inspect examples near the score boundary specifically — not just
   high-input-energy or median cases. Require centered temporal structure, a
   coherent multi-channel footprint, and post-fit normalized channel
   residuals consistent with noise.
3. Keep proposal and final-fit thresholds coupled unless a controlled
   experiment shows a reason to separate them. Never use captured fraction
   alone as the spike/noise gate.

## Empirical noise calibration (planned)

Build a held-out null that preserves the recording's filtered temporal
spectrum and channel noise structure while destroying spike-like
spatiotemporal alignment; run the identical Omega × spatial-bank search and
full xyz-sigma fit on it; report candidate and fitted-score tails at
6–10; pick the lowest threshold whose estimated false-detection burden is
acceptable after the full bank search. A maximized score is not an
uncorrected Gaussian sigma. Cross-checks: score-stratified reconstruction
pages; event-triggered averages by Omega row and pass; source/sigma/rho/
boundary occupancy; time/channel recurrence and refractory-like lags;
per-pass event rate and residual-energy removal.

## Performance work (only after scientific acceptance)

Remove per-fit GPU→CPU geometry-key transfers via stable geometry IDs;
benchmark fit batches 2,048/4,096/8,192 on identical inputs; overlap bounded
CPU read/filter/CMR for chunk n+1 with GPU pursuit for chunk n using pinned
buffers within one allocation; measure real GPU utilization, peak VRAM, host
RSS, warmed chunk time. No artificial GPU-burn processes.

## Promotion gate

No further promotion until the empirical null supports the chosen threshold,
score-boundary examples look spike-like rather than selected noise, every
accepted pass reduces residual energy, cross-pass recurrence stays zero by
construction, the output schema and full plot suite pass, and the
performance configuration sustains the allocation without host OOM or
low-utilization termination.

## Plots added this session

- `plot_temporal_codebook_depth_time_raster.py` now renders a true
  rasterized Omega-colored scatter: fixed point size, per-point opacity
  scaled by normalized `|alpha|`. (Two earlier render attempts, the binned
  raster `16528417` and marker-area scatter `16528656`, were cancelled; the
  opacity correction is ready but was never resubmitted.)
- `plot_0014_full_recording_passes.py` replays each saved chunk on CUDA with
  the exact saved `predictions`, pools each stage with a sign-preserving
  temporal extremum, and renders 5×2: input plus the four residual passes on
  the left, against an empty-then-cumulative opacity-weighted Omega scatter
  on the right.
- Job `16525720` (1m56s) regenerated reconstruction examples with
  `--examples-per-pass 2` (8 columns) and added a codebook-usage plot, both
  under `residuals/out/0014_xyzsig_full_score8/`.
- New `plot_0014_codebook_usage.py` reads the 0014 schema (`omega.npy`,
  `temporal_idx.npy`, `residual_pass.npy`) and renders each row as waveform
  + overall usage + per-pass fraction. Score-8 usage: row 5 dominates
  (24.34% overall, 25.7% P1 → 20.0% P4), row 1 second (22.43%); rows 6/7
  grow with pass (row 7: 7.9% → 17.5%).

## Spatial-spread hypothesis (yours)

First-pass pursuit appears to force fitted sigma toward its lower bound.
Treat this as a fitting pathology to test, not an established localization
result. See `residuals/out/0014_xyzsig_full_score8/sigma_by_residual_pass.png`
together with the per-chunk x/y/z-by-pass heatmaps before changing the
spatial model or selection gate.

## Housekeeping (2026-08-28)

Top-level `runs/` and `out/` moved into `residuals/runs/` and
`residuals/out/` (no name collisions; the 0014 outputs merged into the
existing `residuals/runs/dataset1_p1/`). AGENTS.md documents the
consolidation. The active 0014 sbatch scripts were updated to the new paths
(`run_0014_xyzsig_full.sbatch`, `run_0014_xyzsig_full_score6.sbatch`,
`run_0014_xyzsig_smoke.sbatch`, `run_0014_xyzsig_audit.sbatch`, both plot
sbatch files); older legacy sbatch scripts still point at the old top-level
paths and need the same fix before reuse.

## Links

- [[session-014-xyzsigma-residual-pursuit]]
- [[session-012-rho-implementation-plan]]
- [[session-009-ibl-style-pursuit]]
