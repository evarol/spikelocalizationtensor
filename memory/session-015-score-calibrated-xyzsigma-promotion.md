# Session 015: Score-Calibrated XYZ-Sigma Promotion
**Created:** 2026-08-28
**Last updated:** 2026-08-28 (evening, ~18:30)

## Context

Promote session 0014's fixed `alpha * monopole(x,y,z,sigma) * Omega[q]`
model without starting another model lineage. The remaining question is whether
maximized template scores near the selection boundary are spikes or noise.
Captured fraction is not the answer: it includes irreducible noise energy across
the complete local channel-by-time waveform.

## Job outcome summary (2026-08-28)

### Score-8 chain — COMPLETE
- `16513755` (`xyzsig0014-full`): **CANCELLED+** 13:31 after 2h02m at chunk
  1094/1958 (USR1 signal trap, not a pursuit failure; all passes healthy).
- `16517915` (`xyzsig0014-full`, manual requeue of the same sbatch): **COMPLETED**
  14:30–16:07 (1h36m52s). `--resume` continued from chunk ~1094 and finished the
  full recording. Output: `runs/dataset1_p1/0014_xyzsig_full_score8/`.
- `16517916` (`plot-x8`, `afterok:16517915`): **COMPLETED** 16:08–16:11 (3m10s).
  Eight-panel suite written to `out/0014_xyzsig_full_score8/`.
- **Score-8 full recording + plots are done.**

### Score-6 chain — INCOMPLETE (stopped mid-run)
- `16514699` (`xyzsig0014-full-s6`): **CANCELLED+** 14:31 after 2h27m at chunk
  772/1958 (USR1 signal trap). Its auto-requeue did **not** materialize as a new
  running instance — only the OOD-jupyter job is in the queue as of 16:15.
- `16517890` (`plot-s6`, `afterok:16514699`): **CANCELLED** — its dependency was
  cancelled, so the afterok constraint was unsatisfiable.
- To finish score-6: resubmit `residuals/src/preprocessing/run_0014_xyzsig_full_score6.sbatch`
  (`--resume` picks up at chunk ~772), then reseck the plot job with
  `afterok:<newwid>`.

## Resolved smoke results (2026-08-28)

- `16503279`, `16503283`, `16503486` all completed. Score-8 smoke: 2m56s,
  202,654 events (vs 356,315 in corrected score-6 smoke). Every accepted pass
  reduced residual energy.
- Audit: max direct captured-fraction arithmetic error 1.91e-7; fitted scores
  ≥ 8.00029 on chunk 0; zero later-pass events within the 0.5 ms/48 um
  cross-pass exclusion.
- Eight standard plots at `out/0014_xyzsig_smoke_16503279/`.

## Full-recording runs (queued 2026-08-28)

- Score-8 (`16513755`/`16517915`): unchanged score-8 model, `all` stages, proposal
  and fitted-projection thresholds 8, four residual passes, 2,048-event fit
  batches, one-second chunks, 24 h limit, `--signal=B:USR1@60` + `--requeue`.
  Output `runs/dataset1_p1/0014_xyzsig_full_score8/`, account `torch_pr_60_general`.
- Score-6 (`16514699`): same config, thresholds 6, account `torch_pr_62_general`,
  output `runs/dataset1_p1/0014_xyzsig_full_score6/`. Requires resubmission to
  finish the last ~60% of chunks.

## Immediate decisions

1. Finish the score-6 full run (resubmit with `--resume`), then compare score 8
   directly with score-6 using counts by pass, pass-wise energy drops, fitted
   projection distributions, sigma/rho boundary mass, runtime, and plots.
2. Inspect examples near the score-8 boundary, not only high-input-energy or
   median examples. Require centered temporal structure, coherent multi-channel
   footprint, and post-fit normalized channel residuals consistent with noise.
3. Keep proposal and final-fit thresholds coupled unless a controlled experiment
   shows a reason to separate them. Never use captured fraction alone as the
   spike/noise gate.

## Empirical noise calibration

Construct a held-out null that preserves the recording's filtered temporal
spectrum and channel noise structure while destroying spike-like
spatiotemporal alignment. Run the identical Omega-by-spatial-bank search and
full xyz-sigma fit on that null. Report candidate and final fitted-score tails at
6, 7, 8, 9, 10, then choose the lowest threshold whose estimated false detection
burden is acceptable after the full bank search. Do not interpret a maximized
score as an uncorrected Gaussian sigma value.

Cross-check the chosen threshold with:
- score-stratified reconstruction pages;
- event-triggered averages by Omega row and residual pass;
- source, sigma, rho, and channel-boundary occupancy;
- time/channel recurrence and refractory-like short-lag structure;
- accepted event rate and residual-energy removal per pass.

## Performance work after scientific acceptance

Only after the score gate is selected:
1. Remove per-fit GPU-to-CPU geometry-key transfers by assigning stable geometry
   IDs before upload and reusing the existing GPU footprint cache.
2. Benchmark fit batches 2,048, 4,096, 8,192 on identical inputs; promote only
   exact, memory-safe settings.
3. Overlap bounded CPU read/filter/CMR for chunk `n+1` with GPU pursuit for
   chunk `n`, using pinned buffers within one GPU SLURM allocation.
4. Measure real GPU utilization, peak VRAM, host RSS, warmed chunk time. No
   artificial GPU-burn processes.

## Full-recording promotion gate

Do not promote the pipeline further until:
- the empirical null supports the chosen projection threshold;
- score-boundary examples look spike-like rather than selected noise;
- every accepted pursuit pass reduces residual energy;
- 0.5 ms/48 um cross-pass recurrence remains zero by construction;
- output schema and the complete established plot suite pass;
- the selected performance configuration sustains the allocation without host
  OOM or low-utilization termination.

## Links

- [[session-014-xyzsigma-residual-pursuit]]
- [[session-012-rho-implementation-plan]]
- [[session-009-ibl-style-pursuit]]

## Directory consolidation (2026-08-28, ~18:30)

- Top-level `runs/` and `out/` were moved into `residuals/runs/` and
  `residuals/out/` (no name collisions; `runs/dataset1_p1/0014_*` merged into
  the existing `residuals/runs/dataset1_p1/`). AGENTS.md now documents that all
  new outputs go under `residuals/`.
- The active 0014 sbatch scripts were updated to `$ROOT/residuals/runs/...`:
  `run_0014_xyzsig_full.sbatch`, `run_0014_xyzsig_full_score6.sbatch`,
  `run_0014_xyzsig_smoke.sbatch`, `run_0014_xyzsig_audit.sbatch`, and the two
  plot sbatch files. Other legacy sbatch scripts still point at the old
  top-level paths and need the same fix before reuse.

## Full-recording residual/raster plot (2026-08-28)

- `plot_temporal_codebook_depth_time_raster.py` now renders a literal
  rasterized Omega-coloured scatter plot, with fixed point size and each
  point's transparency scaled by normalized `|alpha|`.
- Added `plot_0014_full_recording_passes.py`: it replays each saved 0014 chunk
  on CUDA using the exact saved `predictions` arrays, pools each stage with a
  sign-preserving temporal extremum, and renders five rows × two columns:
  input plus each of the four residual passes on the left, against an empty
  then cumulative `|alpha|`-transparent Omega scatter plot on the right.
- The binned-raster render `16528417` and the marker-area scatter retry
  `16528656` were cancelled. The opacity-based scatter correction is ready but
  has not been resubmitted.

## Reconstruction + codebook plots (2026-08-28, ~18:20)

- Job `16525720` (`plot-x8-full`, 1m56s) regenerated the reconstruction
  examples with `--examples-per-pass 2` (8 columns: 2 per pass × 4 passes) and
  added a new codebook usage plot. Both are under
  `residuals/out/0014_xyzsig_full_score8/`.
- `plot_raw_residual_reconstructions.py` gained `--examples-per-pass` (default
  1, backward compatible); `choose_examples` now stratifies by input energy
  into per-pass bands and picks the median captured-fraction fit in each.
- New `residuals/src/plots/plot_0014_codebook_usage.py` reads the 0014 schema
  (`omega.npy`, `temporal_idx.npy`, `residual_pass.npy`) and renders each
  codebook row as waveform + overall usage fraction + per-pass fraction.
  Score-8 usage: row 5 dominates (24.34% overall, 25.7% P1 → 20.0% P4), row 1
  second (22.43%); rows 6/7 grow with pass (row 7: 7.9% P1 → 17.5% P4).
