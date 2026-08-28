# Session 015: Score-Calibrated XYZ-Sigma Promotion
**Created:** 2026-08-28
**Last updated:** 2026-08-28

## Context

Promote session 0014's fixed `alpha * monopole(x,y,z,sigma) * Omega[q]`
model without starting another model lineage. The remaining question is whether
maximized template scores near the selection boundary are spikes or noise.
Captured fraction is not the answer: it includes irreducible noise energy across
the complete local channel-by-time waveform.

## Active jobs

- `16503279`: 10-second all-stage CUDA smoke with proposal and final fitted
  projection thresholds both set to 8.
- `16503283`: `afterok:16503279` audit of reconstruction arithmetic, fitted-score
  distribution, gate retention, and cross-pass recurrence.
- `16503486`: `afterok:16503283` established full residual plot suite, writing to
  `out/0014_xyzsig_smoke_16503279/`.

## Resolved job results (2026-08-28)

- `16503279`, `16503283`, and `16503486` all completed successfully. The
  score-8 smoke took 2m56s and saved 202,654 events, compared with 356,315 in
  the corrected score-6 smoke. Every accepted pass reduced residual energy.
- Audit confirmed a 1.91e-7 maximum direct captured-fraction arithmetic error,
  fitted projection scores no lower than 8.00029 on audit chunk 0, and zero
  later-pass events within the 0.5 ms/48 um cross-pass exclusion.
- All eight standard plots are present at
  `out/0014_xyzsig_smoke_16503279/`. The active work is now the score-6/8
  boundary-example review and empirical-null calibration, not job monitoring.

## Full-recording run queued (2026-08-28)

- User authorized the full-recording run before completing the remaining
  score-boundary/null checks. Job `16513755` (`xyzsig0014-full`) is pending on
  `l40s_public` with a 24-hour limit.
- It uses the unchanged score-8 model over the entire recording: `all` stages,
  proposal and final fitted-projection thresholds of 8, exactly four residual
  passes, 2,048-event fit batches, and one-second chunks. Its resumable output
  is `runs/dataset1_p1/0014_xyzsig_full_score8/`.

## Corrected score-6 comparison queued (2026-08-28)

- User requested the valid full-recording score-6 comparison on a separate
  allocation. Job `16514699` (`xyzsig0014-full-s6`) is running on
  `torch_pr_62_general`; score-8 job `16513755` remains on
  `torch_pr_60_general`.
- Both runs use the repaired final projection gate and cross-pass lockout,
  exactly four passes, 2,048-event batches, and one-second chunks. The
  independent score-6 output is `runs/dataset1_p1/0014_xyzsig_full_score6/`.

## Immediate decisions

1. Resolve all three active jobs and compare score 8 directly with corrected
   score-6 run `16502643`, using counts by pass, pass-wise energy drops, fitted
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
6, 7, 8, 9, and 10, then choose the lowest threshold whose estimated false
detection burden is acceptable after the full bank search. Do not interpret a
maximized score as an uncorrected Gaussian sigma value.

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
2. Benchmark fit batches 2,048, 4,096, and 8,192 on identical inputs; promote
   only exact, memory-safe settings.
3. Overlap bounded CPU read/filter/CMR for chunk `n+1` with GPU pursuit for chunk
   `n`, using pinned buffers within one GPU SLURM allocation.
4. Measure real GPU utilization, peak VRAM, host RSS, and warmed chunk time. Do
   not use artificial GPU-burn processes.

## Full-recording promotion gate

Do not submit the 1,957.1908-second run until:

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
