# Session 014: XYZ-Sigma Residual Pursuit
**Created:** 2026-08-27
**Last updated:** 2026-08-28

## Context

Replace the active rho-localization direction with a discrete `(x, y, z, sigma, q)` residual model. Reuse session 0012's GPU detection/codebook initialization and port fit-lattice's geometry-grouped assignment, but retain exactly one temporal codebook waveform and one gain per event.

## Finalized plan

Create `residuals/src/preprocessing/0014_xyzsig_residual.py` as a resumable pipeline with `calibration-detect`, `alternating-fit`, `pursue`, and `all` stages.

1. `calibration-detect` scans the complete requested recording with session 0012's detection/NMS behavior, then writes deterministic isolated-event keys and local geometry metadata. It samples across the whole recording with Q8 defaults: 100,000 events, 32 sampled chunks, 1 ms isolation, and a fixed seed.
2. Initialize Omega using session 0012's existing one-waveform-per-event codebook procedure.
3. `alternating-fit` streams the fixed calibration events without retaining all raw waveforms in memory. Group identical `(local_coords, mask)` configurations, cache their normalized `site × sigma` monopole dictionaries, and assign one stable `(x, y, z, sigma, q)` candidate per event. Refit every Omega row from its assigned events by the closed-form least-squares update, normalize it, retain the prior row if empty, and repeat to tolerance or iteration limit.
4. Save learned `omega.npy`, assignment shards, objective history, row counts, footprint-cache diagnostics, and the complete stage configuration.
5. `pursue` starts fresh from raw recording data with learned Omega frozen. Keep one-second chunks, four residual passes, GPU detection/subtraction, existing gates, and stable peak/subtraction ordering. Use discrete lattice fitting only: no rho and no continuous refinement.

## Invariants and batching

- Reconstruction is `alpha * monopole(x, y, z, sigma) * Omega[q]`: one codebook row and scalar gain per event.
- Save `rho = sqrt(z^2 + sigma^2)` only as a diagnostic. The discrete `z` and `sigma` labels are deterministic dictionary selections, not separately identifiable physical estimates.
- Preserve strict `>` candidate/tie behavior and restore original event order after grouped scoring.
- Default final-pursuit fits to 2,048 events. Benchmark 4,096 and 8,192 on identical inputs, then promote only an exact, memory-safe batch size while retaining one-second residual state.
- Do not reuse session 0013's rho path or `fit_lattice.py`'s rank-Q temporal mixtures. Port only its geometry caching and GEMM-style discrete assignment design.

## Validation and promotion

- Synthetic exact reconstruction, boundary/mask cases, and candidate-tie tests.
- Bitwise grouped-versus-uncached assignment/prediction comparison.
- Non-increasing alternating-fit objective; empty codebook rows remain unchanged.
- Fixed-input 2,048/4,096/8,192 GPU timing, peak-memory, and utilization comparison.
- Short CUDA all-stage smoke, then one-second fresh-pursuit smoke with finite output, valid checkpoints, stable pass order, and positive accepted-pass energy reduction.
- Do not submit a full-recording session-0014 run until those checks pass.

## Progress (2026-08-27)

- Added `residuals/src/preprocessing/0014_xyzsig_residual.py` as the new
  stage-driven entry point. It has deterministic calibration detection shards,
  geometry-keyed cached coarse `(x, y, z, sigma, q)` assignment, closed-form
  one-row codebook updates, iteration assignment shards, resume metadata, and
  one-second frozen-codebook pursuit that routes session-0012 detection and
  subtraction through the grouped localizer.
- The Singularity `py_compile` check passed. No CUDA smoke or full-recording
  run has been submitted; those remain required before promotion.

## CUDA validation and selection repair (2026-08-28)

- Initial CUDA smoke `16500465` exposed an undefined `positions` reference in
  calibration; `16501772` then stopped at the pre-existing partial output.
  Calibration now obtains the channel count from `fit_ids`, and smoke outputs
  are job-specific.
- All-stage smoke `16501854` completed in 3m09s over 10 seconds of recording,
  proving calibration, alternating fitting, cached discrete localization,
  subtraction, rollback, and chunk checkpointing on CUDA. Its original
  score-6/5%-capture selection saved 526,688 events and repeatedly rolled back
  the last residual pass.
- 0014 now emits the established residual-run schema: complete config and
  metadata, channel/neighborhood artifacts, root `omega.npy`, legacy event
  names including `local_coords`, `profile_idx`, and `temporal_idx`, bounded
  memmap consolidation, and `residual_waveforms` by default. Delta-chi-squared
  is intentionally absent. The established plot suite consumes the output.
- Waveform-saving smoke `16502038` completed in 4m36s with 542,894 events.
  Plot job `16502115` produced the main suite before its final spiketensor panel
  hit a non-event-array indexing assumption; that loader was repaired and all
  eight figures were produced under `out/0014_xyzsig_smoke_16502038/`.
- Audit `16502593` verified saved reconstruction arithmetic to `2.24e-7`
  maximum captured-fraction error. The 5% captured fraction was not a useful
  spike/noise discriminator because it divides explained atom energy by the
  full multi-channel waveform energy, including hundreds of noise degrees of
  freedom. Median accepted capture was about 12%, while post-fit normalized
  channel RMSE was near one noise unit.
- The actual selection defects were a missing final-fit projection gate and no
  cross-pass lockout. In the old run, 9.1% of saved fits had final projection
  score below the proposal threshold of 6; 77.8% of pass-2 and 93.1% of pass-3
  events were within 0.5 ms/96 um of an earlier event, and essentially all were
  within the full 3 ms support.
- 0014 now requires `sqrt(captured_energy) >= min_fitted_projection`, carries
  the existing 0.5 ms/48 um NMS exclusion across residual passes, and treats
  captured fraction as a diagnostic (`min_captured_fraction=0`). The atom model,
  Q8 codebook, discrete lattice, sigma bank, and one-gain invariant are unchanged.
- Corrected score-6 smoke `16502643` completed in 3m44s with 356,315 events.
  Accepted counts decayed cleanly by pass (roughly 21k, 10k, 3k, and <1k per
  one-second chunk), and every pass produced a positive residual-energy drop.
  Audit `16502874` confirmed zero later-pass events within 0.5 ms/48 um of an
  earlier event and a minimum saved fitted projection score of exactly 6.
  Plot suite `16502893` completed successfully under
  `out/0014_xyzsig_smoke_16502643/`.
- Because maximizing over time, channels, Q8, nine scales, and the xyz lattice
  creates a substantial multiple-comparisons tail, the conservative current
  defaults are proposal threshold 8 and final fitted-projection threshold 8.
  Score-8 smoke `16503279` is running; dependent audit `16503283` and plot suite
  `16503486` are queued.

## Score-8 validation resolved (2026-08-28)

- The score-8 chain completed cleanly: CUDA smoke `16503279` (2m56s), audit
  `16503283` (7s), and the full eight-panel plot suite `16503486` (1m22s).
  The run is at `runs/dataset1_p1/0014_xyzsig_smoke_16503279/` and its figures
  are at `out/0014_xyzsig_smoke_16503279/`.
- The 10-second score-8 run saved 202,654 events (43.1% fewer than the
  corrected score-6 run's 356,315), with every residual pass accepted and
  producing a positive energy reduction. On audit chunk 0, the saved fitted
  projection score ranged from 8.00029 to 101.22 (median 10.44), and direct
  captured-fraction reconstruction agreed to 1.91e-7 maximum absolute error.
- Cross-pass lockout remained exact: no pass-2/3/4 event lay within
  0.5 ms/48 um of an earlier event. The generated plot suite includes the
  score-boundary data but those examples have not yet been scientifically
  reviewed.

## Current promotion status

- CUDA execution, resumability, output compatibility, arithmetic consistency,
  cross-pass duplicate suppression, and positive pass-wise energy reduction
  have passed short-run validation.
- A full-recording run is still blocked on reviewing score-boundary examples
  against score 6 and calibrating the detection threshold against an empirical
  noise/null control.
- Continue in [[session-015-score-calibrated-xyzsigma-promotion]].

## Links

- [[session-012-rho-implementation-plan]]
- [[session-013-rho-localization-optimization-plan]]
- [[session-015-score-calibrated-xyzsigma-promotion]]
