# XYZ-Sigma Residual Pursuit (0014)
**Created:** 2026-08-27
**Last updated:** 2026-08-28
**Status:** Done — validated through score-8; full runs handed to session 0015

## Why

Replace the rho-localization direction with a discrete `(x, y, z, sigma, q)`
residual model. Reuse 0012's GPU detection and codebook initialization, port
fit-lattice's geometry-grouped assignment, but keep exactly one temporal
codebook waveform and one gain per event.

## Design

`residuals/src/preprocessing/0014_xyzsig_residual.py` is a resumable
pipeline with `calibration-detect`, `alternating-fit`, `pursue`, and `all`
stages:

1. `calibration-detect` scans the whole requested recording with 0012's
   detection/NMS behavior and writes deterministic isolated-event keys plus
   local geometry metadata — 100,000 events over 32 sampled chunks, 1 ms
   isolation, fixed seed, Q8 defaults.
2. Omega initializes from 0012's one-waveform-per-event codebook procedure.
3. `alternating-fit` streams calibration events without holding all raw
   waveforms in memory: group identical `(local_coords, mask)` geometries,
   cache their normalized `site × sigma` monopole dictionaries, assign one
   stable `(x, y, z, sigma, q)` per event, refit each Omega row by
   closed-form least squares over its assigned events, normalize, keep the
   prior row if empty, and repeat to tolerance or iteration limit.
4. Saves learned `omega.npy`, assignment shards, objective history, row
   counts, cache diagnostics, and the full stage configuration.
5. `pursue` starts fresh from the raw recording with Omega frozen:
   one-second chunks, four residual passes, GPU detection/subtraction,
   existing gates, and discrete lattice fitting only — no rho, no continuous
   refinement.

Invariants: reconstruction is `alpha * monopole(x,y,z,sigma) * Omega[q]`;
`rho = sqrt(z² + sigma²)` is saved as a diagnostic only, since z and sigma
are deterministic grid selections, not separately identifiable physical
estimates. Strict `>` candidate/tie behavior with original event order
restored after grouped scoring. Default final fits of 2,048 events (benchmark
4,096/8,192 on identical inputs before promoting anything memory-safe). Do
not reuse 0013's rho path or fit_lattice's rank-Q temporal mixtures — port
only the geometry caching and GEMM-style discrete assignment design.

## What the smokes found and fixed

- Initial CUDA smoke `16500465` hit an undefined `positions` reference in
  calibration; `16501772` then stopped on the pre-existing partial output.
  Fixed both (channel count now from `fit_ids`; smoke outputs job-specific).
- All-stage smoke `16501854` (3m09s over 10 s of recording) proved the whole
  chain on CUDA: calibration, alternating fit, cached discrete localization,
  subtraction, rollback, chunk checkpointing. Its score-6 + 5%-capture
  selection saved 526,688 events and kept rolling back the last pass.
- Audit `16502593` confirmed the saved arithmetic (2.24e-7 max
  captured-fraction error) but showed the 5% captured fraction is a poor
  spike/noise discriminator: its denominator includes hundreds of noise
  degrees of freedom. Median accepted capture was ~12% with post-fit
  channel RMSE near one noise unit.
- The real defects were a missing final-fit projection gate and no
  cross-pass lockout: 9.1% of saved fits scored below the proposal threshold
  of 6, and 77.8% / 93.1% of pass-2 / pass-3 events sat within 0.5 ms/96 µm
  of an earlier event (essentially all within the full 3 ms support).
- Fixes: require `sqrt(captured_energy) >= min_fitted_projection`, carry the
  0.5 ms/48 µm NMS exclusion across residual passes, and demote captured
  fraction to a diagnostic (`min_captured_fraction=0`). Atom model, Q8
  codebook, discrete lattice, sigma bank, and the one-gain invariant
  unchanged.
- Corrected score-6 smoke `16502643` (3m44s, 356,315 events) decayed cleanly
  (~21k / 10k / 3k / <1k accepted per pass per chunk) with positive energy
  drops on every pass; audit `16502874` confirmed zero later-pass events
  within 0.5 ms/48 µm and a minimum saved score of exactly 6; plot suite
  `16502893` completed. (An earlier waveform-saving smoke `16502038` and its
  plot job `16502115` established the output schema and eight-figure suite;
  the spiketensor-panel loader bug found there was repaired.)

## Score-8 validation

Because maximizing over time, channels, Q8, nine scales, and the xyz lattice
creates a large multiple-comparisons tail, defaults moved to proposal
threshold 8 and final fitted-projection threshold 8. The score-8 chain
completed cleanly: smoke `16503279` (2m56s), audit `16503283` (7s),
eight-panel plots `16503486` (1m22s). The 10-second run saved 202,654 events
(43.1% fewer than corrected score 6), with every residual pass accepted and
positive energy reduction; fitted scores on audit chunk 0 ranged 8.00029 to
101.22 (median 10.44), and cross-pass lockout stayed exact. The
score-boundary examples in the plots were never scientifically reviewed —
that is where session 0015 picks up.

## Status at handoff

CUDA execution, resumability, output compatibility, arithmetic consistency,
cross-pass duplicate suppression, and positive pass-wise energy reduction
all passed short-run validation. Full-recording runs were gated on
score-boundary review and empirical-null calibration; see
[[session-015-score-calibrated-xyzsigma-promotion]].

## Links

- [[session-012-rho-implementation-plan]]
- [[session-013-rho-localization-optimization-plan]]
- [[session-015-score-calibrated-xyzsigma-promotion]]
