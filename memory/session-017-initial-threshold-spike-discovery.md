# Initial-Threshold Spike Discovery

## Decision

- The user rejected adding a separate event-level no-spike model before testing
  the simpler IBL-style root: thresholding for initial spike discovery.
- IBL source inspection confirmed that its initial discovery uses negative local
  minima below `spkTh=-6`; its later learned-template pursuit is a distinct
  normalized score threshold and does not contain a no-spike classifier.
- Do not interpret the 0017 outputs before the user has reviewed the complete
  plots, especially localization and saved reconstruction examples.

## Baseline snapshot

- Commit `ef2ae76` (`Add one-hot lattice peeling evaluation`) records the complete
  pre-0017 worktree, including 0016, its corrected plot suite, and the related
  0014 score-9 work.

## 0017 implementation

- Entry point:
  `residuals/src/preprocessing/0017_initial_threshold_peeling.py`.
- 0016 remains unchanged and is loaded as the downstream localization,
  reconstruction, acceptance, subtraction, and rescoring implementation.
- Proposal discovery now scores every sample as
  `-residual_voltage / per-channel_MAD_noise`.
- A proposal must reach threshold 6 and be a spatiotemporal maximum within five
  samples and the existing 48-um detection neighborhood.
- No temporal codebook row, sigma, or analytic spatial template is searched to
  create a proposal. The saved initial sigma/temporal indices are therefore `-1`.
- The first controlled comparison retains the downstream fitted-projection gate
  at 8 and all other 0016 fit/merge gates. This isolates the proposal-detector
  change; it is not a claim that those gates are optimal.
- The focused CPU self-test and syntax validation passed inside Singularity.

## Full run and plots

- Full-recording job `16575057` was submitted on 2026-08-29 using
  `residuals/src/preprocessing/0017_initial_threshold_full.sbatch`.
- Output:
  `residuals/runs/dataset1_p1/0017_initial_threshold6_learned_fitted8/`.
- Initial SLURM state was `PENDING`.
- Plot script:
  `residuals/src/plots/0017_initial_threshold_plots.sbatch`.
- Plot output:
  `residuals/out/0017_initial_threshold6_learned_fitted8/`.
- The plot suite includes peeling/stopping diagnostics, localization cohorts,
  full XYZ density, XYZ-sigma scatter, saved reconstruction boundary/round
  examples, raw residual reconstructions, SpikeTensor-style summaries, and the
  Omega depth-time raster.
- Submit plots only after `summary.json` exists and the full job completes.
