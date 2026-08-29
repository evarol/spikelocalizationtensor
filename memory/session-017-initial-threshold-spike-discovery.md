# Initial-Threshold Spike Discovery
**Created:** 2026-08-29
**Last updated:** 2026-08-29

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
- Job `16575057` was `RUNNING` at the 2026-08-29 status snapshot.
- Plot script:
  `residuals/src/plots/0017_initial_threshold_plots.sbatch`.
- Plot output:
  `residuals/out/0017_initial_threshold6_learned_fitted8/`.
- The plot suite includes peeling/stopping diagnostics, localization cohorts,
  full XYZ density, XYZ-sigma scatter, saved reconstruction boundary/round
  examples, raw residual reconstructions, SpikeTensor-style summaries, and the
  Omega depth-time raster.
- Submit plots only after `summary.json` exists and the full job completes.

## Queued experiment matrix

- Commit `29cec35` adds the full-recording threshold sweep and structural
  checks. All runs use the same recording, learned Omega path, localization,
  reconstruction, fit gates, 48-um spatial neighborhood, and fitted-projection
  threshold 8 unless stated otherwise.
- Threshold sweep array `16575275`:
  - task 0: threshold 5, 60 peeling rounds, output
    `residuals/runs/dataset1_p1/0017_initial_threshold5_learned_fitted8/`;
  - task 1: threshold 7, 60 peeling rounds, output
    `residuals/runs/dataset1_p1/0017_initial_threshold7_learned_fitted8/`;
  - task 2: threshold 8, 60 peeling rounds, output
    `residuals/runs/dataset1_p1/0017_initial_threshold8_learned_fitted8/`.
- Structural array `16575400` at threshold 6:
  - task 0: reject local maxima having another local maximum within ±30
    samples and the 48-um neighborhood, output
    `residuals/runs/dataset1_p1/0017_initial_threshold6_isolated30_learned_fitted8/`;
  - task 1: run exactly one peeling round with no wide isolation, output
    `residuals/runs/dataset1_p1/0017_initial_threshold6_onepass_learned_fitted8/`.
- At the submission snapshot, `16575275_0` was running; threshold-7,
  threshold-8, and both structural tasks were pending.
- The optional isolation implementation defaults to zero and omits the new
  metadata field at zero. This preserves resume compatibility with threshold-6
  job `16575057`, which started before the optional field was added.
- Do not use SLURM `afterok` plot dependencies for these requeueable jobs.
  Submit the complete plot suite after each run has a completed `summary.json`.
- `residuals/src/plots/0017_initial_threshold_matrix_plots.sbatch` maps all six
  outputs to the same full plotting suite as a six-task CPU array. Submit its
  tasks only after the corresponding run summaries exist.
- Plot array `16575437` was submitted with `--hold`. Keep all six tasks held
  until their corresponding `summary.json` files exist, then release the array
  rather than submitting a second plot job.

## Links

- [[session-016-one-hot-lattice-peeling]]
- [[session-009-ibl-style-pursuit]]
