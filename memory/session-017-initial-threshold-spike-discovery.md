# Initial-Threshold Spike Discovery
**Created:** 2026-08-29
**Last updated:** 2026-08-30

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

## Terminal status and final-chunk failure

- Threshold 8, array task `16575275_2`, completed all 1,958 chunks in
  `00:36:24`. Its `summary.json` reports 915,348 events. This number alone is
  not evidence that threshold 8 solves over-detection; wait for the complete
  plot suite and direct result review.
- The other five runs all saved 1,957 of 1,958 chunks and then failed in the
  final partial chunk:
  - threshold 6 (`16575057`): `FAILED`, exit `1:0`, `00:55:27`;
  - threshold 5 (`16575275_0`): `FAILED`, exit `1:0`, `00:53:00`;
  - threshold 7 (`16575275_1`): `FAILED`, exit `1:0`, `00:49:55`;
  - threshold 6 with 30-sample isolation (`16575400_0`): `FAILED`, exit
    `1:0`, `00:43:46`;
  - threshold 6 one-pass (`16575400_1`): `FAILED`, exit `1:0`, `00:33:13`.
- Their completed chunk shards are intact. After correcting the detector bound,
  each run is resumable with `--resume` and should need only the final chunk and
  output consolidation, not recomputation of the first 1,957 chunks.
- The shared failure is a center-coordinate boundary bug in the 0017 detector.
  The 0016 caller already passes valid event-center bounds:
  `valid_start = max(n_before, local_core_start - n_after)` and
  `valid_stop = min(len(data) - n_after + 1, local_core_stop + n_before)`.
  The 0017 detector incorrectly adds `n_before` again when constructing
  `peak_start` and `peak_stop`, shifting the allowed center interval forward by
  45 samples. In the final partial chunk, a selected proposal can therefore
  lie too close to the read boundary, and waveform extraction with offsets
  `[-45, ..., 44]` indexes beyond the residual tensor. The logs show CUDA
  `IndexKernel` out-of-bounds assertions; the one-pass run later surfaced a
  secondary `CUBLAS_STATUS_EXECUTION_FAILED` after the same assertions.
- Correct the 0017 NMS interval to use the center-aligned `valid_start` and
  `valid_stop` directly, then explicitly validate the final partial chunk.
  Resume the five failed runs, confirm all six summaries, and only then release
  plot array `16575437` and review the full localization/reconstruction plots.
- Plot array `16575437` remains `PENDING` with reason `JobHeldUser`.

## Resume check — 2026-08-30

- One-shot `squeue` and `sacct` checks found no state changes: threshold 8
  remains complete, the other five runs remain failed, and plot array
  `16575437_[0-5]` remains held by the user.
- Only the threshold-8 output has a `summary.json`; it still reports 915,348
  events across 1,958 chunks. The other five summaries remain absent.
- The detector boundary fix has not yet been applied: `peak_start` and
  `peak_stop` still add `n_before` to the already center-aligned `valid_start`
  and `valid_stop` values.

## Boundary fix and resumed runs — 2026-08-30

- `0017_initial_threshold_peeling.py` now passes the caller's center-aligned
  `valid_start` and `valid_stop` directly to spatiotemporal NMS.
- The focused self-test now covers 30-kHz waveform boundaries and requires
  centers 45 and 211, while rejecting out-of-support centers 44 and 212, for a
  256-sample buffer. Syntax validation and the self-test passed inside
  Singularity.
- Five saved runs were resumed without rerunning completed threshold 8:
  - `16591825`: threshold 6;
  - `16591826_0`: threshold 5;
  - `16591826_1`: threshold 7;
  - `16591827_0`: threshold 6 with 30-sample isolation;
  - `16591827_1`: threshold 6 one-pass.
- At the submission check, `16591825` and `16591826_0` were running; the other
  three tasks were pending because of `QOSMaxGRESPerUser`.
- Keep plot array `16575437` held until all six run summaries exist.

## Completed resumes and plot replacement — 2026-08-30

- All five resumed jobs completed with exit code `0:0`, and every run now has a
  1,958-chunk summary:
  - threshold 5: 2,835,467 events;
  - threshold 6: 1,784,872 events;
  - threshold 7: 1,247,808 events;
  - threshold 8: 915,348 events;
  - threshold 6 with 30-sample isolation: 1,595,629 events;
  - threshold 6 one-pass: 1,387,714 events.
- SLURM rejected release of held plot array `16575437`, individual task
  release, and a partition-only update with `Unspecified error`. The array had
  zero runtime and no outputs, so it was canceled after approval.
- Replacement plot array `16592119` runs the same six-task plot script. Task 0
  was running at the submission check; tasks 1–5 were pending because of
  `QOSMaxMemoryPerUser`.

## Links

- [[session-016-one-hot-lattice-peeling]]
- [[session-009-ibl-style-pursuit]]
