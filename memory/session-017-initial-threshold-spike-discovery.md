# Initial-Threshold Spike Discovery (0017)
**Created:** 2026-08-29
**Last updated:** 2026-08-30
**Status:** Done — all six runs and plot suites complete; totals await your review of the plots

## Why

Before adding any event-level no-spike model, test the simpler IBL-style
root: plain thresholding for initial spike discovery. Reading the IBL source
confirmed its discovery stage is exactly negative local minima below
`spkTh = -6`; its later learned-template pursuit is a different normalized
score threshold and contains no no-spike classifier. Standing rule: don't
interpret 0017 outputs before the complete plots — especially localization
and saved reconstruction examples — have been reviewed.

## Method

- Entry point:
  `residuals/src/preprocessing/0017_initial_threshold_peeling.py`. 0016 is
  unchanged and provides everything downstream (localization,
  reconstruction, acceptance, subtraction, rescoring).
- A proposal's score is `-residual_voltage / per-channel_MAD_noise`. It must
  reach threshold 6 and be a spatiotemporal maximum within five samples and
  the existing 48 µm detection neighborhood. No codebook row, sigma, or
  analytic spatial template is searched to make a proposal, so the saved
  initial sigma/temporal indices are `-1`.
- The first controlled comparison keeps the downstream fitted-projection
  gate at 8 and every other 0016 gate, isolating the detector change — it is
  not a claim that those gates are optimal.
- Baseline snapshot: commit `ef2ae76` ("Add one-hot lattice peeling
  evaluation") records the complete pre-0017 worktree, including 0016, its
  corrected plot suite, and the related 0014 score-9 work.

## Run matrix

Commit `29cec35` adds the full-recording threshold sweep and structural
checks. All runs share the recording, learned-Omega path, localization,
reconstruction, fit gates, 48 µm neighborhood, and projection threshold 8:

- threshold 6 full run `16575057` →
  `.../0017_initial_threshold6_learned_fitted8/`
- threshold sweep array `16575275`: tasks 0–2 run thresholds 5 / 7 / 8, 60
  peeling rounds each
- structural array `16575400` at threshold 6: task 0 rejects local maxima
  with another maximum within ±30 samples (plus the 48 µm neighborhood);
  task 1 runs exactly one peeling round with no wide isolation

The optional isolation implementation defaults to zero and omits its
metadata field at zero, preserving resume compatibility with the threshold-6
run that predated it.

Process rule learned here: never attach plots to these requeueable runs with
`afterok` — after a requeue the dependency can become unsatisfiable. Submit
the full suite only once each run's `summary.json` exists. The six-task plot
array `16575437` was submitted `--hold` for exactly this reason.

## The final-chunk bug and the fix

Five of the six runs saved 1,957 of 1,958 chunks and then crashed in the
final partial chunk (threshold 8, task `16575275_2`, completed cleanly in
36m24s with 915,348 events). Root cause: the 0016 caller already passes
center-aligned bounds,
`valid_start = max(n_before, local_core_start - n_after)` and
`valid_stop = min(len(data) - n_after + 1, local_core_stop + n_before)`, but
the 0017 detector added `n_before` *again* when building `peak_start` and
`peak_stop`, shifting the allowed center interval forward by 45 samples. In
the final partial chunk a selected proposal could land too close to the read
boundary, and waveform extraction with offsets `[-45, ..., 44]` ran off the
end of the residual tensor — the logs show CUDA `IndexKernel`
out-of-bounds assertions, and the one-pass run later added a secondary
`CUBLAS_STATUS_EXECUTION_FAILED` after the same asserts.

The fix: pass the caller's center-aligned `valid_start`/`valid_stop`
straight through to spatiotemporal NMS. The focused self-test now covers
30 kHz waveform boundaries (requires centers 45 and 211, rejects 44 and 212
in a 256-sample buffer). All completed chunk shards were intact, so each
failed run resumed with `--resume` and only had to finish the last chunk and
consolidate — no recomputation of the first 1,957 chunks.

## Final totals

Resume jobs `16591825` (thr 6), `16591826_0/1` (thr 5/7), and
`16591827_0/1` (thr 6 isolated / one-pass) all completed with exit `0:0`
(a few were initially pending on `QOSMaxGRESPerUser`). Every run now has a
1,958-chunk summary:

| run | events |
| --- | ---: |
| threshold 5 | 2,835,467 |
| threshold 6 | 1,784,872 |
| threshold 7 | 1,247,808 |
| threshold 8 | 915,348 |
| threshold 6, ±30-sample isolation | 1,595,629 |
| threshold 6, one pass | 1,387,714 |

## Plots

The held plot array `16575437` could not be released — SLURM rejected the
array release, per-task release, and a partition-only update with
`Unspecified error`. Since it had zero runtime and no outputs, it was
canceled with your approval. Replacement array `16592119` ran the same
six-task suite and completed all tasks with exit `0:0` (tasks 0–4 emitted
routine NumPy `Mean of empty slice` warnings; no task failed).

Both 0017 suite entry points now also run the reusable
`plot_0014_codebook_usage.py` panel as `temporal_codebook_usage.png` — every
learned Omega waveform with recording-wide and per-round usage — generated
for all six runs. The standalone depth-time Omega plot was restyled to the
same 1,750×960 smoothed categorical-density rendering as the SpikeTensor
raster while keeping its original layout and right-side Omega key; all six
outputs were regenerated.

A later `sacct` audit confirmed all terminal states and that every
`summary.json` still reports 1,958 chunks and exactly the totals above.

## Next steps

- [ ] Review the complete six-way plot matrix — especially localization and
      saved reconstruction examples — before interpreting the event totals
      or choosing a follow-up detector configuration.

## Links

- [[session-018-bipolar-prototype-cone-peeling]]
- [[session-019-all-channel-error]]
- [[session-016-one-hot-lattice-peeling]]
- [[session-009-ibl-style-pursuit]]
- [[feedback_plot_suite_completeness]]
