# Fresh-Raw Global Codebook Pursuit (007)
**Created:** 2026-08-25
**Last updated:** 2026-08-25
**Status:** Done — Q8/Q16/Q24/Q32 matched-threshold pursuit and comparison complete on the 8.748 s window; the full-recording plan was written but never executed

## Idea

Test a non-unit-specific residual model whose temporal vocabulary is learned
directly from fresh detections on the raw recording: fit global temporal
banks at `Q = 8, 16, 24, 32`, pair each with the same ten analytic spatial
scales, and repeatedly detect → localize → reconstruct → subtract → rescore
so colliding spikes can emerge in later pursuit rounds. Separately, benchmark
larger localization superbatches as a way to put more independent solver work
on the GPU at once.

The first submission mistakenly trained from saved
`neighborhood_waveforms.npy`; that entire chain (`16344079`, `16344119`,
`16344132`, `16344187`) was cancelled. The replacement reads the raw
`.ap.bin` through `spikeglx.Reader`, preprocesses one-second chunks,
estimates robust channel noise, detects fresh negative-voltage peaks at
threshold 6, extracts 90-sample neighborhoods, and fits every Q from the
same raw-derived sample. Training scans `[60 s, recording end)` — never
touching old `spike_times` / `spike_channels` / `neighborhood_waveforms` —
and the first 8.748 s are held out for threshold calibration and pursuit
comparison. Up to 250,000 events are sampled with seed 2026; the saved
`training_*.npy` arrays and `training_metadata.json` make the training set
auditable.

## Design notes

- One global (not unit-specific) temporal bank shared by the whole
  recording; each bank pairs with ten analytic monopole scales, so scoring
  searches Q × 10 separable temporal-spatial combinations at every valid
  time and anchor.
- Greedy pursuit handles collisions by scoring the current residual,
  selecting a nonoverlapping group, localizing and subtracting it, then
  rescoring — a spike obscured by a stronger neighbor can surface in a later
  round.
- Maximizing over 10 × Q templates is a look-elsewhere effect, so Q16/Q24/
  Q32 received matched thresholds calibrated to yield the same held-out raw
  candidate count as Q8 before pursuit.
- More rows add waveform expressiveness, not biological identity or a
  correct event count by themselves.
- Support-aware pursuit grouping (merging peaks across the full support)
  was a proposed follow-up, never implemented; the group lockout used
  nonoverlapping temporal support.

## Results

Foundation commits: `945d5fa` (fixed-codebook localization + profiler scopes
in `src/maths.py`; baseline residual extraction, pursuit/rescoring,
frozen-codebook learning, profiling, atomic chunk output in
`raw_residual.py`), `b5364ae` (ablation launch/compare/plot utilities),
`8a6cc0d` (collision + depth-time raster plots), `03ecae0` (repository test
files removed at your request — this project validates via syntax/import
checks and controlled smokes), `89d88f1` / `acaaec7` / `58f3fc7` (global-Q
calibration, pursuit, comparison, corrected fresh-raw training), `bfe0565`
(localization-superbatch benchmark).

All jobs completed: raw codebook fit `16344879` (1h00m55s; 250k-event
training sample + four banks), threshold calibration `16344923` (53s),
pursuit tasks `16344946_0..3` (Q8 4m47s → Q32 6m22s), CPU comparison
`16344947` (13s), superbatch benchmark `16345636` (52s).

- Training nMSE improves monotonically with Q: 0.457843 (Q8), 0.440314
  (Q16), 0.429810 (Q24), 0.422310 (Q32).
- Matched thresholds: 6.00005 / 6.03269 / 6.55326 / 6.69276 — each yielding
  exactly 203,604 held-out candidates over the calibration interval.
- Pursuit event rates barely move: ~14,950 events/s at every Q. Mean
  captured fraction rises 0.2246 (Q8) → 0.2387 (Q32); mean remaining core
  energy falls 0.7069 → 0.6946. Q32 costs ~43% more wall time (241 s →
  345 s over 8.748 s).
- Relative to Q8, Q32 matches 85.4% of events within three samples but only
  39.5% when the anchor channel must also agree — larger Q reshuffles
  spatial assignments despite the flat rate.
- All rows are used (effective pursuit-row counts 7.87 / 14.93 / 22.10 /
  29.12), though banks gain increasingly close row pairs.
- Superbatch: localization throughput 435.9 → 1,869.4 events/s from batch
  256 → 2048, with peak allocated GPU memory 1.68 → 13.04 GiB. Temporal-row
  assignments agree exactly and energy changes are negligible, but the max
  source-coordinate difference reaches 1.52 µm at batches 512/1024 and
  218 µm at 2048. Batch 1024 is the safer performance candidate; 2048 needs
  the spatial outlier distribution diagnosed first.

Plot job `16349352` (11m46s) produced per-Q localization scatters,
reconstruction examples/diagnostics, energy-loss figures, codebook values,
and usage percentages under `out/global_codebook_per_q_16344946/q{8,16,24,32}/`
(all 800 DPI). Reading note for `reconstruction_examples.png`: the bottom
row is a 1-D temporal-fit check — red is the saved local residual projected
through the fitted spatial footprint, green dashed is the selected row
scaled by its fitted amplitude — not a recording-wide reconstruction. The
first energy-figure render was 50,471 × 8,322 px because 59 recurrence
percentages landed in one subplot title; commit `ae11eb4` fixed the layout
and replacement array `16349527` rewrote all 16 energy figures at
12,890 × 7,293 px.

## Full-recording plan (written, never run)

The 8.748 s window was a leftover holdout habit — this is analytic
estimation, not ML, so there is no valid holdout; pursuit should cover the
whole 1,957.2 s recording (58,715,724 samples at 30 kHz, 384 channels). The
CLI already supports it (`--duration-seconds`; `--resume` and USR1 requeue
patterns exist in `raw_residual.py`). Scale vs the small run: ~224× data,
895 chunks at 2.187 s, ~29M events, ~17–18 h (Q8) / ~24 h (Q32) wall time,
~90 GB per Q with `--save-waveforms`. Recommended defaults were: add resume
support, run each Q as its own ~24 h job with `--signal=B:USR1@60` +
`--requeue`, keep 2.187 s chunks and matched thresholds, and drop
`--save-waveforms` unless reconstruction examples are wanted. Open decision
points (chunk size, thresholds, waveform saving) were left for a later
session — which instead went the cheaper peak-channel route in 008.

## Links

- [[session-008-peak-channel-codebook-init]]
- [[session-005-residual-profiler]]
- [[session-004-continuous-residual]]
- [[project_overview]]
