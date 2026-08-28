# Session 007: Fresh-Raw Global Codebook Pursuit
**Created:** 2026-08-25
**Last updated:** 2026-08-25

## Context
Test a non-unit-specific residual model whose temporal vocabulary is learned directly from fresh detections on the raw recording. Fit global temporal banks with `Q = 8, 16, 24, 32`, combine each bank with the same ten analytic spatial scales, and repeatedly detect, localize, reconstruct, subtract, and rescore so colliding spikes can emerge in later pursuit rounds. Separately benchmark larger localization superbatches as a way to put more independent solver work on the GPU at once.

## Correction and data split
- The first submitted implementation incorrectly trained from saved `neighborhood_waveforms.npy`. That entire dependency chain was cancelled: training `16344079`, calibration `16344119`, pursuit array `16344132`, and comparison `16344187`.
- The replacement reads the raw `.ap.bin` through `spikeglx.Reader`, preprocesses one-second chunks, estimates robust channel noise, freshly detects generic negative-voltage local peaks at threshold `6`, extracts their 90-sample neighborhoods, and fits every Q from the same raw-derived sample.
- Codebook training scans `[60 seconds, recording end)`. It never loads old `spike_times`, `spike_channels`, or `neighborhood_waveforms`.
- The first `8.748` seconds are held out for threshold calibration and pursuit comparison. Training and evaluation do not overlap.
- Up to 250,000 events are sampled across the training scan with seed `2026`. The saved `training_*.npy` arrays and `training_metadata.json` make the exact fresh-detection training set auditable.

## Scientific design
- The codebook is global, not unit-specific: one temporal bank is shared by the entire recording.
- Each Q bank is paired with ten analytic monopole scales, so template scoring searches `Q x 10` separable temporal-spatial combinations at every valid time and anchor.
- All Q values use the same fresh raw sample and the existing normal codebook initialization plus alternating localization/codebook updates. Each fitted codebook is frozen before evaluation.
- Greedy pursuit handles collisions by scoring the current residual, selecting a nonoverlapping group, localizing and subtracting it, then rescoring. A spike obscured by a stronger overlapping spike can therefore appear in a later round.
- More rows increase waveform expressiveness and may improve reconstruction, but do not by themselves guarantee biological unit identity or correct event count.
- Maximizing over `10 x Q` templates creates a look-elsewhere effect. Q8 uses threshold `6`; Q16/Q24/Q32 receive matched thresholds chosen to yield the same held-out raw candidate count as Q8 before pursuit.
- Support-aware pursuit grouping remains a proposed follow-up and is not implemented. The current group lockout uses nonoverlapping temporal support.

## Committed foundation and file map
- Commit `945d5fa`: `src/maths.py` contains fixed-codebook localization and profiler scopes; `src/preprocessing/raw_residual.py` contains baseline residual extraction, pursuit/rescoring, frozen-codebook learning, profiling, and atomic chunk output. `raw_template_residual_profile_15m.sbatch` and `raw_template_residual_pursuit_codebook_smoke.sbatch` launch the corresponding profiler and smoke runs.
- Commit `b5364ae`: the pursuit/codebook ablation launch, comparison, and plotting utilities.
- Commit `8a6cc0d`: `plot_raw_residual_collisions.py` and `plot_temporal_codebook_depth_time_raster.py` for collision examples and dense codebook-colored rasters.
- Commit `03ecae0`: removed all repository test files at the user's request. This experiment uses syntax/import checks and controlled smoke runs, not a test suite.
- Commits `89d88f1`, `acaaec7`, and `58f3fc7`: the new global-Q calibration, pursuit, comparison, and corrected fresh-raw training workflow. The saved-waveform training launcher from the first attempt was deleted.
- Commit `bfe0565`: `benchmark_global_localization_superbatch.py` and its launcher benchmark localization batch sizes `256, 512, 1024, 2048` on saved pursuit waveforms while comparing numerical output with a reference batch size.

## New scripts
- `src/preprocessing/fit_raw_global_temporal_codebooks.py`: fresh raw detection, reproducible training-sample capture, and sequential Q8/Q16/Q24/Q32 fitting.
- `src/preprocessing/fit_raw_global_temporal_codebooks.sbatch`: 18-hour GPU launcher for the full raw scan and four fits.
- `src/preprocessing/calibrate_global_codebook_thresholds.py`: held-out per-Q candidate-score calibration.
- `src/preprocessing/run_global_codebook_pursuit.py`: frozen-codebook, 60-round residual pursuit wrapper with matched or fixed thresholds.
- `src/preprocessing/compare_global_codebook_pursuit.py`: Q comparison of rate, captured fraction, remaining core energy, row usage, redundancy, runtime, and event overlap against Q8.
- `src/preprocessing/benchmark_global_localization_superbatch.py`: isolated larger-batch localization timing and equivalence benchmark.
- Each Python entry point has a new `.sbatch` launcher; no legacy script was changed for this experiment.

## Validation and completed jobs
- A controlled CPU smoke run completed the whole raw read -> fresh detection -> waveform extraction -> Q2 fit path: 1,069 fresh detections in 0.1 seconds, 16 sampled events, final nMSE `0.9291789`, and row counts `[8, 8]`. The temporary output was removed.
- Syntax, import, shell-syntax, and whitespace checks passed. No repository tests were added or run after test removal.
- Raw-codebook job `16344879` completed with exit `0:0` in `01:00:55`; its output is `runs/dataset1_p1/raw_global_codebooks_16344879/` and contains the 250,000-event fresh-raw training sample plus all four fitted banks.
- Threshold calibration `16344923` completed with exit `0:0` in `00:00:53` and wrote `global_codebook_thresholds.json` inside the codebook directory.
- Matched-threshold pursuit tasks `16344946_0` through `_3` completed with exit `0:0` in `4:47`, `5:26`, `6:04`, and `6:22` for Q8/Q16/Q24/Q32. Outputs are `runs/dataset1_p1/global_codebook_pursuit_q{Q}_matched_16344946_{task}/`.
- CPU comparison `16344947` completed with exit `0:0` in `00:00:13` and wrote `runs/dataset1_p1/global_codebook_pursuit_matched_16344946/comparison.json`.
- Q32 localization-superbatch benchmark `16345636` completed with exit `0:0` in `00:00:52` and wrote `q32_localization_superbatch.json` beside the comparison.

## Results
- Fresh-raw refined training nMSE improves monotonically with codebook size: Q8 `0.457843`, Q16 `0.440314`, Q24 `0.429810`, and Q32 `0.422310`.
- Matched thresholds are Q8 `6.00005`, Q16 `6.03269`, Q24 `6.55326`, and Q32 `6.69276`; each gives exactly 203,604 initial held-out candidates over the eight-second calibration interval.
- Held-out pursuit event rates remain nearly fixed at `14,947.2`, `14,979.0`, `14,962.6`, and `14,947.5` events/s for Q8/Q16/Q24/Q32.
- Mean accepted-event captured fraction improves from `0.22455` at Q8 to `0.23870` at Q32. Mean remaining core-energy fraction after 60 rounds falls from `0.70692` to `0.69457`.
- Pursuit wall time over 8.748 seconds rises from `241.0 s` at Q8 to `344.9 s` at Q32. Q32 therefore improves reconstruction modestly but costs about 43% more runtime.
- Relative to Q8, Q32 matches `85.36%` of Q8 events within three samples, but only `39.46%` when the same anchor channel is required. Larger Q changes spatial/anchor assignments substantially despite nearly unchanged event rate.
- All codebook rows are used. Effective pursuit-row counts are `7.87`, `14.93`, `22.10`, and `29.12`, so added rows are not simply dead capacity; the banks nevertheless contain increasingly close row pairs.
- Localization throughput rises from `435.9` events/s at batch 256 to `1,869.4` events/s at batch 2048, while peak allocated GPU memory rises from `1.68` to `13.04 GiB`. Temporal-row assignments agree exactly and fitted energy changes are negligible, but the maximum source-coordinate difference is `1.52 µm` at batches 512/1024 and `218 µm` at 2048. Do not promote 2048 without diagnosing the spatial outlier distribution.

## Figures
- The one-off plotting sources created specifically for this result were removed at the user's request. Commit `21253b4` added only `src/plots/plot_global_codebook_existing.sbatch`, an orchestration launcher that calls the repository's established plotting programs unchanged.
- Plot job `16349352` completed with exit `0:0` in `00:11:46`. Per-Q outputs are under `out/global_codebook_per_q_16344946/q{8,16,24,32}/` and include localization XY/YZ/XZ scatter, reconstruction examples and diagnostics, four chunk-level energy-loss figures, temporal codebook values, and temporal-row usage percentages. All figures are saved at 800 DPI.
- In `reconstructions/reconstruction_examples.png`, the bottom row is a one-dimensional temporal-fit check, not the complete recording reconstruction or post-subtraction residual. Red is each accepted event's saved local residual waveform projected through its fitted unit-normalized spatial footprint, `sum_c footprint[c] * measured[c,t]`. Green dashed is the selected global temporal row scaled by its fitted amplitude, `alpha * omega[q,t]`.
- The first energy-diagnostic render was `50,471 x 8,322` because `plot_residual_pass_diagnostics.py`, originally designed for a few passes, inserted all P2-P60 recurrence percentages into one subplot title and made a 59-entry legend. `bbox_inches="tight"` expanded the canvas and squashed the six actual panels.
- Commit `ae11eb4` fixes the 60-pass layout without dropping the underlying diagnostics: all recurrence curves remain plotted, full recurrence percentages remain in stdout, the title is short, the legend shows six representative passes, pass ticks are thinned, bar labels are omitted at high pass counts, and large usage heatmaps omit per-cell text.
- Energy-only array job `16349527`, submitted with `afterok:16349352`, completed all Q8/Q16/Q24/Q32 tasks with exit `0:0` in 43-45 seconds each. It replaced only `recording_energy_loss_after_each_pass_chunk{0,1,2,3}.png`; all 16 corrected files are `12,890 x 7,293`. Stderr contains only the routine read-only overlay FUSE2FS warning.

## Next steps
- Diagnose the 2048-event localization source-coordinate outliers and plot the full displacement distribution; batch 1024 is the safer current performance candidate.
- Build exact sequential collision reconstructions from selected local cross-round pairs before claiming biological collision recovery.
- Decide whether Q32's roughly 1.24 percentage-point extra core-energy removal over Q8 justifies its 43% runtime increase. Q16 may be the better cost-quality knee.
- Only after those checks should a new multi-chunk or support-aware scheduler be implemented, again without editing legacy scripts.

## Full-recording pursuit (queued plan, decisions pending)
- Context: the held-out 8.748 s evaluation window was a leftover holdout-style choice. This project is analytical math, not ML — there is no valid "holdout", so pursuit should run over the entire 1957.2 s recording (58,715,724 samples @ 30 kHz, 384 channels).
- The pursuit CLI already supports this: `run_global_codebook_pursuit.py` takes `--duration-seconds`; the prior run used `8.748`. `run_recording()` in `raw_residual.py` already supports `resume` and `USR1` requeue patterns (see `raw_template_residual_full.sbatch`).
- Full-run scale (vs the 8.748 s run): ~224x data, 895 chunks at 2.187 s (vs 4), ~29M events (vs ~131k), ~17-18 h Q8 / ~24 h Q32 wall time, ~90 GB disk per Q with `--save-waveforms`.
- Decision points presented, not yet chosen: (1) wall time / restartability — recommend `--signal=B:USR1@60` + `--requeue` + add `--resume` flag to the pursuit CLI; (2) chunk size — 2.187 s keeps byte-identical logic, larger chunks speed IO but shift grouping at boundaries; (3) thresholds — calibrated matched (Q8 `6.00005` ... Q32 `6.69276`, equal candidate counts) for comparability vs plain fixed `6.0`; (4) `--save-waveforms` — drives ~90 GB/Q for reconstruction plots, but codebook-values/usage plots only need `omega` + `temporal_idx`.
- Recommended defaults: add resume support, run each Q as its own ~24 h job with requeue, keep 2.187 s chunks and matched thresholds, drop `--save-waveforms` unless reconstruction examples are wanted.

## Links
- [[session-008-peak-channel-codebook-init]]
- [[session-005-residual-profiler]]
- [[session-004-continuous-residual]]
- [[project_overview]]
