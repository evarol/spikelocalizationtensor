# Session 008: Peak-Channel Codebook Initialization
**Created:** 2026-08-25
**Last updated:** 2026-08-26

## Context
Replace the expensive 250,000-event multichannel analytic codebook initialization from session 007 with an IBL-inspired peak-channel sweep. The initialization should learn temporal shapes cheaply before using the existing frozen-codebook greedy residual pursuit.

## Decisions before implementation
- Sweep temporal codebook sizes `Q = 4, 8, 16, 32, 64`.
- Detect fresh raw events and learn each temporal bank from one-dimensional waveforms on the detected peak channel rather than localizing every multichannel training waveform over the analytic spatial grid.
- Keep the existing greedy residual matching pursuit after initialization: template scoring, conflict-free selection, analytic localization and reconstruction, sequential gain refitting, subtraction, and residual rescoring.
- Preserve a reproducible shared training sample across all Q values so the codebook-size comparison changes only Q.
- Record the implementation, validation, exact launch commands, SLURM job IDs, settings, and output paths in this card after jobs are queued.

## Required audit before launch
- Re-read and verify the residual-subtraction path, especially sequential overlap handling, gain refitting against the current residual, the positive-energy and captured-fraction acceptance gates, whole-round rollback, core-versus-margin accounting, and saved fitted amplitudes and waveforms.
- Verify that any new initialization does not mutate the frozen codebook during the comparison pursuit.
- Add controlled equivalence and monotonic-energy checks without restoring the removed repository test suite.

## Follow-up work
- Add optional codebook pruning controlled through argparse.
- Measure pairwise row correlation or absolute cosine similarity and remove redundant temporal rows above a configurable threshold; pruning must be disableable for unpruned Q comparisons.
- Evaluate early stopping for greedy pursuit. The existing no-candidate/no-accepted-fit exits and minimum round-energy-drop control must be reviewed before adding another stopping criterion.
- Report both requested Q and effective post-pruning Q, plus the retained-row mapping, in saved metadata.

## Fixed-threshold decision
- On 2026-08-25, threshold calibration was removed from this experiment. Every learned Q bank now runs greedy pursuit at the same fixed template-detection threshold `6`.
- This intentionally compares the complete detectors at a shared numerical threshold; candidate counts are allowed to differ with Q.
- The generic calibration utility remains available for other experiments, but the dedicated peak-channel calibration launcher was removed and the peak-channel pursuit launcher no longer reads a calibration file.

## IBL denoising finding
- IBL does not directly denoise each detected event. It high-pass filters, spatially destripes, handles bad channels, and locally whitens the recording; generic one-dimensional prototypes are weighted averages over many aligned isolated events; full multichannel seeds are projected into a three-PC temporal subspace, factorized as rank-3 templates, and averaged over assigned events.
- An individual extracted IBL waveform can therefore remain noisy. The denoised objects are the learned generic prototypes and fitted templates/reconstructions.
- The new peak-channel initializer follows the cheap isolated-event correlation-clustering part. It does not add IBL local whitening or rank-3 multichannel projection.

## Implemented initialization
- Commit `882c680` adds `fit_raw_peak_channel_temporal_codebooks.py` and four dedicated SLURM launchers without changing the legacy global-codebook scripts.
- Training reproducibly permutes one-second chunks in `[60 s, recording end)`, detects fresh negative peaks at threshold `6`, keeps events isolated by `1 ms` within the same `48 µm` spatial neighborhood, and extracts the 90-sample peak-channel waveform. The original submitted fit used seed `2026`; on 2026-08-25 the default and launcher were standardized to seed `42` for future fits.
- At most 2,048 isolated events are sampled from a chunk until the shared 100,000-event training sample is full. Exact sampled times, channels, scores, waveforms, and scanned chunk indices are saved.
- Each waveform is temporally centered. Hard assignments maximize absolute correlation with a unit-normalized row; fixed assignments refit each row by its signed least-squares numerator. All Q values use the same nested initialization permutation and ten iterations.
- The fitted banks remain frozen throughout pursuit. `run_global_codebook_pursuit.py` explicitly uses `codebook_learning_chunks=0`.

## Subtraction and stopping audit
- `select_conflict_free_peaks` chooses candidates by descending score and prevents any selected 90-sample supports from overlapping; it then orders the accepted group by time, which is output-equivalent because supports do not overlap.
- Each candidate's analytic prediction is gain-refit against the current residual immediately before subtraction. A subtraction is applied only when it lowers actual residual energy and captures at least 5% of the local input energy.
- Saved alpha includes the sequential gain correction, and saved residual waveforms are the local residual immediately before the accepted subtraction.
- Round energy is measured only on the core interval. If its drop is below the configured threshold, the complete residual is restored and all results from that round are discarded. Chunk margins are processed for boundary context but only core events are saved.
- Existing exits for no detected peaks and no accepted fits remain active. This sweep additionally uses the conservative round-energy threshold `0.002`; prior Q8-Q32 runs usually reached round 60 at this threshold, unlike the aggressive `0.005` threshold that stopped around rounds 14-22.

## Validation
- Python syntax/import and all four shell launchers passed controlled checks in the project Singularity environment.
- Synthetic noisy-waveform clustering recovered all four planted temporal shapes with maximum-row correlations above `0.95`, used every row, and reduced nMSE from `0.533768` to `0.309320`.
- A real-recording Q4/Q8 smoke extracted 128 of 146 isolated events from 0.2 seconds, used every row, and completed codebook fitting. The smoke exposed a physical-voltage scale bug in the zero-energy guard; changing float32 `eps` to `tiny` retained all 128 finite nonzero waveforms instead of only 48.
- A real-waveform Q64 smoke used all 64 rows and completed at nMSE `0.056935` after three iterations.
- The new NPZ banks passed the existing calibration reader during validation, but calibration was subsequently removed from the peak-channel experiment design.
- A controlled overlapping-subtraction check confirmed every accepted subtraction reduced energy and that the measured global energy decrease matched the sum of recorded captured energies within floating-point tolerance.

## Superseded queued experiment
- Dependency chain was submitted on 2026-08-25 and was pending at the first one-shot accounting check.
- Peak-channel initialization job `16357483` writes `runs/dataset1_p1/raw_peak_channel_codebooks_16357483/`.
- Matched-threshold calibration job `16357500` depends on `afterok:16357483` and writes `global_codebook_thresholds.json` inside that codebook directory.
- Frozen greedy-pursuit array `16357508_[0-4]` depends on `afterok:16357500`; tasks map to Q4/Q8/Q16/Q32/Q64 and write `runs/dataset1_p1/peak_channel_codebook_pursuit_q{Q}_matched_16357508_{task}/`.
- CPU comparison job `16357511` depends on `afterok:16357508` and writes `runs/dataset1_p1/peak_channel_codebook_pursuit_matched_16357508/comparison.json`.
- Pursuit settings are the previous 8.748-second comparison interval, 2.187-second chunks, matched Q-specific thresholds referenced to Q8 at 6, 60 maximum rounds, `0.002` round-energy stopping, 10,000 maximum peaks per round, localization batch 1,024, frozen codebooks, saved waveforms, and stage profiling.

## Status update: seed and first calibration attempt
- Fit job `16357483` completed successfully in 5:07 using its submitted seed `2026`; its output remains preserved and must not be relabeled as seed `42`.
- Calibration job `16357500` failed after scoring all banks because the Q8-at-6 target was 202,371 candidates while Q32 had only 201,287 total local maxima. Pursuit and comparison therefore did not run.
- Source defaults and the fit launcher now use seed `42`.
- Dangling comparison job `16357511` was explicitly cancelled after its dependency became unsatisfiable.
- The failed calibration path is superseded rather than repaired: the next experiment should run the five pursuit tasks directly after the seed-42 fit, all with fixed threshold `6`, then compare their outputs.

## Submitted seed-42 fixed-threshold experiment
- Seed-42 peak-channel fit job `16358267` writes `runs/dataset1_p1/raw_peak_channel_codebooks_16358267/`; it was running at the first accounting check.
- Fixed-threshold pursuit array `16358281_[0-4]` depends directly on `afterok:16358267`. Tasks map to Q4/Q8/Q16/Q32/Q64, all use threshold `6`, and write `runs/dataset1_p1/peak_channel_codebook_pursuit_q{Q}_fixed_16358281_{task}/`.
- Comparison job `16358283` depends on `afterok:16358281` and writes `runs/dataset1_p1/peak_channel_codebook_pursuit_fixed_16358281/comparison.json`.
- There is no calibration job or calibration artifact in this chain. All outputs are new job-ID paths; the completed seed-2026 fit is untouched.

## Completed fixed-threshold Q-sweep readout
- The seed-42 fixed-threshold chain completed and produced `runs/dataset1_p1/peak_channel_codebook_pursuit_fixed_16358281/comparison.json`.
- Event rates were high and similar across Q: Q4 `14,274/s`, Q8 `14,437/s`, Q16 `14,959/s`, Q32 `14,924/s`, Q64 `14,992/s`.
- Mean completed pursuit rounds were near the cap: Q4 `57.25`, Q8 `58.00`, Q16 `60.00`, Q32 `59.75`, Q64 `60.00`.
- Median raw local-energy captured fraction was only `18.14–20.90%`, while mean remaining core energy after pursuit stayed around `69.6–72.4%`.
- Round diagnostics show accepted events remain high into late rounds. By late rounds most accepted fits are below `20%` raw local-energy capture, although few are exactly pinned at the `5–10%` floor.
- Generated plots:
  - `out/plots/peak_channel_codebook/peak_channel_q_sweep_16358281.png`
  - `out/plots/peak_channel_codebook/peak_channel_round_diagnostics_16358281.png`
  - `out/plots/peak_channel_codebook/peak_channel_q8_chunk0_round_diagnostics_16358281.png`
  - `out/plots/peak_channel_codebook/peak_channel_q64_chunk0_round_diagnostics_16358281.png`

## Noise-weighted acceptance conclusion
- The `min_captured_fraction=0.05` gate is not an IBL-sorter value. It was an extra analytic-fit safety heuristic requiring local residual energy to decrease by at least 5% of the full local waveform patch energy.
- This raw fraction is a poor significance statistic because its denominator includes expected noise across all neighbor channels and all 90 samples. A threshold-6 whitened projection has `Delta chi^2 ~= 36`; for an 8-channel, 90-sample patch, the expected whitened noise denominator alone is about `720`, giving a raw fraction near `36/(720+36) ~= 4.8%`.
- The cleaner model is `Y_s = alpha_s g_{k_s}(x_s,y_s,z_s) Omega_{a_s}^T + epsilon_s`, with Gaussian noise `epsilon_s ~ N(0, Sigma_s)`. Acceptance should use a noise-weighted likelihood/projection improvement such as `Delta chi^2 >= tau_fit^2`, plus finite parameters and monotone actual residual-energy decrease.
- Formal note added at `docs/noise_weighted_pursuit_acceptance.tex`. `pdflatex` was unavailable on the host, so only the TeX source was validated by inspection.

## Whitening diagnostic job
- To test whether whitening reduces apparent noise in poor reconstructions before changing the probabilistic model, added:
  - `src/plots/plot_whitening_spike_examples.py`
  - `src/plots/plot_whitening_spike_examples.sbatch`
- The diagnostic reloads pass-0 poor-capture events from `runs/dataset1_p1/raw_template_residual_batched_4s_16257633`, reconstructs the fitted analytic model, estimates local covariance whitening on the same preprocessed chunk, and plots four sampled events as unwhitened versus whitened columns.
- Submitted SLURM job `16360429` on 2026-08-26. User `sqme` showed it pending in `Priority` state. Expected output path is `out/plots/whitening_diagnostic/whitening_spike_examples_4s_16360429.png`, with metrics JSON at the same path using `.json`.
- Important caveat: because the 4-second benchmark did not save residual waveforms, this first diagnostic uses pass-0 events from chunk 0 so the exact pre-subtraction waveforms can be reloaded from the preprocessed raw chunk.

## Whitening diagnostic result
- Job `16360429` completed with exit `0:0` in `00:01:07` (2026-08-25, `whiten_spike_examples_16360429.out`). Saved `out/plots/whitening_diagnostic/whitening_spike_examples_4s_16360429.png` (5.7 MB) plus metrics JSON at the `.json` path.
- Selected four pass-0 events with captured fractions `0.094/0.057/0.072/0.070`. Whitening did **not** rescue the poor captures — it made relative error slightly *worse* on every example:
  - unwhitened relative error `[0.857, 0.882, 0.808, 0.983]`
  - whitened relative error `[0.968, 0.953, 0.893, 0.990]`
- Conclusion: whitening the preprocessed chunk does not reduce apparent noise for these reconstructions, so the poor captures are not a whitening artifact. The noise-weighted `Delta chi^2` acceptance model remains the clean path forward. Plot image itself could not be inspected in the terminal; the JSON above is the authoritative readout.

## Links
- [[session-009-ibl-style-pursuit]]
- [[session-007-global-codebook-pursuit]]
- [[session-005-residual-profiler]]
