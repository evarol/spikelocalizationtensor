# Session 010: Whitened Dense GPU Pursuit
**Created:** 2026-08-26
**Last updated:** 2026-08-27

## Context
Replace the current direct raw-residual workflow with an explicitly configurable, GPU-dense pipeline. The design preserves the separable raw-waveform model while separating cheap single-channel proposal generation from full multi-channel scoring, fitting, and subtraction.

## Agreed pipeline

1. Read a user-selected recording interval (`--start-seconds`, `--duration-seconds`).
2. Preprocess and normalize the recording under `--whitening`:
   - `none`: no noise normalization;
   - `diagonal`: divide each channel by its robust noise estimate;
   - `local-zca`: local covariance/ZCA whitening using nearby channels;
   - `zca`: full-probe covariance/ZCA whitening.
   Estimate the covariance from the exact selected recording excerpt used to decide/learn `Omega`, with configurable sampling and regularization.
3. Threshold detections on the selected normalized recording and learn a temporal codebook. Threshold, interval, codebook size `Q`, kernel, and spatial-scale bank are command-line arguments and saved with every output.
4. Run dense, per-channel temporal-template convolution to make inexpensive proposals.
5. Merge proposals that lie within 0.5 ms (15 samples at 30 kHz) and on neighboring channels, retaining the highest-scoring representative. This intentionally permits distinct collisions within the 90-sample waveform window.
6. For each merged event, score the full spatial-scale × temporal-template bank over the 48 um local neighborhood.
7. Run the existing full local solver, then refit spatial scale `sigma` on the complete local waveform. Refit gain `alpha` for each tested `sigma` and select the scale minimizing full-window reconstruction error.
8. Save candidate times, channel IDs, detection scores, selected temporal row, spatial fit, refitted sigma, reconstruction, residual, whitening/noise data, and all configuration values.
9. Evaluate reconstruction quality per valid local channel using both raw-uV and noise-normalized RMSE. Every valid channel in the fixed 48 um neighborhood is a participating channel; do not construct a second, amplitude-based active-channel subset. Use per-channel normalized RMSE as the coherent-fit acceptance diagnostic; do not use Delta chi^2 as the acceptance gate.
10. Subtract accepted full `channels × time` reconstructions from every valid channel in the 48 um neighborhood. Never subtract only the peak channel.

## Whitening-coordinate rule

Follow IBL-sorter: once whitening is enabled, temporal-codebook learning, template scoring, full local fitting, RMSE evaluation, residual updates, and subtraction all operate in the whitened coordinate system. Apply the same channel transform to data and spatial footprints before scoring/reconstruction. Save the whitening matrix and its inverse/pseudoinverse so reconstructions can also be displayed in the original preprocessed-voltage coordinate; do not unwhiten merely to perform subtraction.

## Performance constraint

The hot path must remain GPU-dense: batched tensor convolutions, batched neighborhood gathering, batched spatial/template scores, and batched fitting. Avoid per-event Python loops and frequent CPU/GPU synchronization because sustained GPU utilization below 50% risks SLURM termination.

## IBL reference

IBL-sorter already offers the intended normalization family in `iblsorter/preprocess.py`:

- `zscore`: diagonal per-channel normalization;
- `whitening`: ZCA from the channel covariance;
- finite `whiteningRange` (default 32): geometry-local ZCA, retaining each primary channel's local whitening column.

This project should implement compatible modes independently rather than importing the sorter into the main runtime.

## Implementation update (2026-08-27)

- Added `residuals/src/preprocessing/run_whitened_dense_pursuit.py`, a dedicated χ²-free runner. It accepts `none`, `diagonal`, `local-zca`, and full `zca` whitening; applies the selected transform to each read chunk; saves the whitening matrix, pseudoinverse, and noise estimate; performs dense codebook/spatial-bank proposals; merges candidates within 0.5 ms only when channels are neighbors; fits and subtracts full local channel × time reconstructions; and uses a per-valid-channel normalized-RMSE gate (`--max-channel-normalized-rmse`, default 3.0).
- Added full-window sigma selection across the configured monopole scale bank, saving `sigma.npy`, `profile_idx.npy`, and per-channel `channel_normalized_rmse.npy` with the usual event outputs.
- Added `--learn-omega`: it refines the supplied initial codebook on randomly selected pursuit chunks before final extraction, saving `omega_initial.npy`, `omega_learned.npy`, `omega.npy`, and `codebook_learning_history.json`. It does not yet bootstrap a codebook from a user-selected Q; it requires an initial `omega` file.
- Added dataset-specific `residuals/src/preprocessing/0010_params.py` and `residuals/src/preprocessing/run_0010_whitened_dense_pursuit.sbatch`. The parameters select dataset1_p1, peak-channel Q8 initialization `raw_peak_channel_codebooks_16358267/global_codebook_q8.npz`, local-ZCA, four omega-learning chunks, 8.748 seconds, and 60 pursuit rounds.
- Submitted SLURM job `16454393` (`whitened_dense_0010`) on 2026-08-27, then cancelled it before execution after identifying that its localizer still used raw-coordinate spatial footprints.
- Restored `residuals/src/maths.py` to its pre-session-0010 interface. Added `residuals/src/maths_0010.py` instead: its new localizer maps every raw footprint with the per-event local whitening map before normalization, spatial/temporal candidate scoring, gain fitting, discrete source refinement, full-window scale refit, and reconstruction. The session-0010 path in `raw_residual.py` alone routes to it through `use_0010_math=True`; legacy workflows retain `maths.py`.
- CPU smoke test of `maths_0010.localize_spikes_fixed_codebook` succeeded with a valid `(2, 4, 90)` reconstruction. Replacement GPU job `16454644` failed after 67 seconds during whitening setup, before pursuit: NumPy advanced indexing swapped the spatial-profile and local-channel axes in `transform_detection_footprints`, producing a 5-vs-9 matrix-multiplication mismatch. Corrected the local slice to `footprints[anchor][:, valid]`. Submitted replacement GPU job `16454853` (`whitened_dense_0010`), writing to `residuals/runs/dataset1_p1/whitened_dense_0010_16454853/` with logs in `residuals/slurm_logs/`.
- Submitted dependent plot job `16454903` with `afterok:16454853`. `residuals/src/plots/plot_0010_whitened_dense_pursuit.py` will write separate 800-dpi PNGs to `residuals/runs/dataset1_p1/whitened_dense_0010_16454853/plots/`: `summary.png`, `localizations.png`, `spike_raster.png`, `temporal_codebook.png`, `codebook_usage.png`, and `reconstruction_examples.png`. It follows existing residual plotting conventions and the localization/usage structure in `spiketensor/viz_lattice.py`.

## Resume update (2026-08-27)

- GPU pursuit job `16454853` completed successfully in 20m48s (wall-clock runtime recorded as 1204.31s). Its dependent plot job `16454903` also completed successfully in 46s.
- The local-ZCA Q8 run processed four 2.187s chunks (8.748s total) and saved 382,799 accepted events, or about 43.8k events/s. Each chunk reached negligible residual energy reduction by rounds 16--17; final round drops were 0.000002--0.000004.
- All expected arrays, four waveform shards, whitening artifacts, learned-codebook artifacts, and six 800-dpi plot PNGs were produced under `residuals/runs/dataset1_p1/whitened_dense_0010_16454853/`. Neither completed job wrote stderr output.
- The result has not yet received a scientific-quality review. The very high event rate means the next task is to inspect the plots and event/fit distributions, then compare reconstruction quality against the prior raw-coordinate baseline before considering this whitening pathway validated.

## Remaining caveat

`maths_0010.py` is intentionally separate from the established `maths.py`, but it must be evaluated for throughput and scientific behavior on the submitted dataset1_p1 job before replacing any legacy pathway.

## Links
- [[session-012-rho-implementation-plan]]
- [[session-009-ibl-style-pursuit]]
- [[session-008-peak-channel-codebook-init]]
- [[session-011-identifiable-rho-localization]]
