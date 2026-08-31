# Whitened Dense GPU Pursuit (0010)
**Created:** 2026-08-26
**Last updated:** 2026-08-27
**Status:** Done — local-ZCA run `16454853` + plots completed (382,799 events in 8.748 s) but never validated scientifically; pathway set aside

## Design

A configurable, GPU-dense pipeline that keeps the separable raw-waveform
model but splits cheap single-channel proposal generation from full
multi-channel scoring, fitting, and subtraction:

1. Read a user-selected interval (`--start-seconds`, `--duration-seconds`).
2. Preprocess under `--whitening`: `none`, `diagonal` (per-channel robust
   noise), `local-zca` (nearby channels), or `zca` (full probe). Covariance
   is estimated from the exact excerpt used to decide/learn Omega, with
   configurable sampling and regularization.
3. Threshold-detect on the normalized recording and learn a temporal
   codebook; threshold, interval, Q, kernel, and scale bank are CLI
   arguments saved with every output.
4. Dense per-channel template convolution makes cheap proposals.
5. Merge proposals within 0.5 ms (15 samples at 30 kHz) on neighboring
   channels, keeping the highest-scoring representative — distinct collisions
   inside the 90-sample window are allowed to survive on purpose.
6. For each merged event, score the full scale × template bank over the
   48 µm neighborhood with the full local solver, then refit sigma on the
   complete local window (refitting gain per tested sigma, keeping the scale
   with the best full-window error).
7. Accept per valid channel using noise-normalized RMSE (default cap 3.0) —
   never Δχ², and never a second amplitude-based "active channel" subset;
   every valid channel in the fixed 48 µm neighborhood participates.
8. Subtract accepted full channel×time reconstructions from every valid
   channel in the neighborhood — never only the peak channel.

Whitening rule (following IBL): once whitening is on, codebook learning,
scoring, fitting, RMSE, residuals, and subtraction all happen in whitened
coordinates; save the whitening matrix and its inverse/pseudoinverse so
reconstructions can be displayed in the original voltage coordinate; never
unwhiten merely to subtract. IBL's normalization family lives in
`iblsorter/preprocess.py` (`zscore`, `whitening`, finite `whiteningRange=32`
local ZCA) — implemented independently, not imported. The hot path must stay
GPU-dense (batched convolutions, gathering, scoring, fitting); sustained GPU
utilization below 50% risks SLURM termination.

## Implementation

- `residuals/src/preprocessing/run_whitened_dense_pursuit.py`: the
  χ²-free runner with all four whitening modes, whitening artifacts, dense
  proposals, 0.5 ms neighbor-only merging, full local fits/subtractions, and
  the per-channel RMSE gate.
- Full-window sigma selection over the monopole bank, saving `sigma.npy`,
  `profile_idx.npy`, and per-channel `channel_normalized_rmse.npy`.
- `--learn-omega` refines a supplied initial codebook on random pursuit
  chunks (`omega_initial.npy` / `omega_learned.npy` / `omega.npy` +
  `codebook_learning_history.json`); it requires an initial Omega file and
  does not bootstrap from Q.
- Dataset params in `residuals/src/preprocessing/0010_params.py` +
  `run_0010_whitened_dense_pursuit.sbatch`: dataset1_p1, Q8 peak-channel
  initialization from
  `raw_peak_channel_codebooks_16358267/global_codebook_q8.npz`, local-ZCA,
  four learning chunks, 8.748 s, 60 rounds.
- Localizer split: `residuals/src/maths.py` restored to its pre-0010
  interface; new `residuals/src/maths_0010.py` maps every raw footprint
  through the per-event local whitening transform before normalization,
  scoring, gain fitting, discrete refinement, and full-window scale refit.
  Only the 0010 path in `raw_residual.py` routes to it
  (`use_0010_math=True`); legacy workflows keep `maths.py`.

Job history: first submission `16454393` was cancelled before execution once
its localizer was found to still use raw-coordinate footprints. GPU
replacement `16454644` died after 67 s in whitening setup — NumPy advanced
indexing had swapped the spatial-profile and local-channel axes in
`transform_detection_footprints` (a 5-vs-9 matmul mismatch); fixed to
`footprints[anchor][:, valid]`. Run `16454853` then completed in 20m48s with
dependent plot job `16454903` (46 s) writing six 800-dpi PNGs (`summary`,
`localizations`, `spike_raster`, `temporal_codebook`, `codebook_usage`,
`reconstruction_examples`) into the run's `plots/` directory.

## Result — and why it was set aside

The local-ZCA Q8 run processed four 2.187 s chunks (8.748 s) and accepted
382,799 events (~43.8k events/s), with negligible residual-energy reduction
by rounds 16–17. All arrays, whitening and codebook artifacts, and plots were
produced with clean stderr. But the event rate is implausibly high, and when
0011 audited the output, 83.0% of fits had hit the lower z bound and 86.8%
had picked sigma 2 µm — the model was selecting near-delta spatial profiles.
The pathway was never scientifically validated and was set aside in favor of
the unwhitened and identifiable-rho directions
([[session-011-identifiable-rho-localization]]).

Standing caveat: `maths_0010.py` is intentionally separate from `maths.py`
and must clear throughput and scientific review on dataset1_p1 before
replacing any legacy path.

## Links

- [[session-012-rho-implementation-plan]]
- [[session-009-ibl-style-pursuit]]
- [[session-008-peak-channel-codebook-init]]
- [[session-011-identifiable-rho-localization]]
