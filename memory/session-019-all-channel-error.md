# All-Channel-Error Peeling (0019)
**Created:** 2026-08-30
**Last updated:** 2026-08-30
**Status:** Implemented and validated (CPU self-test + synthetic smoke); full run not yet queued

## Context

0018's fit objective minimizes the WORST channel's error (minimax) and its
acceptance demands only 2 improving channels. This rewards narrow templates:
37.2% of 0018 events pick sigma=2 µm, 57% of those sit within 3 µm of the
probe surface, and 82% of sub-20%-capture events are the three narrowest
widths. The user requires the error to drop across ALL channels. 0019
enforces it.

## What 0019 implements (all decided with the user)

- Fit objective: `mean-channel-rmse` (total error across all channels)
  instead of worst-channel minimax.
- Acceptance: EVERY valid channel in the fit mask must capture >= 20% of its
  OWN noise-normalized energy (`all_channel_min_fraction=0.2`); the bar rises
  0.1 per recording pass (0.2/0.3/0.4). No per-event total-share bar: that is
  mathematically impossible across ~8 channels (sum > 100%).
- Recording passes: 3 passes x 1 peeling round per chunk visit (arg-tunable).
  Detection threshold stays 5 on every pass; only the reconstruction bar
  escalates. Pass 2+ rebuilds each chunk's starting residual on the GPU by
  replaying every saved earlier-pass event (searchsorted time window, batched
  index_put_, no residual files, no CPU-heavy work). The chunk-local duplicate
  prior is preloaded with the replayed events. Early stop if a pass accepts
  < `pass_stop_min_events` (1000) recording-wide.
- Rejection audit: every detected-but-rejected candidate is saved per chunk
  (reason bitmask: 1 gain, 2 rmse, 4 captured, 8 projection, 16 all-channel,
  32 energy, 64 duplicate, 128 rolled-back round) with its fit metrics.
- Score floor stays 8 (single-variable experiment vs 0018).
- Synthetic CPU smoke: spike accepted at bar 0.2 (min channel fraction 0.33);
  at bar 0.9 rejected and logged with reason 0b10011000. Self-test covers the
  replay window logic and pass-fraction escalation.

## Files (uncommitted as of 2026-08-30)

- `residuals/src/preprocessing/0019_allchannel_peeling.py` (from 0018;
  new pursue/process_chunk/replay/load_prior_events/_consolidate)
- `residuals/src/preprocessing/0019_allchannel_full.sbatch` (3 passes, round 1,
  fraction 0.2 step 0.1, mean-channel objective)
- `residuals/src/plots/plot_0019_allchannel_cones.py` (renamed 0018 plot)
- `residuals/src/plots/0019_allchannel_plots.sbatch` (full suite + index.html
  gallery; pointers updated to the 0019 run)
- Run output: `residuals/runs/dataset1_p1/0019_allchannel_pass3_round1_fraction20_step10_fitted8`
- Plot loaders (`0016_onehot_lattice_plots.py`,
  `plot_raw_residual_reconstructions.py`,
  `plot_spiketensor_residual_pursuit.py`) gained a `pass_*/chunk_*.npz`
  fallback so the same suite works for both layouts.

## Key facts

- Chunk layout changed: chunks live in `pass_XX/chunk_NNNNNN.npz`, one
  directory per recording pass; consolidation runs per pass and at the root
  (event-aligned fields only; `rejected_*` audit tables stay sharded).
- `residual_pass.npy` = peeling round; NEW per-event `recording_pass.npy`,
  `all_channel_ok`, `all_channel_fraction`, `min_channel_captured_fraction`.
- Round census from 0018 (2.89M events): round 0 71.2%, round 1 22.0%,
  round 2 5.3%, round 3 1.2%, round 4+ ~0.3% — the productive depth is 4
  layers; 60 in-chunk rounds were over-detecting noise.
- Passes and inner rounds are the same machine state-wise; passes add seam
  healing, parallelism, per-pass audit, and tighter acceptance.

## Next Steps

- [x] Implement all-channel criterion, pass loop, GPU replay, rejection log.
- [x] Validate: py_compile, bash -n, self-test (CPU), synthetic smoke.
- [ ] Queue the full 0019 run + dependent plot suite (SLURM, outside sandbox).
- [ ] After completion: check rejected-reason histogram, sigma usage (did the
  sigma-2 cheat die?), per-pass event counts, and per-channel fractions.

## Links

- [[session-018-bipolar-prototype-cone-peeling]]
- [[session-016-one-hot-lattice-peeling]]
- [[session-017-initial-threshold-spike-discovery]]
- [[feedback_plot_suite_completeness]]
