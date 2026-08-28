# Session 002: Template Residual Detection

## Status
Active on branch `kilosort-template-residual`; the 10-second smoke run completed and exposed threshold, repeated-pass, and localization-continuity problems.

## Objective
Replace the separate SpikeInterface peak-detection and waveform-extraction front end with raw BIN/CBIN processing through `spikeglx.Reader`, while using the analytic localization model as a matching-pursuit reconstruction atom.

The intended loop is:

1. Read and preprocess an overlapping raw chunk.
2. Evaluate codebook-derived templates at every valid waveform window.
3. Select joint space-time local maxima.
4. Run the full spatial localization with the frozen temporal cookbook.
5. Reconstruct the accepted component.
6. Subtract it from the residual.
7. Recompute all template scores and repeat.

## Current implementation
- `src/maths.py` now has `localize_spikes_fixed_codebook`, `reconstruct_spike_fits`, and `build_codebook_detection_footprints`.
- `src/preprocessing/raw_residual.py` reads raw data using only `spikeglx.Reader`.
- Preprocessing is a third-order zero-phase `300–6000 Hz` bandpass followed by per-sample global median reference.
- The detector convolves every channel with every temporal row in `Omega` and combines those projections with anchor-centered spatial codebook footprints.
- The current default detection bank has 8 temporal rows and 10 monopole spatial profiles per anchor.
- Scores are made spatially and temporally exclusive over ±0.5 ms and the 48 µm channel-map neighborhood.
- Accepted events receive the complete coarse-to-integer-1 µm spatial search, not only the reduced detection bank.
- The original smoke output stopped on the integer grid. Continuous raw-residual localization and subtraction safeguards are implemented in [[session-004-continuous-residual]].
- Chunk results are restartable and can save the residual waveform presented to each accepted fit.

## Verification
- `src/test_raw_residual.py` contains five synthetic tests.
- All five tests pass in `/scratch/ap7151/_ENVS/ibl-sorter.ext3` using CPU execution.
- A test caught and fixed suppression that originally compared neighboring channels only at the identical sample; suppression now covers the joint space-time neighborhood.

## Smoke result
- Job `16058517` completed successfully with exit `0:0` in `3:34` on `gl049`.
- Launcher: `src/preprocessing/raw_template_residual_smoke.sbatch`.
- Output: `runs/dataset1_p1/raw_template_residual_smoke_16058517/`.
- The run accepted 353,279 events in 10 seconds. Pass counts were `89,657`, `88,851`, `87,941`, and `86,830`.
- Median captured fraction was `0.152`; 80% of fits were below `0.25`, and 96% were below `0.5`.
- Every saved local source coordinate is exactly integer-valued. The local axes use 301, 301, and 300 distinct positions, confirming that this output never received continuous refinement.
- A 20 ms chunk-0 replay changed RMS from `100%` to `89.9%`, `86.0%`, `87.6%`, and `97.8%` across the four passes. Passes three and four undo most of the initial reduction.

## Plot outputs
- `out/plots/raw_template_residual_smoke_16058517/localizations_by_pass.png`
- `out/plots/raw_template_residual_smoke_16058517/residual_recording_chunk0.png`
- Plot scripts are `src/plots/plot_raw_residual_localizations.py` and `src/plots/plot_raw_residual_recording.py`.
- The recording used by this smoke run is BIN. The replay script reads BIN or CBIN through `spikeglx.Reader` and reconstructs saved atoms; the run did not save a continuous residual recording.

## Pending
- Continued in [[session-004-continuous-residual]].

## Worktree caution
- The new residual files and solver changes are uncommitted.
- `src/plots/plot_amplitude_scatter.py` and `src/plots/plot_monopole_paired_scatter.py` were already untracked before this branch and must not be bundled accidentally. The two raw-residual plot scripts were added later for this branch.

## Links
- [[session-004-continuous-residual]]
