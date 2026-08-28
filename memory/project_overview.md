# Project Overview

## Goal
Localize and reconstruct extracellular spikes with an analytic spatial model, then detect overlapping or missed spikes by repeatedly subtracting those reconstructions from the raw-recording residual.

## Key components / scripts

| File | Purpose |
| --- | --- |
| `src/maths.py` | Masked spatial/temporal solver, fixed-codebook localization, reconstruction, and collision prototypes |
| `src/continuous_refine.py` | Bounded continuous refinement inside the winning 1 µm voxel |
| `src/preprocessing/detect_peaks.py` | Previous SpikeInterface peak detector |
| `src/preprocessing/extract_neighborhoods.py` | Previous SpikeInterface 48 µm waveform extraction |
| `src/preprocessing/raw_residual.py` | Raw SpikeGLX codebook-template matching, localization, reconstruction, subtraction, and repeated residual passes |
| `src/preprocessing/fit_raw_global_temporal_codebooks.py` | Fresh raw detection and global Q8/Q16/Q24/Q32 temporal-codebook fitting |
| `src/preprocessing/run_global_codebook_pursuit.py` | Frozen global-codebook residual pursuit for collision recovery and reconstruction comparison |
| `src/plots/plot_raw_residual_localizations.py` | Probe-global raw-residual localizations split by subtraction pass |
| `src/plots/plot_raw_residual_recording.py` | Raw-to-residual recording replay from saved fitted atoms |

## Current state
- The masked one-hot monopole solver and continuous refinement are working on all 2,303,434 saved spikes.
- The full continuous result has nMSE `0.4926997`, with zero monotonicity violations and zero voxel-cell escapes.
- Branch `residual-smoke-plot` contains the active raw-recording work.
- The new detector reads BIN or CBIN through `spikeglx.Reader`; it does not import or run `iblsorter`.
- Candidate detection evaluates separable codebook templates at every valid time window, then applies coarse-to-integer-1 µm localization and subtracts the fitted reconstruction.
- A separate Q12 experiment improves the old extracted-spike hard-one-hot fit from Q8 nMSE `0.4928` to `0.481783`; see [[archive/session-003-q12-temporal-codebook]].
- Bounded continuous monopole refinement is integrated into Q8 raw-residual localization, followed by overlap-safe gain re-fitting and pass-level energy rollback. No full residual job is currently running; the faster batching benchmark changed scientific output, and the baseline profiler identifies continuous-refinement launch and synchronization overhead as the next performance target; see [[session-004-continuous-residual]] and [[session-005-residual-profiler]].
- Repository test files were removed at the user's request; current validation uses controlled smoke runs plus syntax and import checks.
- GPU smoke job `16058517` completed successfully for the first 10 seconds of `dataset1_p1`; its later residual passes increase replayed RMS.
- Identifiable-width residual figures and dense codebook-colored Q8/Q12 depth-time rasters are recorded in [[session-006-plots]].
- The active raw-only global-codebook experiment and corrected SLURM chain are recorded in [[session-007-global-codebook-pursuit]].

## Links
- [[session-007-global-codebook-pursuit]]
