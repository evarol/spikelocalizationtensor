# Project Overview

## Goal

Localize and reconstruct extracellular spikes with an analytic spatial
model, then detect overlapping or missed spikes by repeatedly subtracting
those reconstructions from the raw-recording residual.

## Where things stand

- Front line: **0019** (all-channel-error peeling, full run `16655016`)
  building directly on **0018** (bipolar prototype-cone peeling, complete).
  See [[session-019-all-channel-error]] and
  [[session-018-bipolar-prototype-cone-peeling]].
- The masked one-hot monopole solver and continuous refinement work on all
  2,303,434 saved spikes; the full continuous result has nMSE 0.4926997 with
  zero monotonicity violations and zero voxel-cell escapes
  ([[archive/session-001-analytic-localization]]).
- The monopole's `(z, sigma)` pair is not identifiable from the data — only
  `rho = sqrt(z² + sigma²)` is ([[session-004-continuous-residual]]'s
  audit). Spatial-width analysis must use rho.
- A separate Q12 experiment improved the old extracted-spike hard-one-hot
  fit from Q8 nMSE 0.4928 to 0.481783
  ([[archive/session-003-q12-temporal-codebook]]).
- Repository test files were removed at your request; validation is
  controlled smoke runs plus syntax/import checks.
- The method lineage runs 002 → 004 → 005 → 007 → 008 → 009 → 010 → 011 →
  012 → 014 → 015 → 016 → 017 → 018 → 019, each card recording what its
  stage changed and why it ended.

## Key components / scripts

| File | Purpose |
| --- | --- |
| `residuals/src/maths.py` | Masked spatial/temporal solver, fixed-codebook localization, reconstruction, collision prototypes |
| `residuals/src/maths_0010.py` | 0010's whitened localizer (separate interface, see [[session-010-whitened-dense-pursuit]]) |
| `residuals/src/continuous_refine.py` | Bounded continuous refinement inside the winning 1 µm voxel |
| `residuals/src/preprocessing/raw_residual.py` | Raw SpikeGLX template matching, localization, reconstruction, subtraction, residual passes |
| `residuals/src/preprocessing/0016_onehot_lattice_peeling.py` | One-hot lattice residual peeling (0016) |
| `residuals/src/preprocessing/0018_prototype_cone_threshold_peeling.py` | Bipolar prototype-cone peeling (0018) |
| `residuals/src/preprocessing/0019_allchannel_peeling.py` | All-channel-error peeling (0019) |
| `residuals/src/plots/plot_raw_residual_localizations.py` | Probe-global localizations split by subtraction pass |
| `residuals/src/plots/plot_raw_residual_recording.py` | Raw-to-residual recording replay from saved atoms |

## Conventions

- The detector reads BIN/CBIN through `spikeglx.Reader`; it never imports or
  runs `iblsorter` (reference only).
- Detection uses full template deconvolution over valid time samples with
  templates from the spatial and temporal codebooks; 48 µm channel-map
  neighborhoods around each anchor channel.
- Every accepted component is localized, reconstructed, subtracted, and the
  residual re-scored.
- Plot suites follow [[feedback_plot_suite_completeness]]; outputs go under
  `residuals/`.
