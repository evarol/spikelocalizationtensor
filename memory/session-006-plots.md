# Residual and Codebook Plots (006)
**Created:** 2026-08-24
**Last updated:** 2026-08-30
**Status:** Done — smoke-run localization/reconstruction figures, exact two-detection collision examples, and dense Q8/Q12 rasters

## Context

Visualize the completed 10-second Q8 residual smoke run without mixing it
with the separate Q12 extracted-spike experiment. Because of 004's
identifiability audit, the plots use the effective width
`rho = sqrt(z² + sigma²)` rather than interpreting fitted `z` and profile
scale `sigma` independently.

Code snapshot: branch `residual-smoke-plot`, commit `a2bae95` ("Add residual
smoke reconstruction plots"). `plot_raw_residual_projections.py` renders
lateral-depth, width-depth, and width-lateral projections colored by fitted
amplitude; `plot_raw_residual_reconstructions.py` renders recording-wide
diagnostics plus one representative saved-waveform reconstruction per
residual pass. Both read
`runs/dataset1_p1/raw_template_residual_smoke_16180032/` without re-reading
the raw recording, at 800 DPI.

## What the smoke figures showed

- The run holds 332,012 accepted fits (82,826 / 83,172 / 83,503 / 82,511 by
  pass).
- Median rho 1.803 µm, p99 141.578 µm, p99.9 523.578 µm under the ten-scale
  Q8 parameterization; median captured fractions decline 0.1914 → 0.1524 →
  0.1277 → 0.1114 by pass.
- A 1,000-fit reconstruction check reproduced the saved captured fractions
  to 2.24e-7 max / 3.45e-8 mean absolute difference.
- Outputs under `out/plots/raw_template_residual_smoke_16180032/`
  (`localizations_xrho.png`, `reconstructions/*.png`). Plot PNGs are
  gitignored; the two scripts are committed.

## Depth-time rasters

`plot_temporal_codebook_depth_time_raster.py` is the generic Q8/Q12
depth-versus-time raster plotter: reads row-aligned `spike_times.npy`,
`centroids.npy`, fitted `sources`, `temporal_idx`, and `omega`; global depth
is `centroids[:, 1] + sources[:, 1]`; time converts from 30 kHz samples to
minutes; each spike takes the RGB color of its hard-selected temporal row
(rainbow-sampled with a discrete `Omega_0...Omega_{Q-1}` colorbar). Rendering
matches the SpikeTensor categorical raster: 1,750 × 960 binning of all
finite spikes, 0.5-pixel Gaussian smoothing of event mass and RGB sums,
density-driven intensity, mixed-row blending within a bin, depth limits at
the 0.2/99.8% quantiles, dark theme, 800 DPI. Generated for all 2,303,434
spikes in both the Q12 output
(`out/plots/temporal_q12/depth_time_codebook_raster.png`) and the Q8 masked
one-hot monopole output
(`out/plots/gpu_fit_voxel_1um_masked_monopole/depth_time_codebook_raster.png`).

## Residual-pass localizations and two-detection collisions

`plot_raw_residual_collisions.py` writes a per-pass localization atlas plus
four sequential collision examples from the same 10-second smoke run, with
the reproducible selection in `collision_selection.json` (under
`.../residual_collisions/`). Median rho by pass: 8.1 / 4.3 / 1.1 / 1.0 µm.
A collision edge requires different residual passes, temporal separation
below the 90-sample waveform length, source separation in [15, 80] µm, and
at least one shared electrode; requiring graph degree one at both endpoints
gives exactly two detections. That definition found 1,078 candidates; the
displayed pairs come from chunks 1 / 2 / 7 / 0 with time offsets 19 / 26 /
31 / 15 samples and source separations 77.7 / 68.7 / 35.9 / 73.4 µm.

Important caveat kept from the analysis: these are algorithmic two-atom
residual decompositions, not proof of two biological neurons. Several
selected fits have broad effective widths, so separating genuine collisions
from repeat decomposition remained a scientific follow-up.

## Merge status (historical)

`kilosort-template-residual` is the direct ancestor of `residual-smoke-plot`
(0 target-only commits, 1 plot-branch-only commit); merge-tree showed no
conflicts and `a2bae95` fast-forwards. The merge was not performed that
session, to avoid touching unrelated working-tree edits in the residual
solver, tests, profiler launcher, and pursuit/codebook smoke launcher.

## Links

- [[archive/session-003-q12-temporal-codebook]]
- [[feedback_plot_suite_completeness]]
- [[session-004-continuous-residual]]
- [[project_overview]]
