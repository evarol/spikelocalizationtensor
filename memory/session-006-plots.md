# Session 006: Residual and Temporal-Codebook Plots
**Created:** 2026-08-24
**Last updated:** 2026-08-30

## Context
Visualize the completed 10-second continuous Q8 residual smoke run without mixing it with the separate Q12 extracted-spike experiment. The monopole spatial audit in [[session-004-continuous-residual]] means the plots use the identifiable effective width `rho = sqrt(z^2 + sigma^2)` instead of interpreting fitted `z` and profile scale `sigma` independently.

## Code snapshot
- Branch: `residual-smoke-plot`.
- Commit: `a2bae95` (`Add residual smoke reconstruction plots`).
- `src/plots/plot_raw_residual_projections.py` plots lateral-depth, effective-width-depth, and effective-width-lateral projections colored by fitted amplitude.
- `src/plots/plot_raw_residual_reconstructions.py` plots recording-wide reconstruction diagnostics and one representative saved-waveform reconstruction from each residual pass.
- Both scripts render at 800 DPI and read `runs/dataset1_p1/raw_template_residual_smoke_16180032/` without rereading the raw recording.

## Artifacts and findings
- Localization figure: `out/plots/raw_template_residual_smoke_16180032/localizations_xrho.png`.
- Reconstruction diagnostics: `out/plots/raw_template_residual_smoke_16180032/reconstructions/reconstruction_diagnostics.png`.
- Reconstruction examples: `out/plots/raw_template_residual_smoke_16180032/reconstructions/reconstruction_examples.png`.
- The run contains `332,012` accepted fits; pass counts are `82,826`, `83,172`, `83,503`, and `82,511`.
- Identifiable width `rho` has median `1.803 µm`, p99 `141.578 µm`, and p99.9 `523.578 µm` under the ten-scale Q8 parameterization.
- Median sequential captured fractions decline by pass: `0.191361`, `0.152439`, `0.127686`, and `0.111379`.
- A 1,000-fit reconstruction check reproduced saved captured fractions with maximum absolute difference `2.24e-7` and mean absolute difference `3.45e-8`.
- Plot PNGs are gitignored; the two reusable plot scripts are committed.

## Temporal-codebook depth-time rasters
- `src/plots/plot_temporal_codebook_depth_time_raster.py` is a generic Q8/Q12 depth-versus-time raster plotter. It reads row-aligned `spike_times.npy`, `centroids.npy`, fitted `sources`, `temporal_idx`, and `omega`; global depth is `centroids[:, 1] + sources[:, 1]`, and time is converted from samples at 30 kHz to minutes.
- Each spike contributes the RGB color of its hard-selected temporal row. The Q colors are sampled at evenly spaced points from Matplotlib's `rainbow` map, with a discrete `Omega_0 ... Omega_{Q-1}` colorbar.
- The rendering now matches the SpikeTensor categorical raster: all finite spikes are binned into a 1,750-by-960 time-depth image, event mass and RGB sums receive 0.5-pixel Gaussian smoothing, density controls intensity, and mixed rows blend within a bin. Depth limits remain at the `0.2%` and `99.8%` quantiles.
- The plot uses the localization-figure dark theme (`#0d0d0d` background, light labels, dark grid) and saves at 800 DPI.
- Q12 output: `out/plots/temporal_q12/depth_time_codebook_raster.png`, containing all `2,303,434` extracted spikes and 12 colors.
- Q8 masked one-hot monopole output: `out/plots/gpu_fit_voxel_1um_masked_monopole/depth_time_codebook_raster.png`, containing the same `2,303,434` spikes and 8 colors.
- The raster PNGs are gitignored. The generic plot script is untracked as of this update and should remain separate from unrelated residual-solver and profiler changes.

## Residual-pass localizations and two-detection collisions
- `src/plots/plot_raw_residual_collisions.py` reads the same 10-second smoke run and writes an 800-DPI localization atlas split by residual pass, four sequential collision examples, and the reproducible case selection in `collision_selection.json`.
- Outputs are in `out/plots/raw_template_residual_smoke_16180032/residual_collisions/`: `residual_localizations_by_pass.png`, `two_detection_collision_examples.png`, and `collision_selection.json`.
- Median effective width by residual pass is `8.1`, `4.3`, `1.1`, and `1.0 µm`; median captured fractions are the previously recorded `19.1%`, `15.2%`, `12.8%`, and `11.1%`.
- A collision edge requires different residual passes, temporal separation below the 90-sample waveform length, source separation in `[15, 80] µm`, and at least one shared electrode. Requiring graph degree one at both endpoints gives exactly two detections under this local overlap definition.
- The strict definition found `1,078` candidates. The displayed cases use one top-quality pair per chunk from chunks `1`, `2`, `7`, and `0`; their time offsets are `19`, `26`, `31`, and `15` samples, and their source separations are `77.7`, `68.7`, `35.9`, and `73.4 µm`.
- These are algorithmic two-atom residual decompositions, not proof of two biological neurons. Several selected fits have broad effective widths, so separating genuine collisions from repeat decomposition remains a scientific follow-up.
- The script is untracked alongside the other current pursuit/codebook work; unrelated working-tree edits were preserved.

## Merge status
- `kilosort-template-residual` is the direct ancestor of `residual-smoke-plot`; the branches differ by `0` target-only commits and `1` plot-branch-only commit.
- The merge-tree check reports no conflicts. Commit `a2bae95` can be fast-forwarded onto `kilosort-template-residual`.
- The merge was not performed during this session. Unrelated working-tree edits in the residual solver, tests, profiler launcher, and pursuit/codebook smoke launcher were left untouched.

## Links
- [[archive/session-003-q12-temporal-codebook]]
- [[feedback_plot_suite_completeness]]
- [[session-004-continuous-residual]]
- [[project_overview]]
