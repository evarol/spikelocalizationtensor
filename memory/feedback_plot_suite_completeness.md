# Plot Suite Completeness
**Created:** 2026-08-30
**Last updated:** 2026-08-30

## Context

Temporal-row indices, color keys, and usage heatmaps do not reveal the learned
waveform shapes, so a plot suite is incomplete without an explicit view of the
codebook itself.

## Key facts

- Every future experiment suite that learns or uses a temporal codebook must
  include a dedicated `temporal_codebook_usage.png` panel showing every Omega
  row waveform and its recording-wide assignment count and fraction.
- When residual passes or peeling rounds exist, the same panel must show usage
  within each pass or round.
- A depth-time categorical raster, a row-selection heatmap, or a colorbar key
  does not substitute for the waveform-and-usage panel.
- For consolidated residual-pursuit runs with `omega.npy`, `temporal_idx.npy`,
  and `residual_pass.npy`, use
  `residuals/src/plots/plot_0014_codebook_usage.py`, include it in suite syntax
  checks and execution, save under `residuals/out/<run>/`, and retain 800 DPI.
- Starting with 0018, every gated plot job must finish by running
  `residuals/src/plots/build_plot_gallery.py` and writing
  `residuals/out/<run>/index.html`.
- The offline browser should follow the SpikeTensor browser's presentation:
  matching color system and typography, panel controls, run detail card,
  full-panel stack, contact-sheet mode, lazy media, and full-resolution links.
- Browser resemblance must not imply state equivalence. Each residual browser
  must name unavailable SpikeTensor panels and state which defining arrays or
  computations are absent instead of silently substituting an approximation.
- For 0018, all 16 generated figures are registered. Exact dense-coefficient
  embeddings, soft/hard readout comparisons, multipole support diagnostics,
  saved DREDge results and corrected families, and the full atom-viewer pack are
  unavailable from the persisted state. Time/depth/amplitude inputs are present
  for a later canonical DREDge computation.

## Links

- [[session-006-plots]]
- [[session-017-initial-threshold-spike-discovery]]
- [[session-018-bipolar-prototype-cone-peeling]]
