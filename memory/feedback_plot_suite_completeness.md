# Plot Suite Completeness
**Created:** 2026-08-30
**Last updated:** 2026-08-30

## The rule

Temporal-row indices, color keys, and usage heatmaps never reveal the
learned waveform shapes, so a temporal-codebook plot suite is incomplete
without the codebook itself. Every future suite that learns or uses a
temporal codebook must include a dedicated `temporal_codebook_usage.png`
panel showing every Omega row waveform with its recording-wide assignment
count and fraction — and, when residual passes or peeling rounds exist,
usage within each pass/round. A depth-time categorical raster, a
row-selection heatmap, or a colorbar key does not substitute for it.

For consolidated residual-pursuit runs with `omega.npy` +
`temporal_idx.npy` + `residual_pass.npy`, use
`residuals/src/plots/plot_0014_codebook_usage.py`, include it in suite
syntax checks and execution, save under `residuals/out/<run>/`, at 800 DPI.

## The offline browser

Since 0018, every gated plot job must finish by running
`residuals/src/plots/build_plot_gallery.py` and writing
`residuals/out/<run>/index.html`. The browser follows SpikeTensor's
presentation: matching color system and typography, panel controls, run
detail card, full-panel stack, contact-sheet mode, lazy media, and
full-resolution links.

Browser resemblance must not imply state equivalence: each browser must
name the SpikeTensor panels it cannot produce and state which defining
arrays or computations are missing, rather than silently substituting an
approximation. For 0018, all 16 generated figures are registered; dense
coefficient embeddings, soft/hard readout comparisons, multipole support
diagnostics, saved DREDge results and corrected families, and the full
atom-viewer pack are unavailable from the persisted state (the
time/depth/amplitude inputs for a later DREDge solve are present).

## Links

- [[session-006-plots]]
- [[session-017-initial-threshold-spike-discovery]]
- [[session-018-bipolar-prototype-cone-peeling]]
