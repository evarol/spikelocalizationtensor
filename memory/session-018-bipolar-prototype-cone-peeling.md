# Bipolar Prototype-Cone Residual Peeling (0018)
**Created:** 2026-08-30
**Last updated:** 2026-08-30
**Status:** Complete — full run `16626415`, plot suite `16626447`, and offline browser all succeeded

## Design decisions

Spikes point either up or down, so 0018's two temporal mother shapes are an
explicit positive/negative pair rather than two negative-going morphological
clusters. Two older conventions are deliberately dropped: the 0016/0017 habit
of flipping every Omega row negative, and 0017's negative-only proposal
detector. For discovery, 0018 follows SpikeInterface's locally-exclusive
detector (source at
`/scratch/ap7151/_REFERENCE/spikeinterface/src/spikeinterface/sortingcomponents/peak_detection/`).

At your request, the implementation was written blind: no syntax check,
self-test, Singularity run, smoke, or SLURM check before the first submission.

## Detection semantics (SpikeInterface-compatible)

- `peak_sign="both"`, threshold 5 in per-channel robust-noise units, 1 ms
  exclusion sweep.
- A positive candidate crosses the inclusive positive threshold, is strictly
  larger than the preceding sample and at least as large as the next;
  negative candidates use the sign-reversed inequalities.
- Positive and negative candidates compete in one pool across the temporal
  and spatial neighborhood, ranked by absolute noise-normalized amplitude.
  Exact score ties keep the earlier sample; equal-time cross-channel ties
  survive. Conservative boundary exclusion is combined with the
  waveform-safe center bounds residual fitting already requires.
- Saved events keep the signed normalized detection score, signed raw peak
  amplitude, detection polarity, fitted prototype family, and fitted
  polarity.
- Where the reference Torch path loses polarity via `abs()` and has
  normalized-amplitude output quirks, 0018 follows the intended signed NumPy
  behavior.

## Implementation

- Entry point:
  `residuals/src/preprocessing/0018_prototype_cone_threshold_peeling.py`
- Full-run script:
  `residuals/src/preprocessing/0018_prototype_cone_threshold_full.sbatch`
- Output:
  `residuals/runs/dataset1_p1/0018_bipolar_threshold5_prototype2_cone35_fitted8/`

Calibration peak-aligns the maximum-amplitude channel waveform, splits it by
signed extremum, and initializes one positive and one negative prototype.
Eight temporal atoms are assigned alternately to the two prototypes
(within-polarity spherical k-means initialization) and constrained to
35-degree one-sided cones. Updates use the residual model's closed-form
temporal sufficient statistics, cone projection, weighted-SVD prototype
refitting, and fixed-assignment objective backtracking. Prototype polarity is
fixed only to resolve label/sign ambiguity; atoms are never flipped across
polarity families during cone projection.

Both calibration discovery and raw-residual proposal discovery use the new
bipolar detector. Everything downstream — the coherent xyz-sigma fit,
acceptance gates, subtraction, residual rescoring — remains the 0016 inner
pursuit. Omega, both prototypes, and atom-to-prototype assignments are
persisted with the run, and the calibrated codebook is frozen for the entire
raw-recording pursuit (no chunk-order-dependent online adaptation; resume
stays consistent).

## Plots and browser

- `residuals/src/plots/plot_0018_prototype_cones.py` renders both prototypes,
  every assigned Omega waveform, and each atom's prototype-cone angle at
  dpi=800.
- `residuals/src/plots/0018_prototype_cone_threshold_plots.sbatch` runs the
  complete localization / reconstruction / codebook-usage / SpikeTensor-style
  / depth-time / prototype-cone suite after a completed summary exists.
- Its final gated step runs `residuals/src/plots/build_plot_gallery.py`,
  producing a SpikeTensor-styled offline browser at
  `residuals/out/0018_bipolar_threshold5_prototype2_cone35_fitted8/index.html`
  with all 16 generated panels registered.
- The saved state covers pursuit/stopping, both prototypes and every Omega
  row, usage, xyz/depth-time localization, reconstruction error, and raw
  observed/predicted/residual examples — plus the time/depth/amplitude inputs
  for a later DREDge solve. It does *not* contain dense SpikeTensor
  coefficient vectors, a soft/hard readout pair, multi-source
  support/conditioning/LOO state, saved DREDge corrections, or the complete
  `atom_viewer.py` pack; the browser names those gaps instead of passing off
  analogues as exact panels.

## Validation and runs

Shell and Python syntax, imports, CLI construction, and the focused CPU
self-test (signed positive/negative extrema, opposite-polarity competition,
signed detector outputs, alternating atom-family assignment, prototype
polarity, the 35-degree cone invariant) all passed inside Singularity.

- First full job `16625553` failed after 39 s: `pytorch.ext3` lacks the
  `spikeglx` module the raw-recording reader needs.
- The sbatch was corrected to the established `ibl-sorter.ext3:ro` runtime,
  and both `torch`/`spikeglx` imports plus the self-test were re-verified
  inside that exact runtime before resubmission.
- Corrected full run `16626415`: exit `0:0` in 1h01m22s, producing
  2,891,519 events over 1,958 chunks.
- Plot job `16626447`: exit `0:0` in 2m38s; the browser generator passed its
  syntax check and registered 16/16 panels.

`docs/0018_optimization_understanding.md` documents bipolar discovery,
prototype-cone calibration, the raw-pursuit bridge, persisted arrays,
optimization structure, and the active configuration; it was not committed,
per your instruction.

## Next steps

- [x] Validate syntax, runtime imports, CLIs, and the detector/cone self-test.
- [x] Queue the corrected full run and dependent plot suite.
- [x] Confirm both jobs completed and the offline browser was built.
- [ ] Review positive/negative proposal balance, prototype/atom usage,
      localization, and saved reconstructions before drawing a model
      conclusion.

## Links

- [[session-017-initial-threshold-spike-discovery]]
- [[session-019-all-channel-error]]
- [[feedback_plot_suite_completeness]]
