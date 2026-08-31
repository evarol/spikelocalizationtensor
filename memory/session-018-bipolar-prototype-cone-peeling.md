# Bipolar Prototype-Cone Residual Peeling
**Created:** 2026-08-30
**Last updated:** 2026-08-30
**Status:** Active — full run, plot suite, and SpikeTensor-style browser complete

## Decision

- 0018 must represent spikes that point either upward or downward. Its two
  temporal mother shapes are therefore an explicit positive/negative pair, not
  two negative-going morphological clusters.
- The 0016/0017 convention that flips every Omega row negative is incorrect for
  this experiment and is not used by 0018.
- The 0017 negative-only proposal detector is also not used. SpikeInterface's
  locally-exclusive detector under
  `/scratch/ap7151/_REFERENCE/spikeinterface/src/spikeinterface/sortingcomponents/peak_detection/`
  is the authoritative discovery design for 0018.
- Per user instruction, the implementation was written without running syntax,
  self-test, Singularity, smoke-run, or SLURM checks.

## SpikeInterface detection semantics

- The detector uses `peak_sign="both"`, threshold 5 in per-channel robust-noise
  units, and a 1-ms exclusion sweep.
- Positive candidates satisfy an inclusive positive threshold, are strictly
  larger than the preceding sample, and at least as large as the following
  sample. Negative candidates use the sign-reversed inequalities.
- Positive and negative candidates compete together within the temporal and
  spatial neighborhood using absolute noise-normalized amplitude.
- Exact score ties retain the earlier sample; equal-time cross-channel ties
  survive. Conservative boundary exclusion is combined with the waveform-safe
  center bounds already required by residual fitting.
- Saved accepted events retain signed normalized detection score, signed raw
  peak amplitude, detection polarity, fitted prototype family, and fitted
  polarity.
- 0018 follows the intended signed NumPy locally-exclusive behavior rather than
  the reference Torch path's version-specific `abs()` polarity loss and
  normalized-amplitude output quirks.

## 0018 implementation

- Entry point:
  `residuals/src/preprocessing/0018_prototype_cone_threshold_peeling.py`.
- Full-run script:
  `residuals/src/preprocessing/0018_prototype_cone_threshold_full.sbatch`.
- Full output target:
  `residuals/runs/dataset1_p1/0018_bipolar_threshold5_prototype2_cone35_fitted8/`.
- Both calibration discovery and raw residual proposal discovery use the new
  bipolar locally-exclusive detector. The downstream coherent xyz-sigma fit,
  acceptance gates, subtraction, and residual rescoring remain the 0016 inner
  pursuit.
- Calibration peak-aligns the maximum-amplitude channel waveform, separates it
  by signed extremum, and initializes one positive and one negative prototype.
  Eight temporal atoms are assigned alternately to the two prototypes,
  initialized by within-polarity spherical k-means, and constrained to 35-degree
  one-sided cones.
- Calibration updates use the residual model's closed-form temporal sufficient
  statistics, cone projection, weighted-SVD prototype refitting, and
  fixed-assignment objective backtracking. Prototype polarity is fixed only to
  resolve label/sign ambiguity; candidate atoms are not flipped across
  polarity families during cone projection.
- Omega, both prototypes, and atom-to-prototype assignments are persisted with
  the run. The calibrated codebook is frozen throughout raw-recording pursuit,
  avoiding chunk-order-dependent online adaptation and preserving resume
  consistency.

## Plot suite

- `residuals/src/plots/plot_0018_prototype_cones.py` plots both prototypes,
  every assigned Omega waveform, and each atom's prototype-cone angle at
  `dpi=800`.
- `residuals/src/plots/0018_prototype_cone_threshold_plots.sbatch` runs the
  complete localization, reconstruction, codebook-usage, SpikeTensor-style,
  depth-time, and prototype-cone suite after a completed summary exists.
- Its final gated step runs `residuals/src/plots/build_plot_gallery.py`, which
  writes a SpikeTensor-style offline browser at
  `residuals/out/0018_bipolar_threshold5_prototype2_cone35_fitted8/index.html`.
  The browser uses SpikeTensor's color system, typography, control groups,
  detail card, full-panel stack, and contact-sheet mode. It registered all 16
  generated 0018 panels.
- 0018 has enough saved state for pursuit/stopping, both prototypes and every
  Omega row, usage, xyz/depth-time localization, reconstruction error, and raw
  observed/predicted/residual examples. It has time/depth/amplitude inputs for
  a later DREDge solve. It does not have dense SpikeTensor coefficient vectors,
  a soft/hard readout pair, multi-source support/conditioning/LOO state, saved
  DREDge corrections, or the complete `atom_viewer.py` input pack. The browser
  discloses these unavailable exact panels rather than mislabeling analogues.

## Validation, runtime correction, and queued jobs — 2026-08-30

- Shell syntax passed for both 0018 SLURM entry points.
- Python syntax, imports, and CLI construction passed for the 0018 preprocessing
  and prototype-cone plot entry points inside Singularity.
- The focused CPU self-test passed. It covers signed positive/negative extrema,
  opposite-polarity local competition, signed detector outputs, alternating
  atom-family assignment, prototype polarity, and the 35-degree cone invariant.
- Scoped whitespace checks found no trailing whitespace in the 0018 code,
  scripts, documentation, or memory card.
- Initial full job `16625553` failed after 39 seconds because `pytorch.ext3`
  does not provide the `spikeglx` module needed by the raw-recording reader.
- The full-run script was corrected to use the established
  `ibl-sorter.ext3:ro` runtime. Both `torch` and `spikeglx` imports and the 0018
  self-test passed inside that exact runtime before resubmission.
- Corrected full-recording job `16626415` completed with exit code `0:0` in
  `01:01:22`, producing 2,891,519 events over 1,958 chunks.
- Complete plot job `16626447` completed with exit code `0:0` in `00:02:38`.
- The HTML browser generator passed its Singularity syntax check and generated
  the current 0018 index with 16 panels, all 16 registered.
- `docs/0018_optimization_understanding.md` documents bipolar discovery,
  prototype-cone calibration, the raw-pursuit bridge, persisted arrays,
  optimization structure, and the active full configuration. It was not
  committed, per user instruction.

## Next Steps

- [x] Validate syntax, runtime imports, CLIs, and the focused detector/cone
  self-test inside Singularity.
- [x] Queue the corrected full 0018 run and complete dependent plot suite.
- [x] Confirm the full and dependent plot jobs complete successfully and create
  the SpikeTensor-style offline browser.
- [ ] Review
  positive/negative proposal balance, prototype/atom usage, localization, and
  saved reconstructions before drawing a model conclusion.

## Links

- [[session-017-initial-threshold-spike-discovery]]
- [[session-019-all-channel-error]]
- [[feedback_plot_suite_completeness]]
