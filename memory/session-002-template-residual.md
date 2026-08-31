# Template Residual Detection (002)
**Status:** Done — superseded by 004; the 10-second smoke exposed exactly the problems 004 set out to fix

## Idea

Replace the separate SpikeInterface peak-detection and waveform-extraction
front end with raw BIN/CBIN processing through `spikeglx.Reader`, using the
analytic localization model as a matching-pursuit reconstruction atom. The
loop: read and preprocess an overlapping raw chunk → evaluate codebook
templates at every valid waveform window → select joint space-time local
maxima → run the full spatial localization with the frozen temporal codebook
→ reconstruct the accepted component → subtract it from the residual →
recompute all template scores and repeat.

## Implementation

- `src/maths.py`: `localize_spikes_fixed_codebook`,
  `reconstruct_spike_fits`, `build_codebook_detection_footprints`.
- `src/preprocessing/raw_residual.py`: raw reading through
  `spikeglx.Reader` only; preprocessing is a third-order zero-phase
  300–6000 Hz bandpass followed by per-sample global median reference.
- Detection convolves every channel with every temporal row of Omega and
  combines the projections with anchor-centered spatial codebook footprints
  (8 temporal rows × 10 monopole profiles per anchor in the default bank),
  made exclusive over ±0.5 ms and the 48 µm channel-map neighborhood.
- Accepted events receive the complete coarse-to-integer-1 µm spatial
  search, not just the reduced detection bank. Chunk results are restartable
  and can save the residual waveform presented to each accepted fit.

Five synthetic tests in `src/test_raw_residual.py` all passed on CPU in the
ibl-sorter overlay; one caught suppression that originally compared
neighboring channels only at the identical sample — it now covers the joint
space-time neighborhood.

## Smoke result

Job `16058517` (3m34s, launcher `raw_template_residual_smoke.sbatch`)
accepted 353,279 events in 10 s (pass counts 89,657 / 88,851 / 87,941 /
86,830); median captured fraction 0.152, with 80% of fits below 0.25. Every
saved source was exactly integer-valued — the output never received
continuous refinement — and a 20 ms chunk-0 replay showed RMS 100% → 89.9%
→ 86.0% → 87.6% → 97.8% across the four passes: passes three and four
undid most of the initial reduction. Those three problems (threshold
calibration, repeated-pass behavior, localization continuity) are exactly
what [[session-004-continuous-residual]] took on.

Plots: `localizations_by_pass.png` and `residual_recording_chunk0.png`
under `out/plots/raw_template_residual_smoke_16058517/`, via
`plot_raw_residual_localizations.py` and `plot_raw_residual_recording.py`.
This run used a BIN recording; the replay script reads BIN or CBIN, and no
continuous residual recording was saved.

Worktree note: the residual files and solver changes were uncommitted at the
time; `plot_amplitude_scatter.py` and `plot_monopole_paired_scatter.py`
predate this branch and were never to be bundled with it.

## Links

- [[session-004-continuous-residual]]
