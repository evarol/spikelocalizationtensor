# IBL-Style Pursuit (009)
**Created:** 2026-08-26
**Last updated:** 2026-08-26
**Status:** Done — Δχ² gate committed and smoke-tested; the first-fit diagnostics traced the one-contact pathology to the spatial dictionary and seeded the 0010+ lineage

## What IBL actually does (reference notes)

Source at `/scratch/ap7151/_REFERENCE/ibl-sorter`; Kilosort analog at
`/scratch/ap7151/_REFERENCE/Kilosort`.

- IBL detects on locally-whitened, destriped, re-referenced data projected
  onto template spatial factors (a matched filter) — not on raw voltage.
  Threading a global spatial whitening rotation through every torch kernel
  would be major surgery; the cheap faithful step is the acceptance gate
  below.
- Detection pipeline: destripe (300 Hz+ order-3, k-filter), local ZCA over
  the 32 nearest channels, `spkTh=-6` isolated-peak template bootstrap, then
  mexMPnu8 matched-filter pursuit with Th=6 (learn) / Th=3 (final) in
  whitened unit-noise units, subtractive peeling up to 500 iterations, and
  kriging 10× subsample alignment.
- `eloss = best − second-best` template margin is computed but not used to
  reject events; `lam=10` is an amplitude prior inside the template score,
  not a residual-quality gate. The real IBL reliability gate is cluster-level
  and post-hoc: lower a per-cluster amplitude/variance threshold until
  refractory/CCG contamination exceeds 10% (or a stricter 1–5% target if
  initially clean), drop low-score events, label clusters good/mua. IBL has
  no per-event spatial-reconstruction coherence test — exactly what this
  project wants.
- Kilosort4 (`9f8e705`): high-pass + median-CAR → local ZCA (32 nearest) →
  drift → universal-template detection (6 unit templates × 5 spatial Gaussian
  sizes, max over |A|, dedup over 100 nearest template centers, local max
  ±20 samples, `Th_universal = 9` in whitened units) → clustering →
  learned-template pursuit (`sqrt(relu(B)²/nm) > 8`, local max ±61 samples,
  ≤100 peels/batch) → merge → save. Denoising comes from cluster-mean PC
  templates, low-rank temporal PC reconstruction, feature construction, and
  alignment/merging — not per-event cleanup. No `lam`, no median smoothing.

## The cheap, faithful step: Δχ² acceptance

The noise-weighted acceptance gate derived in
`docs/noise_weighted_pursuit_acceptance.tex` (`Δχ² = α̂²`, Eq. 109)
replaces/augments the `min_captured_fraction` heuristic. Why the old gate
fails: a threshold-6 whitened projection has `Δχ² ≈ 36`, but the raw
captured fraction divides by the full 8-channel × 90-sample patch whose
expected whitened noise alone is ~720 — so even a clean detection scores
about 4.8%, right at the 5% floor.

Implementation (committed `9111f4b`): `min_delta_chi2` in `ResidualConfig`;
`subtract_predictions_monotone` computes
`Δχ² = (Σ current·atom/σ²)² / Σ atom²/σ²`, requires the noise argument when
the gate is active, keeps the monotone-energy and finite-parameter
requirements, and refits gain with the same diagonal-noise-weighted objective
when gating is active (legacy mode keeps the unweighted gain). Threaded
through both peel kernels and `run_global_codebook_pursuit.py`
(`--min-delta-chi2`, `--min-captured-fraction`). Verified in Singularity:
gate-off is identical to legacy, gate-on rejects correctly, and the
brute-force whitened projection gain matches reported `Δχ²` to 1e-4. One bug
caught along the way: `np.ndarray.square` doesn't exist — `np.square`.

Q8 sample job `16375246` (5m03s; threshold 6, capture gate off, Δχ² ≥ 36,
60 rounds, over the 8.748 s window) completed, with plot commit `d1eaf88`.
The new gate barely moved aggregate round trajectories — aggregate
acceptance statistics alone can't diagnose the bad fits.

## The first-fit diagnostic (the important finding)

You spotted that later pursuit rounds contaminate visual diagnosis, so
commit `8201bdc` added an extractor that freezes each preprocessed chunk and
runs only first-round threshold-6 detection plus localization — no
subtraction, sequential gain refit, or rescoring. For every conflict-free
first-round candidate in the 8.748 s Q8 window it saves observed `Y_c(t)`,
fitted `Ŷ_c(t)`, residual, noise, masks, metadata, and per-channel
`Δχ²_c = 2⟨Y_c,Ŷ_c⟩/σ_c² − ‖Ŷ_c‖²/σ_c²` (the eight values sum to the true
weighted residual improvement; negative channels indicate mismatch).
Extraction `16410844` (59 s) saved 2,179 first fits across four chunks; PNG
gallery `16412097` wrote 28 panels with `gallery_selection.json` mapping
each image to its chunk row and total Δχ². (First attempt `16410352` failed
on the missing `spikeglx` module in the pytorch overlay — the launcher moved
to the verified `ibl-sorter` overlay; gallery attempt `16410899` failed on a
`PdfPages`-as-PNG bug and was replaced by direct PNGs, commit `83467c9`.)

The sentinel `chunk_00_row_0011` stayed a bad one-contact reconstruction
through every spatial fix tried:

- Its normalized footprint is a near-delta on the peak channel (0.995 vs
  0.03–0.06 elsewhere), with the source essentially directly above the
  channel at sigma = 1 µm. The production solver allowed this: `maths.py`'s
  voxel bound `z ≥ 0` plus 1 µm refinement admits `z=0, sigma=1` solutions.
  UnitMatch's solver, using the same monopole formula, searches
  `z = geomspace(1, 300)` — no contact-plane source — and never refines to
  1 µm voxels. SpikeInterface's monopolar triangulation also bounds `z ≥ 1`,
  but at a 20 µm pitch even `z=1` leaves a neighbor at ~1/20 of peak
  amplitude, so that bound alone can't prevent one-contact fits.
- **z=1 test** (commit `8fdc14a`; jobs `16418938` + gallery `16419019`):
  the sentinel moved from (10.620, −0.981, 0) to (10.655, −0.728, 1) µm;
  peak PTP 32.37 → 32.30 µV, other channels 1.1–1.8 → 1.3–2.0 µV against
  observed 29–42 µV. Not the cause.
- **sigma=1 removal** (commit `d1978d7`: spatial dictionary 1..512 → dyadic
  2..512 µm; jobs `16424683` + gallery `16424709`): the sentinel is still
  dominated by channel 10.

Conclusion (your call): keep the separable model
`Y_c(t) ≈ α · g(x,y,z,σ) · Ω_q(t)` and fix the *acceptance* problem instead
— aggregate-error localization lets a strong peak-channel improvement
outweigh worsened companions. Next: a channel-aware full-waveform criterion
or rejection gate so a candidate can't be accepted on peak-channel strength
alone. No denoising, no switch to PTP (diagnostic only). This decision is
the direct ancestor of 0010's per-channel RMSE gate and, eventually,
0018/0019's all-channel requirements.

## Loose ends at close

- True upstream whitening (W applied to data + footprint columns before
  scoring) remains the full IBL analogue and would need `maths.py` surgery;
  Δχ² was deliberately the cheap compatible step first.
- 008's whitening diagnostic had already shown whitening doesn't rescue poor
  captures, which is why Δχ² was the path.

## Links

- [[session-010-whitened-dense-pursuit]]
- [[session-008-peak-channel-codebook-init]]
- [[session-007-global-codebook-pursuit]]
- [[project_overview]]
