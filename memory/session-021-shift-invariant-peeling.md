# Shift-Invariant Temporal Codebook (0021)
**Created:** 2026-08-31
**Last updated:** 2026-08-31
**Status:** Draft plan — not started; written while full run `16655016` is still in progress

## Why shift invariance

Every event is extracted peak-anchored (n_before/n_after around the detection
peak), but the waveform shape around that anchor jitters by a few samples
from spike to spike — the fixed-onset Ω forces all of that jitter into either
the gain, a wrong atom, or a biased σ. 0020 attacks expressiveness with two
atoms; 0021 attacks alignment: keep the model rank-1 but let each event pick
its own integer lag τ from a shift bank, so one atom covers a family of
onsets. The two plans are complementary temporal-expressiveness extensions
and share the same acceptance machinery.

## The SpikeTensor reference

SpikeTensor already built and ran this model; 0021 ports its parameterization
into the sln residual pursuit:

- `spiketensor/unified.py` is the single model file. Its headline equation is
  `Ŷ_s = Σ_r a_r · g(·; μ, σ) · (S_τ ψ_q)ᵀ` with `shape="onehot"` — exactly
  one atom at exactly one lag per spike, amplitude non-negative.
- The shift bank (`shift_bank`, unified.py:147–166) enumerates **integer
  lags only**: τ ∈ [−max_shift, max_shift] (default 10), zero-padded shifts
  of each atom, each row renormalized to unit norm. Zero-padding is
  deliberate — a spike shifted partly out of the window loses that energy,
  which is the honest accounting.
- The per-event lag is an **integer index chosen during pursuit** and stored
  as `source_shift` (int16). There is no sub-sample interpolation, no
  fractional shift, and no continuous shift parameter anywhere in
  SpikeTensor (verified across code and docs).
- Codebook learning undoes each source's lag before accumulating temporal
  sufficient statistics, then applies shift-aligned orthogonal Procrustes
  (`svd(Cacc)` → polar factor); proposals are scored and rolled back if nMSE
  regresses.
- Their headline run is `prior2_shift_M64_R4` (P=2 prototype prior + shift,
  M=64, R=4), referenced by the README schematic and atom viewers. Its
  outputs live upstream (`zncc/runs/onehot_prior/`) and are **not** in this
  workspace snapshot.
- One bug report worth keeping: `source_figures.py:345–377` — reconstructing
  at zero lag instead of the stored lag disagreed with stored SSE by up to
  4×. Lesson: every reconstruction path must apply the stored lag.

SpikeTensor's inference picks (place × bank index) greedily over the full
product dictionary with a spatial shortlist. That is affordable for their
M=64 offline fit; the sln pursuit has a different cost structure (4096 sites
× 9 sigmas already), so 0021 needs a cheaper integration strategy — below.

## The model in sln terms

```
prediction[c,t] = footprint[c](x,y,z,σ) · α · Ω[q, t − τ],   α ≥ 0
```

with τ ∈ {−10, …, 10} samples (max_shift=10 to match SpikeTensor's default;
tune later). Because events are peak-anchored, τ is a shape-jitter
correction, and the gain stays the same closed form as rank-1 given (site, σ,
q, τ): `α = (f·y·Ω_τ) / (‖f‖² · ‖Ω_τ‖²)` — since bank rows are unit-norm
shifted copies, `‖Ω_τ‖ = 1` except at window-edge clipping, which the
renormalization already handles.

## Where the lag enters the search — the key design decision

The naive route multiplies the coarse stage by (2·max_shift+1) = 21:
4,096 sites × 9 σ × 8 atoms × 21 lags. That is a non-starter. The plan is a
two-tier lag search:

1. **Coarse + refine unchanged, zero lag.** The existing coherent assignment
   (16³ × 9σ × Q) and three-level 27-point refine pick (x, y, z, σ, q) as
   today. Justification: the site is a spatial decision driven by the
   footprint-vs-energy pattern; a 2–3 sample jitter almost never flips it.
2. **Lag pass (new).** At the winning (x, y, z, σ), correlate the per-event
   residual waveform against the M × (2·max_shift+1) shift bank, take the
   best (q', τ), and refit α. Accept the lagged fit only if it beats the
   zero-lag objective under a complexity penalty (like 0020's rule), else
   keep the zero-lag result. Cost: C·T·M·21 per event ≈ 120k multiply-adds
   — about 1.1–1.3× per event overall, no footprint-cache changes
   (footprints depend only on site/σ/mask).
3. **Honest approximation caveat, written into metadata:** the coarse site
   was chosen at zero lag. If diagnostics show events whose winning τ sits
   at the ±10 rail with a large objective gain, add an optional
   `--coarse-lag-search` mode that scores the top-K sites at a few lags
   before committing (a controlled cost knob, off by default).

An alternative considered and rejected for v1: fold τ into the refine
coordinate descent as a fourth discrete dimension (27-point × 21 lags per
level). It is cleaner but triples refine cost for a benefit the post-hoc lag
pass should already capture.

## Learning the codebook under shifts

The sln pursuit freezes Ω after calibration (0018 convention, resume
consistency). Two options:

- **v1 (chosen): learn Ω at zero lag, exactly as 0019 does**, and add the
  shift bank only in the pursuit. Cheapest; the risk is that calibration
  atoms absorb shape jitter by smearing, which the pursuit lag then
  double-corrects. Diagnostic to watch: the lag histogram should be
  unimodal near 0; a heavy-tailed or bimodal histogram says the atoms
  themselves are smeared composites.
- **v2 (follow-up if needed): shift-aware calibration** — during calibration,
  assign each event its best lag before accumulating the fixed-assignment
  sufficient statistics, undo the lag in the accumulators, and refit
  prototypes by Procrustes (port SpikeTensor's `basis_proposal` logic).
  This needs rollback-on-regression like their α ∈ {1, ½, ¼} backtracking.

## Acceptance, duplicates, replay, schema

- **Acceptance gates are unchanged in form** — per-channel improvement and
  the all-channel 20% bar are computed from the final prediction, however it
  was formed. But the extra dof again inflates `captured_fraction` and the
  projection score, so the score-8 floor needs the same recalibration
  question as 0020. Decide after the CPU synthetic.
- **Duplicate mask** (0016:494): `temporal_index` stays the atom; add
  `shift_lag` (int16) as a new saved field. Key duplicates on (time,
  channel, atom) as today and check whether equal-atom-different-lag pairs
  within the merge window need a lag-tolerance term.
- **`replay_predictions` (0019:421)** must apply the stored lag when building
  prior-pass predictions: the atom row shifts by τ with zero padding — a
  gather with clamped/zeroed edges, batched exactly like the current
  `omega[q]` lookup. This is the SpikeTensor zero-lag bug territory: a unit
  check must verify replay predictions equal stored-SSE reconstructions lag
  for lag.
- **Chunk npz additions:** `shift_lag`, `shift_objective_gain` (per-event
  objective improvement over zero lag), `at_lag_rail` (bool, |τ| = max_shift)
  for the caveat diagnostic. Rejected-candidate audit gains the same fields.
- Plot loaders and the 0021 plot script follow the completeness lesson: all
  Ω waveforms with usage, lag histogram, lag-vs-depth and lag-vs-amplitude
  scatter (drift manifests as structured lag — a cheap motion probe), and
  explicit disclosure of any exact panels not producible.

## Falsifiable diagnostics (the run is only worth it if)

- Lag histogram: mass near zero with realistic spread (1–3 samples) is the
  win; pile-up at the rails means max_shift is too small or the model is
  abusing lags.
- Objective-gain distribution: what fraction of events take a nonzero lag
  with a meaningful penalty-adjusted gain, and does the accepted-event count
  rise (spikes that failed the all-channel bar at zero lag now pass)?
- σ usage vs 0019: if fixed-onset mismatch was biasing σ narrow, the
  narrow-σ pile should relax.
- Sub-ms double-detection rate vs 0019 (lag freedom should not increase it;
  if it does, the duplicate key needs the lag term).
- Per-channel captured-fraction histograms vs 0019 — the headline metric.

## Sequencing

1. Wait for full run `16655016` and its review (shared with session 019);
   0021 is specced against 0019's results. Order relative to 0020 is open —
   they are independent single-variable extensions of 0019 and can be run in
   either order on the same base.
2. Implement `residuals/src/preprocessing/0021_shift_invariant_peeling.py`
   derived from 0019, `--max-shift` default 10 with 0 = exact 0019
   reproduction (the SpikeTensor convention).
3. CPU synthetic validation: (a) a spike built from an atom at τ = +4 must
   be rejected-or-biased at zero lag and recovered with α within tolerance at
   the true lag; (b) replay-with-lag unit check against stored SSE (the
   SpikeTensor bug class); (c) zero-lag path bit-identical to 0019's fit when
   `--max-shift 0`.
4. Full run + dependent plot suite in the 0019 sbatch pattern
   (ibl-sorter.ext3 runtime, USR1 requeue trap, `afterok`-held plots
   released only after review).

Files follow the established layout: script and sbatch in
`residuals/src/preprocessing/`, plots in `residuals/src/plots/`, run output
under `residuals/runs/dataset1_p1/0021_shiftinvariant_maxshift10_fitted8/`,
figures in `residuals/out/`.

## Next steps

- [ ] Review 0019 run `16655016` when it completes (shared with session 019).
- [ ] Decide the lag-acceptance rule: penalty for the extra dof and whether
      the fitted-projection floor is recalibrated.
- [ ] Implement the post-refine lag pass inside a copy of `fit_grouped`,
      plus replay-with-lag.
- [ ] CPU synthetic: τ-recovery, replay unit check, `--max-shift 0`
      equivalence.
- [ ] Full run + held plot suite; compare lag histogram, σ usage, and
      per-channel fractions against 0019.
- [ ] If the lag histogram is smeared/bimodal, port shift-aware calibration
      (Procrustes) as the v2 follow-up.

## Links

- [[session-019-all-channel-error]]
- [[session-020-rank2-temporal-peeling]]
- [[session-018-bipolar-prototype-cone-peeling]]
- [[session-016-one-hot-lattice-peeling]]
- [[feedback_plot_suite_completeness]]
