# Rank-2 Temporal Peeling (0020)
**Created:** 2026-08-31
**Last updated:** 2026-08-31
**Status:** Draft plan — not started; written while full run `16655016` is still in progress

## Why rank 2

0019 fixes the acceptance side of the fit (every channel must capture 20% of
its own energy) but keeps the model rank-1: one monopole footprint times one
temporal atom, α·f(x,y,z,σ)·Ω[q]. Real biphasic spikes — a trough plus a
delayed peak whose amplitude ratio no single atom matches — are the dominant
rank-1 failure mode. Under the all-channel bar such a spike either gets
rejected outright or drags σ/ρ off-physical trying to absorb the temporal
mismatch. 0020 gives each spike a two-atom temporal basis so biphasic shapes
are exactly expressible, while keeping the spatial factor rank-1 (one source,
one footprint) so localization stays identifiable.

## The model

```
prediction[c,t] = footprint[c] · (α₁·Ω[q₁,t] + α₂·Ω[q₂,t]),   α₁, α₂ ≥ 0
```

Both atoms share the same (x, y, z, σ) and hence the same footprint. Because
Ω holds both polarity cones (0018) and the gains are non-negative, a
trough-plus-peak shape is a two-atom sum through one monopole — no new
spatial machinery.

The gains stay closed-form. Given the site, σ, and atom pair, with
`b_k = Σ_c f_c · projected[c, q_k]` (the per-event atom correlations the
code already computes as `projected`), `G = Σ_c f_c²`, and cosine
`m = Ω[q₁]·Ω[q₂]` (Ω is normalized, so no time sum needed):

```
α₁ = (b₁ − m·b₂) / (G·(1 − m²)),   α₂ = (b₂ − m·b₁) / (G·(1 − m²))
```

Two-variable NNLS is exact by active set: if one gain comes out negative,
zero it and the surviving gain reduces to the rank-1 answer. The per-channel
improvement and all-channel acceptance code (0019 lines 787–813) is agnostic
to how the prediction was formed and stays untouched — only `fit_grouped`
internals change.

## The search must stay greedy, not exhaustive

Exhaustive atom pairs are Q² per (site, σ); at the coarse stage that is
4,096·9·Q² evaluations and a non-starter. The plan is OMP-style greedy with a
shared footprint:

1. **Coarse (unchanged, rank-1).** The existing coherent coarse assignment
   (16³ sites × 9 sigmas × Q) picks the site. The site choice almost never
   depends on the second atom.
2. **Refine (rank-1, unchanged).** The existing three-level 27-point
   coordinate descent fixes (x, y, z, σ, q₁, α₁) as today.
3. **Second-atom pass (new).** Form the rank-1 prediction, compute the
   per-event residual, project it onto Ω, and take the best-scoring atom q₂
   (q₂ may equal q₁ only if the residual picks a delayed copy — start by
   forbidding q₂ = q₁ and check whether that matters). Refit both gains
   jointly with the 2×2 NNLS and re-score. Accept the rank-2 fit only if it
   beats the rank-1 objective under a complexity penalty (one extra dof must
   earn its keep, e.g. a fixed SSE-improvement floor per event), otherwise
   fall back to the rank-1 result. Rank-1 events keep their old fields
   exactly.

Expected cost is roughly 1.3–1.6× per event — no Q² blowup, no footprint
cache changes (footprints depend only on site/σ/mask).

## Design risks and their guards

- **Collinearity.** Gains scale as 1/(1 − m²). Cone projection already
  spreads atoms, but forbid atom pairs with |cos| above ~0.95 (regularizing
  or rejecting) and report the pair-cosine distribution in diagnostics.
- **Score floor inflation.** Two extra degrees of freedom make
  `captured_fraction` and the projection score easier to pass, so the score-8
  floor and the 20%/channel bar would silently loosen if kept as-is. The
  per-channel all-channel bar is a built-in guard (a second atom only helps
  where it genuinely improves each channel), but the fitted-projection floor
  needs recalibration for a fair single-variable comparison. Decide after the
  CPU synthetic: either keep the floor and accept the slight looseness
  (documented), or raise it to match rank-1's null distribution.
- **Duplicate mask.** `duplicate_mask` (0016:494) keys on one
  `temporal_index` per event. Convention: `temporal_index` stays the
  dominant atom (larger |α|), plus a new `temporal_index2` field. Keep the
  duplicate key on the dominant atom and check sub-ms double-detection rates
  before and after.

## Schema and blast radius

New per-event arrays in the chunk npz: `alpha2`, `temporal_index2`,
`rank2_used` (bool), `rank2_objective_gain`. Everything downstream of the
stored arrays must learn the new fields:

- `replay_predictions` (0019:421): prediction becomes
  f ⊗ (α₁Ω[q₁] + α₂Ω[q₂]); `load_prior_events` (0019:386) loads the new
  columns.
- Consolidation and the rejection audit: rejected events also need their
  rank-2 fit metrics if we want to audit "rank-2 would have saved it" cases.
- The three plot loaders that gained the `pass_*/chunk_*.npz` fallback
  (0016 one-hot, raw-residual, spiketensor) and the 0019 plot script —
  per the plot-suite completeness lesson, the 0020 suite must render every
  Ω waveform with usage and disclose any exact panels it cannot produce.

## Falsifiable diagnostics (the run is only worth it if)

- α₂/α₁ distribution: mass concentrated near zero means rank-2 is
  noise-fitting and the honest conclusion is "rank-1 suffices".
- Fraction of events where rank-2 wins the penalized objective.
- σ usage histogram vs 0019: did the σ-2 µm cheat die, and did narrow-σ
  events shift toward plausible widths?
- Sub-ms double-detection rate: two detections for one true spike should
  collapse when one rank-2 event can cover both.
- Per-channel captured-fraction histograms vs 0019 (the metric 0019 exists
  to fix).

## Sequencing

1. Wait for full run `16655016` to finish and review its rejection-reason
   histogram, σ usage, and per-pass counts (the open 0019 next-step). 0020
   is specced against 0019's results, not in parallel with them.
2. Implement `residuals/src/preprocessing/0020_rank2_temporal_peeling.py`
   derived from 0019, with `--rank2` off by default so a rank-2-off run
   reproduces 0019 numbers exactly before turning it on.
3. CPU synthetic validation: build a biphasic spike as two atoms through one
   monopole; verify rank-1 fails the all-channel bar (or lands a biased
   σ/ρ) while rank-2 passes with the correct gains; verify the closed-form
   NNLS against a brute-force grid and the fallback path when one gain is
   negative.
4. Full run + dependent plot suite in the 0019 sbatch pattern
   (ibl-sorter.ext3 runtime, USR1 requeue trap, `afterok`-held plots
   released only after review).

Files and outputs follow the established layout: script and sbatch in
`residuals/src/preprocessing/`, plot script and sbatch in
`residuals/src/plots/`, run output under
`residuals/runs/dataset1_p1/0020_rank2_...`, figures in `residuals/out/`.

## Next steps

- [ ] Review 0019 run `16655016` when it completes (shared with session 019).
- [ ] Decide the rank-2 acceptance rule: penalized-objective floor value and
      the pair-collinearity cutoff.
- [ ] Implement the second-atom pass inside a copy of `fit_grouped`.
- [ ] CPU synthetic: biphasic recovery, NNLS closed form vs brute force,
      negative-gain fallback.
- [ ] Full run + held plot suite; compare σ usage, α₂ mass, and
      double-detection rate against 0019.

## Links

- [[session-019-all-channel-error]]
- [[session-018-bipolar-prototype-cone-peeling]]
- [[session-016-one-hot-lattice-peeling]]
- [[feedback_plot_suite_completeness]]
