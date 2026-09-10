# Shift-Invariant Temporal Codebook (0021)
**Created:** 2026-08-31
**Last updated:** 2026-09-09
**Status:** Implementation started 2026-09-09 (codebook + preflight; see addendum below) — pursuit-side lag pass not yet built (specced against the completed 0019 run `16655016`)

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

## Addendum: four priors (polarity × droop speed) alongside shift invariance (2026-09-09)

The user wants the two-prototype calibration generalized to **four priors** —
2 polarities × 2 droop speeds (slow/wide vs fast/narrow decay) — and asked
whether that composes with the lag machinery. It does; the two live on
different layers, and the current code is much closer to 4-prior-ready than
the hard-coded "two-prototype" wording suggests:

- **What is actually hard-coded to 2:** only the init path.
  `initialize_codebook` (0019:1321) splits the calibration pool by binary
  extremum polarity and deals atoms round-robin; `peak_aligned_waveforms`
  (0019:1274) emits the binary label; a RuntimeError demands both groups.
- **Already general:** the alternating refit `prototype_cone_proposal`
  (0019:1518) loops over `len(prototypes)` — per-group SVD re-derivation,
  polarity fix, cone re-projection — and acceptance/duplicate/localization
  speak atom indices, never priors. `fix_polarity`'s "even index positive"
  convention works for any count if groups are ordered
  `prototype_index = polarity + 2*speed` (0 = pos-slow, 1 = neg-slow,
  2 = pos-fast, 3 = neg-fast) — even indices are exactly the positive ones.
- **The one new ingredient:** a per-waveform speed label. Polarity is a free
  hard label (extremum sign); droop speed needs a feature and a 2-way split.
  Primary feature: trough-to-peak width in samples on the peak-aligned,
  unit-norm waveform (main extremum → opposite-sign extremum after it);
  fallback half-decay width where no opposite extremum exists. Split by 1-D
  2-means per polarity (deterministic seed), threshold recorded — never a
  hand-picked cutoff. Bin 0 = narrow/fast, bin 1 = wide/slow.
- **Composition with the lag machinery:** priors structure how atoms are
  learned (grouping + cone constraints); the lag machinery changes how each
  group's statistics accumulate (undo each source's τ before summing,
  shift-aligned proposal) and adds a per-event lag pass after the spatial
  fit. With 4 priors the lag-undone accumulation runs once per prior group —
  same code, four calls. The shift bank is per-atom, indifferent to priors.
- **Conventions verified against the reference implementation** (2026-09-09,
  `spiketensor/unified.py` in-repo): their `shift_bank` (lines 147-166,
  zero-pad + renormalize, τ ≥ 0 delayed), their lag undo in
  `basis_proposal` (lines 389-399, per-unique-τ head-drop zero-pad
  accumulation), and their cone block (lines 406-423, cone-project
  accumulated profiles, amplitude-weighted SVD prototypes, re-project after
  prototypes move) are exactly the machinery 0019/0021 already use. The one
  deliberate difference: their acceptance guard re-runs full inference on
  the candidate basis, so their statistics need no 1/‖S_τΩ‖ rescale; 0021
  keeps 0019's closed-form `fixed_assignment_objective` guard, where the
  rescale is what makes the accumulated statistics exact. Their
  `shift_nonzero_fraction` summary field is a diagnostic worth mirroring
  (0021 records the full lag histogram + mean |τ| per iteration).
- **The one real interaction — identifiability:** both lag and droop reshape
  the waveform's time course (lag translates, droop changes decay), so a
  misaligned slow-droop spike and a fast-droop atom at a compensating lag
  can fit comparably. Mitigations already in the plan: lag chosen only
  after (x, y, z, σ, q), lagged fits accepted under a complexity penalty,
  and the cone projection keeping the four families separated.
- **Cheap sanity check first:** within-cone k-means may already place
  fast- and slow-droop atoms inside the same 35° cone. Reading the 026
  master codebook's atom widths tells us whether hard 4-prior structure
  adds anything beyond what the data already produced; 4 priors earn their
  keep when the prior *gates assignment* (SpikeTensor's `prior2` trick),
  not merely by having both shapes present (027: Q32→Q64 bought +2–4%, so
  capacity is not the bottleneck).
- **Pursuit-side changes needed: none.** Peeling, acceptance, duplicate,
  replay/resume are atom-index agnostic; only calibration learns priors.

## Implementation (started 2026-09-09)

Same file format as the other lineages, `0021_*.{py,sbatch}`:

- `0021_shift_invariant_codebook.py` — the IBL master codebook (reuses 026's
  808 harvested shards, no re-harvest): 4-prior init with the 2-means speed
  split, then alternating fit with optional lag-aligned accumulation
  (zero-pad undo of each event's lag before adding to the per-atom
  numerator; raw input energy so the guard stays a true SSE; lag chosen by
  renormalized shift-bank correlation at the winning site). Flags:
  `--priors {2,4}` and `--shift-learning/--no-shift-learning` so both ways
  are runnable from the same pool.
- `0021_preflight.py` — per-recording preflight: extracts a fresh pool from
  a recording (SpikeGLX `.bin` via spikeglx.Reader or primate `.nwb` via
  0025's NwbReader) with the production recipe (locally exclusive thr 5,
  1 ms isolation), then fits both priors × both shift settings and writes a
  comparison JSON (per-variant objective, dead atoms, lag histogram, speed
  split quality, per-atom widths). Purpose: decide 2 vs 4 priors per
  recording before committing, with objectives comparable across variants
  (same pool, same events; lag learning minimizes over a superset of the
  zero-lag assignments).
- `0021_master_codebook.sbatch`, `0021_preflight.sbatch` — l40s_public,
  ibl-sorter overlay (has spikeglx + h5py + torch; the pytorch overlay has
  neither spikeglx), USR1/TERM requeue trap.

Learning-side lag accounting note: the backtracked guard treats the atom's
canonical frame as lag-free (prediction in the original frame is
α·footprint·S_τΩ, and ‖S_τΩ‖ < 1 by edge loss is ignored in the guard — a
second-order effect at |τ| ≤ 10 on T = 90). The pursuit-side lag pass must
use the exact bank-row norms; that is 021's phase 2, not built yet.

Validation (2026-09-09, both CPU): py_compile clean; the codebook self-test
builds lagged biphasic synthetic shards (2 speeds × 2 polarities, footprints
matching the production σ/sqrt(d²+ρ²+σ²) kernel) and passes — fixed-onset
objective 1133 → 732 vs shift objective 379 → 340, recovered lags within
median 2 samples of the applied ones (max_shift 4). The preflight self-test
fits 2-prior, 4-prior, and 4-prior+shift on one synthetic pool: final SSE
523 → 374 → 267, i.e. the two dials contribute independently. A smoke run of
the preflight against the real dataset1_p1 SpikeGLX file (2000 events, 30 s,
Q8, 1 iteration) exercised the full reader→pool→variants path end to end.
Getting there caught three convention bugs worth remembering: the synthetic
shapes must be biphasic or trough-to-peak width is noise (pure gaussians
have no opposite-sign extremum), the synthetic spatial kernel must match the
footprint family or fit_grouped assigns garbage atoms, and the lag
convention is "delayed by τ" — y(t) = Ω(t−τ), so undoing is y(t+τ).

Queued 2026-09-09 evening: the headline master codebook (4-prior +
shift learning, Q32) as job `17290203`, plus a Q64 twin (`17291113`)
after the user asked for both codebook sizes; the per-recording preflight
was cancelled unsubmitted-at-user-request (no dial-ranking wanted — 4
priors is the chosen model). `17290203` FAILED in 43 s on gl052: the
sbatch passed `--priors 4` but the script's auto-generated argparse only
exposed `--prototype-priors`. Fixed by adding a `--priors` alias whose
None default is dropped from the Config overrides, and the sbatch now
parameterizes Q via `Q=<n>` (output dirs include the q tag). Resubmitted
Q32 as `17291246`; `17291113` (Q64) was submitted after the sbatch fix
and reads the python file at runtime, so it needs no resubmission.

Run status 2026-09-09 ~21:00: both fits RUNNING concurrently (Q64 on
gl059, Q32 on gl031), writing to separate output trees
(`master_codebook_0021_q{32,64}_p4_shift/`) so nothing can overwrite
anything else; 026's dir is read-only input. Q64 was at iteration 5/10
after 33 min (~6.1 min/iteration), Q32 finished iteration 1 (~2.2
min/iteration); every basis proposal accepted at full step, objectives
decreasing monotonically. Two early observations from the histories.
First, the lag structure is real, not noise: roughly half of the pooled
events pick a nonzero lag (Q32 iteration 1: 115,529 of 202,000 at zero,
Q64 converging to ~48% nonzero), the histogram peaks sharply at zero
with heavy ±1–2 shoulders and a thin tail to ±10, mean |lag| ≈ 0.7–1.1
samples — exactly the onset jitter the fixed-onset Ω was forcing into
gains and σ. Second, the four cones split the pool as the polarity
imbalance predicts: the two negative-polarity cones carry 3–5× the
events of the positive ones, matching the 2.59M/0.64M harvest ratio,
and no cone is starved.

Codebook plot suite `residuals/src/plots/0021_shift_codebook_plots.py`
(2026-09-09, family `residuals/out/0021_shift_codebook/`, hub-registered):
per-cone atom galleries, widths vs 026 reference, lag histograms,
convergence, occupancy. Two findings drove the next step. (1) The lag
structure is real and the fit grows it: 57% zero-lag at iteration 1 → 44%
finally, ±1-sample shoulders ~17% each, mean |τ| ≈ 1.4. (2) The user
spotted the flaw: the learned "slow" atoms don't look slow — several
slow-cone atoms are 0.17–0.3 ms wide, inside the fast regime. Cause: the
cones are 35° wide and the prototypes are re-derived from their own
atoms' SVD each iteration, so atoms (and their prototypes with them)
drift across the init speed split; nothing pins the classes. Fix in
0021: `--freeze-prototypes` (prototypes stay at the init split's group
means; the proposal becomes a pure cone projection of the accumulated
statistics, `frozen_cone_proposal`) plus a tighter
`--prototype-cone-deg 20`. Self-test extended to assert the frozen
prototypes never move and atoms stay within their cones. Queued the
tight variant for both sizes as jobs `17296151` (Q32) and `17296152`
(Q64) → `master_codebook_0021_q{32,64}_p4f20_shift/`; the plot suite
takes `--tag` so the tight codebooks can be plotted and compared against
the loose ones (`p4_shift`) when they land.

Update (2026-09-09, later): all four codebook jobs completed. The loose
pair (`17291246` Q32 24:49, `17291113` Q64 1:04:05) and the frozen-prototype
tight pair (`17296151` Q32 24:55, `17296152` Q64 1:03:33) each ran the full
10 iterations with the basis accepted at full step every time, objectives
monotone down, and no dead atoms (tight-Q32 minimum row count 1749). All
four output dirs hold complete `omega.npy`/`prototypes.npy` artifacts
(`master_codebook_0021_q{32,64}_{p4f20,p4}_shift/`). Next: run
`0021_shift_codebook_plots.py` with `--tag p4f20_shift` on the tight pair
and check the slow-cone atoms are actually slow now — that was the whole
point of freezing the prototypes.

Update (2026-09-09, night): the tight-pair plot suite ran inline (six
`*_p4f20_shift.png` figures in `residuals/out/0021_shift_codebook/`, hub
index refreshed) and the frozen-prototype fix is confirmed working. Atom
widths now show a clean fast/slow separation with no cross-contamination:
at Q32 the fast cones span 0.13–0.43 ms and the slow cones 0.58–1.3 ms
(Q64 fast 0.13–0.43, slow 0.46–1.3), whereas the loose fit had slow-cone
atoms down at 0.17–0.3 ms inside the fast regime. The shift structure
survived the freeze unchanged in shape — 52–53% of events at zero lag,
mean |τ| ≈ 0.8 samples, sharp zero peak with ±1–2 shoulders and a thin
tail — just slightly tighter than the loose fit's 44% zero / mean 1.4,
as expected from pinning the prototypes at the init split. Both sizes
converged with no dead atoms, so `p4f20_shift` is the current candidate
codebook family for the 0021 peeling runs.

## Links

- [[session-019-all-channel-error]]
- [[session-020-rank2-temporal-peeling]]
- [[session-018-bipolar-prototype-cone-peeling]]
- [[session-016-one-hot-lattice-peeling]]
- [[feedback_plot_suite_completeness]]
