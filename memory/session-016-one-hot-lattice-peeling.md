# One-Hot Lattice Peeling (0016)
**Created:** 2026-08-28
**Last updated:** 2026-08-29
**Status:** Done — full learned-Omega run and corrected plot suite complete; results await review

## Why

0014's score-8 run accepted 37,791,971 events from the full recording —
roughly an order of magnitude above the expected scale. That is model
over-detection, not a threshold-tuning problem. 0016 replaces the 0014
detection path with a design kept as close as practical to SpikeTensor's
lattice fit while retaining a hard one-hot temporal code per event, so that
detection, localization, subtraction, and rescoring form one coherent
residual-pursuit model: find the codebook atom and source location that best
explain a spike, subtract it, and repeat until what remains looks like noise.

## Locked design decisions

- Hard one-hot temporal assignment: exactly one Omega row per event; no
  SpikeTensor-style temporal mixture coefficients.
- Per-channel reconstruction error, not total SSE — total SSE can tolerate a
  great fit on the peak channel while other channels stay bad, which is
  exactly the observed low-sigma / low-z failure mode.
- Spatial fitting neighborhood fixed at exactly 48 µm.
- One scalar event gain, closed form, for every candidate atom.
- Integer coarse-to-fine xyz localization: 16³ coarse search over the 301³
  integer lattice plus discrete refinement; no continuous optimization (an
  optional terminal 1×1×1 µm refinement may be exposed as a flag later).
- Omega may stay frozen during pursuit, with a path to initialize it from a
  SpikeTensor-derived prior rather than an online update.
- No whitening.
- SpikeInterface detections are a competing detection model, not ground
  truth — useful for architecture and count-scale context only.
- No scientific verdicts from 5–10-second prefixes; the early recording is
  not representative. Compare plausible methods on the complete recording,
  even when that costs substantial GPU time.
- New work only in `0016_*.py` / `0016_*.sbatch`; prior implementations and
  job scripts untouched.

## Diagnosis carried into the design

- The within-pass step was local-max suppression, not a real event merge
  after fitted locations and temporal supports are known.
- A large batch was fit against one stale residual and subtracted
  afterwards, so nearby proposals could claim the same waveform before
  either subtraction was visible to the other.
- 0014's grouped coarse assignment used an aggregate captured-energy score
  while discrete refinement used per-channel SSE — coarse and fine searches
  optimized different objectives.
- Permanent cross-pass exclusion blocked genuine collisions from re-emerging
  at the same place after the first waveform was subtracted.
- The template-bank score had no explicit no-event decision calibrated after
  maximization over time, channel, temporal code, spatial scale, and
  location.

## What 0016 implements

Implemented in `residuals/src/preprocessing/0016_onehot_lattice_peeling.py`,
with full real and empirical-null job scripts plus `0016_compare_real_null.py`:

1. One coherent per-channel minimax objective for coarse assignment and every
   discrete refinement level; closed-form positive one-hot gains.
2. Collapse of codebook/channel hypotheses to one winning proposal over each
   full waveform-support interval before fitting — merging matching peaks
   across nearby channels and preventing a stale batch from repeatedly
   claiming one waveform.
3. Subtract the conflict-free winners, then recompute all proposal scores
   from the new residual; no permanent time/location lockout across rounds.
4. Acceptance requires positive marginal residual-energy reduction and the
   per-channel fit criterion on the current residual.
5. Optional external frozen Omega (SpikeTensor-compatible), with the
   calibration route retained as an alternative.
6. Per-round diagnostics: proposed, fitted, accepted, marginal energy
   removed, score distributions, stopping reason.

## Promotion conditions

Coarse and fine localization share one auditable objective; accepted events
are conflict-free within the active round; every subtraction improves
marginal energy on the residual its amplitude was estimated from; peeling
stops because nothing passes the model/noise decision, not because a round
cap ran out; and count scale, per-round decay, residual energy, per-channel
error, and spatial-boundary occupancy are plausible — i.e. nothing like the
37.8M-event pathology.

## Full run and stopping audit (2026-08-29)

Full learned-Omega job `16541332` completed all 1,958 one-second chunks in
2h36m06s with exit `0:0`, saving 3,115,098 events (1,591.6 events/s) to
`residuals/runs/dataset1_p1/0016_onehot_lattice_learned_score8/` — 91.8%
fewer than 0014's 37.8M, a 12.1× reduction. Every saved event passed the
numerical gates: fitted projection ≥ 8.00048, max channel-normalized RMSE ≤
2.99999, at least two improving channels, strictly positive raw marginal
energy drop. Round counts decayed 295,027 → 124,020 (r10) → 37,603 (r20) →
1,740 (r40) → 125 (r59). The sigma=2 lower boundary still held 902,898
events (29.0%); sigma=512 held 26,248 (0.84%).

The stopping audit found no chunk that ran out of proposals. In 1,896 chunks
the terminal round still had events passing all fit gates, but the
accumulated cross-round duplicate mask rejected every one
(`no_accepted_events`); the other 62 chunks hit the 60-round cap with events
still accepted in round 59. Across all rounds, 4,912,187 fits passed before
merging and 1,794,849 were rejected as duplicates. The mechanism is explicit
in `process_chunk`: accepted times, channels, and temporal codes accumulate
in `prior_*` across peeling rounds, and `duplicate_mask` permanently
excludes later nearby events with a sufficiently similar Omega row — which
contradicts the design intent of letting a genuine collision re-emerge after
subtraction. Whether it materially harms the results is a question for the
plots, not something to conclude now.

Also still open: the matched empirical-null scripts exist but no null job
ever ran, and the external-Omega-prior ablation hasn't run. All four
`0016_*` implementation files and this card were untracked in Git at the
time.

## Next steps

- [ ] Review corrected plot job `16569753` under
      `residuals/out/0016_onehot_lattice_learned_score8/` (3m37s, exit
      `0:0`): full XYZ and XYZ-sigma panels, reconstruction examples across
      rounds and near the score boundary, SpikeTensor-style panels, and
      peeling/stopping diagnostics. (First attempt `16563803` produced only
      four panels — missing legacy metadata crashed two plot commands while
      the inner shell masked their errors.)
- [ ] Decide from the results whether cross-round duplicate suppression
      should be removed, retained, or revised.
- [ ] If the model changes: rerun the complete-recording learned-Omega job
      and re-audit its stopping behavior.
- [ ] Run the matched empirical null and the `0016_compare_real_null.py`
      threshold comparison.
- [ ] Run the external SpikeTensor-compatible Omega-prior ablation when the
      prior is available.
- [ ] Re-audit count scale, round decay, residual energy, per-channel error,
      and boundary occupancy before any promotion.

No model-quality, promotion, redesign, or rerun conclusion until the
complete-run plots have been reviewed.

## Links

- [[session-018-bipolar-prototype-cone-peeling]]
- [[session-019-all-channel-error]]
- [[session-017-initial-threshold-spike-discovery]]
- [[session-015-score-calibrated-xyzsigma-promotion]]
- [[session-014-xyzsigma-residual-pursuit]]
- [[session-009-ibl-style-pursuit]]
