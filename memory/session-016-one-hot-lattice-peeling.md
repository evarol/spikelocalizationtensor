# Session 016: One-Hot Lattice Peeling
**Created:** 2026-08-28
**Last updated:** 2026-08-29
**Status:** Active

## Objective

Replace the 0014 detection path with a new 0016 implementation whose core is
as close as practical to SpikeTensor's lattice fit while retaining a hard
one-hot temporal code per event. Detection, localization, subtraction, and
rescoring must form one coherent residual-pursuit model: find the codebook atom
and source location that best explain a spike, subtract it, and repeat until
the remaining matched structure is consistent with noise.

The 0014 score-8 result accepted 37,791,971 events from the full recording,
roughly an order of magnitude above the expected scale. Treat this as model
over-detection, not as a threshold-only problem.

## Locked design decisions

- Keep a hard one-hot temporal assignment. For each event, choose exactly one
  row of `Omega`; do not introduce SpikeTensor's temporal mixture coefficients.
- Keep the per-channel reconstruction-error route. A total-SSE objective can
  tolerate a good fit on the peak channel while leaving large errors on the
  other channels, which is the observed low-`sigma`, low-`z` failure mode.
- Keep the spatial fitting neighborhood at exactly 48 um by design.
- Fit one scalar event gain in closed form for every candidate atom.
- Keep integer-index coarse-to-fine xyz localization: a 16^3 coarse search over
  the 301^3 integer lattice followed by discrete refinement. Do not introduce
  continuous optimization. A separate terminal 1 x 1 x 1 um refinement may be
  exposed as an optional flag later.
- It is acceptable for `Omega` to remain frozen during pursuit. Add a path for
  initializing it from a SpikeTensor-derived prior rather than requiring an
  online update.
- Do not add whitening.
- Do not use SpikeInterface detections as truth. They are a competing detection
  model, useful for architecture and count-scale context but not labels.
- Do not evaluate scientific changes on 5- or 10-second prefixes. The early
  recording is not representative. Compare plausible methods on the complete
  recording, even when that consumes substantial GPU time.
- Implement the redesign only in new `0016_*.py` and `0016_*.sbatch` files;
  leave prior implementations and job scripts unchanged.

## Diagnosis carried forward

- The current within-pass operation is local-max suppression, not a complete
  event merge after fitted locations and temporal supports are known.
- The current code fits a large batch against one stale residual and subtracts
  the whole batch afterward. Nearby proposals can therefore claim the same
  waveform before either subtraction is visible to the other.
- The 0014 grouped coarse assignment uses an aggregate captured-energy score,
  while discrete refinement uses the configured per-channel SSE score. The
  coarse and fine searches are optimizing different objectives.
- Permanent cross-pass exclusion prevents duplicate rediscovery but also blocks
  a genuine collision from emerging at the same place after the first waveform
  has been subtracted.
- The template-bank score has no explicit no-event decision calibrated after
  maximization over time, channel, temporal code, spatial scale, and location.

## 0016 implementation target

1. Use the same per-channel objective for coarse xyz assignment and every
   discrete refinement level. Preserve closed-form event gain estimation.
2. Collapse codebook/channel hypotheses to one winning proposal over each full
   waveform-support interval before fitting. This merges matching peaks across
   nearby channels and prevents a stale batch from repeatedly claiming one
   waveform.
3. Subtract the conflict-free winners, then recompute all proposal scores from
   the new residual. Do not permanently lock the same time/location across
   peeling rounds; residual rescoring must reveal real collisions.
4. Require every accepted event to produce a positive marginal residual-energy
   reduction and pass the per-channel fit criterion on the current residual.
5. Support a frozen `Omega` loaded from an external SpikeTensor-compatible
   array, while retaining the existing calibration route as an alternative.
6. Save enough per-round diagnostics to explain the final count: proposed,
   fitted, accepted, marginal energy removed, score distributions, and stopping
   reason.
7. Run full-recording ablations only. At minimum compare the 0014 behavior with
   full-support merging plus residual rescoring, and with the external `Omega`
   prior when available. Judge methods using residual behavior and noise-null
   calibration, not agreement with SpikeInterface detections.

## Promotion conditions

- Coarse and fine localization use one auditable per-channel objective.
- Accepted events are conflict-free within the active peeling round.
- Every subtraction has positive marginal energy improvement on the residual
  from which its amplitude was estimated.
- Peeling stops because no candidate passes the model/noise decision, not only
  because a small fixed pass count was exhausted.
- The complete-recording event count, events per peeling round, residual energy,
  per-channel error, and fitted spatial-boundary occupancy are plausible and do
  not reproduce the 37.8-million-event pathology.

## Full learned-Omega run and stopping audit (2026-08-29)

- Implemented the redesign in
  `residuals/src/preprocessing/0016_onehot_lattice_peeling.py`, with full real
  and empirical-null job scripts plus `0016_compare_real_null.py`. The new path
  uses one coherent per-channel minimax objective for coarse and refined
  lattice fitting, positive one-hot gains, full-support temporal proposal
  collapse, residual rescoring, per-channel acceptance gates, an optional
  external Omega prior, and per-round diagnostics.
- Full learned-Omega job `16541332` (`onehot0016-full`) completed all 1,958
  one-second chunks in 2h36m06s with exit `0:0`. It saved 3,115,098 events to
  `residuals/runs/dataset1_p1/0016_onehot_lattice_learned_score8/`, or
  1,591.6 events/s. This is 91.8% fewer events than 0014's 37,791,971-event
  score-8 run, a 12.1x count reduction.
- Every saved event passed the configured numerical gates: fitted projection
  score was at least 8.00048, maximum channel-normalized RMSE was at most
  2.99999, at least two channels improved, and raw marginal energy drop was
  strictly positive. Counts decayed from 295,027 events in peeling round 0 to
  124,020 in round 10, 37,603 in round 20, 1,740 in round 40, and 125 in round
  59. The sigma=2 lower boundary held 902,898 events (29.0%); sigma=512 held
  26,248 (0.84%).
- The stopping audit found that no chunk stopped because there were no
  proposals. In 1,896 chunks, the terminal round still had events that passed
  all fit gates, but the accumulated cross-round duplicate mask rejected every
  one and produced `no_accepted_events`. The other 62 chunks still accepted
  events in round 59 and stopped at the 60-round cap. Across all rounds,
  4,912,187 fits passed before merging and 1,794,849 were rejected as
  duplicates. The scientific meaning of this pattern is pending visual review.
- The cause is explicit in `process_chunk`: accepted times, channels, and
  temporal codes accumulate in `prior_*` across peeling rounds, and
  `duplicate_mask` permanently excludes later nearby events with a sufficiently
  similar Omega row. This differs from the stated target of allowing a genuine
  collision to re-emerge at the same place after subtraction, but whether it
  materially harms these results must be judged after inspecting the event,
  reconstruction, localization, and round-diagnostic plots. No promotion or
  redesign conclusion has been made.
- The matched empirical-null implementation and sbatch file exist, but no null
  job or output was found. The external-Omega-prior ablation has also not run.
  All four `0016_*.py`/`0016_*.sbatch` implementation files and this session
  card are currently untracked in Git.

## Next steps

- [ ] Scientifically review corrected plot job `16569753` under
  `residuals/out/0016_onehot_lattice_learned_score8/`. It completed in 3m37s
  with exit `0:0` and produced full XYZ and XYZ-sigma localization panels,
  reconstruction examples across rounds and near the score boundary,
  SpikeTensor-style panels, and peeling/stopping diagnostics. Initial job
  `16563803` produced only four panels because missing legacy metadata caused
  two plotting commands to fail while the inner shell masked their errors.
- [ ] Decide from the results whether cross-round duplicate suppression should
  be removed, retained, or revised.
- [ ] If the model changes, repeat the complete-recording learned-Omega real
  run and re-audit its stopping behavior.
- [ ] Run the matched complete-recording empirical null and generate the
  threshold comparison with `0016_compare_real_null.py`.
- [ ] Run the external SpikeTensor-compatible Omega-prior ablation when the
  prior is available.
- [ ] Re-audit count scale, round decay, residual energy, per-channel error,
  and spatial-boundary occupancy before promotion.

No model-quality, promotion, redesign, or rerun conclusion should be made until
the complete-run plots have been reviewed by the user.

## Links

- [[session-015-score-calibrated-xyzsigma-promotion]]
- [[session-014-xyzsigma-residual-pursuit]]
- [[session-009-ibl-style-pursuit]]
