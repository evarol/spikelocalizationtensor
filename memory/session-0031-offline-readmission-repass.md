# Offline Re-Admission and the Bar-Repass (0031)
**Created:** 2026-09-09
**Last updated:** 2026-09-09
**Status:** ACTIVE — pass-0 re-admission implemented and VALIDATED on the
20% min-rule run (0019's original default bar, not a decided production recipe; method confirmed exact to ~0.1%); the 20% run
was the wrong demonstration target. **The next agent must run it on
`0019_allchannel_trimmed_mean_f005_q32`** — the Kilosort-closest run (0.94
good-time coverage, 2.27M events). Field caveats for that run are in
"Target correction" below. GPU repass (replay-seeded) designed but not
built.

## Target correction (2026-09-09, user-directed)

**The demonstration run for this method is the mean f₀ 0.05 run, not the
20% min-rule run.** Next agent: run
`kilosort_overlap_census/readmit_pass0.py` pointed at
`residuals/runs/dataset1_p1/0019_allchannel_trimmed_mean_f005_q32`
(singularity pytorch overlay, CPU is fine).

Field-availability facts verified in
`0019_allchannel_peeling.py` (`all_channel_acceptance`, lines ~355–403,
and the save sites ~1060–1160) before running:

- The mean rule's deciding variable — the **simple mean of per-channel
  captured fractions** — is computed inside `all_channel_acceptance()` but
  **never saved**, not for accepted events and not for rejections. The
  saved `min_channel_captured_fraction` is the worst-channel value
  (returned as the diagnostic for every rule), and `rejected_captured_
  fraction` / `captured_fraction` is the **energy-weighted** mean of the
  per-channel fractions (fit-level, different aggregate). The
  `all_channel_fraction` column is just the per-pass bar repeated
  (`np.full`), not per-event data. So exact mean-rule re-admission from
  saved tables is impossible for this run; the §8.5 per-channel save
  (`channel_input` next to `channel_improvement`) is what would make it
  exact in future runs.
- Workable approach for the mean run: re-admit its pass-0 rejections with
  `rejected_captured_fraction ≥ bar` (energy-weighted proxy) and rmse ≤ 3,
  clearly labeled as a proxy for the mean rule, and report the bar sweep
  0.05/0.025/0.01 alongside. Also report the expectation up front: this
  run's pass-0 rejections number only 65,202 (13.9% bar-fails ≈ 9k, 86%
  rmse-fails), so re-admission gains are small by construction — the mean
  run's binding constraint is the pass-1+ duplicate wall (4.46M rejections,
  96% dup-flagged), which offline re-admission cannot touch. The honest
  framing: the mean f005 run mostly needs either the §8.5 save (for exact
  offline re-evaluation in future runs) or the GPU repass (to attack the
  duplicate wall), and the re-admission demo on it is mostly a
  rule-adapter exercise.

## The idea

The kilosort overlap census (in [[session-022-kilosort-baseline]]) proved the
under-detection is purely acceptance-side: the detector nominates ~95–100%
of Kilosort-good spike times, the bar rejects 99.3% of those proposals at
Kilosort-active times, and lowering the bar recovers them (mean f₀ 0.05 →
0.94 coverage). Re-running the f₀ sweep for every new bar costs GPU hours,
but the rejected proposals already store the exact numbers the gate
consumed (`rejected_min_channel_fraction`, `rejected_max_channel_rmse`,
reason bits, pass, times/channels). So a lower bar can be *re-evaluated
offline* on an existing run's rejections — no re-detection, no re-fitting.

## What is exact and what is stale

- **Exact (pass 0, min-rule runs):** the fit does not depend on the bar, so
  re-admitting pass-0 rejections with
  `min_channel_fraction ≥ b' AND rmse ≤ 3 AND ¬(dup|rollback)` is an exact
  evaluation of what bar `b'` would have accepted *for that run's own
  codebook and detection*. Pass-0 proposals are mutually exclusive
  (1 ms sweep), so re-admission creates no internal duplicates, and pass 0
  had zero rollbacks/duplicates in these runs.
- **Stale (pass 1+):** later-pass proposals were detected on a residual
  shaped by the earlier passes' subtractions. Under a lower pass-0 bar that
  residual would differ, so pass-1+ rejections cannot be re-evaluated
  offline. Duplicate-flagged rejections (bit 64) must stay rejected.
- **Subtraction gap:** re-admitted spikes were never subtracted in the
  original run, so the updated set is a *candidate* set. Re-admitted
  pass-0 events can overlap later-pass accepted events (double-subtraction
  risk); the overlap is quantified per run.

## The two products

1. **Offline re-admission (this card, working):** a CPU script that takes a
   run's consolidated tables + a target bar and outputs the updated event
   set (original accepted ∪ re-admitted pass-0), its Kilosort coverage, and
   the overlap audit. Extends the f₀ dial to bars the sweep never ran.
2. **GPU repass (designed):** build a new run directory whose consolidated
   pass-0 events are the expanded set (original pass-0 accepted ∪
   re-admitted, replayable in the standard layout), then let 0019's
   existing GPU-replay machinery rebuild the residual from it and run fresh
   passes at the new bars. Needs a small "seed-from-old-run" entry point;
   everything else (replay, resume, exhaustion, audit) already exists.

## First run: the 20% min-rule run (2026-09-09) — VALIDATED

Run: `0019_allchannel_pass3_round1_fraction20_step10_fitted8` (568,889
accepted, 4.39M rejections, Q8). Bars re-evaluated: 0.15 / 0.10 / 0.05.
Outputs: `residuals/runs/dataset1_p1/readmission/0019_allchannel_pass3_
round1_fraction20_step10_fitted8_pass0/` (`readmission_bar{15,10,05}.npz`
+ `summary.json`); script
`residuals/runs/dataset1_p1/kilosort_overlap_census/readmit_pass0.py`
(run dir, bars CSV, out dir as args).

Results, with the real lower-bar runs as approximate twins (per-run
codebooks, so exact equality is not expected):

| bar | re-admitted | updated total | twin run's pass 0 | coverage ours / twin |
|---|---|---|---|---|
| 0.15 | 153,511 | 722,400 | (no 15% run) | 0.562 / — |
| 0.10 | 332,395 | 901,284 | 899,862 | **0.6406 / 0.641** |
| 0.05 | 536,563 | 1,105,452 | 1,103,359 | **0.7114 / 0.712** |

The offline re-admission reproduces the real 10% and 5% runs to ~0.1% in
both event counts and Kilosort-good coverage. The method is effectively
exact for pass 0. Re-admitted spikes match KS-active times at 99.7% and
KS-good at ~78%. Overlap between re-admitted events and this run's
later-pass accepted events is negligible (95–169 events — later passes
accepted only 453 here); runs with bigger late-pass sets (024's ~18k)
would show more, and the npz flags keep them reconcilable.

## Next steps

- [x] Run the pass-0 re-admission on the 20% min-rule run; validate
      against the 10%/5% twins — done, reproduced to ~0.1% (table above),
      but see Target correction: this was the wrong demonstration run.
- [ ] **Next agent:** run the re-admission on
      `0019_allchannel_trimmed_mean_f005_q32` (the Kilosort-closest run) at
      bars 0.05/0.025/0.01 with the energy-weighted proxy gate, labeled as
      approximate; expect small gains (its pass-0 bar rejections ≈ 9k).
- [ ] Extend to the trimmed min/flat Q32 runs (exact — they save the right
      field) and bars below 0.05.
- [ ] Build the GPU repass: seed a fresh run dir from the expanded pass-0
      set, replay, run passes 1–2 at the new bar schedule.
- [ ] Decide the production recipe from the sweep + census (note: 20% was never a decided recipe, only 0019's founding default): the mean rule at
      f₀ 0.05 (0.94 coverage) vs re-admission paths on min-rule runs.

## Links

- [[session-022-kilosort-baseline]] — the overlap and rejection censuses motivating this
- [[session-027-acceptance-gate-census-trim]] — the trimmed gate stack the re-admission replicates
- [[session-023-acceptance-rule-variants]] — the mean/k-of-n rules and their gate variable
- [[session-028-neural-components-plan]] — phase 1 trains on exactly this re-evaluation set
- [[session-019-all-channel-error]] — the rejection audit tables being re-read
