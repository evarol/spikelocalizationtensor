# Acceptance-Gate Census, Gate Trim, and the Q32 Bar Sweep (027)
**Created:** 2026-09-05
**Last updated:** 2026-09-05
**Status:** gate trim implemented and validated; census recorded in the math doc; Q32 f₀ sweep 14/20 tasks COMPLETE with per-family totals recorded (2026-09-08) — the min and mean families are fully done, mean f₀ 0.05 at 2.27M events is the largest run in the project. Eight tasks were policy-killed, not two: tasks 10, 11, 12, 14, 15, 17 (kofn 05/10/15/30, flat 05/15) died at ~2h15–2h30 on the pre-fix script and were never resubmitted — they need `--resume` resubmission. Pending: those six tasks, plot-suite decision, the sigma-mix/near-surface quality audit, and possibly the alternating-codebook refit.

## Why this card exists

The acceptance stack had accreted one gate per past fix (0012's quality
gates, 0014's projection floor, 0016's duplicate gate, 0019's all-channel
bar), and the user suspected some of it was dead weight. This card records
the measurement that settled the question, the trim that followed, the
mathematical reference doc written alongside it, and the new sweep that the
trimmed gate stack makes possible.

## The census: which gates actually decide anything

The consolidated rejection tables in every 0019 run record, per rejected
proposal, the full reason bitmask. Counting proposals that fail **only**
one gate — exactly the proposals that would flip to accepted if that gate
were removed — over the six Q=8 runs (~27M logged proposals) gives the
answer directly. Four of the seven gates are inert on this data:

- **gain (bit 1)**: never fires at all. The closed-form NNLS on
  threshold-detected peaks never returns a non-positive gain.
- **captured fraction (bit 4)**: never fires. It turns out to be disabled
  anyway — the runs inherit `min_captured_fraction = 0.0` from 0014's
  Config override, not 0012's 0.05 (this corrected an error in the math
  doc, which had claimed 0.05).
- **energy drop (bit 32)**: fired 4–6 times in ~27M proposals, never as
  the sole cause.
- **projection ≥ 8 (bit 8)**: fires 32–42k times but always alongside
  another failure (zero flip counts). It sits in a valley of the projection
  distribution — the weakest accepted events clear it by +4.2 to +8.5 — so
  at threshold 8 it is empirically inert, though re-tuning it above the
  valley (≥ ~12) would re-activate it.

The stack that actually decides is three gates: the **all-channel bar**
(27–80% of proposals, sole cause), the **duplicate gate** (14–62%), and
**rmse ≤ 3** (a minor trimmer at 0.4–2.5%). The multi-gate view: only
6–8% of rejected proposals fail two or more gates, with the top pairs
being rmse+allchan (2.8–4.1%), rmse+dup (up to 4.1% on mean20),
proj+allchan (0.7–0.9%), and allchan+dup (0.5–2%). Gate interactions are
minor, so removing the inert four changes essentially nothing about which
proposals the remaining gates see.

## The trim

`residuals/src/preprocessing/0019_allchannel_peeling.py`: the acceptance
conjunction in `process_chunk` is now exactly max-channel-RMSE ≤ 3, the
all-channel bar, and the duplicate test. The gain, captured-fraction,
projection, and raw-energy-drop conditions are gone from `accepted` and
their reason bits (1, 4, 8, 32) are gone from the audit bitmask — the
bitmask is now {2 rmse, 16 all-channel, 64 duplicate, 128 rollback}. The
metrics themselves (alpha, captured_fraction, fitted_projection_score,
raw_energy_drop, …) are still computed and saved per event and per
rejected proposal as diagnostics, so any future re-activation or census
stays possible. `output_metadata` records the trim with a pointer to the
census. The config keys (`min_captured_fraction`, `min_fitted_projection`,
`min_raw_energy_drop`) still exist and the sbatches still pass them — they
are accepted but do nothing, which keeps configs comparable with the
completed sweeps. py_compile and the CPU self-test pass. The trim is
provably a no-op on the six census runs (flip counts 0–2 per run), so old
and new runs are directly comparable; the user chose to skip requeueing
the old matrix on that basis.

## The math doc

`docs/0019_acceptance_mathematics.md` (new) formalizes the decision layer:
per-channel bookkeeping (the improvement is a signed quadratic in the
amplitude ratio, `f_c = (2β−1)/β²`, so amplitude underestimates are
punished far more than overestimates; the event-level captured fraction is
exactly the energy-weighted mean of the per-channel fractions), the gate
table, the three all-channel rules as aggregation functionals with their
exact nesting/incomparability relations, the escalation schedule, the
duplicate-wall arithmetic (a leftover re-detects iff
`f_c ≤ 1 − 25/A²`, so most accepted events leave ≥5σ leftovers), and §8's
design space for new criteria (power means `M_p` from min to mean to max,
fractional-ρ quantile rules, energy-weighted `F_β`, lower-tail and clipped
means). §3.1 holds the census. §8.5 notes the one cheap save that would
make every future rule evaluable offline: storing per-channel
`channel_input` next to `channel_improvement`.

## Provenance note

The coarse-assignment optimizer (closed-form scalar-gain NNLS over
(site, σ, q) with channel-SSE objectives) was introduced in
`residuals_0012.py` (commit `7e46ccd`, 2026-08-27), not ported from
`spiketensor/fit_lattice.py`, whose optimizer is a Rayleigh-quotient
assignment with free per-spike coefficient vectors and weighted-PCA refit.
Per-event, the two agree under a one-hot constraint given the same frozen
codebook and noise metric; the learned codebooks differ because the refit
blocks and data differ. Alternating codebook refit during peeling (per-pass
Ω updates) was designed in conversation — refit between recording passes,
duplicate gate re-tested against stored prediction windows instead of atom
indices, exhaustion reset per refit, NMSE-guarded updates — but is NOT
implemented yet.

## The Q32 bar sweep (2026-09-05)

Array job `16989541` (20 tasks, l40s_public, `torch_pr_60_general`,
48G mem, USR1-requeue trap): {min, mean, kofn, flat} × f₀ ∈ {0.05, 0.10,
0.15, 0.20, 0.30} at Q = 32 under the trimmed gates. The min-channel
family at f₀ = 0.20 with step 0.1 is the trimmed-era "base"; min at the
other bars is the "frac" family; flat runs step 0. Everything else matches
the completed Q-sweep (threshold 5, projection flag 8 — now an inactive
gate — 3 passes × 1 round, mean-channel-rmse objective, event-merge 0.5
ms, RMSE 3). Run dirs:
`residuals/runs/dataset1_p1/0019_allchannel_trimmed_<rule>_f0<tag>_q32`
(`<tag>` = 05/10/15/20/30). Sbatch:
`residuals/src/preprocessing/0019_allchannel_trimmed_f0_sweep.sbatch`
(one file, array index → rule/f₀ mapping). Task 0 (min, f₀ 0.05) started
immediately; tasks 1–19 queued behind the 024 convolving runs and
`16986952` (026 master codebook). Plot suites deliberately not queued —
decide after the event counts land, same as the 023 sweep.

## Policy kills and the TERM-trap fix (2026-09-05)

About two hours into the sweep, the <50% GPU-utilization policy started
killing tasks. Tasks 6 and 7 (min f₀ = 0.15/0.30) died at ~2h04 mid-pass-1
with SIGTERM, and the cause was a known trap gap: the utilization policy
sends TERM, the 025 session had already learned to trap `USR1 TERM` for
exactly this, but the trimsweep sbatch was copied from the older q_sweep
pattern that only traps USR1 — so those two died instead of requeueing.
The sbatch now traps `USR1 TERM`, and the killed tasks were resubmitted as
`16991857` (they `--resume` from their consolidated pass 0 and redo only
partial pass-1 work).

The low utilization itself is the duplicate wall, not a malfunction: pass
1 re-detects thousands of leftover proposals per chunk (task 0's log shows
~1,600 proposed and ~840 duplicate-rejected per chunk), so most time goes
to CPU-side rejection logging, npz writes, and FUSE reads between short
GPU bursts, and per-job GPU utilization sags below the 50% bar during
passes 1–2. Nothing about the runs is unhealthy — min f₀ = 0.05 (task 0)
was at pass 1, chunk 1580/1958 when checked. Residual risk: the tasks that
were already running when the trap was fixed keep the old script in
memory, so any that get killed later need a manual resubmission, which is
safe and cheap because every run resumes from its last consolidated pass.

## Sweep state on 2026-09-08

Fourteen of the twenty tasks are COMPLETE, all exit 0. The task→config
mapping (confirmed from the sbatch arrays) is rule blocks of five in order
min, mean, kofn, flat — so the card's earlier "tasks 6–7 (min f₀ 0.15/0.30)"
was mislabeled: 6–7 are mean f₀ 0.15/0.20. Totals so far, n_events
(`stopping_reason: all_passes_complete` on every finished run):

| rule \ f₀ | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| min | 1,256,018 | 1,052,923 | 863,953 | 694,578 | 419,167 |
| mean | 2,271,336 | 2,182,316 | 2,049,274 | 1,899,946 | 1,519,955 |
| kofn | — | — | — | 1,431,736 | — |
| flat | — | 1,067,203 | — | 704,695 | 424,346 |

The min f₀ 0.20 run (694,578) matches the untrimmed base Q32 (694,753) to
within 175 events, empirically re-confirming the trim's no-op. Everything
else lands where the Q8 sweeps predict: monotone decline in f₀ within each
rule, mean > kofn > flat ≈ min, and Q32 inflating its Q8 twin by ~10–18%
(min f₀ 0.05: 1.26M vs the Q8 5% bar's 1.11M; mean f₀ 0.20: 1.90M vs
1.72M). Mean f₀ 0.05 (2.27M) is the largest run in the project and the
presumed worst case for the pending sigma-mix audit.

The policy kill tally is eight tasks, not two. Beyond the known 6–7 (which
completed on the resubmission `16991857`), tasks 10, 11, 12, 14, 15, 17 —
kofn 05/10/15/30 and flat 05/15 — were SIGTERM-killed at ~2h15–2h30. They
were queued before the TERM-trap fix landed, so they ran the old script and
died instead of requeueing. Nothing was resubmitted for them; their run
dirs hold consolidated pass-0 output and resume cleanly. Six `--resume`
resubmissions (via the same sweep sbatch with the fixed trap) would finish
the matrix. **Done 2026-09-08:** the six tasks were resubmitted as array
`17221954` (tasks 10, 11, 12, 14, 15, 17 through the fixed sbatch; pending
on `QOSGrpGRES` at submission, queued behind 024's perchannel5-Q64).

A parameterized plot suite for the trimmed runs is now queued:
`residuals/src/plots/0019_allchannel_trimmed_f0_sweep_plots.sbatch` — an
array job with the identical task→(rule, f₀) mapping, the same 14-figure
body as the q-sweep suite (it computes each run's own most-subtractive
chunk and renders the full-recording replay inline), run guard on
`summary.json`, 64G on cpu_short. Submitted for the fourteen completed
tasks as array `17222019` (`--array=0-9,13,16,18,19`); the six pending
runs' suites should be submitted with the same array indices once their
runs land.

## Next steps

- [ ] When the six resubmitted tasks (`17221954`) land: record the missing
      cells in the totals table, then submit their plot suites
      (`--array=10,11,12,14,15,17` on the trimsweep plots sbatch).
- [ ] When `17222019` lands: verify all 14 galleries carry the full
      14-figure set.
- [ ] Sigma-mix / near-surface quality audit across the sweep (the open
      item inherited from 019/023), worst case expected at mean f₀ = 0.05
      (softest rule × loosest bar).
- [ ] Optional: implement the per-pass alternating codebook refit (design
      in the provenance note above); optional: save `channel_input` per
      event to enable offline rule evaluation (math doc §8.5).

## Links

- [[session-019-all-channel-error]]
- [[session-023-acceptance-rule-variants]]
- [[session-024-convolving-detection-peeling]]
- [[session-028-neural-components-plan]] — the census is phase 1's training data
- `docs/0019_acceptance_mathematics.md` — the math reference with the census (§3.1)
