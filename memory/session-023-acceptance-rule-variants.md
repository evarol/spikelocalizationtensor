# Acceptance-Rule Variants (0023)
**Created:** 2026-09-01
**Last updated:** 2026-09-01
**Created:** 2026-09-01
**Last updated:** 2026-09-02
**Status:** COMPLETE on both dials — the four acceptance-rule runs and the full 5×3 codebook sweep (15 runs) all ended `all_passes_complete`, all 15 galleries built with the full panel set. Totals across Q=8→64: base 569k→743k, flat10 906k→1.12M, step20 900k→1.09M, mean20 1.72M→1.96M, kofn20 1.25M→1.49M. More atoms add events monotonically at every rule; rule ordering (mean > kofn > flat ≈ step > base) is invariant to Q; escalation still weak. Pending: sigma-mix/near-surface quality audit on the big mean20 runs.

## Why 0023 exists

0019's fraction sweep moved only the bar height while the aggregation rule
stayed fixed at "the worst valid channel decides". The known failure mode is
the duplicate wall: passes 1–2 accept almost nothing because replayed
earlier-pass fits leave ≥5σ leftovers that re-detect and die as duplicates.
The user asked for runs with different acceptance criteria to see how the
aggregation rule and the escalation schedule interact with that wall, and
picked all four variants below. This card records that work so
[[session-019-all-channel-error]] stops growing; it is a direct extension of
0019 and inherits its machinery (GPU replay, chunk exhaustion, rejection
audit, consolidated tables).

## What changed in the code

`residuals/src/preprocessing/0019_allchannel_peeling.py`:

- New config fields: `all_channel_rule` ("min-channel" | "mean-channel" |
  "k-of-n", default min-channel) and `all_channel_required_share` (default
  0.875, used only by k-of-n). Both auto-expose as CLI flags
  (`--all-channel-rule`, `--all-channel-required-share`) through 0016's
  dataclass-driven argparse.
- The per-channel gate now lives in one function, `all_channel_acceptance()`,
  which takes the improvement/input/mask tensors, the pass bar, and the
  config, and returns the pass mask plus each event's worst-channel captured
  fraction (the saved diagnostic is unchanged for every rule). `process_chunk`
  calls it; no acceptance math is inlined anymore.
- The min-channel branch keeps the historical energy-floor form
  (`improvement >= input*bar - 1e-6`) rather than the ratio form, so its
  accepted events stay directly comparable with the completed fraction sweep.
- Near-zero-input channels (input energy ≤ 1e-8) are excluded from the mean
  and count as passing under min/k-of-n — the same treatment the floor form
  gives them, so a channel with no signal can neither help nor kill an event.
- `output_metadata` records the rule name plus, for k-of-n, the required
  share; `validate_config` rejects unknown rules and shares outside (0, 1].
- Self-test extended: synthetic tensors cover all three rules (min fails an
  event the mean and k-of-n@0.5 accept; k-of-n at the default share fails it
  again) and the near-zero-input edge that would poison a naive mean with
  inf.

Validation before submission: py_compile of both modules, `bash -n` on all
four sbatches, and the CPU self-test in the pytorch overlay — all passed.
Code and sbatches were uncommitted at submission time.

## The four runs

Each variant is single-variable against an existing sweep run; everything
else is identical (threshold 5, projection 8, RMSE 3, 3 passes × 1 round,
mean-channel-rmse objective, event-merge 0.5 ms).

| Job | Name | Run directory suffix | Change |
|---|---|---|---|
| `16762080` | flat10 | `0019_allchannel_pass3_round1_fraction10_step0_fitted8` | bar 0.10, `--pass-fraction-step 0.0` — no escalation |
| `16762081` | step20 | `0019_allchannel_pass3_round1_fraction10_step20_fitted8` | bar 0.10, step 0.2 → bars 0.10/0.30/0.50 |
| `16762082` | mean20 | `0019_allchannel_pass3_round1_fraction20_step10_meanchannel_fitted8` | mean captured fraction ≥ bar decides |
| `16762083` | kofn20 | `0019_allchannel_pass3_round1_fraction20_step10_kofn875_fitted8` | ceil(0.875·n) of valid channels must clear the bar (~7 of ~8) |

What each isolates:

- **flat10 vs. the 10% run** tests whether escalation contributes anything at
  all, given that pass 0 is bit-identical between the two (pass 0's bar is
  the base fraction regardless of step).
- **step20 vs. the same 10% run** steepens escalation from the same pass-0
  bar; together with flat10 the trio maps escalation speed at fixed pass-0
  difficulty.
- **mean20 vs. the 20% run** swaps the aggregation rule at the same numbers,
  so one weak channel no longer kills an otherwise good event.
- **kofn20 vs. the 20% run** keeps a per-channel floor but tolerates exactly
  one weak channel per event.

New sbatches, each a copy of `0019_allchannel_full.sbatch` with only the job
name, run directory, and criterion flags changed:
`0019_allchannel_flat10.sbatch`, `0019_allchannel_step20.sbatch`,
`0019_allchannel_mean20.sbatch`, `0019_allchannel_kofn20.sbatch`.

## Launch state

All four were running on l40s_public at submission (queued straight onto
GPUs, no waiting), through calibration and into the alternating fit within
two minutes, objectives converging, stderr empty beyond the usual FUSE mount
warning. Expected finish ~1.5 h each based on the completed sweep.

## Results (2026-09-01, night)

All four completed exit 0 in 1:14–1:57, every one ending
`stopping_reason: all_passes_complete` under the exhaustion code. Totals
against their baselines:

| Run | Pass 0 | Pass 1 | Pass 2 | Total | Baseline total |
|---|---|---|---|---|---|
| flat10 | 899,822 (bar .10) | 6,178 (.10) | 52 (.10) | 906,052 | 10%: 901,334 |
| step20 | 899,873 (bar .10) | 575 (.30) | 0 (.50) | 900,448 | 10%: 901,334 |
| mean20 | 1,721,112 (bar .20) | 3,124 (.30) | 4 (.40) | 1,724,240 | 20%: 568,889 |
| kofn20 | 1,244,448 (bar .20) | 1,740 (.30) | 2 (.40) | 1,246,190 | 20%: 568,889 |

Findings:

- **Escalation is a weak dial.** flat10 (906,052), the original 10% run
  (901,334), and step20 (900,448) land within 6k events of each other despite
  bars 0.10/0.10/0.10 vs 0.10/0.20/0.30 vs 0.10/0.30/0.50. A flat bar lets
  passes 1–2 grind out a few thousand more events (6,178 + 52 versus 1,472 +
  0); a steep bar kills them harder (575 + 0). Pass 0 dominates every total;
  the duplicate wall holds under every schedule. Note flat10's pass-0 count
  differs from the 10% run's by 40 events (899,822 vs 899,862) — the config
  is identical, so this is GPU nondeterminism in the pursuit, worth
  remembering when comparing "identical" passes across jobs.
- **The aggregation rule is the strong dial.** mean-channel triples the 20%
  bar's yield (1,724,240 — beyond even the 5% run's 1,105,917) and k-of-n
  more than doubles it (1,246,190). "Which channels decide" admits far more
  than "how high the bar sits". Whether those extra events are real spikes or
  junk is exactly the pending quality audit: mean20 must be checked against
  0019's founding motivation (narrow-sigma near-surface cheats), since
  softening the per-channel floor is precisely what could re-admit them.

## Codebook-size sweep on top of the rule variants (2026-09-01, night)

The user asked to cross the acceptance-criteria work with a second dial: the
temporal codebook size `q` (8 everywhere so far). One parameterized sbatch,
`0019_allchannel_q_sweep.sbatch` (env vars `VARIANT=base|flat10|step20|
mean20|kofn20` and `Q=16|32|64`, flags per variant via a case statement),
runs 5 configs × 3 sizes = 15 runs into run dirs suffixed `_q16/_q32/_q64`
(the Q=8 baselines are the existing sweep). `q` is generic in the code —
atoms are assigned to the two prototypes by `q modulo 2`, and the k-means,
cone projection, and duplicate machinery all take `config.q` — but the
0016→0019 lineage had only ever run at Q=8, so the base q=16 run went first
as a GPU canary (`16780702`, completed 1:49): it cleared calibration and was
accepting ~335 events/chunk at bar 0.2 by chunk 51, so the path works. The
remaining 14 went in as `16780929–42` (flat10/step20/mean20/kofn20 ×
q16/32/64, plus base q32/q64). The user's unrelated download job blew the
scratch quota and the scheduler then killed everything running: base-q16
`16780702` and step20-q16 `16780930` had completed (exit 0) before the
quota blowup; the other 13 died with exit 1 mid-run (I/O deaths, no
traceback — the base-q32 log shows a healthy 32-row alternating fit, the
flat10-q16 resume was mid-pass-2, so not a q>8 code bug). Once the quota
was writable again, all 13 were requeued with `--resume` as `16794259` and
`16794261–72` (the sweep sbatch counts a completed pass by its
`consolidation.json`, counts existing chunk files as visits, and atomic-npz
means no half-written chunks, so mid-pass deaths resume cleanly;
step20-q16 `16794260` was also requeued by the loop and self-terminates
from its completed state). Queue state at reporting: all 14 pending on
Priority. A CPU
smoke of the full pipeline is impossible — extraction is CUDA-only by
design (`residuals_0012.validate_config`), which is why the canary ran on
GPU. Everything else stays identical to the Q=8 runs (threshold 5,
projection 8, RMSE 3, 3 passes, 48G mem for the bigger footprint cache).
One watch-item for the audit: with more atoms per cone, the duplicate
machinery's temporal-correlation gate (Q×Q, threshold 0.9) has more pairs
that can look "correlated", so pass-1+ duplicate rejections may shift with
Q for reasons unrelated to the acceptance rule.

## Plot suites

The first submission attempt (`sbatch --dependency=afterok:…`) failed with a
SLURM "Job dependency problem" because the runs had already completed by the
time the suites went in; resubmitted plainly as `16768253` (flat10),
`16768254` (step20), `16768255` (mean20), `16768256` (kofn20). New sbatches:
`0019_allchannel_{flat10,step20,mean20,kofn20}_plots.sbatch`, each a copy of
the frac10 suite with new run/plot paths and gallery title. One suite-code fix
first: `plot_0019_recording_replay.py` hardcoded the escalation as
`bar + 0.1 * pass`; it now reads the run's own `pass_fraction_step`
(capped at 0.9, matching `pass_all_channel_fraction`), so the suptitle shows
0.10/0.10/0.10 for flat10 and 0.10/0.30/0.50 for step20.

Those first suites were based on the frac10 template and carried only 12
panels each — the replay panels the older 5/10/20% galleries have (the
most-subtractive-chunk replay and the three-column full-recording figure)
were missing. Per the user's request the two missing panels were added: each
suite sbatch now renders `recording_replay_chunk001629.png` (chunk 1629 is
the most-subtractive chunk in **all four** variant runs, 60.9–67.9% of its
local energy captured — chunk 1580 was the 20%-sweep's peak, but the
variants peak at 1629) and chains the existing
`0019_full_recording_replay.sbatch` as a dependent job so the full-recording
figure lands in the same gallery (suppression via `SKIP_FULLREC=1`). The
upgraded suites are jobs `16775165` (flat10), `16775166` (step20),
`16775168` (mean20), `16775169` (kofn20).

Plot suites were deliberately not queued: decide after seeing event counts
and rejection histograms which variants earn galleries.

## Plot suites for the q sweep (2026-09-02)

Queued with one new parameterized sbatch,
`residuals/src/plots/0019_allchannel_q_sweep_plots.sbatch` (same
`VARIANT`×`Q` env-var scheme as the run sweep; run/plot dirs derived from
the same `DIRTAG` mapping). It renders the standard panels plus chunk-0
replay, then **computes each run's own most-subtractive chunk at runtime**
(max of `captured_energy.sum()/input_energy.sum()` over `pass_00` chunks)
instead of assuming chunk 1629, renders that replay, builds the gallery,
and renders the full-recording replay inline (no chained job, since the
sweep runs finish at different times). Supporting change:
`build_plot_gallery.py` now labels any unregistered
`recording_replay_chunk*.png` as "recording replay (chunk N, most
subtractive)" in the reconstruction group instead of dumping it into
"other" — the chunk number varies per run and codebook. The replay
suptitle already reads `pass_fraction_step` from the run config, so flat10
and step20 show correct bar schedules.

Two submission waves were needed: the first dependent submission passed
`Q=q64` (letter included) by accident, so those suites failed fast on the
missing-run guard (`_qq64` dir); and another scheduler purge wave
CANCELLED-by-0 the whole first batch of pending plot suites. The final
state: 12 of 15 runs complete (only kofn20-q64 `16794270` and base-q64
`16794272` still running; mean20-q32 was killed mid-pass-2 and resumed as
`16816964`). Plot suites resubmitted with correct Q: `16816966–77`
immediate for the completed runs, `16816987` (`afterok:16816964`),
`16816988` (`afterok:16794270`), `16816989` (`afterok:16794272`) for the
three stragglers. Most suites queue on `QOSMaxMemoryPerUser` (64G
request), so they serialize; base-q16's suite was the first to start.

## Q16 runs complete; mean20-q32 killed by QOS (2026-09-02)

All five q16 runs are done and end `all_passes_complete`: base 635,256, flat10 987,956
(`16794259`), step20 974,666 (`16794260`, self-terminating from its completed state as
designed), mean20 1,808,003 (`16794261`), kofn20 1,338,731 (`16794262`). Q=16 inflates
every variant's total by roughly 5–12% over its Q=8 twin (base 635k vs 569k, flat10
988k vs 906k, step20 975k vs 900k, mean20 1.81M vs 1.72M, kofn20 1.34M vs 1.25M), and
the rule hierarchy is unchanged: mean > kofn > flat > step > base. Three q32 runs also
landed — flat10 `16794263`, step20 `16794264`, kofn20 `16794266`, all COMPLETED — but
**mean20-q32 `16794265` was SIGTERM-killed at 15:31 by `QOSMaxGRESPerUser`** while
mid-pass-2 at chunk 123/1958, every visited chunk accepting zero (all proposals
duplicate-rejected with rollback, reasons 192/144/146 — the duplicate wall is total
for the mean rule at Q32). It was requeued the same day as `16817267`
(`sbatch --export=VARIANT=mean20,Q=32` on the sweep sbatch; resume picks up at
pass-2 chunk 123 with passes 0–1 consolidated: 1,857,163 + 42,794 events). Still running: the four q64 variants `16794267–70`, base-q32 `16794271`,
base-q64 `16794272`. The q16 plot suites `16802326–30` are pending on Priority and can
go at any time since their runs are complete; the q32/q64 suites `16802377–91` stay on
`afterok` dependencies.

## Sweep results: all 15 runs and galleries complete (2026-09-02)

Every run exited 0 with `stopping_reason: all_passes_complete`, and all 15
galleries built (14 PNGs each, including each run's own most-subtractive
chunk replay and the full-recording replay). Totals per (config, Q), with
the Q=8 baselines for comparison:

| config | Q=8 | Q=16 | Q=32 | Q=64 |
|---|---|---|---|---|
| base (bar .2, min) | 568,889 | 635,256 | 694,753 | 742,738 |
| flat10 | 906,052 | 987,956 | 1,067,274 | 1,121,706 |
| step20 | 900,448 | 974,666 | 1,043,539 | 1,094,827 |
| mean20 | 1,724,240 | 1,808,003 | 1,900,546 | 1,955,846 |
| kofn20 | 1,246,190 | 1,338,731 | 1,431,211 | 1,494,851 |

Findings:

- **More codebook atoms add events monotonically at every acceptance
  rule.** Base +30% from Q=8→64, variants +8–20%. The gain comes from
  better-shaped fits passing the same gates, not from the rules changing.
- **The rule ordering is invariant to Q**: mean > kofn > flat ≈ step > base
  at every codebook size. The two dials act independently.
- **Escalation still barely matters** — flat10 and step20 remain within ~3%
  of each other at every Q.
- The pending quality question compounds: the mean20 Q=64 run's ~1.96M
  events (3.4× the base run) need the sigma-mix/near-surface audit more
  than ever.



- [x] When `16762080–83` land: per-pass event counts, rejection-reason
      histograms (expect mean20/kofn20 to shift rejections from reason 16
      toward smaller counts), and the duplicate wall under each rule.
- [ ] Rejection-reason histogram across the four runs (the totals above are
      event counts; the audit tables are consolidated per run).
- [ ] Accepted-event quality vs the sweep: captured fraction, worst-channel
      fraction, sigma mix, and the near-surface share of accepted sigma-2
      events — does softening the rule re-admit the narrow-sigma junk 0019
      was built to remove?
- [x] Verify plot suites `16775165–69` land with the upgraded panel set:
      all four suites and their chained full-recording jobs (`16775228`,
      `16775652–53`, `16775656`) completed exit 0; every gallery now carries
      14 PNGs including `recording_replay_chunk001629.png` and
      `recording_replay_full_recording.png`.
- [ ] When the 15 q-sweep runs land (survivors `16780702`, `16780930`;
      requeued as `16794259`, `16794261–72`): totals per (variant, Q), the
      duplicate-wall shift with Q, and whether bigger codebooks raise
      captured fraction at fixed acceptance rules.
- [x] Queue plot suites for the q-sweep runs — done as one parameterized
      sbatch (see the q-sweep plot section): immediate suites `16802326–30`
      for the completed q16 runs, dependent suites `16802377–91` for the
      rest. Verify all 15 galleries land with the full panel set.

## Resume verification (2026-09-04)

Checked the queue and `sacct` on resume: every run and suite this card was
waiting on has landed. The mean20-q32 resume `16817267` completed in 43:12
exit 0 (passes 0–1 kept, 1,857,163 + 42,794 events, plus whatever pass 2 at
the 0.40 bar admitted), the late-landing plot suites `16816987–89`
(mean20-q32, kofn20-q64, base-q64) all completed exit 0, and the surviving
first-batch suites that show FAILED/CANCELLED in sacct (`16802326–30`,
`16802377–91`) are the superseded Q-typo/purge wave, already replaced.
On disk all 15 q-sweep galleries carry 14 PNGs and every run's
`summary.json` ends `all_passes_complete`. Nothing from this card remains in
the queue. The only open item is the accepted-event quality audit below.

## Next steps

- [x] When the 15 q-sweep runs land: totals per (variant, Q) recorded above;
      monotone Q gains, invariant rule ordering, escalation still weak.
- [ ] Accepted-event quality across the sweep: sigma mix and near-surface
      share of accepted sigma-2 events, worst on mean20 q64 (1.96M events).

## Links

- [[session-019-all-channel-error]]
- [[session-021-shift-invariant-peeling]]
- [[session-024-convolving-detection-peeling]]
- [[session-027-acceptance-gate-census-trim]]
