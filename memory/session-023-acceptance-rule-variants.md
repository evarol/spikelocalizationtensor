# Acceptance-Rule Variants (0023)
**Created:** 2026-09-01
**Last updated:** 2026-09-01
**Status:** all four acceptance-rule runs complete exit 0, all ending `all_passes_complete`; totals — flat10 906,052, step20 900,448, mean20 1,724,240, kofn20 1,246,190; escalation speed is a weak dial (all three 10%-bar runs within 6k of each other), the aggregation rule is the strong dial (mean-channel 3.0×, k-of-n 2.2× the 20% run); plot suites verified complete with the full panel set (14 PNGs each, per-chunk + chunk-1629 + full-recording replays, jobs `16775165–69` and chained `16775228`/`16775652–53`/`16775656`); per-variant quality audit pending

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

## Next steps

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
- [ ] When the 15 q-sweep runs (`16780702`, `16780929–42`) land: totals per
      (variant, Q), the duplicate-wall shift with Q, and whether bigger
      codebooks raise captured fraction at fixed acceptance rules.
- [ ] Queue plot suites for the q-sweep runs once their results are in —
      copy the upgraded variant-suite sbatches (chunk-0 + chunk-1629 replays
      + chained full-recording render), find each run's own most-subtractive
      chunk instead of assuming 1629, and verify the replay suptitle's
      per-pass bars, which now read `pass_fraction_step` from the run
      config.

## Links

- [[session-019-all-channel-error]]
- [[session-021-shift-invariant-peeling]]
- [[session-024-convolving-detection-peeling]]
