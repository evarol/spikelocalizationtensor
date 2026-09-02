# Acceptance-Rule Variants (0023)
**Created:** 2026-09-01
**Last updated:** 2026-09-01
**Status:** four acceptance-rule runs launched and healthy (`16762080–83`, all in the alternating fit within two minutes of submission, ~1.5 h each expected); code, sbatches, self-test coverage, and validation all done; results and plot-suite decision pending

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

Plot suites were deliberately not queued: decide after seeing event counts
and rejection histograms which variants earn galleries.

## Next steps

- [ ] When `16762080–83` land: per-pass event counts, rejection-reason
      histograms (expect mean20/kofn20 to shift rejections from reason 16
      toward smaller counts), and the duplicate wall under each rule.
- [ ] Compare accepted-event quality against the sweep: captured fraction,
      worst-channel fraction, sigma mix — does softening the rule re-admit
      the narrow-sigma junk 0019 was built to remove?
- [ ] Decide which variants earn plot suites (and extend the suite's
      per-pass bars in the suptitle, which assumes one bar per pass — flat10
      keeps that trivially, step20/mean20/kofn20 fit the existing pattern).

## Links

- [[session-019-all-channel-error]]
- [[session-021-shift-invariant-peeling]]
- [[session-024-convolving-detection-peeling]]
