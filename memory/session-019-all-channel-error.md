# All-Channel-Error Peeling (0019)
**Created:** 2026-08-30
**Last updated:** 2026-08-31
**Status:** fraction20 run `16655016` complete (568,888 events) and plotted; 5% run resumed as `16678849` — passes 0 (1.10M events) and 1 (2,557) done, pass 2 pending on productive chunks; 10% middle-ground run `16679282` queued behind it, plots `16679287` dependent

## Why 0019 exists

0018's fit objective minimizes the *worst* channel's error (a minimax), and
its acceptance rule asks for just two improving channels. That combination
rewards suspiciously narrow templates: 37.2% of 0018 events pick sigma = 2 µm,
57% of those sit within 3 µm of the probe surface, and 82% of events that
capture under 20% of their local energy come from the three narrowest widths.
The fix, agreed for 0019: the error must drop across **all** channels, not
just the worst one.

## What 0019 does

- **Objective.** Fit cost is the mean RMSE across all channels, replacing the
  worst-channel minimax.
- **Acceptance.** Every valid channel in the fit mask must capture at least
  20% of its own noise-normalized energy (`all_channel_min_fraction=0.2`),
  and the bar rises by 0.1 per recording pass (0.2 / 0.3 / 0.4). There is no
  per-event total-share bar — across ~8 channels that is mathematically
  impossible (the sum would exceed 100%).
- **Passes.** 3 recording passes × 1 peeling round per chunk visit
  (tunable). Detection threshold stays 5 on every pass; only the
  reconstruction bar escalates. From pass 2 on, each chunk's starting
  residual is rebuilt on the GPU by replaying every saved earlier-pass event
  (searchsorted time windows, batched `index_put_`; no residual files, no
  CPU-heavy work), and the chunk-local duplicate prior is preloaded with the
  replayed events. A pass that accepts fewer than 1,000 events recording-wide
  (`pass_stop_min_events`) triggers an early stop.
- **Rejection audit.** Every detected-but-rejected candidate is saved per
  chunk with its fit metrics and a reason bitmask (1 gain, 2 rmse,
  4 captured, 8 projection, 16 all-channel, 32 energy, 64 duplicate,
  128 rolled-back round).
- **Score floor** stays 8 — this is a single-variable comparison against
  0018.
- **Validation.** The synthetic CPU smoke accepts a spike at bar 0.2 (worst
  channel captures 0.33) and rejects it at bar 0.9, logging reason
  `0b10011000`. The self-test also covers the replay-window logic and the
  per-pass escalation.

## Files and outputs

All uncommitted as of 2026-08-30.

- `residuals/src/preprocessing/0019_allchannel_peeling.py` (derived from
  0018; new pursue / process_chunk / replay / load_prior_events /
  _consolidate)
- `residuals/src/preprocessing/0019_allchannel_full.sbatch` (3 passes,
  round 1, fraction 0.2 step 0.1, mean-channel objective)
- `residuals/src/plots/plot_0019_allchannel_cones.py` (renamed 0018 plot)
- `residuals/src/plots/0019_allchannel_plots.sbatch` (full suite +
  index.html gallery, pointed at the 0019 run)
- Run output:
  `residuals/runs/dataset1_p1/0019_allchannel_pass3_round1_fraction20_step10_fitted8`
- Plot loaders `0016_onehot_lattice_plots.py`,
  `plot_raw_residual_reconstructions.py`, and
  `plot_spiketensor_residual_pursuit.py` gained a `pass_*/chunk_*.npz`
  fallback so the same suite works for both output layouts.

## Data layout and diagnostics

- Chunks now live in `pass_XX/chunk_NNNNNN.npz`, one directory per recording
  pass. Consolidation runs per pass and at the root; only event-aligned
  fields are consolidated — the `rejected_*` audit tables stay sharded.
- New per-event arrays: `recording_pass.npy`, `all_channel_ok`,
  `all_channel_fraction`, `min_channel_captured_fraction`.
  `residual_pass.npy` still holds the peeling round.
- Round census from 0018 (2.89M events): round 0 — 71.2%, round 1 — 22.0%,
  round 2 — 5.3%, round 3 — 1.2%, round 4+ — ~0.3%. Four layers carry
  essentially all the productive depth; the old 60 in-chunk rounds were
  mostly re-detecting noise.
- Passes and inner rounds are the same machine state; passes add seam
  healing, parallelism, a per-pass audit, and the tighter acceptance bar.

## Queue history

The first full job, `16654819`, died after 42 s (exit 2): the sbatch passed
`--all-channel-improvement true`, but the flag is argparse `store_true`, so
`true` arrived as an unrecognized positional. Fixed the sbatch to the bare
flag and re-ran the self-test. Its dependent plot job `16654823` died with
DependencyNeverSatisfied and was canceled. The resubmitted full run
`16655016` uses the same output directory (`--resume` is safe because the
failed attempt wrote no chunks). Plot suite `16655046` waits on
`afterok:16655016` and is held at your instruction — release it with
`scontrol release 16655046` after reviewing the run.

## Progress 2026-08-31

Job `16655016` is 17 minutes in and healthy: pass 0 reached chunk 621/1958
(about 36 chunks/minute, so roughly 55 minutes per pass and around three
hours for all three passes). Chunks are landing in
`pass_00/` as expected (628 files on disk) and the stderr log is empty.
The rejection histogram is dominated by reason 16 (all-channel fraction)
with small counts of 2 (rmse) and 18/24 combos — exactly what the new
criterion should produce. Per-chunk acceptance is roughly 25–30% of
proposals (230–440 events per chunk), and each chunk drops 4–5% of its
full energy. No pass-1 replay yet; that starts after pass 0 finishes.

## Run complete and 5% rerun (2026-08-31)

Job `16655016` finished in 1:17:52 with exit 0 and an empty stderr. Pass 0 at
the 0.20 bar accepted 568,436 events from ~2.28M proposals with zero rollbacks
and zero duplicate rejections; 1.62M proposals (71%) failed on the all-channel
bar alone (reason 16), so the new criterion is doing essentially all of the
filtering. Chunks lost about 4.4% of their energy per round on average.

Pass 1 at the 0.30 bar collapsed: 1,527 of 1,958 rounds rolled back, 452
events survived, and the 1,000-event early stop fired, so pass 2 never ran.
The rollback audit explains the mechanism. Of pass 1's 2.22M proposals,
520k were killed as duplicates (reason 192/64) and another 1.2M died as
all-channel-fail-plus-rollback (reason 144). The replayed pass-0 fits were
allowed to capture as little as 20% of a channel's energy, so the replay
residual keeps ≥5σ leftovers at pass-0 event sites; those re-detect, are
correctly rejected as duplicates, and the round rolls back. A weaker pass-0
bar therefore makes pass 1 harder, not easier — the escalation design assumes
strong earlier-pass fits. Accepted events' median worst-channel captured
fraction was 0.327 against the 0.20 bar.

Total yield: 568,888 events (568,436 + 452) versus 0018's 2.89M — the
all-channel criterion cut acceptance roughly 5×. The sigma mix barely moved:
39.8% of accepted events still pick sigma = 2 µm (0018: 37.2%), so the bar
removed many narrow-width events without changing the width preference. The
near-surface breakdown of the sigma-2 events (0018's other diagnostic) is
still unchecked.

On the ops side, `scontrol release` fails from the agent sandbox with an
unspecified error; the working pattern (confirmed with the user) is to
`scancel` a held job and `sbatch` it again. The held plot job `16655046` was
cancelled and resubmitted as `16675906`, which completed in 3:03; the gallery
is at `residuals/out/0019_allchannel_pass3_round1_fraction20_step10_fitted8/index.html`.

**5% rerun.** Per the user's instruction the acceptance bar drops from 20% to
5% while the detection threshold stays at 5, "saving the rollbacks" — the
rejected/rollback audit is kept and is now consolidated instead of living only
inside per-chunk npz shards. `_consolidate` in
`0019_allchannel_peeling.py` writes consolidated `rejected_*.npy` tables per
pass and at the run root, and `n_rejected` joins `n_events` in the pass and
root summaries. Validated with the 0019 self-test and a synthetic two-chunk
consolidation round-trip (including a zero-rejection chunk). The run config in
`0019_allchannel_fraction5.sbatch` is identical to the 20% run except
`--all-channel-min-fraction 0.05`, so the per-pass bars are 0.05 / 0.15 / 0.25,
into run directory
`residuals/runs/dataset1_p1/0019_allchannel_pass3_round1_fraction5_step10_fitted8`.
Queued as `16676130` with the plot suite `0019_allchannel_fraction5_plots.sbatch`
dependent.

**Early stop replaced by chunk exhaustion.** The recording-wide pass floor
(`pass_stop_min_events`) stopped the whole recording when a pass accepted
under 1,000 events, which cut the fraction20 run off after pass 1; the user
asked to keep the recording going instead. The floor is removed (config field,
validate check, and both sbatch flags), and a chunk visit that accepts zero
events now marks the chunk *exhausted*: later passes skip it entirely. The
justification is determinism — an exhausted chunk's interior residual is
unchanged (its own subtractions are nil; neighbors can only touch seam margins)
and the per-channel bar only escalates, so re-detection could only reproduce
the same rejected proposals. `exhausted_chunks()` rebuilds the set on resume
from the last consolidated pass (missing or zero-event chunk file ⇒ exhausted),
which makes skip state survive requeues; pass summaries gained
`n_chunks_visited` / `n_chunks_exhausted`, and the loop ends naturally with
`stopping_reason=all_chunks_exhausted` when nothing productive remains. The
self-test covers the rebuild across skipped passes.

**5% results so far (job `16676130`, cancelled mid-pass-2 to pick up the new
code).** Pass 0 at the 0.05 bar accepted 1,103,359 events with 1.18M rejections
— roughly double the 20% bar's 568k, with no rollbacks. Pass 1 at the 0.15 bar
accepted just 2,557 events from 2.19M rejections: the duplicate wall is
confirmed and much harder than at the 20% bar, exactly as the memory analysis
predicted (weak 5%-bar replay fits leave large ≥5σ leftovers that re-detect
and die as duplicates). The run resumes as `16678849` (plots `16676258` were
cancelled with it; the new dependent suite is `16678863`): completed passes 0–1
are kept, pass 2 runs at the 0.25 bar visiting only chunks that were productive
in pass 1, and the loop ends when every chunk is exhausted.

**10% middle ground.** The user framed the acceptance bar as the dial that was
too low in 0018 (no per-channel floor, 2.89M events with junk) and too high in
0019's 20% run (569k), and asked for a middle setting. The 5% run already
occupies the loose end, so the new run uses `--all-channel-min-fraction 0.10`
(per-pass bars 0.10 / 0.20 / 0.30), detection threshold and every other gate
unchanged, into
`residuals/runs/dataset1_p1/0019_allchannel_pass3_round1_fraction10_step10_fitted8`.
It is job `16679282` with plot suite `16679287` dependent, queued behind the
5% resume on GPU quota. The fraction sweep is now 5% (`16678849`), 10%
(`16679282`), and 20% (`16655016`) at otherwise identical settings.

## Next steps

- [x] Implement the all-channel criterion, pass loop, GPU replay, rejection log.
- [x] Validate: py_compile, `bash -n`, CPU self-test, synthetic smoke.
- [x] Queue the full run + dependent plot suite (SLURM, outside the sandbox).
- [x] After completion: check the rejection-reason histogram, sigma usage
      (did the sigma-2 cheat die?), per-pass event counts, and per-channel
      fractions.
- [x] Replace the recording-wide pass floor with per-chunk exhaustion skipping
      so the recording always runs to natural completion.
- [ ] When `16678849` finishes: pass-2 yield at the 0.25 bar, the full-run
      sigma mix, and comparison of the 5% vs 20% acceptance-quality trade.
- [ ] Near-surface share of accepted sigma-2 events (0018's 57%-within-3 µm
      diagnostic) on both runs.

## Links

- [[session-018-bipolar-prototype-cone-peeling]]
- [[session-016-one-hot-lattice-peeling]]
- [[session-017-initial-threshold-spike-discovery]]
- [[feedback_plot_suite_completeness]]
