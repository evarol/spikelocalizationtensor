# All-Channel-Error Peeling (0019)
**Created:** 2026-08-30
**Last updated:** 2026-09-01
**Status:** fraction sweep complete (20% `16688480` resumed → 568,889 events, 10% `16679743` → 901,334, 5% `16679719` → 1,105,917, all ending `all_passes_complete` under the exhaustion code); cross-fit `16685303` complete (my fitter on his 2.31M spikes: captured 0.42, 23.8% passing the 20% bar); plot suite carries per-chunk, most-subtractive-chunk, and two-column full-recording replays; the 5%/10% resubmitted galleries (`16722284`/`16722285`) verified complete with replay panels; three-column full-recording renders `16732015–17` and dependent full suites `16732100`/`16732101` in flight; everything in `residuals/out/` reachable from the new hub `residuals/out/index.html` (`build_out_index.py`)

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

The exhaustion loop's first version crashed on resume (`16678849` died after
65 s): the resumed-chunk branch read the saved event count but fell through to
the compute tail without `core_stop` bound. The fix (`8be20df`) gives resumed
chunks their own `continue` after counting the visit and applying exhaustion
marking, so a resumed zero-event chunk also exhausts. The 10% attempt
`16679282` was cancelled one minute in for running the unfixed module (a
requeue would have crash-looped it) and resubmitted. Working tree fully
committed through `8be20df`; the fraction sweep now runs as 5% `16679719`
(resume; passes 0–1 kept), 10% `16679743` (fresh), with plot suites `16679742`
and `16679749` dependent.

## Channel × time recording replay added to the suite (2026-08-31)

The user asked for a channel-by-time amplitude plot in the 0019 plot suite,
and after two rejected drafts (a binned amplitude-mass histogram, then a
per-event scatter — "i want some continuity") settled on a continuous
recording view like the 0018-era `residual_recording_chunk0.png` example.
Final panel: `residuals/src/plots/plot_0019_recording_replay.py`, registered
in `build_plot_gallery.py` as `recording_replay_chunk0.png` (reconstruction
group) and invoked by all three 0019 sbatch suites after the Omega raster.
It renders the preprocessed input for a 20 ms window of one chunk plus one
panel per recording pass, subtracting each pass chunk's saved `predictions`
exactly as the run did, in the red-blue `RdBu_r` colourway on white with
depth-ordered channels and a voltage/robust-noise colorbar; the window is
auto-placed where pass-0 events are densest, and pass bars in the suptitle
come from the run config so one script serves the 5%, 10%, and 20% runs. On
the 20% run's chunk 0 the RMS drops 100% → 97.8% after the pass-0 replay,
matching the run's own per-chunk energy accounting (pass 1 accepted nothing
there, so its panel is identical).

The debugging detour is worth remembering: the first renders were garbage
because the raw `.imec0.ap.bin` was memmapped as 384 columns, but SpikeGLX
ap files carry 384 AP **plus one sync column** (`snsApLfSy=384,0,1`, 385
columns total), so a 384-column memmap is byte-misaligned from the very
first sample — symptoms were ~2× wrong robust noise and near-zero waveform
correlations that looked exactly like a preprocessing mismatch. The script
now parses the adjacent `.meta` for `snsApLfSy`, memmaps all file columns,
slices the first `n_channels`, and scales int16 → volts by 2.34375e-06 (the
NP1.0 AP factor the ibl `spikeglx.Reader` applies). Preprocessing is the
standard 300–6000 Hz butter order-3 bandpass plus per-channel time-median
(`preprocessing.raw_residual.preprocess_voltage`); no whitening, matching
the pipeline. The spikeglx package itself was avoided on purpose — stacking
`ibl-sorter.ext3` with `pytorch.ext3` breaks scipy/numpy ABI, and the plot
env (pytorch overlay) has everything else needed. A per-chunk subtractiveness ranking (summed per-event `captured_energy` over
`input_energy`, binned by `spike_times // 30000`) picked chunk 1580 as the
heaviest chunk (693 events, 60.2% of its local energy captured); its replay
drops RMS 100% → 88.6% with 90 events in the 20 ms view and the big
multi-channel spikes visibly carved out, while passes 1–2 add essentially
nothing (2 events, 0 in view) — the duplicate wall in picture form. Both
`recording_replay_chunk0.png` and `recording_replay_chunk001580.png` are
registered in the gallery (19 panels); the overall recording-wide captured
fraction is 65.6% of binned input energy.

The user then asked for a full-recording version, which is
`residuals/src/plots/plot_0019_full_recording_replay.py` with its own sbatch
`0019_full_recording_replay.sbatch` (cpu_short, ~1 h; the 20% run is job
`16693050`). It preprocesses every chunk (300–6000 Hz bandpass + per-channel
time-median, volts via the 2.34375e-06 factor), replays every saved pass
chunk's `predictions` into a rolling residual (chunks are visited in order,
events of earlier passes accumulate, events within ±45 samples of the core
window are subtracted), normalizes by each chunk's saved `noise`, and
decimates with signed block extrema (5000-sample bins) before stacking the
input panel plus one panel per recording pass. A 30-chunk inline test
confirmed 100% → 94.5% RMS after pass 0; the first version silently did no
subtraction because the volts scaling was dropped in the rewrite — worth
remembering that `preprocess_voltage` takes volts, not raw int16 counts.
The sbatch takes the run directory as its first argument (20% run default).
The figure's columns evolved with the user's requests: column 2 holds a
cumulative depth-time spike raster per pass row (fitted-depth µm from
`global_sources`, pass-colored smoothed density using the
`plot_temporal_codebook_depth_time_raster` binning technique recolored for
white), with the input row blank ("no events yet"), and column 3 — added
next — shows only that pass's own events ("pass 0 events (568,436)",
"pass 1 events (452)", …), the direct picture of how little the escalation
passes contribute. The three-column renders are jobs `16732015`
(20%), `16732016` (5%), and `16732017` (10%), with the stale frac5/frac10
plot suites resubmitted as `16732100`/`16732101` dependent on them (earlier
suite resubmissions `16722284`/`16722285` and `16730632`/`16730633` were
superseded — the first predates the replay panels, the second was cancelled
to avoid rendering the two-column figure twice); their
per-chunk
most-subtractive replays also exist — every fraction peaks at the same
chunk 1580 (5%: 1,072 events, RMS 100→87.2%; 10%: 923 events, 100→87.3%;
20%: 693 events, 100→88.6%), and the recording-wide captured fraction is
58.0% (5%), 60.6% (10%), and 65.6% (20%) of input energy.

Rejection audit over the consolidated tables (per proposal): the 5% run
rejected 5,222,940 of 6,328,857 proposals (82.5%), the 10% run 4,715,134 of
5,616,468 (84.0%), and the 20% run 4,394,432 of 4,963,321 (88.5%). Pass 0
rejections are almost purely the all-channel bar (reason 16 alone: 1.14M /
1.35M / 1.70M across 5/10/20%), with rmse (~74k) and projection (~14.6k)
flat across fractions; passes 1–2 are dominated by the duplicate flag
(64 bit; up to 1.07M) almost always combined with round rollback (128 bit),
and rollback counts scale with the bar (pass-1 rollbacks 578k → 996k →
1.71M) — the weak-earlier-fits mechanism in numbers.

## 20% run resumed under the exhaustion code (2026-08-31)

The 20% run `16655016` had early-stopped by the old recording-wide floor
(`stopping_reason: pass_1_below_floor` — pass 1 at the 0.30 bar accepted 452
events, pass 2 never ran), and was never re-run with the new code. Per user
instruction it was resumed as job `16688480` (9 min 23 s, exit 0), keeping
passes 0–1 and running pass 2 at the 0.40 bar over only the 399 chunks that
were productive in pass 1. The result confirms the duplicate wall tightens
with the bar: pass 2 accepted exactly 1 event (340 of 1,306 proposals killed
as duplicates in the last logged round, the rest as all-channel-plus-rollback
combos), so the 20% run's final yield is 568,889 events and the run now ends
`all_passes_complete` instead of an artificial floor stop. One migration fix
was needed first: the old run's `pass_summaries.json` still carried the legacy
`"stopped": true` flag from the removed floor, and `pursue()` honored it,
which would have skipped pass 2 immediately. Because the exhaustion code never
writes that flag, the `stopped_after` gate was deleted from
`0019_allchannel_peeling.py` (legacy markers are now inert); py_compile passed
and the sbatch is the unchanged `0019_allchannel_full.sbatch` with `--resume`.

## Cross-comparison with the SLT collision repo (2026-08-31)

To put 0019's fit quality next to the other implementation
(`/scratch/ap7151/_SYMLINKS/am15577-paths/UnitMatch/SLT_ICLR/collision/base_implementation`,
the "SLT/spiketensor" multipole decomposition), the two error metrics are being
cross-evaluated on each other's spike lists. His metric is nMSE over per-spike
C=10 TPCA-denoised × T=90 windows (DC-removed, ÷100) divided by variance
4.242133873e-4; his per-spike `sse`/`captured` arrays live in
`runs/base_M64_R4/multipole_uo_monop_M64_R4_s10_P2c35.npz` and his 2,310,868
spike times/channels in `extraction/results/dataset1_p1/`. Both pipelines read
the same raw binary. Two facts already established:

- **His metric on my spikes.** 86.8% of my 568,888 accepted events match one of
  his spikes (same channel, ±0.5 ms). On that matched subset his base fit gets
  nMSE 8.236, VE 91.3%; a random size-matched control subset of his own spikes
  gets 8.107 / 85.8%. So at my event sites his model explains more energy than
  average — my sites are strong clean spikes.
- **My metric on his spikes** is running as job `16685303`
  (`0019_cross_hisspikes_fit.py` + `0019_cross_hisspikes.sbatch`): the frozen
  0019 model (codebook, neighborhoods, filter from the 20% run) fit once on raw
  windows at each of his 2.31M sites, no detection/peeling. The harness is
  validated by a self-check that re-fits the run's own pass-0 events through
  the identical path and reproduces the consolidated metrics to ~1e-4. The
  2-segment smoke on his spikes: mean-channel nRMSE 1.24 (like mine) but
  captured fraction 0.42 (mine: 0.61) and only ~23% passing the 20%
  all-channel bar (reason 16 dominant).

The fraction sweep on my own spikes is now complete: 5% bar → 1,105,917
events (captured 0.527, VE 0.580), 10% bar `16679743` → 901,334 events
(captured 0.557, VE 0.606; pass 2 at 0.30 accepted zero and the run ended
`all_chunks_exhausted`), 20% bar → 568,888 (captured 0.614, VE 0.656).

The reverse cross-fit job `16685303` completed in 39:29 over all 2,310,868 of
his spikes, and the smoke numbers held at scale: my frozen 20% model achieves
mean-channel nRMSE 1.279 (median 1.168 — like my own events) but captured
fraction only 0.421 mean / 0.411 median (mine: 0.614), and the median
worst-channel captured fraction is 0.048, so his spike sites spread their
energy much less uniformly across channels than mine. Under the 20% all-channel
bar just 550,136 of his 2.31M sites would be accepted (23.8%); 1.64M die on the
all-channel criterion alone (reason 16), 89.6% pass the detection gate, and
essentially all pass rmse/projection. Both directions of the comparison agree:
his decomposition matches spikes my detector never proposes (weaker, less
spatially concentrated), while at my event sites his fit is as good as on his
own spikes — the pipelines find overlapping but not identical event sets.

## 5% run finished under the exhaustion code; plot suites resubmitted (2026-09-01)

The 5% resume `16679719` completed in 14:17 with exit 0 and, crucially, ended
with `stopping_reason: all_passes_complete` — the loop-completion semantics the
user asked for, not a recording-wide early stop. Pass 2 at the 0.25 bar visited
1,365/1,958 chunks, marked 1,957 exhausted, and accepted a single event, so the
final 5% yield stands at 1,105,917 events with 5.22M logged rejections (pass 0:
1,103,359, pass 1: 2,557). A chunk visit that accepts zero events marks the
chunk exhausted and later passes skip it, which is why the whole resume cost 14
minutes instead of a full third pass.

The dependent plot suites `16679742` (5%) and `16679749` (10%) both completed
exit 0, but as predicted their sbatch copies predated the replay-panel script:
the galleries registered the base 16 panels with no
`recording_replay_chunk0.png` (11 PNGs on disk versus 19 in the 20% gallery).
Both suites were resubmitted from the current scripts — `16722284` for the 5%
run, `16722285` for the 10% run — each with a 64G memory request; the 10% job
sat pending on `QOSMaxMemoryPerUser` behind the running 5% job and was not yet
verified complete at the time of writing (the user declined a wait-and-check).

## Plot-suite state on 2026-09-01 (afternoon)

The superseded-gallery chain is now: the three-column full-recording renders
`16732015` (20%, running), `16732016` (5%) and `16732017` (10%) (both pending
on `QOSMaxMemoryPerUser`), with the full frac5/frac10 suites `16732100`/`16732101`
pending on `afterok:16732016`/`16732017` respectively. The 0018/0019 galleries
plus every earlier-session figure set are now reachable from one hub page:
`residuals/out/index.html`, generated by `residuals/src/plots/build_out_index.py`
(stdlib-only, re-run any time), which walks out/, joins each gallery to its
backing run's `summary.json`/`config.json` (events, rejections, threshold,
escalating bars, stopping reason), and lists all 23 collections and 482
collection links with every target verified to exist. Also unrelatedly in the
queue: `ibl_ap_c` conversion job `16730433` running and `16730850`–`58` held on
dependency.

## Next steps

- [x] Implement the all-channel criterion, pass loop, GPU replay, rejection log.
- [x] Validate: py_compile, `bash -n`, CPU self-test, synthetic smoke.
- [x] Queue the full run + dependent plot suite (SLURM, outside the sandbox).
- [x] After completion: check the rejection-reason histogram, sigma usage
      (did the sigma-2 cheat die?), per-pass event counts, and per-channel
      fractions.
- [x] Replace the recording-wide pass floor with per-chunk exhaustion skipping
      so the recording always runs to natural completion.
- [x] Confirm the 5% resume finished the exhaustion way: `16679719` ended
      `all_passes_complete` with pass 2 at the 0.25 bar accepting 1 event.
- [x] Verify the resubmitted 5%/10% plot suites (`16722284`, `16722285`)
      produced the replay panels in their galleries. Both completed exit 0
      (4:26 / 5:17) with 17 panels each, including `recording_replay_chunk0.png`
      (5%: pass events [542, 1, 0], RMS 100 → 96.3%; 10%: [456, 0],
      100 → 96.8%). They lack the full-recording and chunk-20 figures, which is
      what the superseding three-column chain delivers.
- [ ] Full-run sigma mix across the fraction sweep, and the 5% vs 20%
      acceptance-quality trade.
- [ ] Near-surface share of accepted sigma-2 events (0018's 57%-within-3 µm
      diagnostic) on both runs.
- [x] Record the cross-fit results of `16685303` (my fitter on his 2.31M
      spikes) into the cross-comparison section above.

## Links

- [[session-018-bipolar-prototype-cone-peeling]]
- [[session-016-one-hot-lattice-peeling]]
- [[session-017-initial-threshold-spike-discovery]]
- [[feedback_plot_suite_completeness]]
