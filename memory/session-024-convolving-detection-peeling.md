# Convolving Detection Peeling (0024)
**Created:** 2026-09-01
**Last updated:** 2026-09-02
**Status:** Full runs in flight — four single-variable jobs `16773522–25` all RUNNING; no empirical null (user decision, see below); results pending

## Threshold decision: no empirical null (2026-09-02)

The plan had mandated calibrating `--proposal-threshold` on an empirical
spike-free surrogate before any full run. Before building that, the IBL
reference was audited for how it calibrates thresholds, and the answer is:
it doesn't. The explore audit of `/scratch/ap7151/_REFERENCE/ibl-sorter`
found no empirical-null, false-positive-rate, or spike-free-score-tail
machinery anywhere: whitening covariance comes from 25 fixed 0.4 s windows
that freely contain spikes (`preprocess.py:139-168`); `Th = [6, 3]` is a
hard-coded Kilosort1 convention (`params.py:156`, "like in Kilosort1");
`lam = 10` plus the `max(0, ·)` clamp is an analytic no-spike decision (the
score is exactly zero for non-positive projections, `mexMPnu8.cu:286-288`),
but the numeric threshold on top is convention. The only data-driven
threshold selection is post-hoc and cluster-level: `set_cutoff`
(`postprocess.py:1148-1184`) sweeps each cluster's amplitude cutoff from 6
down to 3 and stops when the cluster's autocorrelogram shows refractory
contamination above 10% (5% for initially clean units) — output-side
spike-timing evidence, not a voltage-domain null.

The user's decision: skip the null entirely and run the four full
configurations at IBL's convention threshold 6.0, judging quality on the
output side — the all-channel bar (which already rejects 71–88.5% of
proposals in 0019), the rejection audit, and a recording-wide
autocorrelogram of accepted events as the IBL-inspired contamination check
(excess short-lag mass would expose noise re-detections without clusters).

## Full runs queued (2026-09-02)

All four launch configurations submitted single-variable against the 0019
20%-bar baseline (`--all-channel-min-fraction 0.2 --pass-fraction-step 0.1`,
detection threshold 5, fitted projection 8, 3 passes × 1 round, min-channel
rule):

| config | job | output |
| --- | --- | --- |
| perchannel, lockout 5 (017-parity control) | `16773522` | `024_convolving_perchannel_lockout5` |
| perchannel, lockout 60 (IBL lockout) | `16773523` | `024_convolving_perchannel_lockout60` |
| gaussian 30 µm, lockout 60 (IBL sigmaMask) | `16773524` | `024_convolving_gaussian30_lockout60` |
| growsum, lockout 60 (IBL adaptive extent) | `16773525` | `024_convolving_growsum_lockout60` |

**Codebook-size extension (2026-09-02).** Twelve more runs queued, the same
four detector configurations at `--q` 16/32/64: jobs `16780896–98`
(perchannel lockout5), `16780899–901` (perchannel lockout60), `16780902–04`
(gaussian30), `16780905–07` (growsum), sbatches
`024_convolving_<config>_q<16,32,64>.sbatch`, outputs
`024_convolving_<config>_q<N>`. Q is a dataclass field (`config.q`, default
8), so the flag needed no code changes — calibration assigns the extra
atoms alternately to the two polarity cones and the matched filter just
adds rows. Two expectations: at fixed threshold 6.0, more rows raise the
max-over-rows tail (007 saw candidates grow with Q at fixed thresholds),
and fit/filter cost scales roughly linearly in Q — lockout5 + Q64 is the
slowest run in the family. The user separately queued a 0019-detector Q
sweep for matched-detector controls (`16780702` base-q16 plus
`16780929–32`, with `16780933–42` pending).

## First full-run results (2026-09-02, ~5h in)

- **gaussian30 Q8 `16773524` COMPLETED** (5h18m57s, exit 0:0,
  `all_passes_complete`): **618,630 events** — pass 0 at the 0.2 bar
  accepted 600,005 from 13.97M rejected, pass 1 (0.3) accepted 18,570 from
  14.31M, pass 2 (0.4) accepted 55 with 1,903 chunks exhausted.
- **growsum Q8 `16773525` COMPLETED** (4h41m35s, exit 0:0,
  `all_passes_complete`): **746,961 events** — 728,433 / 18,490 / 38 by
  pass, 1,920 exhausted in pass 2.
- Against the 0019 20% run (568,888 total; pass 1 accepted 452): the
  merged configs accept +8.7% (gaussian) and +31.3% (growsum) more events,
  and the pass-1 duplicate wall collapses from 452 accepted to ~18.5k —
  matched-filter proposals leave a residual in which later passes still
  find real fits. Pass-2 near-nothing and the exhaustion accounting match
  the 0019 pattern.
- gaussian30 Q16 `16780898` was SIGTERMed at 2h16m (chunk 936/1958 of
  pass 0) and left CANCELLED — no auto-requeue; resubmitted as
  **`16783761`** with `--resume`, picking up at chunk 936.
- Still running (all clean stderr): perchannel60 Q8 `16773523` in pass 2 at
  1633/1958; perchannel5 Q8 `16773522` at 1261/1958; growsum Q16
  `16780899` at 1499/1958; gaussian30 Q32 `16780902` at 912/1958;
  perchannel60 Q16/Q32 `16780897`/`16780901` at 1052/611; perchannel5
  Q16/Q32 `16780896`/`16780900` at 216/95 — the latter accepting ~776 and
  ~1,550 events/chunk, the strongest look-elsewhere signal so far
  (lockout5 × high Q at fixed threshold 6.0). Pending: all four Q64s
  (`16780903–907`) and the user's remaining 0019 Q-sweep controls
  (`16780933–42`).

## Quota-kill incident and resubmission (2026-09-02 evening)

An unrelated download job blew through the user's scratch quota, and every
job that tried to write died with exit 1:0 — all thirteen active 024 runs
FAILED (the two completed Q8 runs were untouched). Scratch was verified
writable again and all thirteen resubmitted as **`16794200–12`**:
`16794200` perchannel-lockout5 Q8, `16794201–03` the lockout5/lockout60/
gaussian Q16s, `16794204–06` the lockout5/lockout60/gaussian Q32s,
`16794207–12` growsum Q32 plus the four Q64s. Each sbatch carries
`--resume`, so the killed runs pick up from their checkpointed chunks
(lockout5 Q8 at pass-0 chunk ~1261, perchannel60 Q8 in pass 2, gaussian30
Q16 at 936, the other Q16/Q32 variants at their last chunks); the four
Q64s start fresh, having died before their first checkpoint. All PENDING
on Priority at submission. The user's own 0019 Q-sweep controls were not
touched (their jobs, their resubmission).

(lockout5 × high Q at fixed threshold 6.0). Pending: all four Q64s
  (`16780903–907`) and the user's remaining 0019 Q-sweep controls
  (`16780933–42`).

Two started immediately on gl028/gl030; gaussian30 and growsum queued behind
them on `QOSGrpGRES`. Plot suites deliberately not queued until runs land
and [[feedback_plot_suite_completeness]] applies.

Progress at ~2h (2026-09-02): all four healthy. gaussian30 and growsum
finished pass 0 and are in pass 1 (124/1958 and 398/1958); perchannel60 at
1684/1958 of pass 0; perchannel/lockout5 trails at 451/1958 because it
keeps ~50,000 proposals per chunk (the ±5-sample lockout cannot absorb the
filter response's temporal ringing) versus 7,400–10,800 at lockout 60, so
it runs ~14 s/chunk versus ~2–3 s and needs ~10–12 h total. The
duplicate-merging numbers from the logs are the design working: raw
matched-filter crossings per chunk run 0.7–3.7M, collapsing to 7–11k
proposals after merge plus NMS (~280× for gaussian30, ~500× for growsum).
No cap enforcement, no errors. A method document was written at
`docs/0024_convolving_detection.md` in the style of `0019_end_to_end.md`
(matched-filter and merge-layer linear algebra included).

## Validation (2026-09-02)

Self-test sbatch `024_convolving_selftest.sbatch` (py_compile → CPU self-test →
convolving CUDA smoke → bipolar parity smoke, 10 s each at the 0019 flag set).

- First submission `16764952` failed at 44 s: the self-test passed `"rec.bin"`
  as str where 0014's `output_metadata` needs a Path. Fixed (3 call sites).
- Second submission `16765590` "passed" but was a false positive: 0019's
  `pursue` calls the bare name `process_chunk` inside its own module
  namespace, so 024's patched `process_chunk` (the convolving router) was dead
  code in the pursue path — both smokes ran the 018 detector end to end and
  differed only through calibration Omega drift (max |ΔΩ| 0.0058); audit bit
  256 never appeared. The event-count difference (3403 vs 3407) was noise,
  not routing. Lesson: patching `PIPELINE.X` is not enough when the base
  module's functions call `X` by bare name — rebind on ALLCHANNEL too, and
  validate a detector swap by its audit-bit footprint, not by exit code.
- Fixes applied: `ALLCHANNEL.process_chunk`/`ALLCHANNEL.detect_events` are
  now rebound in `main()` (with `_0019_detect_events` captured at import so
  the bipolar branch cannot recurse into the router); `reasons` starts at
  zero and bit 256 is OR-ed in only at save time (`logged_reasons`), so
  `logged = reasons != 0` cannot log accepted fits as rejections.
- Validated submission `16772092` (7m14s, exit 0:0): CPU self-test passed;
  convolving growsum smoke accepted 4,656 events in 10 s vs bipolar 3,402,
  only 924 shared — the detector set genuinely differs; rejection audit now
  carries bit 256 (min value 258), 221,068 rejected proposals logged; bipolar
  smoke reproduced 0019 behavior. The invalid smoke outputs from `16765590`
  were deleted before the rerun.

## Why revisit the convolving detector

Session 002's detector nominated spikes by convolving every channel with the
temporal codebook rows; 017 replaced it with plain thresholding after 014's
score-8 run accepted 37.8M events. That verdict fused the proposal generator
with the acceptance model of its day: the over-detection came from maximized
template scores with no calibrated no-spike decision, judged by a
captured-fraction gate later proven meaningless (008/009). Since then the
acceptance side was rebuilt twice (016's coherent per-channel minimax fit,
0019's all-channel bar), and 0019's rejection audit shows the all-channel bar
alone kills 71–88.5% of proposals depending on the bar. Under the current
architecture the detector only nominates (t, c) candidates — the fit and the
all-channel bar decide — so a more sensitive proposal generator cannot
reproduce the 37.8M pathology at acceptance; it can only spend compute.

The scientific motivation is 0019's cross-comparison with the SLT collision
repo: my threshold detector never even proposes many of his sites, and his
spikes are weaker and less channel-concentrated than mine (my fitter on all
2.31M of his spikes, job `16685303`: captured fraction 0.421, 23.8% passing
the 20% bar). A noise-normalized temporal matched filter integrates evidence
over the full 90-sample window, so it should nominate weak, correctly-shaped
spikes that never produce a single-sample voltage extremum on any channel.

## Hypothesis

Convolving proposals + 0019's all-channel acceptance recover additional
genuine spikes relative to the 018-detector-based 0019 runs at the same
acceptance bar, without re-opening over-detection — measured as newly
accepted events that look spike-like under 015's boundary criteria (centered
temporal structure, coherent multi-channel footprint, post-fit channel
residuals consistent with noise), not as raw event count.

## IBL reference findings (what iblsorter actually does)

From `/scratch/ap7151/_REFERENCE/ibl-sorter` (explored 2026-09-01). The
governing lesson: IBL merges duplicate channel detections *inside the score*,
with two explicit dedup passes around it — not by per-channel thresholding.

1. **The detection score already merges channels.** Seed detection
   (`mexGetSpikes2.cu:3-38` `sumChannels`, called from `learn.py:904` with
   `Nsum = min(Nchan, 7)`): for each channel, a growing sum over its nearest
   channels trying every prefix size j, keeping the max of the
   count-normalized energy `Cf*Cf/(1+j)` — the filter picks the best spatial
   extent before thresholding. The template pursuit score is per-template
   over 32 nearest channels (`mexMPnu8.cu:47-49`), and `bestFilter`
   (`mexMPnu8.cu:294-305`) keeps exactly one argmax over templates per time
   point, so one spike yields one winner and there is no per-channel
   threshold left to over-fire.
2. **Temporal lockout on the merged score** (`cleanup_spikes`,
   `mexMPnu8.cu:367-390`, `nt0 = 61` from `params.py:190`): a crossing
   survives only if no sample within ±(nt0−1) = ±60 samples is strictly
   greater — the full template support, earliest max winning ties by
   construction. Same logic at seed detection (`mexGetSpikes2.cu:116-143`).
3. **Cross-channel dedup** (`cleanup_heights`, `mexGetSpikes2.cu:158-200`):
   a peak is dropped if a peak within template-id distance 5 (≈100 µm on
   NP1.0/2.0) has larger amplitude. Index-based heuristic — any port must
   use µm distances from the channel map instead.
4. **Pursuit hygiene** (`mexMPnu8`): the winning template's whole spatial
   footprint is subtracted (`mexMPnu8.cu:621`) so the same spike cannot
   re-fire; only `UtU`-overlapping templates are re-scored; 60 iterations
   (`learn.py:895` → `Params[3]`); thresholds Th = [6, 3] learn/final tested
   as `err > Th*Th` in whitened units (`params.py:156`, `learn.py:910,
   1161-1163`); `lam = 10` amplitude penalty with a `max(0,·)` clamp inside
   the score (`mexMPnu8.cu:285-288`); `maxFR` per-batch caps as a hard
   over-detection backstop.
5. **Post-detection dedup is cluster-level only** (`find_merges` with
   refractory veto `postprocess.py:682-772`; `set_cutoff`
   `postprocess.py:1116-1232`); there is no per-spike duplicate-time removal
   anywhere.

Caveats: `cleanup_heights`' d<5 assumes spatially ordered channel ids;
`spikedetector3` (5-scale spatial detector) is dead code in this tree; the
drift-stage DARTsort detector uses a 100 µm neighbor index with a 75 µm
spatial-dedup index (`datashift2.py:474-493`), confirming the ~75–100 µm
merge scale.

## Design — four detector configurations as argparse dials

Derive `residuals/src/preprocessing/024_convolving_detection_peeling.py`
from `0019_allchannel_peeling.py`, which already carries 023's acceptance
dials (`all_channel_rule`, `all_channel_required_share`,
`all_channel_acceptance()` at line 350) — all of them stay exposed so the
acceptance rule can be tuned later without touching 024 again. Everything
downstream of proposals is unchanged: 0016 coherent xyz-sigma fit, 023
acceptance dials, all-channel bar, pass replay, chunk exhaustion, rejection
audit.

- `--detector {bipolar,convolving}` — `bipolar` routes to the existing 018
  detector and must reproduce 0019 exactly (default; the new convolving-only
  arguments are rejected by validation in bipolar mode).
- Convolving proposal path (GPU-dense per chunk, torch, 0019 style):
  1. Dense temporal matched filter of the current residual against all Ω
     rows: `s[c,t,q] = ⟨r[c], Ω_q⟩ / (σ_c · ‖Ω_q‖)` in one batched conv1d,
     σ_c the per-channel robust noise; `S[c,t] = max_q |s|` with the signed
     value kept (0018 bipolar rows, both polarities compete).
  2. `--score-merge {perchannel,gaussian,growsum}` — the duplicate-merging
     layer (the key over-detection control):
     - `perchannel`: S as computed (no spatial merge).
     - `gaussian`: fixed-kernel spatial merge over the 48 µm neighborhood,
       `M[c,t] = Σ_{c'} w(c,c') S[c',t] / sqrt(Σ w²)` with
       `w = exp(-d²/2σ²)`, `--gaussian-sigma-um` default 30 (IBL's
       `sigmaMask`, `params.py:207`).
     - `growsum`: IBL's growing prefix sum over the `--growsum-channels`
       default 7 nearest channels by distance, keeping the best prefix size
       under count normalization `M/√j` (the `Cf²/(1+j)` analog).
  3. Spatiotemporal NMS on the merged score: local max within the 48 µm
     neighborhood and `--proposal-lockout` samples (default 60 = IBL's
     full-support lockout; 5 = 017 parity), strict `>` comparisons so the
     earliest max wins.
  4. `--proposal-threshold` default 6.0 on the merged noise-normalized score
     (IBL's learn-phase Th); to be replaced by the empirical-null
     calibration before any full run. `--max-proposals-per-pass` cap as the
     maxFR-analog backstop.
  5. Proposals carry no temporal or sigma prior (indices −1, the 017
     convention); the fit searches the full Q × sigma × lattice product.
- The four launch configurations (`024_*.sbatch`, all else identical to a
  matched 0019 run):
  1. `perchannel` + lockout 5 — the 017-parity control.
  2. `perchannel` + lockout 60 — isolates the IBL lockout.
  3. `gaussian` + lockout 60 — IBL sigmaMask-style intrinsic merge.
  4. `growsum` + lockout 60 — IBL seed-detector-style adaptive extent.

## Known failure modes from history, with guards

- Look-elsewhere over Q rows (014/016 diagnosis): accepted at the 6.0
  convention threshold with no null calibration; the output-side gates
  (all-channel bar, rejection audit, ACG check) carry the burden instead —
  watch the per-pass decay and the ACG short-lag mass closely.
- Stale-residual batch claims (016's diagnosis): 0019's conflict-free winners
  per round plus post-subtraction rescoring already prevent it.
- Over-proposal compute: acceptance lives in the all-channel bar, not the
  proposal score; log proposals per round and enforce the per-pass cap.
- Duplicate re-detection after subtraction: 0019's duplicate prior and
  chunk-exhaustion skipping cover it; watch rejection reasons 64/192.
- 002's original smoke failures (threshold calibration, repeated-pass RMS
  regression, integer-only localization) are each addressed by the modern
  stack: null calibration, 004's monotone gain refit, and 016/0019's
  discrete-plus-refit pipeline respectively.

## Falsifiable diagnostics (the run is only worth it if)

- Event-set diff versus the matched 0019 run (same bar, same threshold
  regime): events 024 accepts that 0019 never proposes — the target
  population — reviewed as reconstruction examples, not counts.
- Sigma usage: does matched-filter nomination relax the sigma = 2 µm pile
  (019: 39.8% of accepted events still at the lower bound)?
- Recording-wide autocorrelogram of accepted events (the IBL set_cutoff
  idea, cluster-free): excess mass at short lags means noise re-detections
  or double-hits are getting through.
- Per-pass decay, exhaustion, and rollback behavior unchanged versus 0019.

## Sequencing

1. [x] Implement the 024 module + four sbatches (files prefixed `024_`),
   following 0019's config/self-test structure; CPU self-tests cover the
   planted-weak-spike nomination, NMS/lockout geometry, merge math versus
   brute force, and bipolar-mode equivalence guards.
2. [x] Singularity validation outside the agent sandbox (imports, self-tests,
   short CUDA smokes) — job `16772092`, all green.
3. [x] Threshold decided without a null: run at proposal threshold 6.0
   (IBL convention), output-side quality judgment (user decision above).
4. [x] Full runs `16773522–25` land: ACG contamination check, event-set
   diff versus 0019, sigma-2 pile, rejection-reason histograms. Plot
   suites then, per [[feedback_plot_suite_completeness]]: every Omega
   waveform with usage, plus explicit disclosure of any exact panels not
   producible.
5. [ ] Plot job queue (submit per run after its `summary.json` exists;
   never `afterok` on requeueable runs — cancel-and-resubmit if needed):

   - **Per-run 024 suites** (16 runs): derive the 0019 suite sbatch
     (`0019_allchannel_plots.sbatch` pattern) per config, pointed at each
     `024_convolving_*` output dir. Must include, per
     [[feedback_plot_suite_completeness]]: every Omega waveform with
     recording-wide and per-pass usage (`plot_0014_codebook_usage.py` —
     Q>8 runs show all rows, not the first 8), localizations,
     reconstruction examples across passes and near the score boundary,
     the SpikeTensor-style browser via `build_plot_gallery.py`, plus the
     three recording replays (chunk 0, most-subtractive chunk,
     full-recording three-column).
   - **`temporal_codebook_usage.png` across Q** (the 12 Q-variant runs +
     the 0019 Q-sweep controls): the codebook-usage panel becomes the
     Q-scaling evidence — which atoms the extra rows carve out, and
     whether late rows are pure noise-fitting (all-rows histogram, per-pass
     split).
   - **Event-set diff figure, 024 vs 0019** (new script): for each of the
     four base configs, accepted events that fall outside 0019's
     proposal set (same channel, ±0.5 ms) — the target population —
     rendered as reconstruction-example pages, not counts.
   - **Recording-wide ACG contamination panel** (new script): all accepted
     events per run, 1 ms bins out to ±0.5 s; excess short-lag mass
     = noise re-detections. One figure, all 16 runs side by side plus
     the 0019 20% baseline as reference line.
   - **Sigma-usage bar chart across the 16 runs + 0019 baseline**: does
     matched-filter nomination (and higher Q) relax the σ = 2 µm pile?
     Include the near-surface share of σ-2 events (0018's diagnostic).
   - **Rejection-reason stacked bars per pass, all runs**: bit-256-tagged
     convolving-specific rejections separated from the 0019 reason codes;
     watch reason 16 (all-channel bar) dominance and the duplicate wall
     (64/192) across Q.

## Links

- [[session-002-template-residual]]
- [[session-017-initial-threshold-spike-discovery]]
- [[session-019-all-channel-error]]
- [[session-023-acceptance-rule-variants]]
- [[session-016-one-hot-lattice-peeling]]
- [[session-015-score-calibrated-xyzsigma-promotion]]
- [[feedback_plot_suite_completeness]]
- [[project_overview]]
