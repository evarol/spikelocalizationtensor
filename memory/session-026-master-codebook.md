# Master Codebook from All IBL Mice (0026)
**Created:** 2026-09-04
**Last updated:** 2026-09-04
**Status:** harvest COMPLETE (808/812 shards + `harvest_summary.json`/`skipped_clips.json` in `residuals/runs/ibl_bwm/master_codebook_q32/`); init crashed on a 0019 device bug (fixed 2026-09-08, see below) before any Omega was written, so no codebook exists yet — resubmitted as `17222215`. Outputs, once the fit lands: `omega.npy`, `initial_omega.npy`, prototypes/assignment/history, and `omega_source.json` in that directory, with an atomic checkpoint per fit iteration.

## Why

Every run in the 0018/0019/0024/0025 lineage calibrates its own temporal codebook
Omega from a 100k-event pool drawn from the single recording being peeled. That
makes each run's detector depend on its own recording's calibration and burns
compute on calibration every time. The user's decision: learn one master codebook
(Q = 32, two bipolar prototype cones, same shape family as 0018+) from the pooled
event waveforms of **all 812 public IBL Brain-Wide-Map probe recordings**, then
reuse it across recordings (and species transfers, alongside the 0025 primate
work).

## The data (verified 2026-09-04)

`/scratch/ap7151/_RAW_DATA/ibl-data/` — first ~100 s AP-band clips for every
public IBL BWM probe insertion, kept in the original mtscomp SpikeGLX triplet
format: `<stem>.ap.cbin` + `.ap.ch` + `.ap.meta` (plus a `.clipok` marker).
Downloaded via openalyx 2026-09-01/02; full recordings were verified then
deleted, only clips kept.

- **812 probe recordings** across 525 sessions, 156 subjects, 12 labs
  (per-lab counts in the README: angelaki 79, cortexlab 83, churchland 62,
  churchland_ucla 52, ...).
- Path layout: `ibl-data/<lab>/<subject>/<YYYY-MM-DD>_<eid8>/<probe>/_spikeglx_ephysData_g0_t0.imec{0,1}.ap.cbin`
- raw int16, 384 AP channels + 1 sync column, 29.9–30.0 kHz, 81.2M samples
  total (~22.5 h of ephys); clips are 99.0–100.0 s (whole recording if shorter).
- The `.meta` files were patched for the clip (fileTimeSecs/fileSizeBytes
  describe the clip, fileSHA1 removed); the `.ch` header carries the clip's
  chunk table with `"chopped": true`.
- Loading: `spikeinterface.extractors.read_cbin_ibl(<probe dir>)` — needs
  `mtscomp`, no ibllib. Returns geometry, uV gains, and inter-sample shifts
  from the meta; sync dropped unless `load_sync_channel=True`. Verified: fs 30k,
  location (384, 2) µm, raw int16 counts via `get_traces(return_in_uV=False)`,
  ~1.8 s to read 10 s of data.

## Implementation

`residuals/src/preprocessing/026_master_codebook.py` + `026_master_codebook.sbatch`.
Stages: `harvest` (per clip: read_cbin_ibl → preprocess → 0019's
locally-exclusive peaks at threshold 5, both polarities, 1 ms sweep + isolation →
48 µm neighborhood waveforms → one atomic npz shard per clip, self-contained with
its own anchor-relative offsets and mask), `init` (pool all shards, peak-align +
normalize, then 0019's `initialize_codebook` — Q atoms alternate between the two
polarity cones), `fit` (0019's alternating loop verbatim: fixed-Omega
`fit_grouped` accumulation, `prototype_cone_proposal`, `backtracked_update`,
atomic checkpoint per iteration). Config inherits 0019's full production Config
chain (threshold 5, lattice 16³, sigma bank 2–512, cone 35°).

Design points that matter:
- Waveform window fixed in **samples** (2 × 45 at a 30 kHz reference) so every
  clip and the final Omega share T=90 despite the 29.9–30.0 kHz spread.
- Each event carries its own probe's anchor-relative offsets, and batches never
  mix shards, so NP1.0 (width ~19 neighborhoods) and NP2.0 (width 8) pool
  without rescaling. NP2.0's 48 µm neighborhood is exactly 8 channels.
- Harvest runs 8 spawn-worker processes in parallel (~3.6 clips/min, full pass
  ≈ 4 h); each worker re-imports the 0019 chain. Resume skips clips whose shard
  exists; shard summaries can be reconstructed from disk if the summary JSON
  lags behind (this recovered ~4.5 h of work after the first OOM kill).
- Reused 0019 via importlib; two namespace traps: `FootprintCache` lives on
  `PIPELINE.OLD` (the 0014 module), not on 0016.

## Run history

- `16956537` OOM-killed at 11 min: 8 workers × (torch + CUDA context + scipy
  intermediates) exceeded the 64G cgroup. Fix: `--mem=120G`, and
  OMP/MKL_NUM_THREADS=2 (each spawned worker was spawning 16 BLAS threads).
- `16963532` harvested 808/812 in 13 min, then died on a worker AssertionError:
  the four **NR_0029** clips are four-shank probes saved with a 96-channel
  subset (`snsSaveChanSubset=0:96`), and probeinterface cannot attach the full
  -shank geometry to a 96-channel recording (`set_channel_gains` size
  mismatch). Decision: skip them with a logged reason (0.5% of data) — harvest
  failures are now soft and recorded in `skipped_clips.json`.
- `16984979` accidentally submitted while a compile error was still in place —
  cancelled immediately (lesson: `&&` then `;` still runs the second command).
- `16985008` sat pending on `QOSMaxGRESPerUser` under `torch_pr_60_general`
  (the two 024 stragglers hold the account's GPU quota); `scontrol update
  Account=` is rejected on this cluster, so the move was done by scancel +
  resubmit with `sbatch --account=torch_pr_62_general` — command-line account
  override, sbatch file untouched.
- `16986952` is the live run (torch_pr_62_general, l40s_public).

- `16986952` FAILED after 2:33 (2026-09-08 discovery): it reached `run_init`
  cleanly (harvest resume skipped all existing shards within minutes) and
  crashed on the first k-means draw inside 0019's `spherical_kmeans` —
  `torch.randint(len(values), (1,), generator=generator)` builds its output
  on CPU by default, but the generator is created on `values.device`, and
  026 is the first caller that hands `initialize_codebook` CUDA tensors
  (0019's own calibration keeps the pool on CPU). RuntimeError: "Expected a
  'cpu' device type for generator but found 'cuda'". Fix: the randint call
  now passes `device=values.device`, which is byte-identical behavior on
  the CPU path (same generator, same draw) and lets the CUDA path work —
  so all completed runs stay comparable. py_compile plus a CPU
  spherical_kmeans round-trip pass in the pytorch overlay. Resubmitted as
  `17222215` (`sbatch --account=torch_pr_62_general`; the sbatch file
  itself is untouched and already carries the `USR1 TERM` trap, so the
  utilization-policy kills can't strand it). Pending on Priority at
  submission. The init stage never wrote `initial_omega.npy`, so resume
  will redo init from the pooled shards — the expensive part (harvest,
  ~808 clips) is all on disk.

## Links

- [[session-018-bipolar-prototype-cone-peeling]] — the prototype-cone codebook shape this inherits
- [[session-019-all-channel-error]] — calibration detect + alternating fit being reused
- [[session-024-convolving-detection-peeling]] — Q sweep context; Q16→Q32 saturation observation
- [[session-0025-primate-nwb-peeling]] — the other out-of-family transfer, and the importlib/singularity conventions
- [[session-013-rho-localization-optimization-plan]] — run-fingerprint/resume consistency concern
- [[project_overview]]
