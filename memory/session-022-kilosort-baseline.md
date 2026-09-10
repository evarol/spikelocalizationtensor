# Kilosort Baseline Run (iblsorter)

**Created:** 2026-09-01
**Last updated:** 2026-09-01

## Context

DATASET1_P1 now has a kilosort ground-truth-ish baseline: IBL's pykilosort 2.5 port
(package `ibl-sorter`, import name `iblsorter`) run on the full 1957 s recording, so
future residual-pursuit results can be compared against a standard sorter. The
`ibl-sorter` library remains forbidden inside the residual pipeline itself (see
[[user_me]]); this is a standalone baseline run through its own overlay, which is the
sanctioned way to touch it.

## How it runs

The ibl-sorter overlay is `/scratch/ap7151/_ENVS/ibl-sorter.ext3` (package
`ibl-sorter 1.13.0`, plus torch 2.11+cu128, cupy-cuda12x, spikeinterface). CUDA only
materializes on GPU-allocated nodes — `torch.cuda.is_available()` is False on login
nodes even with `--nv`. The entry point is `iblsorter.ibl.run_spike_sorting_ibl`,
which builds IBL-default parameters via `ibl_pykilosort_params(bin_file)`; probe
geometry (NP1.4, 385 saved channels) is auto-loaded from the SpikeGLX meta, so no
channel map files are needed. Pipeline: IBL destriping → whitening → drift
correction → Kilosort 2.5 clustering → merges/splits/cutoff → phy + ALF conversion.
`extract_waveforms` stays off — dense waveforms would be ~500 GB.

- Driver: `residuals/src/preprocessing/run_kilosort_iblsorter.py` (optional
  `--stop-after <stage>` for smoke runs)
- Sbatch: `residuals/src/preprocessing/run_kilosort_iblsorter.sbatch`
  (l40s_public, 8 CPU/64G/1 GPU, 24 h, USR1 requeue trap, commit `134da47`)
- Outputs: `residuals/runs/dataset1_p1/kilosort_iblsorter/` with `iblsorter/`
  (raw pykilosort), `alf/` (ALF spikes/clusters), `scratch/` (temp, deleted on
  success), `kilosort_summary.json`, `.complete` marker. Delete the run dir to rerun.

## Why requeue-resume works

`iblsorter.main.run` guards every stage with `if "<stage>" not in ctx.timer.keys()`,
persisted in the `.kilosort` context under `scratch/`. The 43 GB destriped
`proc.dat` lives in the same context, and `decompress_destripe_cbin` truncates and
rewrites it from scratch on restart, so an interrupted preprocess is safe. Because
scratch lives inside the fixed run dir (no job id in the name), a USR1
kill-and-requeue resumes from the last completed stage. This mirrors the trap
pattern of the 0019 sbatch.

## Run history

First submission `16735505` ran 20:39 (whitening 23 s, destriping 954 s, died early
in `drift_correction`) and was killed by an outside agent's `scancel`, reason
`QOSGrpGRES` — the account's group GPU cap. Resubmitted clean as `16740477`, which
COMPLETED on gl066 in 1:06:41 (exit 0:0, commit `134da47`).

## Results

`kilosort_summary.json`: 6,180,912 spikes over the full 1957 s (about 3,158 spikes/s
across the probe), 839 clusters — 149 labeled good, 690 mua, 0 noise by KSLabel.
ALF and phy outputs are complete, including per-cluster sparse waveforms,
`templates.waveforms`, drift traces, and the QC pngs. The 43 GB scratch context was
cleaned up automatically; the run dir is ~a few GB.

## Plot suite

`residuals/src/plots/kilosort_baseline_plots.py` (plus a `cpu_short` sbatch wrapper)
renders nine dpi=800 panels into `residuals/out/0022_kilosort_baseline/` and writes
its own SpikeTensor-style `index.html` (the shared `build_plot_gallery.py` hardcodes
a residual-pursuit header, which would be false for a standard sorter). Panels:
depth×time raster, four 20 s raster windows, firing rates, cluster depths and
trough-to-peak widths, a 12-unit template gallery drawn on probe geometry
(time vertical, waveform horizontal — the first attempt drew flat lines by plotting
waveform against constant depth), drift estimates, amplitudes in µV, KSLabel and
ContamPct quality summary, and a distribution-level cross-comparison against the
0019 20%-bar run (rate over time and depth density — kilosort keeps 6.18M spikes
versus 568,889 accepted; 0019's rejected proposals run 100–120k per 50 s).
`build_out_index.py` picks the gallery up as family 0022 automatically.

## Overlap census: post-0018 runs vs the Kilosort spike times (2026-09-09)

Census script `/state/partition1/job-17271501/opencode/ks_overlap_census.py`
(+ `ks_good_time_recall.py`), results in
`residuals/runs/dataset1_p1/kilosort_overlap_census/{summary,good_time_recall}.json`.
Matching is per-channel and time-windowed at ±0.5 ms (sample units, 30 kHz),
the same convention as the 019/SLT cross-fit; a second time-only pass drops
the channel condition entirely. KS `spikes.amps` and `clusters.amps` are all
zero in this pykilosort output, so no amplitude stratification is possible —
the good/mua cluster split is the only quality axis.

Reference set: 6,180,912 KS spikes, of which 1,402,613 sit on the 149
good clusters.

**No over-detection.** Time-only precision is 0.98–1.00 for every run —
essentially every accepted event (pass 0 and pass 1+ alike, every rule,
every Q, both detector families) coincides with some KS spike time within
±0.5 ms. Chance level is ~10% (105 KS spikes/s × 1 ms window), so this is
a real alignment, and it holds even for the 2.4M-event perchannel5-q32 run
(0.977). The all-channel bar is filtering noise effectively.

**Under-detection at the 0019-default settings** (the 20% bar is 0019's founding default and the sweeps' baseline, never a decided production recipe — the census points at mean f₀ 0.05 as the closest-to-Kilosort operating point). Time-only recall of KS-good
spikes: mean-rule runs 0.86–0.94 (mean f₀ 0.05 = 0.939), kofn 0.73–0.90,
min/flat at f₀ 0.05–0.10 ≈ 0.64–0.76, the 0019-default 20% bar 0.38–0.57
across Q, strictest runs 0.38. So the 0019-lineage 20% bar leaves ~half of
KS-good spike times unpeeled; only mean f₀ ≤ 0.10 approaches full coverage.
Recall against ALL KS spikes (including mua noise) is 0.29–0.82, and the
same-channel numbers are much lower still (recall 0.09–0.42, precision
0.44–0.48) — but time-only precision ≈ 1.0 while same-channel precision is
0.45, which means our detection channel rarely equals KS's cluster peak
channel: the same-channel deficit is channel-assignment bookkeeping, not
missing detection.

**Pass-1+ events are not junk by this measure** — they match KS times at
0.98+ like pass-0 events, but they are so few (the duplicate wall) that
they add almost no coverage. The residual-under-detection levers, in
order: the acceptance bar (f₀ sweep maps it directly), the 5σ proposal
floor (caps what is ever nominated), and the duplicate wall (caps pass-1+
recovery — 028's phase-2 leftover-vs-event discriminator is the designed
fix). Caveats: KS-good is not ground truth, and time-coincidence tolerates
±0.5 ms and any channel, so "recall" here bounds event-time coverage, not
1:1 spike correspondence.

**Rejection census (same day, `rejection_census.json`).** Matching the
consolidated `rejected_*` tables against KS times splits under-detection
into proposal-side vs acceptance-side, per run (scripts
`ks_rejection_census.py`/`ks_overlap_census.py`/`ks_good_time_recall.py`
live in the census dir; compute nodes cannot see `/state/…/job-*/` scratch,
so sbatch-run scripts must live in the repo).

- **The detector nominates essentially all of Kilosort's good spikes.**
  KS-good coverage by proposals (accepted ∪ rejected) is 0.945–0.948 for
  every 0019-lineage run and 0.999–1.000 for every convolving run —
  under-detection is NOT a detection problem.
- **The acceptance bar is the whole story in the 0019 lineage.** The same
  ~95% proposal coverage converts to 0.38–0.94 accepted depending on
  rule/bar; the bar's cost ("proposed but rejected") runs 0.56 at the
  strict f₀ 0.30 bar down to 0.009 at mean f₀ 0.05.
- **99.2–99.3% of 0019-lineage rejections sit at KS-active times** — the
  bar is killing real-spike proposals, not noise. Among those real-spike
  rejections at the 20% bar, the all-channel bit is present in 85%
  (pass 0: 99.2%), while at mean f₀ 0.05 pass 0 rejections collapse to 65k
  and pass 1+ rejections are 96% duplicate-flagged — at loose settings the
  duplicate wall, not the bar, is the residual under-detection mechanism.
  kofn f₀ 0.20 splits ~48/52 between the bar and the wall.
- **024 convolving rejections are a different beast**: 38–410M per run
  (the detector nominates far more), 91% at KS-active times, and 99%
  killed by the all-channel bar (bit 256 marks convolving proposals). Its
  proposal coverage of KS-good is 1.000 and accepted coverage 0.46–0.78 —
  the bar turns its extra sensitivity into a rejection flood rather than
  yield.

## Links

- [[session-009-ibl-style-pursuit]] — where the ibl-sorter reference source lives and how its preprocessing inspired 009/010
- [[project_overview]] — dataset and pipeline context
