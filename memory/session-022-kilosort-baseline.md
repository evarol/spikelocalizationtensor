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

## Links

- [[session-009-ibl-style-pursuit]] — where the ibl-sorter reference source lives and how its preprocessing inspired 009/010
- [[project_overview]] — dataset and pipeline context
