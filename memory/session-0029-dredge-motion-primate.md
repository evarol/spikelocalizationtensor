# DREDge Motion Estimation on Primate Recordings (0029)

**Created:** 2026-09-09
**Last updated:** 2026-09-09 (peaks OOM + stuck dependency recorded)

## Why this work exists

Macaque3 was recorded while the probe was being inserted into the brain, so the
raw fitted-depth spike raster shows the whole activity sheet sliding down the
probe by roughly 2.5–3 mm over the first ~17 minutes. The user wanted that
sweep registered away with DREDge (the Varol/Paninski-lab method), so that
layers sit flat and the registered depth axis extends to the full insertion
travel (in the DREDge Nature Methods 2025 paper, Fig 4b, a comparable acute
insertion session registers out to ~25000 µm of depth). The paper's figure was
reviewed with the user at `/scratch/ap7151/dredge_pages/dredge_page-07.png`
(page 7, Fig 4b) and the target rendering is that figure's tall
time-by-depth strips with horizontal unit lines after registration.

## What was learned about the SI DREDge machinery (installed 0.104.1)

The pytorch overlay's SpikeInterface ships the full DREDge port at
`spikeinterface/sortingcomponents/motion/dredge.py` and a high-level
`spikeinterface.preprocessing.correct_motion()` with native presets
`dredge` and `dredge_fast`. The stock `dredge` preset is: locally_exclusive
detection (peak_sign neg, threshold 8, exclude_sweep 0.8 ms, radius 80 µm),
monopolar_triangulation localization, `dredge_ap` nonrigid estimation with
gaussian windows (step 400, scale 400), and kriging interpolation with
border_mode force_extrapolate. `dredge_fast` swaps the localizer for
grid_convolution. The preset dicts live in
`/ext3/miniforge3/lib/python3.13/site-packages/spikeinterface/preprocessing/motion.py`
(read it with singularity, not bare paths).

Two traps cost us one wasted GPU run each and are now understood:

- **Amplitude units.** `dredge_ap` weights its correlation matrix by
  `log1p(amplitude)` with a default threshold of 0.2 that silently assumes
  µV. Feeding residual-run `alpha` values in volts (~1e-3) makes every
  window fall below the gate, the Newton solve gets zero weights, and the
  entire displacement comes back exactly 0.0 while D/C matrices look healthy.
  The fix is to multiply by 1e6 when building the peaks array. Note that
  `make_2d_motion_histogram` abs()'s amplitudes, so sign is never the issue.
- **Search radius.** The default `max_disp_um=150` can only track jitter.
  The insertion sweep needs a search radius on the order of the tissue
  displacement; `max_disp_um=2500` registers the sweep cleanly. Even with
  that, the whole dredge_ap call takes only ~6–7 min of GPU on a 43-min
  384-ch recording (2556 time bins × 18 windows), so 4 h walltime is plenty.

The peak/localization round trip also taught: `np.save` appends `.npy` to
tmp names that lack it (name temps `*.tmp.npy`), and the residual run's
`global_sources.npy` is the global-µm fitted position while `sources.npy`
stays in the ±150 local lattice frame.

## What was run and what it showed

The first pass used the 0025 residual run's own accepted events as peaks
(6,134,349 events, alpha × 1e6 as amplitude, global_sources as locations)
into `residuals/runs/primate/0025_dredge_ap_macaque3_base_disp2500/`, via
`0025_dredge_ap.py` (stages: prepare/motion) + `0025_dredge_peaks.sbatch` +
`0025_dredge_motion.sbatch` (env vars SUBJECT/VARIANT/MAXDISP/RUNSUFFIX).
That produced a corrected raster whose insertion diagonal fully collapses;
the rotated view (time on y, depth on x, `spike_raster_corrected_rotated.png`)
makes the straightened layers obvious, matching the user's prediction from
the 0024 omega raster. Outputs are under
`residuals/out/0025_dredge_ap_macaque3_base_disp2500/`.

After feedback from the paper's authors, the user wants the stock SI
pipeline instead ("the way god intended — use only spikeinterface"): CPU-side
bandpass 300–6000 + median CMR, locally_exclusive detection, and SLT
(monopolar_triangulation) localization, then GPU `dredge_ap`. That chain
lives in `residuals/src/preprocessing/0025_sidredge.py` (stages:
peaks/motion/raster) plus `0025_sidredge_peaks.sbatch` (cpu_short,
tandon_priority) and `0025_sidredge_motion.sbatch` (l40s_public GPU,
USR1/TERM requeue trap). The raster stage renders in the paper's Fig-4b
style (tall strips, gray density, per-panel normalization, depth axis
extended by the registered span rather than clipped to the probe).

Note that SI's `read_nwb_recording` returns traces in µV by default, so with
SI-native detection the amplitudes are already in the units dredge wants —
the volts-to-µV fix was only needed for the residual-alpha path.

## The CPU OOM saga and the one-pass fix (2026-09-09)

The peaks stage was OOM-killed twice on cpu_short, both times inside SI's
process-pool job framework, and both turns taught something worth keeping.

The first kill (job 17282581, 120G, 32 workers) happened while
`write_binary_recording` cached the preprocessed float32 binary — 118 GB on
disk — a step that was never needed: SI is lazy, detection and localization
stream straight through the filter chain. Removing the cache was the user's
call ("why are we saving the recording") and also cut the disk cost.

The second kill (job 17291851, 120G, 12 workers) revealed a cpu_short
partition rule the hard way: sbatch rejects jobs whose memory-per-CPU
exceeds roughly 4 GB/core (513G node / 128 cores), so 150G ÷ 16 cores fails
with "partition 'cpu_short' is not valid" while 120G ÷ 32 cores submits.
The real footprint is SI's chunk budget (~1.3 GiB) plus ~2–3 GB of
interpreter/numba/torch RSS per spawn worker, so the safe shape is few
workers on 1 s chunks with the 32-core/120G sbatch shell.

The actual rewrite follows the user's own proven recipe from
`/scratch/ap7151/sln-v2/src/localizations/pipeline.py` + `core.py` (they
pointed me at it as "this worked"): lazy bandpass 300–6000 float32 + global
median CMR, TPCA fitted on the first 300 s (7 components, 10k spikes,
5 ms waveforms via a quick `detect_peaks` on a `frame_slice`), then ONE
`run_node_pipeline` pass with a live `LocallyExclusivePeakDetector`
(threshold 8, sweep 0.8 ms, radius 80, noise_levels injected explicitly via
`get_noise_levels(rec, return_in_uV=False, method="mad")` because
`run_node_pipeline` does not supply them) → `ExtractDenseWaveforms` (5 ms) →
`TemporalPCADenoising` → `LocalizeMonopolarTriangulation` (r75, d150, ptp),
returning peaks and locations together. The user was explicit that the work
stays inside SpikeInterface's proper API — no hand-rolled numpy pipelines.
Dredge itself cannot be a streaming node (it needs the complete peak set for
its global xcorr), so it stays a separate GPU stage right after, which is
the same structure SI's own `correct_motion` uses (pipeline → estimate).
The GPU motion sbatch passes `--max-disp-um 2500` and renders the Fig-4b
raster + motion heatmap at the end.

## State

The residual-sourced disp2500 chain is complete and its rasters verified.
The one-pass SI pipeline is proven at full scale: on 2026-09-09 the peaks
job 17295223 completed the entire detect+localize pass over the 43-min
macaque3 recording in 853 s (12 workers, 1 s chunks, peak RSS 46.5 GB —
comfortably inside the 120G request) and found 1,200,296 peaks with sane
depth percentiles (1st–99th: 124–3777 µm) and a median amplitude of
46.6 µV. The run then failed on something trivial: the rewritten run_peaks
no longer created its output directory, so the first np.save into
`residuals/runs/primate/0025_sidredge_macaque3/` threw FileNotFoundError
after the 853 s pass — all the computed peaks were lost. The mkdir call is
back in. Resubmitted as 17296293 (peaks, cpu_short tandon_priority) with
dredge+raster 17296296 dependency-gated behind it.

Update (2026-09-09, later): the peaks rerun 17296293 completed cleanly in
19:15 — 1,226,587 peaks detected+localized in 710 s, depth percentiles
123–3773 µm, median amplitude 46.2 µV — and this time the outputs saved
(`peaks.npy`, `peak_locations.npy`, `peaks_meta.json` in
`residuals/runs/primate/0025_sidredge_macaque3/`). But the dependent
dredge job 17296296 FAILED after 1:53 on an sbatch argument-order bug:
`--max-disp-um` consumed the following `--output-dir` value (argparse
rejection "invalid float value: .../0025_sidredge_macaque3"), so the
motion stage never ran and the raster stage crashed on the missing
`motion.npz`. No data lost — just fix the sbatch's argument quoting/order
and resubmit the motion+raster stage against the saved peaks. Once it
lands, compare its motion estimate and raster against the residual-sourced
one and pick the one for the paper; the 25000 µm depth-axis framing
matters for any figure mirroring Fig 4b.

Second fix (2026-09-09, night): the first resubmission (17298129) failed
at 55 s on a second latent gap — the motion sbatch never passed
`--recording-path`/`--electrical-series-path` (the motion stage rebuilds
the lazily-preprocessed recording to feed `estimate_motion`), and the
inner `bash -c` had no `set -e`, so a failed motion stage cascaded into a
confusing raster failure. Fixed both (per-subject SERIES wiring copied
from the peaks sbatch; recording passed positionally with `set -e` inside)
and resubmitted as **17298551**, now RUNNING: all 1,226,587 peaks loaded
and DREDge's cross-correlation pass was ~2/3 done at ~6 s/window when
last checked, with the raster rendering after.

## Links

- [[session-0025-primate-nwb-peeling]]
- [[session-024-convolving-detection-peeling]]
- [[project_overview]]
