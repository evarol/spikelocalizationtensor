# Session 004: Continuous Residual Subtraction
**Created:** 2026-08-22
**Last updated:** 2026-08-24

## Code snapshot
- Current solver commit: `79b8beb5733075199ce2f2d2638278a42ac80774` (`Add continuous raw residual solver`).

## Context
The first raw residual smoke run localized every source on the integer 1 µm grid and later subtraction passes increased replayed RMS. This work integrates bounded continuous refinement, adds safeguards against over-subtraction, and evaluates the Q8 raw-recording pipeline. The original and optimized full runs were cancelled for low GPU utilization. A more heavily batched benchmark completed much faster but changed the scientific output, so no full residual job is currently running.

## Implementation
- `localize_spikes_fixed_codebook` can refine fixed monopole fits continuously inside the winning ±0.5 µm voxel cell.
- The continuous source, gain, captured energy, prediction, displacement, and energy gain are recomputed before subtraction. Both continuous `sources` and integer `sources_grid` are saved.
- Each proposed subtraction has its gain re-fit against the current residual after earlier overlapping atoms have been removed.
- An atom is accepted only if that sequential subtraction lowers actual residual energy and still meets `min_captured_fraction`.
- A complete pass is rolled back and residual peeling stops when core energy drops by less than `min_pass_energy_drop_fraction`, default `0.01`.
- Template score reduction and temporal/spatial peak selection now remain on the GPU; only selected event candidates are transferred to CPU. The template time batch is `4096` instead of `512`.
- Fixed coarse spatial footprints are cached across fit batches, residual passes, and recording chunks instead of being rebuilt repeatedly for each anchor configuration.
- Coarse localization can batch up to 32 channel-neighborhood configurations in one tensor operation, replacing separate small `einsum` launches for each configuration. The control is `--localization-config-batch-size`, default `32`.
- Nine direct synthetic checks pass in the project Singularity environment, including exact CPU/GPU peak-selection parity, continuous off-grid localization, cached-fit equivalence, serial-versus-configuration-batched equivalence, overlap-safe gain fitting, rollback, and the residual pipeline.

## Kilosort reference
- The local IBL sorter uses regularized template maxima above `Th²`, with default `Th=[6, 3]`, an amplitude penalty `lam=10`, and 60 matching-pursuit iterations per batch.
- It has no whole-pass energy rollback. Its learned oriented templates, projection threshold, amplitude penalty, and fixed iteration cap constrain subtraction.
- IBL starts with 6 generic temporal prototypes but learns up to roughly `4 × channels` localized, unit-specific rank-3 templates for subtraction. Our residual model instead uses analytic spatial footprints times one hard-selected global temporal row.
- Our threshold `6` is not numerically equivalent because the analytic generic-template bank and score normalization differ.

## Environment
- `pytorch.ext3` has NumPy, SciPy, CUDA-enabled PyTorch, Matplotlib, SpikeInterface, and ProbeInterface, but lacks `spikeglx` and `pytest`.
- `ibl-sorter.ext3` contains the full raw-recording runtime, including `spikeglx`; the pipeline does not import the `iblsorter` library.

## Jobs and outputs
- Continuous smoke job `16180032` completed successfully with exit `0:0` in `8:42` on `gl018`.
- Full Q8 residual job `16180078` was cancelled by request after `1:57:17` on `gl045`; SLURM recorded `CANCELLED` with exit code `0:0`.
- The first 127 one-second chunks (`0–126`) remain checkpointed under `runs/dataset1_p1/raw_template_residual_continuous/chunks/`.
- Full launcher: `src/preprocessing/raw_template_residual_full.sbatch`.
- The full 1,957.1908-second recording launcher uses resumable one-second chunks, a 24-hour limit, `USR1@60` requeueing, and does not save per-event waveform shards.
- Baseline profiling job `16254844`, GPU-peak job `16254987`, and cached-localization job `16255034` each processed the same one-second recording chunk successfully.
- Optimized full job `16255197` was submitted with `src/preprocessing/raw_template_residual_full_optimized.sbatch`. It was cancelled by UID 0 after `2:07:36` on `gl049`, consistent with cluster low-GPU-utilization enforcement; this is the untouched one-second-chunk baseline.
- The optimized full run writes to `runs/dataset1_p1/raw_template_residual_continuous_optimized_q8/`. It starts fresh and does not reuse or modify the original 127 checkpoints.
- The optimized full output contains 267 valid one-second checkpoints (`0–266`) with 8,902,883 events, covering 267 of 1,958 chunks. The interrupted next chunk was not written, and no consolidated arrays were produced.
- The cancellation and final progress for job `16255197` are recorded in `slurm_logs/raw_residual_opt_q8_16255197.{out,err}`.
- Configuration-batched four-second benchmark job `16257633` completed successfully in `14:08` with exit `0:0`. It used 20 seconds of recording, four-second chunks, 40,000 peaks per pass, 8,192-event fit batches, 32-configuration localization batches, and stage profiling.
- Benchmark `16257633` writes to `runs/dataset1_p1/raw_template_residual_batched_4s_16257633/` and contains 667,189 events across five chunks. It does not touch the full-run output.

## Smoke results
- The run accepted 332,012 events, or 33,201.2 events/s. Counts by pass were `82,826`, `83,172`, `83,503`, and `82,511`.
- `99.92%` of saved sources are noninteger. Median continuous displacement is `0.503 µm`, with the maximum constrained to the voxel-corner distance `0.866 µm`.
- Median sequential captured fraction declined by pass: `0.191`, `0.152`, `0.128`, and `0.111`.
- Core-energy drops per pass across chunks were approximately `17.0–20.6%`, `9.3–11.5%`, `6.5–7.8%`, and `5.2–6.2%`.
- No pass met the 1% rollback condition. Cumulative remaining RMS after four passes ranged from `78.0%` to `81.7%`, with mean `79.7%`.
- Monotone gain re-fitting fixed the earlier increase in residual RMS, but nearly flat event counts and a material fourth-pass energy drop show that biological over-subtraction is not yet ruled out.

## Full-run performance finding
- Completed chunks advanced steadily at about `55.7 s` per one-second recording chunk, so the job was not stuck. That rate projects to roughly `30–31 h` for all 1,958 chunks: one 24-hour requeue would handle the scheduler limit, but not the low-GPU-utilization policy.
- A live 15-second `nvidia-smi dmon` sample on `gl045` measured mean SM utilization `22.9%` (range `0–30%`), only `1,585 MiB / 46,068 MiB` framebuffer use, and `106–119 W`. This is materially below the cluster's sustained 50% GPU-utilization requirement.
- Baseline steady-state pass timing was about `9.75 s`: localization used about `7.86 s` (`81%`), CPU peak selection `1.56 s` (`16%`), and sequential subtraction about `0.16 s` (`<2%`).
- GPU-native peak selection preserved the baseline artifact exactly and reduced steady-state peak selection to `0.004–0.005 s`, bringing pass time to about `8.13 s`.
- Caching coarse localization footprints reduced steady-state localization to about `6.75 s` and total pass time to about `7.08 s`, a roughly `27%` reduction from the profiled baseline.
- The cached path retained the same `33,357` saved core events and nearly identical pass energy drops, but its output is not elementwise identical to the uncached artifact. It is therefore not numerically compatible with the original checkpoints.
- Two post-optimization samples on `gl049` measured sustained SM utilization around `38%`; the later 30-second sample averaged `38.5%`, median `40%`, range `0–44%`, with `3,679 MiB` framebuffer use. The optimization improved throughput but remains below the cluster's 50% policy target.
- At the later check, job `16255197` had completed 110 of 1,958 chunks in 54 minutes, averaging about `29.5 s` per one-second chunk and projecting to roughly `15.1 h` remaining. This was a point estimate, not a guaranteed completion time.
- The 127 checkpoints are valid partial Q8 results. They may be reused only by a numerically compatible resumed implementation; a Q12 run or scientifically changed detector/subtractor must restart in a separate output directory.

## Four-second benchmark result
- After the one-time first-pass startup/compilation cost of about `475 s`, the four steady-state passes averaged about `21.56 s` per four-second recording chunk, or `5.39 s` per recorded second. The one-second baseline checkpoint interval averaged `28.42 s`, so the steady-state benchmark was about `5.27×` faster.
- SLURM accounting reported whole-job average GPU utilization and peak GPU memory of `40%` and `3,670 MiB` for the baseline, versus `37%` and `21,602 MiB` for the benchmark. The benchmark utilization average is dominated by its long startup and is not a clean steady-state utilization measurement.
- The benchmark is not scientifically output-equivalent to the baseline. Over the same first 20 seconds it saved 667,189 events versus 662,379 (`+4,810`); only 446,331 time/channel/pass keys were shared (`67.38%` of baseline events), with per-pass overlap falling from `88.46%` on pass 1 to `49.91%` on pass 4.
- Event keys were unique in both outputs, and the unmatched events were not concentrated near one-second boundaries. Among shared keys, only `64.13%` had the same coarse source grid and `2.22%` had exactly identical continuous sources. The larger chunk and fit batches alter the sequential residual state, so these settings must not replace the baseline launcher without redesign or explicit acceptance as a changed model.

## Residual-pass diagnostic
- `src/plots/plot_residual_pass_diagnostics.py` writes `out/plots/raw_template_residual_smoke_16255034/pass_diagnostics.png` at 800 DPI.
- Saved core counts by pass were `8,304`, `8,381`, `8,402`, and `8,270`; accepted counts remain nearly flat while median captured fraction declines from about `0.19` to `0.11`.
- Fractions of later-pass events within `0.5 ms` and `48 µm` of any earlier-pass event were `14.6%`, `25.7%`, and `33.5%` for passes 2–4. Later passes increasingly revisit earlier event neighborhoods, but most events are still outside that strict recurrence window.
- Temporal-row usage shifts materially across passes, especially row 1 increasing from `7.1%` in pass 1 to `19.9%` in pass 4.

## Monopole identifiability audit
- For the isotropic monopole with a free event gain, normalized spatial shape depends on axial depth `z` and profile scale `sigma` only through the identifiable effective width `rho = sqrt(z^2 + sigma^2)`. The data cannot separately identify `z` and `sigma`.
- The ten-scale grid is therefore a redundant sampling of effective widths, not ten distinct physical source classes. The live run's saved `z`, `sigma`, and profile index must not be interpreted independently; use `rho` for spatial-width analysis.
- A joint continuous free-range optimization over both `z` and `sigma` is ill-posed because the objective has an exact ridge. The proper free-range replacement directly optimizes `(x, y, rho)` with one `16^3` coarse lattice, hierarchical refinement, and bounded continuous refinement.
- If physical depth is required, fix one externally justified `sigma_0` and optimize `z` conditionally. A prior can select a point on the ridge but does not make both variables identifiable from the recording.
- Mathematical source: `docs/residual_run_math.tex`; compiled six-page note: `out/docs/residual_run_math.pdf`.
- Keep job `16255197` labeled as the Q8 ten-scale baseline. Any `(x, y, rho)` implementation is a changed model and must restart in a fresh output directory.

## Pending
- Implement and validate the identifiable `(x, y, rho)` free-range solver as a separate experiment; do not retrofit it into the running ten-scale job.
- Build a controlled performance experiment that isolates chunk size and fit-batch size while preserving or explicitly redefining sequential subtraction semantics; do not promote the completed four-second benchmark settings to the full launcher.
- Measure sustained GPU utilization after startup on a longer controlled benchmark. The completed job's whole-job SLURM average cannot distinguish compilation overhead from steady-state utilization.
- If utilization remains low, keep the residual and waveform extraction/subtraction on GPU and pipeline CPU filtering/I/O; coarse/discrete localization remains the dominant measured stage.
- Calibrate threshold `6`, captured-fraction cutoff `0.05`, and the 1% pass criterion before treating residual output as scientific output.
- Compare detections across passes and determine whether later atoms are repeat decompositions of one biological spike or genuine overlaps.
- Only after the Q8 residual architecture is efficient should it be combined with the Q12 codebook from [[archive/session-003-q12-temporal-codebook]].

## Links
- [[session-007-global-codebook-pursuit]]
- [[session-005-residual-profiler]]
- [[session-006-plots]]
- [[session-002-template-residual]]
- [[archive/session-003-q12-temporal-codebook]]
- [[project_overview]]
