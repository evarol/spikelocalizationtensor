# Session 005: Residual CPU/GPU Profiler
**Created:** 2026-08-24
**Last updated:** 2026-08-24

## Context
Keep the current Q8 ten-scale continuous-residual model and profile its existing one-second-chunk execution before changing solver semantics. The goal is to separate CPU orchestration, transfers, and GPU kernels inside the localization stage, which currently accounts for roughly 95% of steady-state pass time.

## Profiler implementation
- The raw runtime overlay contains PyTorch `2.11.0+cu128`; `torch.profiler`, Kineto, and both CPU and CUDA profiler activities are available. The interactive node was CPU-only, so CUDA capture is deliberately validated again inside the allocated SLURM GPU job.
- `src/preprocessing/raw_residual.py` can profile one complete recording chunk and write CPU and GPU operator tables, metadata, and a Chrome/Kineto `trace.json`.
- High-level record scopes cover SpikeGLX reading, preprocessing, residual passes, template scoring, peak selection, waveform extraction, localization, subtraction, checkpoint writing, and pass setup/finalization.
- Localization scopes in `src/maths.py` separate CPU configuration grouping, host-to-device input transfer, temporal projection, coarse assignment, discrete refinement, continuous refinement, and the reconstruction round trip.
- The second one-second chunk is profiled after the first chunk warms CUDA state and the persistent coarse-footprint cache. Only that chunk receives operator-level tracing; even so, the 4.2 million recorded launches produced a 12 GB trace and about 33 minutes of profiler aggregation/export overhead.
- The instrumentation preserves the baseline scientific settings: one-second chunks, 1,024-event fit batches, four residual passes, Q8, ten monopole scales, `16^3` coarse sites, six grid-refinement levels, and bounded continuous refinement.
- At the time of this session, all 12 residual equivalence, pursuit, codebook-update, and safety checks passed in the project Singularity environment; the repository test files were later removed by request in commit `03ecae0`.
- The profiler and pursuit/codebook foundation was committed as `945d5fa` after solver snapshot `79b8beb`.

## Job
- Profiling job `16272887`, launched with `src/preprocessing/raw_template_residual_profile_15m.sbatch`, ran from `05:10:15` to `07:30:52` on 2026-08-24. SLURM recorded `CANCELLED by 0`, elapsed `02:20:37`, and reason `QOSGrpGRES`.
- SLURM accounting recorded average GPU utilization `43%` and GPU memory `12,024 MiB`, consistent with the repository's warning that sustained utilization below 50% is subject to cancellation.
- The job targets recording seconds `[0, 900)` in a separate output directory and does not modify the existing baseline checkpoints.
- Run output: `runs/dataset1_p1/raw_template_residual_profile15m_16272887/`.
- Profiler output: `out/profiles/raw_template_residual_profile15m_16272887/`.
- Logs: `slurm_logs/raw_residual_profile15m_16272887.{out,err}`.
- All expected profiler files were written: `cpu_operators.txt`, `gpu_operators.txt`, `metadata.json`, and the 12 GB `trace.json`. The trace begins with the Kineto schema/device metadata and ends with a closed trace object, but it has not been fully parsed as JSON because of its size.
- The run completed 169 checkpoint files, `chunk_000000.npz` through `chunk_000168.npz`. It was terminated during pass 1 of the next chunk, so no partial `chunk_000169.npz` was written.

## Profile result
- Steady-state log timings are about `9.1–9.3 s` per residual pass, of which localization uses about `8.7–9.0 s` (`95–97%`).
- In the profiled chunk, residual peeling used `51.722 s` inclusive CPU time and localization used `50.057 s` (`96.8%`). Continuous refinement dominated localization at `42.397 s`; discrete refinement used `4.489 s`, coarse assignment `2.347 s`, and the reconstruction round trip `0.681 s`.
- The capture contains `4,203,289` `cudaLaunchKernel` calls using `11.081 s` self CPU time, plus `152,120` `cudaStreamSynchronize` calls using `2.843 s`. Total self CUDA kernel time was only `9.753 s`, so the path is dominated by CPU orchestration, tiny launches, and synchronization rather than a few long GPU kernels.
- The continuous solver makes 40 localization calls per chunk. Each call can execute 80 refinement iterations with 30 line-search backtracks, and `_line_search` evaluates `bool(live.any())` on the GPU inside the backtrack loop. The resulting device-to-host control synchronization is the primary output-preserving optimization target.
- Device arithmetic is fragmented across hundreds of thousands of `div`, `mul`, `sum`, masking, indexing, and small `bmm` calls. Increasing chunk or fit-batch sizes is not assumed safe: the completed four-second benchmark changed sequential residual state and failed output equivalence.

## IBL-style pursuit and temporal-codebook experiment
- The baseline path remains the default when `pursuit_rounds=0`. The experimental path is enabled explicitly and must use a fresh output directory because it changes the scientific algorithm.
- Each pursuit round scores the current residual, chooses a deterministic score-greedy set of time-separated peaks, sorts the selected events by time, localizes and monotonically subtracts them, then rescores the updated residual. The default pursuit lockout is one complete waveform, and the fixed round cap can reproduce IBL sorter's 60-round structure.
- This removes the stale-residual problem of enlarging arbitrary fit batches: all events in one pursuit group have nonoverlapping temporal support, while conflicting events are reconsidered after residual subtraction in later rounds.
- Optional temporal-codebook learning is a separate seeded learning phase over randomly selected recording chunks. Accepted core events contribute weighted least-squares sufficient statistics to their selected temporal row; rows update only after a minimum event count, align sign to the previous row, blend with configurable momentum, and are renormalized.
- Codebook updates occur between learning chunks. The learned codebook is then frozen for final extraction over every requested chunk, matching IBL sorter's learn-then-extract structure rather than mixing changing codebooks into saved checkpoints.
- Learned-codebook runs write `omega_initial.npy`, `omega_learned.npy`, `omega.npy`, and `codebook_learning_history.json`. Resume is rejected if checkpoints exist without the frozen `omega_learned.npy`.
- `src/preprocessing/raw_template_residual_pursuit_codebook_smoke.sbatch` runs a four-second smoke test with two seeded learning chunks, Q8, one-second extraction chunks, 60 pursuit rounds, full continuous refinement, and baseline fit/localization batch sizes.
- Smoke job `16299673` completed successfully in `00:05:13` on 2026-08-24. Its output is `runs/dataset1_p1/raw_template_residual_pursuit_q8_16299673/`, with logs at `slurm_logs/raw_residual_pursuit_q8_16299673.{out,err}`.
- The smoke extraction emitted 59,892 events over four seconds, or 14,973 events/s. Its 240 extraction rounds accepted a mean 249.55 events each; every chunk reached the 60-round cap with 211–249 accepted events still present in round 60.
- The product of the reported per-round core-energy reductions left a mean `0.6924` residual-energy fraction after 60 rounds. Median captured fraction fell from about `0.59` in round 1 to `0.15` in round 60, so late rounds fit substantially weaker structure and the fixed cap is active rather than a convergence condition.
- Both learning chunks updated all eight rows using 2,983–4,540 events per row in total. With momentum `0.9`, learned rows moved only `0.8–1.5` degrees from initialization, which establishes stable updates but not a held-out improvement.
- Steady-state pursuit rounds averaged `0.7146 s`: localization used `0.5979 s` (`83.7%`), template scoring `0.0676 s`, and peak selection `0.0092 s`. SLURM recorded average GPU utilization `40%` and GPU memory `3,676 MiB`, so pursuit scheduling alone did not fix the localization utilization bottleneck.

## Paired 65,610-sample ablation
- Use exactly `65,610 = 729 x 90` samples per chunk, the smallest 90-sample multiple at least as large as IBL sorter's 65,600-sample default. At 30 kHz this is `2.187 s`; four complete chunks span 262,440 samples or `8.748 s`.
- The frozen control uses the initial analytic Q8 codebook, 60 pursuit rounds, and no codebook learning. The learned condition uses the same extraction settings and two seeded learning chunks, then freezes the learned Q8 codebook for all four extraction chunks.
- Codebook benefit is evaluated primarily on the two chunks excluded from seeded learning; all-chunk metrics remain descriptive. Compare event counts, captured-fraction mean and median, inferred remaining core-energy fraction, and one-to-one event overlap within three samples, both time-only and requiring the same anchor channel.
- A third condition isolates energy stopping from learning: it loads the learned condition's frozen `omega_learned.npy`, performs no additional learning, and rolls back/stops at the first round with less than `0.005` core-energy reduction. In the one-second smoke chunks this threshold first fired around rounds 18–21, providing a meaningful contrast to the 60-round control.
- Run each GPU condition as a separate SLURM job so wall time and `sacct` GPU utilization remain attributable. A dependent CPU comparison job writes `comparison.json` and the three GPU jobs' accounting table after all conditions complete.
- Frozen-control job `16304090`, learned-codebook job `16304091`, and learned-codebook energy-stop job `16304092` all completed successfully. Their outputs are `runs/dataset1_p1/raw_template_residual_{frozen,learned,learned_stopped}_65610_<jobid>/`.
- Seed 42 selected chunks 2 and 3 for learning, leaving chunks 0 and 1 as the 4.374-second held-out set. On held-out data, frozen and learned full pursuit produced 14,972.1 and 15,000.5 events/s, mean captured fractions `0.23259` and `0.23298`, and mean remaining core-energy fractions `0.69562` and `0.69497`. These changes are negligible at this learning strength.
- Frozen and learned outputs had 61,152 time matches within three samples: `93.38%` of frozen events and `93.20%` of learned events, with Jaccard `0.87425`. Requiring the same anchor left 54,217 matches, about `82.7%` of each output and Jaccard `0.70519`; learning shifts some anchor assignments despite nearly unchanged aggregate metrics.
- The learned rows moved only `0.679–1.402` degrees from initialization. This confirms stable conservative updates but provides no meaningful held-out energy or captured-fraction improvement after two learning chunks with momentum `0.9`.
- The `0.005` energy stop retained 19 rounds on both held-out chunks and emitted 4,807.5 events/s, `32.05%` of the full learned events. Every stopped event matched a full learned event within three samples with the same anchor. Its stronger subset had mean/median captured fractions `0.33081/0.30042`, while leaving `0.79593` core energy versus `0.69497` after 60 rounds.
- SLURM wall time and average GPU utilization were `4:20/40%` for frozen, `6:09/39%` for learned including its two learning chunks, and unexpectedly `7:50/28%` for the 19-round stopped run. The stopped runtime does not scale with its reduced round count and needs log/node-level investigation before treating early stopping as a speed optimization.
- Dependent comparison job `16304093` failed on a `stopped` versus `learned_stopped` dictionary-key typo. The utility was corrected and run inline; its codebook guard now tolerates the expected loader renormalization roundoff (`2.98e-8` maximum absolute change). Final metrics are in `runs/dataset1_p1/raw_template_residual_ablation_65610_16304090_16304091/comparison.json`.
- Final 800-DPI figures are in `out/pursuit_ablation_65610/`: `temporal_codebook_update.png`, `heldout_pursuit_trajectories.png`, and `heldout_ablation_summary.png`. The plotting source is `src/plots/plot_pursuit_ablation.py`.

## Next steps
- Inspect the stopped run's per-round stage timings and node behavior to explain why 19 rounds took longer and used less GPU than the 60-round frozen extraction.
- Treat the current two-chunk momentum-`0.9` codebook update as stable but scientifically neutral. Before scaling it up, test either more learning chunks or lower momentum against the same frozen held-out control.
- Treat the `0.005` stop as a quality/coverage choice, not yet a performance optimization: it selects the first 32% of full-pursuit events with higher captured fractions but leaves about ten percentage points more core energy.
- Treat pursuit output as a new experiment: do not compare it to the baseline as checkpoint-compatible and do not resume existing Q8 residual checkpoints from it.
- First, benchmark an output-equivalent continuous-refinement change that removes the per-backtrack `bool(live.any())` synchronization while retaining the same accepted steps and fixed scientific settings.
- Compare the old and candidate refiners directly on deterministic inputs, then run all nine residual equivalence and safety checks. Require exact or explicitly justified numerical equivalence before using existing checkpoints.
- Measure one-second steady-state stage timing and sustained GPU utilization with a lightweight capture. Avoid another full-chunk Kineto trace until event volume is reduced; the current trace is already sufficient to locate the bottleneck.
- If synchronization removal is insufficient, fuse or compile the fixed-shape score, gradient, masking, and line-search operations. Preserve the 80-iteration/30-backtrack decisions before testing larger fit batches.
- As a separate utilization experiment, run multiple unchanged one-second chunks concurrently on one GPU so more work is in flight without enlarging a chunk or fit batch. First verify chunk independence, atomic checkpoint writes, memory headroom, and exact per-chunk output equivalence.
- Keep the identifiable `(x, y, rho)` solver as a separate scientific experiment with a fresh output directory; it is not a performance patch for this Q8 ten-scale baseline.

## Links
- [[session-013-rho-localization-optimization-plan]]
- [[session-008-peak-channel-codebook-init]]
- [[session-007-global-codebook-pursuit]]
- [[session-004-continuous-residual]]
- [[project_overview]]
