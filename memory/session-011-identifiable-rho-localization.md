# Session 011: Identifiable Rho Localization
**Created:** 2026-08-27
**Last updated:** 2026-08-27

## Context
Replace the monopole solver's non-identifiable `(z, sigma)` parameter pair with an identifiable continuous `(x, y, rho)` spatial model, where `rho = sqrt(z^2 + sigma^2)`. This is a separate model path from the legacy raw solver and from session 0010's local-ZCA experiment.

## Starting point

- Session 0010 local-ZCA run `16454853` is not scientifically usable: 83.0% of fits reached the lower z bound and 86.8% selected sigma 2 um, while its 0.5 ms lockout and zero energy-drop stopping rule admitted 382,799 events in 8.748 s.
- Restored continuous sub-voxel refinement to `maths_0010.py` and submitted the controlled diagonal-whitening smoke job `16475670`.
- `16475670` completed successfully in 2m51s. Its fixed-codebook, full-waveform-lockout, four-round control accepted 36,070 events in 1 s, with round energy reductions of 15.75%, 9.98%, 7.18%, and 5.10%.
- Plot job `16476027` is rendering established reconstruction, inferno-localization, collision, and Omega-raster panels for that control output.
- The identifiable-rho synthetic check recovered rho 13.998 um from a 14 um source, recovered the off-grid lateral location, and reached relative reconstruction error `2.9e-9`.
- Full-recording jobs are queued with independent output directories: `16476587` reproduces original local-ZCA session 0010; `16476588` is repaired diagonal-continuous session 0010; and `16476589` is diagonal identifiable-rho session 0011, dependent on successful completion of smoke job `16476387`.
- The cluster lacks `scontrol`; these 24-hour jobs checkpoint completed chunks and must be manually resubmitted with `--resume` after an allocation ends.
- Gated full-run plot jobs are queued: `16476698` after original local-ZCA `16476587`, `16476699` after repaired diagonal-continuous `16476588`, and `16476700` after identifiable-rho `16476589`. Each writes established residual reconstruction/localization/diagnostic/raster plots plus residual-local adaptations of spiketensor's spike-example, density, and categorical-raster panels.

## Full-recording failure and rerun decision

- The original three full jobs and their gated plots were cancelled. The renewed full diagonal-continuous run `16476952`, local-ZCA run `16476996`, and identifiable-rho run `16476997` used the session-0006-style GPU native peak-selection path whenever no explicit spatial lockout is configured. This removed the accidental Python O(N²) conflict-free peak loop.
- `16476996` (local-ZCA, 11m37s) and `16476997` (diagonal rho, 8m11s) both failed with SLURM host-memory OOM (`ExitCode 0:125`), not GPU OOM. Their gated plots were therefore cancelled/unsatisfiable. `16476952` and its plot dependency were then cancelled as part of switching the experiment.
- Cause: `raw_residual.run_residual_extraction` loads the entire 1,957.1908-second selected recording for whitening (`reader[first_sample:requested_stop]`) and then materializes its filtered float array before `estimate_whitening` subsamples. The raw 384-channel int16 recording is about 45 GB; its float32 filtered version is about 90 GB, before filtering work buffers/copies, exceeding the 128 GB SLURM request.
- Disabling whitening alone did not fix the OOM because the generic runner performed that whole-recording calibration prepass unconditionally, even for `whitening="none"`. The three unwhitened attempts (`16477444`--`16477446`) consequently also OOMed; their plot dependencies were cancelled.
- Repaired calibration on 2026-08-27: `sample_preprocessed_calibration` filters at most `whitening_max_samples` (default 300,000) frames sampled across the requested interval. It replaces the whole-recording load for every whitening mode, including the noise estimate needed by the RMSE gate. This restores session-006's bounded-memory, chunked extraction behavior.

## Active full-recording flavors

All three use the same raw preprocessed coordinate (`whitening="none"`), Q8 input codebook, GPU-native temporal/spatial peak selection, 1,957.1908-second recording, chunk checkpoints with `resume=True`, and the established plus adapted-spiketensor plot suite on successful completion.

- **Continuous session-0010** (`16478822`): diagonal-control parameterization without whitening; discrete `(x, y, z, sigma)` initialization followed by restored continuous position refinement. Output: `residuals/runs/dataset1_p1/unwhitened_continuous_0010_full/`. Gated plots: `16478841`.
- **Local session-0010** (`16478823`): session-0010's broader pursuit flavor, retaining its local-model settings and Omega learning but executed without whitening; it now also uses the GPU-native peak-selection path because no custom lockout is configured. Output: `residuals/runs/dataset1_p1/unwhitened_local_0010_full/`. Gated plots: `16478842`.
- **Identifiable-rho session-0011** (`16478824`): continuous `(x, y, rho)` monopole model, eliminating the non-identifiable depth/scale split. Output: `residuals/runs/dataset1_p1/unwhitened_rho_0011_full/`. Gated plots: `16478843`.
- **Vectorized identifiable-rho session-0011** (`16482462`, pending): independent full run on `torch_pr_60_tandon_advanced`. Its monopole discrete refinement is vectorized across fits, avoids per-event Python and GPU-to-CPU profile-index synchronization, and uses `fit_batch_size=2048`. Output: `residuals/runs/dataset1_p1/unwhitened_rho_0011_vectorized_full/`. Commit: `20eb268`.

## Next steps

- [ ] Track full jobs `16478822`, `16478823`, `16478824`, and vectorized rho `16482462`; retain the original gated plot suites `16478841`--`16478843` only on successful completion.
- [x] Replace whole-recording whitening setup with bounded sampled chunks before filtering/covariance estimation.
- [ ] Compare unwhitened continuous and rho outputs against the raw-coordinate baseline before judging localization quality.

## Resume status (2026-08-27)

- `16478822` (continuous), `16478823` (local), and `16478824` (identifiable rho) are RUNNING with no errors in their latest logs. At inspection, they had reached chunks 107, 67, and 65 of 1,958, respectively.
- Their gated plot jobs `16478841`--`16478843` remain PENDING on successful completion, as intended.
- Vectorized rho full run `16482462` was submitted on `torch_pr_60_tandon_advanced`; its exact monopole formula check and CPU functional smoke passed before submission. GPU validation will occur when it is allocated.

## Performance lineage and current constraint

- The original unwhitened full jobs `16478822` (continuous), `16478823` (local), and `16478824` (rho) were all cancelled together after about 2h22m with `SIGNAL Terminated`; this is consistent with the cluster's low-GPU-utilization enforcement, not a Python or memory failure. Their completed chunk checkpoints remain available.
- Session 005's real CUDA profile is the governing performance evidence: continuous refinement consumed 42.397 of 50.057 localization seconds, emitted 4,203,289 kernel launches and 152,120 stream synchronizations per profiled chunk, and used only 9.753 seconds of CUDA kernel time. The primary output-preserving target is removal of the per-backtrack GPU-to-host condition in the continuous line search (`bool(live.any())`), followed if needed by fusion/compilation of fixed-shape refinement operations.
- The rho implementation in `maths_0010.py` is a separate gradient-refinement path and therefore bypasses the established `continuous_refine.py` implementation and its prior batched-eigensolve work (`20e203d`). The vectorized rho run `16482462` removes per-event discrete-refinement Python and profile-index synchronization and raises `fit_batch_size` to 2048, but this is not the primary known bottleneck.
- Historical 4-second / 8,192-fit superbatch benchmark (`bfe0565`, job `16257633`) achieved about 5.27x better steady-state throughput but was scientifically non-equivalent: only 67.38% of baseline events matched and continuous locations differed materially. Do not adopt large-chunk/superbatch settings as an output-compatible baseline optimization.
- Do not submit additional full rho jobs based solely on batching. First implement and validate continuous-refinement synchronization removal or fusion against deterministic reference inputs and collect a short GPU utilization benchmark.

## Links
- [[session-010-whitened-dense-pursuit]]
- [[session-004-continuous-residual]]
- [[session-006-plots]]
