# Session 013: Rho Localization Optimization Plan
**Created:** 2026-08-27
**Last updated:** 2026-08-27

## Context

Output-preserving optimization of session 011's identifiable-rho localization path by reusing the batching structure of session 003's Q12 solver. Phase 1 is implemented and validated below; do not submit a replacement full-recording job until the remaining phases pass deterministic equivalence checks and the combined result clears the promotion gate.

The existing vectorization commit `20eb268` already reduced steady-state localization from `14.4631 s` to `3.5196 s` per pursuit round (4.1x), and total round time from `14.9755 s` to `3.8740 s`. Localization still occupies about 91% of round time.

One inspected production chunk, `unwhitened_rho_0011_vectorized_full/chunks/chunk_000100.npz`, contains 34,614 accepted events but only 12 distinct `(local_coords, mask)` geometry configurations. Individual 2,048-event blocks contain 9--12 configurations, with the largest configuration repeated 520--755 times. The current `maths_0010._choose` path nevertheless rebuilds the same coarse footprints independently for every event.

## Scope and invariants

- Preserve one-second chunks, four pursuit rounds, Q8, `fit_batch_size=2048`, peak ordering, batch boundaries, sequential subtraction, acceptance gates, and checkpoint format.
- Preserve candidate enumeration, strict comparison operators, tie ordering, and the current first-improving continuous line-search rule.
- Do not adopt the historical 8,192-fit/four-second superbatch settings. That experiment changed residual state and matched only 67.38% of baseline events.
- Optimize the active `whitening="none"` identifiable-rho path first. Generalized whitening support is a later extension with separate equivalence checks.
- Keep the active job's output untouched. All validation and timing runs use independent directories under `runs/` and logs under `residuals/slurm_logs/`.

## Progress (2026-08-27)

- Reference benchmark `16491074` completed in 47 s. On chunk 100's fixed 2,048-event fixture, the explicit per-event identity-transform reference had median wall time 0.701725 s and peak allocated CUDA memory 1,941,805,056 bytes. All ten repeated calls were bitwise identical.
- Phase 1 is implemented in commit `d181044`: `raw_residual.localize_configured` passes `spatial_transform=None` for `whitening="none"`, and `maths_0010` bypasses identity `einsum` calls while retaining the nonidentity path unchanged.
- Identity-fast-path benchmark `16492029` completed in 53 s. Its candidate had median wall time 0.621564 s (11.4% lower) and peak allocated CUDA memory 1,907,726,336 bytes. Every captured field was bitwise identical to the explicit-identity reference in all ten candidate repetitions: sources, grids, profile/temporal indices, sigma, rho, alpha, captured energy, predictions, and continuous-refinement outputs.
- Accidental full-rho resume submission `16491972` was cancelled before it started. It did not modify the checkpointed full-run output.
- Full-rho submission `16492372` failed after 32 s (`ExitCode 1`) because it used the `pytorch` overlay, which lacks the runner's `spikeglx` dependency. It did not process recording data or alter checkpoints.
- Corrected full-rho resume `16492664` is PENDING on `a100_tandon` with reason `QOSGrpGRES`. It uses the `ibl-sorter` overlay and resumes `unwhitened_rho_0011_vectorized_full` with the Phase-1 identity path.
- The prior full jobs were terminated after about 2 h 22 min under suspected low-GPU-utilization enforcement. Phase 1 reduces fixture wall time by 11.4% but does not remove the synchronization-heavy rho backtracking or establish improved GPU utilization, so the same SIGTERM risk remains material.
- Phase 1 alone does not meet the 20% promotion gate. Continue with the GPU-constant workspace, geometry footprint cache, and ordered-backtrack vectorization before considering a full-recording run.

## Phase 0: Reference capture and stage instrumentation

1. Add lightweight timers inside `maths_0010.localize_spikes_fixed_codebook` for temporal projection, coarse assignment, six-level discrete refinement, profile refit, continuous refinement, prediction assembly, and device-to-host output.
2. Construct deterministic localization inputs from an existing saved chunk: 2,048 waveforms, their local coordinates and masks, the frozen Q8 codebook, and identity spatial transforms. The fixture need not reproduce a saved pursuit decision; it must run both reference and candidate localizers on exactly the same values.
3. Capture all reference fields before optimization: `sources`, `sources_grid`, `profile_idx`, `sigma`, `rho`, `temporal_idx`, `alpha`, `captured_energy`, `prediction`, `continuous_displacement_um`, and `continuous_energy_gain`.
4. Record peak CUDA memory and warmed stage times. Do not use another full Kineto trace; the session-005 trace already establishes the small-kernel/synchronization problem.

## Phase 1: Identity-transform fast path

The current unwhitened run receives an exact global identity whitening matrix, but `raw_residual.local_whitening_transforms` constructs an `N x 8 x 8` identity-like tensor for every fit batch. `maths_0010` then applies those identities through `einsum` during coarse assignment, discrete refinement, every continuous proposal evaluation, and final reconstruction.

Implementation:

1. In `raw_residual.localize_configured`, pass an explicit identity/none transform mode when `config.whitening == "none"`; do not build per-event matrices.
2. Allow `maths_0010.localize_spikes_fixed_codebook` to accept `spatial_transform=None` as the identity transform.
3. In `_atoms`, `_selected_monopole_atoms`, `_continuous_refine`, and `_continuous_refine_rho`, normalize `raw * mask` directly for the identity case and retain the existing transform multiplication for nonidentity modes.
4. Keep the nonidentity implementation unchanged in this phase.

Acceptance:

- Discrete indices and temporal assignments must be identical.
- Fitted states, predictions, energies, and accept/reject decisions must be bitwise equal where practical; otherwise require documented float32 roundoff with no threshold or ordering changes.

Status: passed on the deterministic 2,048-event CUDA fixture; end-to-end one-second raw-recording validation remains pending after the combined optimization.

## Phase 2: Cache GPU constants and fixed tensors

Create a job-scoped rho-localizer workspace rather than rebuilding constants on every 2,048-fit call.

Cache:

- `16^3` lattice sites.
- The ordered 27-point `[-1, 0, 1]^3` refinement stencil.
- Voxel bounds and the initial refinement step.
- Profile sigmas and other kernel constants.
- Normalized frozen Omega and its row energies.
- Identity-transform mode and reusable row-index tensors.

Design constraints:

- Pass the workspace explicitly through the existing job-scoped `coarse_footprint_cache`; avoid mutable module globals.
- Key cached objects by device, dtype, lattice size, kernel/profile specification, and codebook identity so resume or a different experiment cannot reuse stale tensors.
- Treat cached tensors as immutable.
- Hoist `torch.cartesian_prod` and repeated `temporal.square().sum(...)` computations out of the six refinement levels.

Acceptance:

- Repeated calls reuse the same tensor allocations and return equivalent results.
- A changed codebook, device, dtype, or spatial model causes a cache miss rather than stale reuse.

## Phase 3: Q12-style geometry grouping and footprint cache

Port the Q12 solver's central optimization: compute spatial dictionaries once per repeated neighborhood geometry and apply them to every event in that group.

Implementation for `whitening="none"`:

1. Build a deterministic configuration ID for every event from exact `local_coords` and `mask` values. Preserve original event row order for outputs.
2. Group rows by configuration without changing candidate, profile, temporal-row, or event ordering.
3. For each configuration, compute and cache normalized coarse atoms over the same ordered `16^3 x profile` candidates. For 12 production configurations, the full float32 `12 x 4096 x 8 x 9` spatial cache is about 14 MiB.
4. Retain candidate blocking for response/score tensors so `batch x candidates x Q x profiles` intermediates remain bounded.
5. Reuse the cached atoms across fit batches, pursuit rounds, and recording chunks.
6. Restore results to original event order before discrete refinement and output.

Later nonidentity extension:

- A cache key must include the effective local transform, or an equivalent stable neighborhood/whitening identifier. Do not assume equal local geometry implies equal transforms under diagonal, local-ZCA, or full-ZCA whitening.

Tie-order invariant:

- Candidate traversal remains site-major with the existing profile and temporal order. Use strict `>` updates exactly as the reference path does so equal scores keep the same earlier candidate.

Acceptance:

- Cache-enabled and cache-disabled assignment indices are identical on deterministic batches covering all 12 observed configurations, boundary masks, and repeated ties.
- Instrumentation confirms one footprint construction per configuration/cache key rather than per event.

## Phase 4: Vectorize all 30 line-search backtracks

### Meaning of careful ordering

The current continuous rho line search does not choose the maximum-energy proposal among 30 step sizes. For every event and outer iteration it tests, in order,

`0.5, 0.25, 0.125, ..., 0.5 * 2^-29`

and accepts the first proposal whose energy is strictly greater than the current energy. Once an event accepts, later backtracks cannot replace that choice. A vectorized implementation must therefore select the lowest backtrack index with a strict improvement. Selecting the best-scoring candidate, the last improving candidate, or using `>=` would change fitted locations.

Implementation:

1. Materialize the 30 exact power-of-two step sizes in their current order.
2. Form all clamped proposals as a `batch x 30 x 3` tensor for the current gradient direction.
3. Evaluate their energies in one batched operation, producing `batch x 30` scores.
4. Form `improving = proposal_energy > current_energy[:, None]` using the existing strict comparison.
5. For each event, select the first true index; if none exists, retain the current state and energy unchanged.
6. Preserve the current per-event clamping bounds and fallback to the discrete grid when final energy is invalid or worse.
7. Replace the host `bool(improved.any())` termination with a fixed, masked 80-iteration schedule so the refinement graph has no per-iteration device-to-host branch. Once no proposal improves an event, its state remains frozen. Confirm that continuing frozen iterations cannot change the reference state.
8. After equivalence, test `torch.compile` or CUDA graph capture on the fixed-shape refinement. Pad the final short fit batch with an explicit validity mask rather than compiling many dynamic shapes.

Memory:

- At batch size 2,048 and eight channels, the proposal state and raw atom tensors for 30 backtracks are only a few MiB in the identity path. Keep a backtrack-block fallback if measured peak memory is unexpectedly larger because of autograd intermediates.

Acceptance:

- First-improving indices match a scalar/reference implementation on synthetic cases covering improvement at backtracks 0, 1, 29, and no improvement.
- Final states and energies match the original loop on random and real deterministic batches.
- No event crosses an acceptance, RMSE, captured-fraction, or energy-drop threshold solely because of the optimization.

## End-to-end validation

Run validation in the project Singularity environment only.

1. CPU functional checks for identity and nonidentity transform branches, cache invalidation, candidate ties, boundary clamping, first-improving ordering, and no-improvement behavior.
2. A short CUDA function benchmark comparing the unchanged reference localizer and each optimization phase on identical 2,048-event inputs after warmup.
3. A separate one-second raw-recording smoke run with production settings and an independent output directory.
4. Compare event keys `(spike_time, spike_channel, residual_pass)`, counts by pass, sources, rho, temporal/profile indices, gains, captured energies, predictions, and round energy drops.
5. Require identical event keys, discrete decisions, pass counts, and stopping decisions. Float differences must remain within a predeclared tolerance and must not alter any gate.
6. Benchmark at least ten warmed pursuit rounds. Report median and range for every internal localization stage, total round time, peak GPU memory, and one-shot SLURM GPU utilization.

Do not submit another full-recording run unless the combined implementation is output-equivalent and produces a material warmed localization improvement. A reasonable promotion gate is at least a 20% median localization reduction without higher peak memory that threatens the allocation.

## Commit sequence

Keep changes reviewable and independently reversible:

1. Add localization stage instrumentation and the deterministic equivalence harness.
2. Add identity-transform handling.
3. Add the job-scoped GPU-constant workspace.
4. Add geometry grouping and cached coarse footprints.
5. Add first-improving vectorized backtracks and synchronization removal.
6. Add compilation only after the eager vectorized implementation passes equivalence.

Do not combine scientific changes, larger batches, direct-rho dictionary redesign, or pursuit-lockout changes with these performance commits.

## Target files

- `residuals/src/maths_0010.py`
- `residuals/src/preprocessing/raw_residual.py`
- A session-0013 validation/benchmark launcher under `residuals/src/preprocessing/`
- Independent validation outputs under `runs/` and logs under `residuals/slurm_logs/`

## Links

- [[session-014-xyzsigma-residual-pursuit]]
- [[session-011-identifiable-rho-localization]]
- [[session-005-residual-profiler]]
- [[archive/session-003-q12-temporal-codebook]]
