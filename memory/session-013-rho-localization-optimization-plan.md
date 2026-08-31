# Rho Localization Optimization Plan (0013)
**Created:** 2026-08-27
**Last updated:** 2026-08-27
**Status:** Phase 1 (identity fast path) done and bitwise-verified, 11.4% faster; phases 2–4 never started

## Situation

Output-preserving speedup of 0011's identifiable-rho localization by reusing
003's batching structure. Commit `20eb268` had already cut steady-state
localization from 14.46 s to 3.52 s per pursuit round (4.1×), with total
round time 14.98 s → 3.87 s — but localization still consumed ~91% of round
time. One inspected production chunk
(`unwhitened_rho_0011_vectorized_full/chunks/chunk_000100.npz`) held 34,614
accepted events but only 12 distinct `(local_coords, mask)` geometries —
yet `maths_0010._choose` was rebuilding identical coarse footprints for
every event.

Ground rules for the whole plan: preserve one-second chunks, four pursuit
rounds, Q8, 2,048-event fit batches, peak ordering, batch boundaries,
sequential subtraction, acceptance gates, and checkpoint format; preserve
candidate enumeration, strict comparisons, tie ordering, and the
first-improving line-search rule; never adopt the historical 8,192-fit /
four-second superbatch settings (they changed residual state and matched
only 67.38% of baseline events); optimize the `whitening="none"` path first,
with whitened extensions later under separate equivalence checks; never touch
the active job's output — all validation and timing go to independent
directories under `runs/`, logs under `residuals/slurm_logs/`.

## Phase 1: identity-transform fast path — DONE

The unwhitened run receives an exact identity whitening matrix but was
building an N×8×8 identity-like tensor per fit batch and pushing it through
`einsum` in coarse assignment, discrete refinement, every continuous
proposal evaluation, and final reconstruction. Fix (commit `d181044`):
`raw_residual.localize_configured` passes `spatial_transform=None` when
`whitening == "none"`, and `maths_0010` normalizes `raw * mask` directly for
the identity case while keeping the nonidentity path unchanged.

Verification: on the deterministic 2,048-event CUDA fixture, the fast path
ran 0.6216 s median wall time vs 0.7017 s for the explicit-identity
reference (11.4% faster; peak allocated CUDA memory 1.908 GB vs 1.942 GB)
and was bitwise identical in every captured field across ten repetitions.
Benchmark jobs: reference `16491074` (47 s), fast path `16492029` (53 s).

Job history around it: accidental full-rho resume submission `16491972` was
cancelled before it started and touched nothing. Full-rho resume `16492372`
failed after 32 s on the `pytorch` overlay (missing `spikeglx`); corrected
resume `16492664` went to `a100_tandon` on the `ibl-sorter` overlay. Caveat
kept visible: earlier full jobs were killed after ~2h22m under suspected
low-GPU-utilization enforcement, and an 11.4% fixture win neither clears the
20% promotion gate nor removes the synchronization-heavy backtracking, so
that risk stayed live.

## Phase 2: cache GPU constants (planned, not started)

Replace per-2,048-fit-call constant rebuilding with a job-scoped workspace
holding the 16³ lattice sites, the ordered 27-point `[-1,0,1]³` refinement
stencil, voxel bounds and initial step, profile sigmas, normalized frozen
Omega with row energies, identity-mode flags, and reusable index tensors.
Pass it explicitly through the existing `coarse_footprint_cache` (no mutable
module globals); key cached objects by device, dtype, lattice size, kernel
spec, and codebook identity so a resume or different experiment cannot reuse
stale tensors; treat cached tensors as immutable; hoist `cartesian_prod` and
repeated `temporal.square().sum(...)` out of the six refinement levels.
Acceptance: repeated calls reuse the same allocations with equivalent
results; any change of codebook, device, dtype, or spatial model causes a
cache miss.

## Phase 3: geometry grouping and footprint cache (planned)

Port Q12's central trick — compute spatial dictionaries once per repeated
neighborhood geometry and apply them to every event in that group. Build a
deterministic configuration ID from exact `local_coords` + `mask`; group
rows without changing candidate, profile, temporal-row, or event ordering;
cache normalized coarse atoms per configuration over the ordered 16³ ×
profile candidates (12 production configurations ≈ 14 MiB float32); keep
candidate blocking for response/score tensors; reuse across fit batches,
rounds, and chunks; restore original event order before refinement and
output. Tie order stays site-major with strict `>` so equal scores keep the
earlier candidate. A later whitened extension must put the effective local
transform in the cache key. Acceptance: cache on/off gives identical
assignment indices on batches covering all 12 observed configurations,
boundary masks, and repeated ties; instrumentation shows one footprint
construction per configuration, not per event.

## Phase 4: vectorize the 30 line-search backtracks (planned)

The current continuous line search is *first-improving*, not
best-improving: step sizes 0.5, 0.25, ..., 0.5·2⁻²⁹ are tested in order and
the first strictly-better proposal wins, after which later backtracks cannot
replace it. A vectorized version must therefore select the lowest improving
index — picking the best-scoring candidate, the last improving one, or using
`>=` would change fitted locations. Materialize the 30 exact step sizes,
form all clamped proposals as batch×30×3, evaluate energies in one batched
op, take each event's first improving index (else keep state), preserve
clamping and grid fallback, and replace the host `bool(improved.any())`
termination with a fixed masked 80-iteration schedule so the refinement
graph has no per-iteration device-to-host branch (frozen events stay
frozen). Only after equivalence, consider `torch.compile` or CUDA graphs on
the fixed-shape refinement, padding the short final batch with a validity
mask. Memory at 2,048 × 8 channels is a few MiB in the identity path, with a
backtrack-block fallback if autograd intermediates inflate peak memory.
Acceptance: first-improving indices match a scalar reference on cases
improving at backtracks 0, 1, 29, and never; final states and energies match
the original loop; no event crosses any gate because of the optimization.

## End-to-end validation before any full run

Run in the project Singularity environment only: CPU functional checks
(identity/nonidentity branches, cache invalidation, ties, clamping,
first-improving, no-improvement); a short CUDA benchmark comparing the
unchanged reference localizer with each phase on identical 2,048-event
inputs after warmup; a one-second raw-recording smoke with production
settings in an independent output directory; then compare event keys
`(spike_time, spike_channel, residual_pass)`, counts by pass, sources, rho,
indices, gains, energies, predictions, and round energy drops — requiring
identical keys and decisions, with float differences inside a predeclared
tolerance that never flips a gate. Benchmark at least ten warmed rounds and
report per-stage medians, total round time, peak GPU memory, and SLURM GPU
utilization. Promotion gate: ≥20% median localization reduction without
peak memory that threatens the allocation.

Commit sequence keeps each change independently reversible: instrumentation
→ identity path → workspace → geometry cache → vectorized backtracks →
compilation last. Never mix scientific changes, larger batches, rho
redesign, or lockout changes into these performance commits.

Target files: `residuals/src/maths_0010.py`,
`residuals/src/preprocessing/raw_residual.py`, a session-0013
validation/benchmark launcher under `residuals/src/preprocessing/`, and
validation outputs under `runs/` with logs in `residuals/slurm_logs/`.

## Links

- [[session-014-xyzsigma-residual-pursuit]]
- [[session-011-identifiable-rho-localization]]
- [[session-005-residual-profiler]]
- [[archive/session-003-q12-temporal-codebook]]
