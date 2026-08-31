# Residual CPU/GPU Profiler (005)
**Created:** 2026-08-24
**Last updated:** 2026-08-24
**Status:** Done — bottleneck identified (continuous-refinement launch/sync overhead); two-chunk codebook learning is held-out neutral; 0.5% stopping keeps a stronger 32% subset but was unexpectedly slower

## Why

Before changing solver semantics, profile the existing Q8 ten-scale
continuous-residual model's one-second-chunk execution, separating CPU
orchestration, transfers, and GPU kernels inside the localization stage —
which already accounted for ~95% of steady-state pass time.

## Setup

The overlay has PyTorch 2.11.0+cu128 with Kineto (the interactive node was
CPU-only, so CUDA capture was re-validated inside the SLURM GPU job).
`raw_residual.py` can profile one complete chunk end to end — CPU/GPU
operator tables, metadata, and a Chrome/Kineto `trace.json` — with scopes
for reading, preprocessing, passes, scoring, peak selection, extraction,
localization, subtraction, and checkpointing; `maths.py` scopes localization
into configuration grouping, host-to-device transfer, temporal projection,
coarse assignment, discrete refinement, continuous refinement, and the
reconstruction round trip. The second chunk is profiled after CUDA warmup
and cache warm. Baseline scientific settings preserved throughout:
one-second chunks, 1,024-event fit batches, four passes, Q8, ten monopole
scales, 16³ coarse sites, six refinement levels, bounded continuous
refinement.

Profiling job `16272887` ran 2h20m before being cancelled on the QOS GPU
limit (SLURM average GPU utilization 43%, memory 12,024 MiB — itself a
warning sign vs the 50% policy). It completed 169 chunks and wrote all
profiler artifacts, including a 12 GB `trace.json` (4.2M kernel launches ≈
33 min of aggregation/export overhead). The trace is structurally complete
but was never fully parsed as JSON due to size.

## The answer

- Steady state: ~9.1–9.3 s per residual pass, ~8.7–9.0 s of it localization.
- In the profiled chunk: peeling 51.7 s inclusive CPU; localization 50.1 s
  (96.8%), of which continuous refinement 42.4 s, discrete refinement 4.5 s,
  coarse assignment 2.3 s, reconstruction round trip 0.7 s.
- 4,203,289 `cudaLaunchKernel` calls and 152,120 `cudaStreamSynchronize`
  calls, but only 9.753 s of CUDA kernel time — the path is dominated by
  tiny launches, CPU orchestration, and synchronization, not long kernels.
- The smoking gun: the continuous solver makes 40 localization calls per
  chunk, each up to 80 refinement iterations × 30 line-search backtracks,
  and `_line_search` evaluates `bool(live.any())` on the GPU inside the
  loop — a device-to-host sync per backtrack. That is the primary
  output-preserving optimization target.
- Device arithmetic is fragmented across hundreds of thousands of tiny
  `div` / `mul` / `sum` / masking / indexing / small `bmm` calls. Bigger
  chunks or fit batches are not safe by default: the earlier four-second
  benchmark changed sequential residual state and failed output equivalence.

## The pursuit + codebook experiment riding along

An IBL-style pursuit mode (explicit opt-in with a fresh output directory —
it changes the algorithm): each round scores the current residual, picks a
deterministic score-greedy set of time-separated peaks, sorts by time,
localizes and monotonically subtracts them, then rescores the updated
residual. All events in one group have nonoverlapping support, so the
stale-residual problem of big fit batches disappears; conflicts are
reconsidered after subtraction in later rounds. Optional seeded codebook
learning updates rows between learning chunks (minimum event count, sign
alignment, momentum blend, renormalize), then freezes for extraction —
matching IBL's learn-then-extract structure. Learned runs write
`omega_initial.npy`, `omega_learned.npy`, `omega.npy`, and
`codebook_learning_history.json`; resume is rejected if checkpoints exist
without the frozen learned bank.

Smoke `16299673` (5m13s): 59,892 events over 4 s (14,973/s); every chunk hit
the 60-round cap with 211–249 events still accepted in round 60; residual
energy 0.6924 after 60 rounds; median captured fraction fell 0.59 → 0.15,
so late rounds fit weaker structure and the fixed cap — not convergence —
stopped the run. Both learning chunks updated all eight rows (2,983–4,540
events each); rows moved only 0.8–1.5° at momentum 0.9 — stable, but not
evidence of held-out improvement. Steady rounds averaged 0.715 s (83.7%
localization); GPU utilization 40% — pursuit scheduling alone didn't fix
the bottleneck.

## The 65,610-sample ablation

Chunks of exactly 65,610 = 729 × 90 samples (2.187 s at 30 kHz; the
smallest 90-sample multiple ≥ IBL's 65,600 default), four chunks = 8.748 s.
Three GPU conditions as separate jobs so wall time and utilization stay
attributable: frozen control `16304090`, learned codebook `16304091` (two
seeded learning chunks), learned + 0.005 energy-stop `16304092`; CPU
comparison `16304093` failed on a `stopped` vs `learned_stopped` key typo
and was rerun inline after fixing (the codebook guard now tolerates
loader renormalization roundoff, max 2.98e-8). Seed 42 chose chunks 2–3
for learning, leaving 0–1 held out.

On held-out data, frozen vs learned were indistinguishable: 14,972 vs
15,001 events/s, captured fractions 0.2326 vs 0.2330, remaining core energy
0.6956 vs 0.6950. Time-matching within three samples covered 93.4% / 93.2%
of events (Jaccard 0.874); requiring the same anchor dropped to ~82.7%
(Jaccard 0.705) — learning shifts some anchor assignments without changing
aggregate metrics. Rows moved 0.68–1.40° from initialization. Verdict:
two-chunk momentum-0.9 learning is stable but scientifically neutral; test
more chunks or lower momentum before scaling it.

The 0.005 energy-stop kept 19 rounds and emitted 4,808 events/s (32.05% of
full pursuit), with every stopped event matching a full-pursuit event within
three samples and the same anchor. Its subset was stronger (mean/median
captured fraction 0.331 / 0.300) but left 0.796 core energy vs 0.695.
Strangely it ran 7m50s at 28% GPU — longer and cooler than the 6m09s / 39%
learned run — so early stopping is a quality/coverage choice, not yet a
speed optimization; the per-round timings need log/node investigation.
Figures in `out/pursuit_ablation_65610/` via `plot_pursuit_ablation.py`.

## Carry-forward

The profile evidence defines the performance program in
[[session-013-rho-localization-optimization-plan]]: remove the per-backtrack
synchronization output-preserving first, then fuse/compile fixed-shape
operations; no more full-chunk Kineto traces until event volume shrinks.
Pursuit output is a new experiment — never checkpoint-compatible with the
baseline, never resumed from existing Q8 checkpoints. And the identifiable
`(x, y, rho)` solver stays a separate scientific path, not a performance
patch.

## Links

- [[session-013-rho-localization-optimization-plan]]
- [[session-008-peak-channel-codebook-init]]
- [[session-007-global-codebook-pursuit]]
- [[session-004-continuous-residual]]
- [[project_overview]]
