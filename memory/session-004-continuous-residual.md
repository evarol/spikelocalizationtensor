# Continuous Residual Subtraction (004)
**Created:** 2026-08-22
**Last updated:** 2026-08-24
**Status:** Done — continuous refinement works and is audited; full runs were blocked by low GPU utilization; the identifiability audit reframed the spatial model

## What was built

On top of 002's integer-grid smoke (solver commit `79b8beb`), 004 added
bounded continuous refinement inside the winning ±0.5 µm voxel cell
(`localize_spikes_fixed_codebook`), recomputing the continuous source, gain,
captured energy, prediction, displacement, and energy gain before every
subtraction; both continuous `sources` and integer `sources_grid` are saved.
Safeguards against over-subtraction: each atom's gain is re-fit against the
current residual after earlier overlapping atoms are removed, and an atom is
accepted only if that sequential subtraction lowers actual residual energy
and still meets `min_captured_fraction`. A pass whose core-energy drop falls
below `min_pass_energy_drop_fraction` (default 0.01) is rolled back
entirely and peeling stops.

Performance work: template scoring and temporal/spatial peak selection moved
onto the GPU (only selected candidates cross to CPU; template time batch
4096 instead of 512), coarse spatial footprints are cached across fit
batches, passes, and chunks, and coarse localization can batch up to 32
channel-neighborhood configurations per tensor operation
(`--localization-config-batch-size`, default 32) instead of separate small
`einsum` launches. Nine synthetic checks passed in Singularity: CPU/GPU
peak-selection parity, off-grid continuous localization, cached-fit and
config-batch equivalence, overlap-safe gain fitting, rollback, and the
pipeline end to end.

Reference points: the local IBL sorter uses regularized template maxima
above Th² (defaults Th = [6, 3]), an amplitude penalty lam = 10, and 60
matching-pursuit iterations per batch, with no whole-pass energy rollback;
it learns up to ~4 × channels unit-specific rank-3 templates, whereas this
model uses analytic spatial footprints × one hard-selected global temporal
row — so threshold 6 is not numerically comparable. Environment:
`pytorch.ext3` lacks `spikeglx` and `pytest`; the full raw-recording runtime
is `ibl-sorter.ext3` (used without importing `iblsorter`).

## Runs

- Continuous smoke `16180032`: exit 0:0 in 8m42s. 332,012 events (33,201/s);
  pass counts 82,826 / 83,172 / 83,503 / 82,511; 99.92% of saved sources
  non-integer with median displacement 0.503 µm (max capped at the
  voxel-corner distance 0.866 µm); captured fraction declining 0.191 →
  0.152 → 0.128 → 0.111; no pass triggered the 1% rollback; cumulative
  remaining RMS 78.0–81.7% (mean 79.7%). Monotone gain re-fitting fixed
  002's RMS regression, but nearly flat pass counts plus a material
  fourth-pass energy drop mean biological over-subtraction was not ruled
  out.
- Full Q8 job `16180078` was cancelled by request after 1h57m; the first 127
  one-second checkpoints survive under
  `runs/dataset1_p1/raw_template_residual_continuous/chunks/`. They are
  valid partial results, reusable only by a numerically compatible resumed
  implementation — any changed detector/subtractor restarts in a fresh
  directory.
- Optimized full job `16255197` was admin-cancelled after 2h07m on the
  cluster's low-GPU-utilization policy, at 267 of 1,958 chunks with
  8,902,883 events. Steady state was ~55.7 s per chunk (projecting ~30 h)
  with SM utilization 22.9% on `gl045` — far below the 50% requirement. The
  GPU-native peak selection and footprint caching brought pass time to
  ~7.08 s (~27% faster) and utilization to ~38%, but the cached path, while
  event-equivalent, is not bitwise-compatible with the uncached checkpoints.
- Four-second benchmark `16257633` (4 s chunks, 8,192-event batches,
  40,000 peaks/pass): 5.27× faster at steady state (~5.39 s per recorded
  second) after ~475 s of one-time startup, but scientifically
  non-equivalent — 667,189 vs 662,379 events on the same 20 s, only 67.38%
  shared event keys, 64.13% same coarse grid, 2.22% identical continuous
  sources. Its settings must not replace the baseline launcher without a
  redesign or explicit acceptance as a changed model. (This benchmark did
  save residual waveforms, which 008's whitening diagnostic later reused.)

Residual-pass diagnostics (`plot_residual_pass_diagnostics.py`): core
counts per pass 8,304 / 8,381 / 8,402 / 8,270; later-pass events within
0.5 ms / 48 µm of an earlier-pass event: 14.6% / 25.7% / 33.5% for passes
2–4; temporal-row usage shifts materially (row 1: 7.1% → 19.9%).

## The identifiability audit (the lasting result)

For the isotropic monopole with a free event gain, the normalized spatial
shape depends on depth z and scale sigma only through the effective width
`rho = sqrt(z² + sigma²)` — the data cannot separate them. The ten-scale
grid is therefore a redundant sampling of effective widths, not ten
distinct physical source classes; saved `z`, `sigma`, and profile index
must never be interpreted independently (use rho). A joint free-range (z,
sigma) optimization is ill-posed because the objective has an exact ridge;
the proper replacement optimizes `(x, y, rho)` directly, and if physical
depth is needed, one externally justified `sigma_0` must be fixed.
Mathematical source: `docs/residual_run_math.tex` (compiled six-page note
at `out/docs/residual_run_math.pdf`). This audit is what motivated 0011's
identifiable-rho solver.

## Handoff

The pending items — calibrate threshold 6, the 0.05 capture cutoff, and the
1% pass rule before treating output as scientific; determine whether
later-pass atoms are repeat decompositions or genuine overlaps; pipeline
CPU I/O if utilization stays low — all flowed into
[[session-005-residual-profiler]] and the rho lineage.

## Links

- [[session-007-global-codebook-pursuit]]
- [[session-005-residual-profiler]]
- [[session-006-plots]]
- [[session-002-template-residual]]
- [[archive/session-003-q12-temporal-codebook]]
- [[project_overview]]
