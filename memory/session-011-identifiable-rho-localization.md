# Identifiable Rho Localization (0011)
**Created:** 2026-08-27
**Last updated:** 2026-08-27
**Status:** Done — superseded by 0014's discrete xyz-sigma pursuit; vectorized full run reached chunk 339/1,958 while localization-bound

## Why

The monopole solver's `(z, sigma)` pair is not identifiable from the data —
only `rho = sqrt(z² + sigma²)` is (see 004's audit). 0011 replaces it with a
continuous `(x, y, rho)` spatial model, as a separate path from the legacy
raw solver and 0010's local-ZCA experiment.

## Starting point

0010's local-ZCA run `16454853` was not scientifically usable: 83.0% of fits
hit the lower z bound, 86.8% picked sigma 2 µm, and its 0.5 ms lockout plus
zero-energy-drop stopping admitted 382,799 events in 8.748 s. The
identifiable-rho synthetic check, meanwhile, recovered rho 13.998 µm from a
14 µm source, recovered the off-grid lateral position, and reached relative
reconstruction error `2.9e-9`. Restored continuous sub-voxel refinement in
`maths_0010.py` was validated by the diagonal-whitening control `16475670`
(2m51s; 36,070 events in 1 s; round energy drops 15.75% / 9.98% / 7.18% /
5.10%), with plot job `16476027`.

## Host-memory OOM and the fix

The first wave of full jobs died with SLURM host-memory OOM (`ExitCode
0:125`), not GPU OOM: `raw_residual.run_residual_extraction` loaded the
entire 1,957.19 s recording for whitening (`reader[first_sample:stop]`,
≈45 GB int16) and materialized its filtered float32 array (≈90 GB) before
`estimate_whitening` subsampled — past the 128 GB request before any work
buffers. Disabling whitening alone didn't help because the calibration
prepass ran unconditionally, so three unwhitened attempts (`16477444`–
`16477446`) OOMed too. Fix (2026-08-27): `sample_preprocessed_calibration`
now filters at most 300,000 frames sampled across the requested interval,
for every whitening mode including the noise estimate the RMSE gate needs —
restoring 006's bounded, chunked behavior.

## The runs

All flavors share the raw preprocessed coordinate (`whitening="none"`), Q8
input codebook, GPU-native temporal/spatial peak selection (replacing an
accidental Python O(N²) conflict-free peak loop), the full 1,957.1908 s
recording, chunk checkpoints with `resume=True`, and the established plus
adapted-SpikeTensor plot suite on completion:

- continuous 0010 (`16478822`) → `.../unwhitened_continuous_0010_full/`, plots `16478841`
- local 0010 (`16478823`) → `.../unwhitened_local_0010_full/`, plots `16478842`
- identifiable-rho 0011 (`16478824`) → `.../unwhitened_rho_0011_full/`, plots `16478843`
- vectorized rho (`16482462`, commit `20eb268`, on
  `torch_pr_60_tandon_advanced`) → `.../unwhitened_rho_0011_vectorized_full/`

The first three were cancelled together after ~2h22m with SIGTERM —
consistent with the cluster's low-GPU-utilization enforcement, not a code or
memory failure; their chunk checkpoints survive. The cluster lacks
`scontrol`, so these 24-hour jobs must be resubmitted manually with
`--resume` after an allocation ends.

## Why it stalled (performance lineage)

Session 005's CUDA profile is the governing evidence: continuous refinement
consumed 42.397 of 50.057 localization seconds per profiled chunk, emitting
4,203,289 kernel launches and 152,120 stream synchronizations, with only
9.753 s of actual CUDA kernel time. The primary output-preserving target is
removing the per-backtrack GPU→host condition in the continuous line search
(`bool(live.any())`), then fusing/compiling fixed-shape refinement — taken
up in [[session-013-rho-localization-optimization-plan]]. The vectorized rho
run removed per-event discrete-refinement Python and profile-index
synchronization and raised `fit_batch_size` to 2048, but that was never the
main known bottleneck. (The historical 4 s / 8,192-fit superbatch benchmark
`bfe0565` was ~5.27× faster but scientifically non-equivalent — only 67.38%
of baseline events matched — and stays inadmissible.)

When last checked, vectorized run `16482462` was 1h34m in at chunk 339/1,958
(17.3%) with no stderr. Steady-state rounds took ~3.85–3.87 s, of which
localization was ~3.50–3.52 s — still localization-bound. Recent chunks
accepted ~34.3–34.7k fits across four rounds with representative energy
drops of ~18–19%, 10–11%, 7–8%, 5–6%. The 0014 discrete `(x,y,z,sigma,q)`
pursuit then replaced this direction; see
[[session-014-xyzsigma-residual-pursuit]].

## Links

- [[session-013-rho-localization-optimization-plan]]
- [[session-012-rho-implementation-plan]]
- [[session-010-whitened-dense-pursuit]]
- [[session-004-continuous-residual]]
- [[session-006-plots]]
