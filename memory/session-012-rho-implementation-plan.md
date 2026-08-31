# Standalone Residual Pursuit (0012)
**Created:** 2026-08-27
**Last updated:** 2026-08-27
**Status:** Done — implemented and CPU-validated; CUDA smoke `16489425` queued behind the 0011 full run

## Why

The residual implementations kept getting terminated, so 0012 replaces them
with one standalone, GPU-dense SpikeGLX pipeline that imports nothing from
the repository and avoids the host-memory and GPU-utilization failures that
ended 0011's runs.

## What it does

`residuals/src/preprocessing/residuals_0012.py`:

1. Reads bounded recording chunks, applies a 300–6000 Hz Butterworth
   bandpass and global common-median reference, and learns a fresh
   peak-channel temporal codebook from isolated events (argparse `--q`,
   default 8).
2. Convolves every channel with the full `Omega × sigma` detection bank
   (sigma log-spaced 2–512 µm by default) and merges time-adjacent proposals
   on geometrically neighboring channels with GPU spatiotemporal NMS.
3. Extracts each merged event's complete 48 µm channel neighborhood and
   scores reconstruction per channel in robust-noise units, defaulting to
   the worst valid channel so a one-channel-only explanation can neither win
   nor pass acceptance.
4. Jointly tests every `16³ lattice site × sigma × Omega` combination, fits
   gain in closed form, and hierarchically refines the winning site with
   batched 27-point integer-lattice searches. Continuous refinement is
   deliberately excluded.
5. Subtracts accepted full channel×time predictions on the GPU and repeats
   detection for four outer passes by default, rolling back any pass that
   fails to reduce core residual energy.

The production path is CUDA-only; SpikeGLX read, SciPy zero-phase
filtering/CMR, and checkpoint serialization stay on CPU. Processing is
bounded and resumable per chunk, with sharded outputs under `chunks/` — the
script never loads the full recording or all event outputs into host memory.
Defaults: `--q 8`, `--outer-passes 4`, `--n-scales 9`, `--lattice-size 16`,
`--radius-um 48`, `--spatial-score max-channel-rmse`,
`--max-events-per-pass 40000`.

The script saves `x, y, z, sigma`, and `rho = sqrt(z² + sigma²)`. Because
the exhaustive monopole search has an exact z/sigma ridge, only rho is
identifiable — z and sigma are deterministic minimum-error grid choices, not
separate physical estimates.

## Validation

Singularity syntax check passed. A CPU synthetic test passed proposal
merging and reconstruction at relative error `1.98943e-4`. CUDA smoke
`16489425` (0.25 s of real recording, full `16³ × 9 × Q8` search, 128-event
cap) was submitted but stayed pending on `QOSGrpGRES` while the 0011 full
run occupied the GPU allocation.

## Links

- [[session-014-xyzsigma-residual-pursuit]]
- [[session-011-identifiable-rho-localization]]
- [[session-010-whitened-dense-pursuit]]
