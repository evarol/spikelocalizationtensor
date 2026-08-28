# Session 012: Standalone Residual Pursuit
**Created:** 2026-08-27
**Last updated:** 2026-08-27

## Context
Replace the repeatedly terminated residual implementations with one standalone, GPU-dense SpikeGLX pipeline. The implementation must not import any repository module and must avoid the host-memory and GPU-utilization failures recorded in session 011.

## Implementation plan

1. Read bounded recording chunks, apply a 300--6000 Hz Butterworth bandpass, and apply global common-median reference. Learn a fresh peak-channel temporal codebook from isolated events with argparse-selectable `Q`, default 8.
2. Convolve every channel with the full `Omega × sigma` detection bank. Sigma is log-spaced from 2 to 512 um by default. Merge time-adjacent proposals on geometrically neighboring channels using GPU spatiotemporal nonmaximum suppression.
3. Extract every merged event's complete 48 um channel neighborhood. Score reconstruction per channel in robust-noise units, defaulting to the worst valid channel so a one-channel-only explanation cannot win or pass acceptance.
4. Jointly test every `16^3 lattice site × sigma × Omega` combination, fit gain in closed form, and hierarchically refine the winning site with batched 27-point integer-lattice searches. Continuous refinement is intentionally excluded.
5. Subtract accepted full channel-by-time predictions on the GPU and repeat detection for four outer passes by default. Roll back a pass that does not reduce core residual energy.

## Implementation

- Standalone file: `residuals/src/preprocessing/residuals_0012.py`.
- The production path is CUDA-only. SpikeGLX read, SciPy zero-phase filtering/CMR, and checkpoint serialization remain on CPU; convolution, merging, all-combination fitting, refinement, gating, and accumulated subtraction remain on CUDA.
- Processing is bounded and resumable by recording chunk. Outputs remain sharded under `chunks/`; the script never loads or consolidates the full recording or all event outputs in host memory.
- Default controls include `--q 8`, `--outer-passes 4`, `--n-scales 9`, `--lattice-size 16`, `--radius-um 48`, `--spatial-score max-channel-rmse`, and `--max-events-per-pass 40000`.
- The script saves `x, y, z, sigma`, and `rho = sqrt(z^2 + sigma^2)`. Because the requested exhaustive monopole search has an exact z/sigma ridge, only rho is identifiable; z and sigma are deterministic minimum-error grid choices, not separately physical estimates.

## Validation

- Singularity syntax check passed.
- CPU synthetic test passed proposal merging and reconstruction with relative error `1.98943e-4`.
- CUDA real-recording smoke job `16489425` was submitted for a 0.25-second interval with the full `16^3 × 9 × Q8` search and 128-event cap. It is pending with reason `QOSGrpGRES` while the existing full session-011 job occupies the available GPU allocation.

## Links
- [[session-014-xyzsigma-residual-pursuit]]
- [[session-011-identifiable-rho-localization]]
- [[session-010-whitened-dense-pursuit]]
