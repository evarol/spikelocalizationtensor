# Continuous monopole refinement

`continuous_refine.py` adds one final localization step after the solver in
`maths.py`. It does not replace the existing fit. The complete spatial search is:

1. Search the coarse lattice in `maths.py`.
2. Search shrinking discrete neighborhoods until each spike has a winning
   integer `1 um` voxel center.
3. Optimize a continuous xyz position inside that voxel.

The third step removes the remaining grid quantization without allowing a spike
to jump to a different voxel.

## What remains frozen

For every spike, continuous refinement keeps these values from the saved fit:

- the monopole scale `sigma`;
- the temporal cookbook `Omega`;
- the selected temporal row `temporal_idx`;
- the selected spatial profile `profile_idx`;
- the valid-channel mask.

Only xyz and the corresponding closed-form scalar gain are updated. The code
does not refit `Omega`, change the temporal assignment, change `sigma`, change
the kernel, or revisit another voxel.

## Search cell

If the final discrete source is `mu_grid` and the saved voxel size is `h`, the
continuous search is restricted coordinate-wise to

$$
\operatorname{lower}
= \max(\operatorname{mu\_grid} - h/2,\ \operatorname{grid\_lower}),
$$

$$
\operatorname{upper}
= \min(\operatorname{mu\_grid} + h/2,\ \operatorname{grid\_upper}).
$$

For the current fit, `h = 1 um`, so an interior source can move at most
`0.5 um` in each direction. Cells on the edge of the full grid are clipped to
the global bounds.

## Frozen monopole objective

For source position

$$
\mu_s = (x_s, y_s, z_s),
$$

the masked monopole footprint on channel `c` is

$$
g_{sc}(\mu_s)
= m_{sc}
\frac{\sigma_s}
{\sqrt{(o^x_{sc}-x_s)^2 + (o^y_{sc}-y_s)^2
+ z_s^2 + \sigma_s^2}},
$$

where `m` is one for a real channel and zero for padding. Let the selected
temporal row be

$$
\omega_s = \Omega_{q_s},
$$

and project the measured waveform through that row:

$$
p_{sc} = \sum_t Y_{sct}\omega_{st}.
$$

After eliminating the scalar gain, the captured energy optimized over xyz is

$$
E_s(\mu_s)
= \frac{\left(g_s(\mu_s)^T p_s\right)^2}
{\left\lVert g_s(\mu_s)\right\rVert_2^2
 \left\lVert\omega_s\right\rVert_2^2}.
$$

This is the same gain-eliminated score used by the final discrete search in
`maths.py`. The gain at the accepted continuous position is recomputed as

$$
\alpha_s
= \frac{\widehat g_s^T p_s}{\left\lVert\omega_s\right\rVert_2^2},
\qquad
\widehat g_s = \frac{g_s}{\lVert g_s\rVert_2}.
$$

## Continuous step

The optimizer begins at the winning integer voxel center. It computes the
analytic gradient of the captured energy, projects outward-facing gradient
components off active voxel faces, and scales the remaining direction by the
cell widths. A backtracking line search clamps every proposal to the search
cell and accepts it only when the captured energy strictly increases and meets
the Armijo condition.

An individual spike stops when no tested step improves its score or after the
iteration limit. Consequently, the returned score cannot be worse than the
score at the integer voxel center, apart from the reported numerical tolerance.

The Hessian is evaluated once at the final position for diagnostics only. Its
three eigenvalues are computed with a closed-form symmetric `3 x 3` formula;
the optimization does not call a batched eigensolver. For an energy-drop
fraction `delta`, the reported local curvature width is

$$
w_{si}
= \sqrt{\frac{2\,\delta\,E_s}{|\lambda_{si}|}},
$$

where `lambda` is a Hessian eigenvalue. Large widths indicate that the score is
nearly flat in that local direction.

## Dataset runner

`refine_dataset_continuous.py` reads the saved monopole fit and the matching
session arrays in chunks. It writes an NPZ containing:

- integer and continuous sources, displacements, and global localizations;
- grid and continuous gains and captured energies;
- voxel bounds and boundary flags;
- final Hessian eigenvalues and curvature widths;
- the frozen assignments needed to align the result with the original fit.

The adjacent JSON summarizes displacement, energy gain, reconstruction nMSE,
boundary hits, flat cells, frozen-energy parity, runtime, and invariant checks.
Files are first written under partial names and renamed only after all chunks
finish.

The full dataset launcher is
`src/refine_dataset_continuous_monopole.sbatch`. A lightweight CPU invocation,
which must still run inside the project Singularity environment, is:

```bash
python src/refine_dataset_continuous.py \
    runs/dataset1_p1 \
    runs/dataset1_p1/gpu_fit_voxel_1um_masked_monopole.npz \
    out/continuous_smoke.npz \
    --kernel monopole --device cpu --spike-chunk 256 --max-spikes 256
```

## Checks

`test_continuous_refine.py` checks the masked objective, analytic derivatives,
closed-form eigenvalues, monotone improvement, voxel constraints, and recovery
of synthetic off-grid monopole sources. The real-data 256-spike smoke test had
zero score regressions and zero voxel escapes.
