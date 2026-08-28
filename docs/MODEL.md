# The model, precisely

## Reconstruction

For spike `s` with DC-removed waveform `Y_s ∈ R^{C×T}` (C=10 nearest contacts, T=90
samples at 30 kHz):

```
Ŷ_s  =  Σ_{r=1..R}  a_r · g_{n_r}  ⊗  (S_{τ_r} ψ_{q_r})        (one-hot shapes)
Ŷ_s  =  Σ_{r=1..R}  g_{n_r}  ⊗  (b_rᵀ Ω)                        (free shapes)
```

**Spatial atoms.** A dictionary of `N` atoms with learned centres `μ_n ∈ R³` and scales
`σ_n`. The footprint on a contact at offset `r_c` is an analytic kernel

```
g_n(r_c) = κ( ‖r_c − μ_n^{xy}‖² + (μ_n^z)² ; σ_n ),     ‖g_n‖ = 1 over the C contacts
```

with `κ` monopole `σ/√(d²+σ²)`, Gaussian `exp(−d²/2σ²)`, or any other entry of
`fit_lattice.KERNELS`. Because `κ` is analytic, a fitted footprint **evaluates anywhere on
the probe**, including contacts the fit never saw — this is what the 3-D viewer's
all-384-channel mode plots.

**Shapes.** A codebook `Ω = {ψ_q}_{q=1..M}`, `‖ψ_q‖ = 1`, shared by every spike.
`S_τ` is the zero-padded shift by `τ ∈ [−τ_max, τ_max]`, renormalised to unit norm.
Under `shape="onehot"` each source uses exactly one atom at one lag with amplitude
`a_r ≥ 0`; under `shape="free"` it uses a signed coefficient vector `b_r ∈ R^M`.

**Prototype prior.** With `P > 0`, each atom is tied to one of `P` *learned* prototypes
`φ_p` by a hard cone constraint `∠(ψ_q, φ_{p(q)}) ≤ θ_max`. Prototype 0 is pinned
depolarizing, prototype 1 hyperpolarizing.

Non-negativity costs nothing: atoms may take either polarity, so `a_r ≥ 0` removes the
sign redundancy of the (atom, amplitude) pair rather than expressiveness.

## Why orthonormality and the prior cannot coexist

`M` unit vectors cannot all lie within a small cone of `P ≪ M` prototypes *and* be
mutually orthogonal. `Config.validate()` raises instead of dropping one silently.

This is also the motivation for the prior. An orthonormal codebook behaves like PCA: the
first atoms are spike-like, and once those directions are used the only orthogonal ones
left are oscillatory. A large free codebook therefore drifts into a Fourier-like basis
that reconstructs well but has no interpretation as cell types.

## Inference

Matching pursuit over the product dictionary (place × shape × lag). With correlations
`A[n,k] = g_nᵀ Y_s b_k`, greedily take the `(n,k)` maximising captured energy —
positive part only when `nonneg` — then refit **all** amplitudes on the support Gram

```
G_ij = (g_{n_i}ᵀ g_{n_j}) · (b_{k_i}ᵀ b_{k_j})
```

by NNLS (one-hot) or ridge least squares (free), subtract, and repeat up to `R` times
subject to a minimum spatial separation between sources.

Two structural facts make this cheap:

* **Sources on different temporal atoms at equal lag are exactly orthogonal**, however
  much their footprints overlap, so only same-atom sources interact in the refit.
* A **spatial shortlist** (default 24 candidates, ranked by unshifted basis energy) keeps
  the product search tractable: `N × M × (2τ_max+1)` is ~1.4 GB of correlations per batch
  at `M=64` with 21 lags.

## Learning

Inference alternates with one exact codebook block, chosen by configuration:

| configuration | codebook update |
|---|---|
| free + orthonormal | weighted PCA of the residual time-scatter (eigendecomposition) |
| one-hot + orthonormal | shift-aligned orthogonal Procrustes |
| one-hot + prototypes | closed-form cone projection, then prototype refit |

Only the first is exact. Once shifted copies of an atom overlap, the alternation is
approximate, so **every basis step is a proposal**: it is scored on the fit pool, accepted
under a backtracked step length (α ∈ {1, ½, ¼}), and rolled back otherwise.

Atoms are initialised by spherical k-means over real peak-aligned waveforms of the
matching polarity. Random initialisation inside the cone made fits **non-monotone in M**
— because the projection pins most atoms to the cone boundary, the initial direction
decides which deformation each atom commits to.

## Evaluation

Variance explained is computed directly:

```
VE = 1 − Σ_s ‖Y_s − Ŷ_s‖² / Σ_s ‖Y_s‖²
```

**Not** `1 − nMSE`. The project's `nMSE` normaliser is 2.68× the true mean square of the
DC-removed target (predicting zero scores nMSE 0.374, not 1.0), so `1 − nMSE` overstates
variance explained by roughly 15 points.
