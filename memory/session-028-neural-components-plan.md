# Neural Components in the Peeling Pipeline (028)
**Created:** 2026-09-05
**Last updated:** 2026-09-05
**Status:** draft plan — nothing implemented; spec written to `docs/TICKET_neural_components.md`

## The idea in one paragraph

Every decision in the peeling pipeline is hand-designed: detection at a
fixed 5σ threshold, acceptance by a hand-picked aggregation rule over
per-channel captured fractions, fitting by frozen one-atom NNLS. Each of
those decision points is a slot where a small learned component could go,
and the pipeline's structural ceilings are exactly the things a learned
component could attack: the 5σ floor caps yield and produces the ≥5σ
leftovers that re-detect and die as duplicates (the duplicate wall), and
the acceptance boundary is currently explored one scalar at a time by grid
sweeps while ~27M labeled proposals with full features sit unused on disk.

## The three insertion points, in risk order

**Phase 1 — learned acceptance rule (recommended first, CPU-only).**
Replace the trimmed gate stack (rmse ≤ 3 + all-channel bar + duplicate)
with a classifier over proposal-level features: the per-channel fraction
vector `f`, event-level `F`, worst-channel fraction, RMSE, detection
score, sigma, pass index. The 0019 runs already log every proposal with
these features (the 027 census), so the dataset exists with no new GPU
hours; the one missing tensor is per-channel `channel_input` (the §8.5
save). Start with logistic regression on ~10 scalars, then a small MLP on
the raw `f` vector; the hand rules (min/mean/k-of-n/power means, math doc
§4/§8) are the baselines, and the running Q32 f₀ sweep is the yield/quality
curve the learned rule has to dominate. The load-bearing risk is label
provenance: our own gate's decisions carry selection bias, so independent
labels are mandatory — synthetic injections through the real pipeline, or
cross-labels from the Kilosort baseline (`session-022`), with our gate's
labels usable only as a comparison set. Post-hoc evaluation cannot measure
pass-1+ behavior (acceptance feeds back into the residual), so any
surviving rule needs one closed-loop GPU run.

**Phase 2 — learned detector / leftover-vs-event discriminator (GPU).**
Attack the duplicate wall directly: either detect at 4σ and let a small
spatiotemporal CNN filter proposals (relax the yield ceiling while keeping
precision), or classify each pass-1+ re-detection as *distinct event* vs
*shadow of a prior fit* using the local residual and the prior event's
stored prediction. The second variant is the direct mechanism to unlock
the pass-1+ yield every sweep so far concedes. The 024 convolving
detection machinery is the natural detector backbone.

**Phase 3 — learned waveform predictor (proposal only).** A network
mapping the fit window to the full predicted spike field — the
`shape="free"` model of `MODEL.md` with an encoder producing `b_r`. Replay
and resume survive unchanged (they need saved predictions, which a
predictor still emits), but this abandons the interpretable
`(x, y, z, σ, q, α)` parameterization the localization story rests on —
only sensible as an additive correction on top of the analytic fit.

**Out of scope, deliberately:** end-to-end learned peeling (breaks
replay/exhaustion determinism and the rejection audit), a learned codebook
(Q32→Q64 bought only +2–4%, the basis is not the bottleneck), and
replacing the subtraction arithmetic (exact given a model).

## First steps when picked up

1. Save `channel_input` next to `channel_improvement` in 0019 (math doc
   §8.5 — one cheap change, enables all offline rule evaluation).
2. When the 027 Q32 f₀ sweep lands, export proposal-level feature tables
   from its consolidated rejected/accepted tables.
3. Logistic gate on scalars; compare offline against the sweep curve
   before any GPU run.

## Links

- [[session-027-acceptance-gate-census-trim]] — the census is phase 1's training data
- [[session-022-kilosort-baseline]] — independent labeling source
- [[session-024-convolving-detection-peeling]] — phase 2 detector backbone
- `docs/TICKET_neural_components.md` — the same plan as an actionable ticket
- `docs/0019_acceptance_mathematics.md` — gate math and §8 design space
