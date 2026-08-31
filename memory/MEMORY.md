# MEMORY.md
ANALYTIC_APPROACH

## Project Map
- [Project Overview](project_overview.md) — analytic spike localization, reconstruction, and raw-recording residual deconvolution
- [User Profile](user_me.md) — working style and constraints, including outside-sandbox SLURM commands and memory-only session notes

## Lessons
- [Plot Suite Completeness](feedback_plot_suite_completeness.md) — every future temporal-codebook suite must show all Omega waveforms with total and per-pass/round usage; 0018+ gated plot jobs also write a SpikeTensor-style offline index and explicitly disclose unavailable exact panels.

## Active
- [Shift-Invariant Peeling (0021)](session-021-shift-invariant-peeling.md) — draft plan: integer-lag shift bank (SpikeTensor unified.py parameterization, max_shift=10) with a post-refine lag pass; `--max-shift 0` reproduces 0019; replay-with-lag unit check mandated after their zero-lag 4× SSE bug.
- [Rank-2 Temporal Peeling (0020)](session-020-rank2-temporal-peeling.md) — draft plan: two non-negative temporal atoms through one monopole (exact 2×2 NNLS, greedy second-atom pass), specced against finished 0019 results; nothing implemented yet.
- [All-Channel-Error Peeling (0019)](session-019-all-channel-error.md) — acceptance-bar sweep at threshold 5 with otherwise identical settings: 20% bar gave 568,888 events (pass 1 collapsed on the replay duplicate wall), 5% bar gave 1.10M in pass 0 but 2,557 in pass 1 (wall confirmed worse); recording-wide pass floor replaced by per-chunk exhaustion skipping (zero-accept visits skipped later, rebuild-safe on resume); 5% run resumed as `16678849`, 10% middle-ground run `16679282` queued, plots dependent; `scontrol` release fails from the sandbox — `scancel` + `sbatch` instead.
- [Bipolar Prototype-Cone Residual Peeling (0018)](session-018-bipolar-prototype-cone-peeling.md) — full job `16626415` and plot job `16626447` completed successfully; the 16-panel output now has a SpikeTensor-style offline browser with exact-data limitations disclosed.

## Done
- [Initial-Threshold Spike Discovery (017)](session-017-initial-threshold-spike-discovery.md) — all six 1,958-chunk runs and plot suites complete; threshold-5/6/7/8 totals 2,835,467/1,784,872/1,247,808/915,348 events.
- [One-Hot Lattice Peeling (016)](session-016-one-hot-lattice-peeling.md) — learned-Omega full job `16541332` and corrected 14-panel plot job `16569753` completed.
- [Score-Calibrated XYZ-Sigma Promotion (015)](session-015-score-calibrated-xyzsigma-promotion.md) — score-8 full run complete (`16517915` + gated plots `16517916`); score-9 full run `16529272` queued.
- [XYZ-Sigma Residual Pursuit (014)](session-014-xyzsigma-residual-pursuit.md) — CUDA-valid discrete xyz-sigma pursuit with projection gating, cross-pass duplicate suppression, and a completed score-8 validation chain.
- [Rho Localization Optimization (013)](session-013-rho-localization-optimization-plan.md) — identity-transform fast path bitwise-equivalent and 11.4% faster.
- [Standalone Residual Pursuit (012)](session-012-rho-implementation-plan.md) — standalone GPU-dense `Omega × sigma` detection and exhaustive lattice pursuit implemented; CPU synthetic validation passed.
- [Identifiable Rho Localization (011)](session-011-identifiable-rho-localization.md) — rho solver passed synthetic recovery.
- [Whitened Dense GPU Pursuit (010)](session-010-whitened-dense-pursuit.md) — corrected local-ZCA run `16454853` and plots completed; 382,799 events in 8.748s.
- [IBL-Style Pursuit (009)](session-009-ibl-style-pursuit.md) — delta-chi2 gate committed and Q8 smoke completed; exhaustive first-fit extraction `16410844` plus 28 first-fit PNG panels.
- [Peak-Channel Codebook Init (008)](session-008-peak-channel-codebook-init.md) — fixed-threshold Q sweep `16358281` and whitening diagnostic `16360429` completed; whitening made poor-capture error worse.
- [Fresh-Raw Global Codebook Pursuit (007)](session-007-global-codebook-pursuit.md) — raw-only Q8/Q16/Q24/Q32 global temporal banks, matched-threshold collision pursuit, localization-superbatch benchmark.
- [Residual and Codebook Plots (006)](session-006-plots.md) — residual-smoke localization/reconstruction figures, exact two-detection collision examples, dense Q8/Q12 rasters.
- [Residual CPU/GPU Profiler (005)](session-005-residual-profiler.md) — 65,610-sample ablation: two-chunk codebook learning held-out neutral; 0.5% stopping kept a stronger 32% event subset but was slower.
- [Continuous Residual Subtraction (004)](session-004-continuous-residual.md) — baseline `16255197` admin-cancelled at 267/1,958 chunks; fast benchmark `16257633` failed output equivalence.
- [Template Residual Detection (002)](session-002-template-residual.md) — completed 10-second smoke run, residual diagnostics, missing continuous refinement.

## Archived
- [Analytic Localization](archive/session-001-analytic-localization.md) — non-residual masked solver, kernel comparison, continuous refinement
- [Q12 Temporal Codebook](archive/session-003-q12-temporal-codebook.md) — non-residual old extracted-spike one-hot method, Q8 nMSE 0.4928 → Q12 0.481783
