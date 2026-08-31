# MEMORY.md
ANALYTIC_APPROACH

## Project Map
- [Project Overview](project_overview.md) — analytic spike localization, reconstruction, and raw-recording residual deconvolution
- [User Profile](user_me.md) — working style and constraints, including outside-sandbox SLURM commands and memory-only session notes

## Lessons
- [Plot Suite Completeness](feedback_plot_suite_completeness.md) — HIGH PRIORITY: every future temporal-codebook suite must show all Omega waveforms with total and per-pass/round usage; 0018+ gated plot jobs also write a SpikeTensor-style offline index and explicitly disclose unavailable exact panels.

## Sessions
- [All-Channel-Error Peeling (0019)](session-019-all-channel-error.md) — ACTIVE: full run `16655016` queued after fixing a store_true sbatch bug (`16654819` died in 42 s); plot suite `16655046` is HELD pending user review; mean-channel objective, 20%/channel bar escalating 0.1/pass, 3 passes x 1 round, GPU replay, rejection audit.
- [Bipolar Prototype-Cone Residual Peeling](session-018-bipolar-prototype-cone-peeling.md) — ACTIVE: full job `16626415` and plot job `16626447` completed successfully; the 16-panel output now has a SpikeTensor-style offline browser with exact-data limitations disclosed.
- [Initial-Threshold Spike Discovery](session-017-initial-threshold-spike-discovery.md) — ACTIVE: all six 1,958-chunk runs and plot suites are complete; threshold-5/6/7/8 totals are 2,835,467/1,784,872/1,247,808/915,348 events, and dedicated temporal-codebook waveform/usage panels were added to the suite.
- [One-Hot Lattice Peeling](session-016-one-hot-lattice-peeling.md) — HIGH PRIORITY: learned-Omega full job `16541332` and corrected 14-panel plot job `16569753` completed; full XYZ/XYZ-sigma and saved-reconstruction figures await user review before any promotion, redesign, or rerun decision.
- [Score-Calibrated XYZ-Sigma Promotion](session-015-score-calibrated-xyzsigma-promotion.md) — HIGH PRIORITY: score-8 full run COMPLETE (`16517915` + gated plots `16517916`); score-9 full run `16529272` queued; first-pass sigma lower-bound forcing is an active fitting-pathology hypothesis; score-boundary review and empirical-null calibration pending.
- [XYZ-Sigma Residual Pursuit](session-014-xyzsigma-residual-pursuit.md) — HIGH PRIORITY: CUDA-valid discrete xyz-sigma pursuit has final projection gating, cross-pass duplicate suppression, compatible waveform outputs, and a completed score-8 validation chain.
- [Rho Localization Optimization Plan](session-013-rho-localization-optimization-plan.md) — HIGH PRIORITY: identity-transform fast path is bitwise-equivalent and 11.4% faster; cache constants/geometries and vectorize ordered backtracks next.
- [Standalone Residual Pursuit](session-012-rho-implementation-plan.md) — HIGH PRIORITY: standalone GPU-dense `Omega × sigma` detection and exhaustive lattice pursuit implemented; CPU synthetic validation passed and CUDA smoke `16489425` is pending.
- [Identifiable Rho Localization](session-011-identifiable-rho-localization.md) — HIGH PRIORITY: rho solver passed synthetic recovery; vectorized full rho run `16482462` is running and reached chunk 339/1,958 with localization still dominating runtime.
- [Whitened Dense GPU Pursuit](session-010-whitened-dense-pursuit.md) — HIGH PRIORITY: corrected local-ZCA run `16454853` and its plots completed; 382,799 events in 8.748s require scientific-quality review against the raw-coordinate baseline.
- [IBL-Style Pursuit](session-009-ibl-style-pursuit.md) — HIGH PRIORITY: delta-chi2 gate committed and its Q8 smoke completed; exhaustive first-fit extraction `16410844` plus 28 first-fit PNG panels distinguish temporal, spatial, alignment, and fragmentation errors before further pursuit changes
- [Peak-Channel Codebook Initialization](session-008-peak-channel-codebook-init.md) — HIGH PRIORITY: fixed-threshold Q sweep `16358281` completed; whitening diagnostic `16360429` completed and showed whitening makes poor-capture error worse, so noise-weighted Delta chi^2 acceptance is next
- [Fresh-Raw Global Codebook Pursuit](session-007-global-codebook-pursuit.md) — HIGH PRIORITY: raw-only Q8/Q16/Q24/Q32 global temporal banks, matched-threshold collision pursuit, and localization-superbatch benchmark; full-recording pursuit queued over entire 1957 s with decisions pending
- [Residual and Codebook Plots](session-006-plots.md) — residual-smoke localization/reconstruction figures, exact two-detection collision examples, and dense Q8/Q12 temporal-codebook rasters
- [Residual CPU/GPU Profiler](session-005-residual-profiler.md) — 65,610-sample ablation completed: two-chunk codebook learning was held-out neutral; 0.5% stopping kept a stronger 32% event subset but was unexpectedly slower
- [Continuous Residual Subtraction](session-004-continuous-residual.md) — HIGH PRIORITY: rho identifiability; baseline 16255197 was admin-cancelled at 267/1,958 chunks, and fast benchmark 16257633 failed output equivalence
- [Template Residual Detection](session-002-template-residual.md) — completed 10-second smoke run, residual diagnostics, and missing continuous refinement

## Archived
- [Analytic Localization](archive/session-001-analytic-localization.md) — non-residual masked solver, kernel comparison, continuous refinement
- [Q12 Temporal Codebook](archive/session-003-q12-temporal-codebook.md) — non-residual old extracted-spike one-hot method, Q8 nMSE 0.4928 → Q12 0.481783
