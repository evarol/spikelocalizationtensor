# MEMORY.md
ANALYTIC_APPROACH

## Project Map
- [Project Overview](project_overview.md) — analytic spike localization, reconstruction, and raw-recording residual deconvolution
- [User Profile](user_me.md) — working style and constraints, including outside-sandbox SLURM commands and memory-only session notes

## Sessions
- [Rho Localization Optimization Plan](session-013-rho-localization-optimization-plan.md) — HIGH PRIORITY: output-preserving plan to port Q12-style geometry caching, skip identity transforms, cache GPU constants, and vectorize first-improving rho line searches.
- [Standalone Residual Pursuit](session-012-rho-implementation-plan.md) — HIGH PRIORITY: standalone GPU-dense `Omega × sigma` detection and exhaustive lattice pursuit implemented; CPU synthetic validation passed and CUDA smoke `16489425` is pending.
- [Identifiable Rho Localization](session-011-identifiable-rho-localization.md) — HIGH PRIORITY: rho solver passed synthetic recovery; vectorized full rho run `16482462` is running and reached chunk 339/1,958 with localization still dominating runtime.
- [Whitened Dense GPU Pursuit](session-010-whitened-dense-pursuit.md) — HIGH PRIORITY: corrected local-ZCA run `16454853` and its plots completed; 382,799 events in 8.748s require scientific-quality review against the raw-coordinate baseline.
- [IBL-Style Pursuit](session-009-ibl-style-pursuit.md) — HIGH PRIORITY: delta-chi2 gate committed and its Q8 smoke completed; exhaustive first-fit extraction `16410844` plus 28 first-fit PNG panels distinguish temporal, spatial, alignment, and fragmentation errors before further pursuit changes
- [Peak-Channel Codebook Initialization](session-008-peak-channel-codebook-init.md) — HIGH PRIORITY: fixed-threshold Q sweep `16358281` completed; whitening diagnostic `16360429` completed and showed whitening makes poor-capture error worse, so noise-weighted Delta chi^2 acceptance is next
- [Fresh-Raw Global Codebook Pursuit](session-007-global-codebook-pursuit.md) — HIGH PRIORITY: raw-only Q8/Q16/Q24/Q32 global temporal banks, matched-threshold collision pursuit, and localization-superbatch benchmark; full-recording pursuit queued over entire 1957 s with decisions pending
- [Residual and Codebook Plots](session-006-plots.md) — residual-smoke localization/reconstruction figures, exact two-detection collision examples, and dense Q8/Q12 temporal-codebook rasters
- [Residual CPU/GPU Profiler](session-005-residual-profiler.md) — 65,610-sample ablation completed: two-chunk codebook learning was held-out neutral; 0.5% stopping kept a stronger 32% event subset but was unexpectedly slower
- [Continuous Residual Subtraction](session-004-continuous-residual.md) — HIGH PRIORITY: rho identifiability; baseline 16255197 was admin-cancelled at 267/1,958 chunks, and fast benchmark 16257633 failed output equivalence
- [Q12 Temporal Codebook](session-003-q12-temporal-codebook.md) — old extracted-spike one-hot method improved from Q8 nMSE 0.4928 to Q12 nMSE 0.481783
- [Analytic Localization](session-001-analytic-localization.md) — completed masked solver, kernel comparison, continuous refinement, and controlled localization plots
- [Template Residual Detection](session-002-template-residual.md) — completed 10-second smoke run, residual diagnostics, and missing continuous refinement
