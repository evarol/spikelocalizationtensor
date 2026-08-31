# Peak-Channel Codebook Initialization (008)
**Created:** 2026-08-25
**Last updated:** 2026-08-26
**Status:** Done — fixed-threshold Q sweep `16358281` complete; whitening diagnostic `16360429` showed whitening makes poor captures worse; Δχ² acceptance adopted next

## Why

Session 007's codebook initialization needed 250,000 multichannel waveforms
localized on the analytic spatial grid — expensive. 008 replaces it with an
IBL-inspired peak-channel sweep: learn temporal shapes cheaply from 1-D
waveforms on the detected peak channel, then hand the frozen bank to the
existing greedy residual pursuit unchanged (template scoring, conflict-free
selection, analytic localization and reconstruction, sequential gain refit,
subtraction, rescoring).

Decisions going in: sweep `Q = 4, 8, 16, 32, 64`; learn each bank from
fresh raw events without localizing training waveforms; use one
reproducible shared training sample so only Q varies; audit the subtraction
path (sequential overlap handling, gain refit vs current residual, energy
and capture gates, whole-round rollback, core-vs-margin accounting) before
launch; verify the frozen codebook is never mutated during comparison
pursuit. On 2026-08-25 threshold calibration was removed from this
experiment entirely — every bank runs greedy pursuit at the same fixed
detection threshold 6, letting candidate counts differ with Q. (The generic
calibration utility survives for other experiments.)

An IBL finding recorded along the way: IBL does not denoise individual
events. It filters/destripes/whitens the recording; the denoised objects are
the learned prototypes and fitted templates. The peak-channel initializer
copies only the cheap isolated-event correlation-clustering part — no IBL
whitening, no rank-3 multichannel projection.

## Implementation (commit `882c680`)

`fit_raw_peak_channel_temporal_codebooks.py` plus four dedicated SLURM
launchers, legacy global-codebook scripts untouched. Training permutes
one-second chunks in `[60 s, recording end)`, detects fresh negative peaks
at threshold 6, keeps events isolated by 1 ms within the 48 µm neighborhood,
and extracts 90-sample peak-channel waveforms — up to 2,048 per chunk until
the shared 100,000-event sample is full (exact sampled times, channels,
scores, waveforms, and scanned chunks saved for audit). Waveforms are
temporally centered; hard assignments maximize absolute correlation with a
unit-normalized row; fixed assignments refit each row by its signed
least-squares numerator; all Q share one nested initialization permutation
and ten iterations. Seed standardized to 42 (an earlier fit used seed 2026;
its output is preserved and not relabeled).

## Readout

The seed-42 fixed-threshold chain (fit `16358267` → pursuit array
`16358281_[0-4]` → comparison `16358283`; an earlier matched-threshold chain
`16357483`/`16357500`/`16357508`/`16357511` died in calibration — the
Q8-at-6 target wanted 202,371 candidates but Q32 had only 201,287 local
maxima — and was superseded, not repaired) completed and wrote
`runs/dataset1_p1/peak_channel_codebook_pursuit_fixed_16358281/comparison.json`:

- Event rates nearly identical across Q: 14,274/s (Q4) to 14,992/s (Q64).
- Rounds ran essentially to the cap (57.25–60); accepted events stayed high
  into late rounds, with most late fits below 20% raw local-energy capture
  (though few pinned exactly at the 5–10% floor).
- Median captured fraction only 18.14–20.90%; mean remaining core energy
  69.6–72.4% after 60 rounds.

Plots: `peak_channel_q_sweep_16358281.png`,
`peak_channel_round_diagnostics_16358281.png`, and Q8/Q64 chunk-0 round
diagnostics under `out/plots/peak_channel_codebook/`.

Validation performed: syntax/import checks plus all four launchers in
Singularity; synthetic clustering recovered all four planted shapes
(correlations > 0.95, every row used, nMSE 0.534 → 0.309); real Q4/Q8 smoke
extracted 128 of 146 isolated events from 0.2 s and exposed a physical-voltage
scale bug in the zero-energy guard (float32 `eps` → `tiny` kept all 128
finite waveforms instead of 48); real Q64 smoke used all 64 rows (nMSE
0.0569); a controlled overlapping-subtraction check confirmed every accepted
subtraction reduced energy, with the measured global decrease matching the
sum of recorded captured energies within float tolerance.

## Whitening diagnostic — negative result

To test whether whitening reduces apparent noise in poor reconstructions
before changing the acceptance model, `plot_whitening_spike_examples.py` +
sbatch (job `16360429`, 1m07s) reloaded four pass-0 poor-capture events from
the 4-second benchmark run (captured fractions 0.094 / 0.057 / 0.072 /
0.070), rebuilt the fitted model, estimated local covariance whitening on
the same preprocessed chunk, and compared unwhitened vs whitened relative
error:

- unwhitened: [0.857, 0.882, 0.808, 0.983]
- whitened:   [0.968, 0.953, 0.893, 0.990]

Whitening made every example slightly *worse* — poor captures are not a
whitening artifact. (Caveat: the 4-second benchmark saved no residual
waveforms, so the diagnostic used pass-0 events from chunk 0, where the
exact pre-subtraction waveforms can be reloaded from raw.) Output:
`out/plots/whitening_diagnostic/whitening_spike_examples_4s_16360429.png`
plus metrics JSON.

Conclusion carried forward: the raw captured fraction is a poor significance
statistic because its denominator includes expected noise across all
neighbor channels and all 90 samples. The clean model is
`Y_s = α_s g(x_s,y_s,z_s) Ω_{a_s}ᵀ + ε_s` with acceptance via a
noise-weighted projection improvement `Δχ² ≥ τ²` — formalized in
`docs/noise_weighted_pursuit_acceptance.tex` (`pdflatex` unavailable on the
host; source validated by inspection) and taken up in
[[session-009-ibl-style-pursuit]].

## Links

- [[session-009-ibl-style-pursuit]]
- [[session-007-global-codebook-pursuit]]
- [[session-005-residual-profiler]]
