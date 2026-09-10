# Multi-Stage SBatch Wiring
**Created:** 2026-09-09
**Last updated:** 2026-09-09

## Context

A multi-stage sbatch wrapper (peaks → motion → raster) fails one latent gap at
a time: each fix reveals the next gap only after a full resubmission cycle.
The 0025 SI-DREDge chain burned three submissions this way before running
clean, wasting two GPU/queue round trips on errors that were visible in
advance by reading the code.

## Notes

The 0025 dredge chain failed three times on three separate gaps, each one
hidden behind the previous. First an argparse order bug let `--max-disp-um`
swallow the following `--output-dir` value. Once that was fixed, the motion
stage failed because the sbatch never passed `--recording-path` at all — the
stage needs the recording even though the peaks stage had already saved its
outputs. Once that was fixed, the raster stage failed the same way: its
`correct_motion_on_peaks` call also needs the recording, and nobody had wired
it. On top of that, the inner `bash -c` had no `set -e`, so a failed motion
stage cascaded into a confusing raster error that masked the real cause.

The lesson for any future multi-stage sbatch: before submitting, read every
python stage the wrapper will call and trace what each one actually needs —
arguments, env vars, input files from earlier stages — and wire all of it in
one pass, instead of assuming later stages inherit what the first one needed.
Put `set -e` at the top of the inner bash script so a stage failure stops the
chain and surfaces the real error. Pass values into `bash -c '...' bash "$A"
"$B"` as positional args ($1, $2) rather than interpolating them into the
quoted string, which is where the argparse-swallowing bug came from.

The pattern that made all three resubmissions cheap: every stage checks for a
completion marker (`_peaks_done`, `_motion_done`) plus its output files, and
skips itself when they exist. Write that guard into every stage from the
start — it turns a failed 90-minute job into a 30-second retry of just the
missing stage, and it is what let the last resubmission run raster-only.
Prefer one stage per sbatch with SLURM dependencies when stages are heavy;
the single-wrapper form is what created the arg-wiring trap in the first
place.

## Links

- [[session-0029-dredge-motion-primate]] — the chain this lesson came from
- [[feedback_plot_suite_completeness]]