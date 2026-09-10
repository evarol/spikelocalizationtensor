# Kilosort on Macaque3 for a Head-to-Head Residual Comparison (0030)
**Created:** 2026-09-09
**Status:** Active — conversion and smoke jobs both completed clean; full Kilosort run submitted (17295617); census script validated against dataset1_p1.

## Why this exists

After [[session-0025-primate-nwb-peeling]] finished its eight primate residual-pursuit
runs, the user asked to also run Kilosort on the macaque data, so the two pipelines
can be compared head to head — both as a spike-time overlap census (the same kind
of comparison [[session-022-kilosort-baseline]] already did for dataset1_p1) and as
a literal residual-voltage-trace comparison (raw minus each pipeline's own
reconstructed spikes, plotted side by side). The user picked macaque3 (already the
active recording in [[session-0029-dredge-motion-primate]]) and chose to compare
against the mean20 q32 residual-pursuit run (9,165,599 accepted events) rather than
base.

## The blocker and how it was solved

iblsorter's `run_spike_sorting_ibl` always builds its own `spikeglx.Reader(dat_path)`
inside `main.run()`, with no parameter for handing it an in-memory recording or
explicit geometry — so the macaque3 NWB file cannot be sorted directly. The fix is
to manufacture a real SpikeGLX `.bin`/`.meta` pair from it. This turned out to be
cheap rather than risky, for two reasons.

First, a SpikeGLX `.bin` file is itself raw int16 samples scaled by a separate
conversion factor recorded in the meta — exactly the representation NWB already
uses — so copying the trace is a bit-exact column copy with no unit math at all.

Second, ibllib's `spikeglx.py` supports a hardware-agnostic geometry field,
`~snsGeomMap` (`shank:x:y:flag` per channel, in real micrometers), as an
alternative to the Neuropixels-specific `~snsShankMap` (column/row on a fixed
pitch grid). `_map_channels_from_meta` prefers `snsShankMap` when both are
present, so the synthetic meta drops that field entirely and carries only
`snsGeomMap`. This was verified directly against the real installed `spikeglx.py`
(not just read, actually executed): feeding it a synthetic meta built from
macaque3's real geometry (x spans 0–68 µm, y spans 0–3820 µm) and the sample rate
30000.380672 Hz, the parser came back with the exact same conversion factor,
2.34375e-06, that dataset1_p1's own real meta produces — confirming the reused
gain table (`~imroTbl`, `apGain=500` for every channel, left completely unchanged
from the dataset1_p1 template) really does match what the primate NWB files use,
just as the 0025 investigation had already found from the NWB side. The fractional
sample rate parsed fine as a plain float, no special-casing needed.

One quirk worth remembering: `geometry_from_meta` applies fixed corrections meant
to invert real Neuropixels manufacturing coordinates back to metal-can-relative
ones (`x = 70 - x`, `y += 20`, only for a "major_version == 1" probe). On a
non-Neuropixels probe these just mirror and shift the geometry uniformly, which
preserves every pairwise channel distance — harmless for spike sorting, so there
was no need to pre-invert them before writing the meta. The synthetic file also
drops the sync channel entirely (`snsApLfSy=<n>,0,0`, `nSavedChans=384` instead of
385) rather than carrying a dummy all-zero column — `_get_sync_trace_indices_from_meta`
reads directly off `snsApLfSy[2]`, so there is nothing else in ibllib's parsing
code that assumes a sync channel is physically present.

## What was built

- `residuals/src/preprocessing/0030_nwb_to_spikeglx.py` — writes the raw int16
  trace straight from the NWB's h5py dataset (via 0025's `build_nwb_reader`, reused
  for geometry resolution rather than re-deriving it) into a flat `.ap.bin`, and
  builds the matching synthetic `.ap.meta` from dataset1_p1's real meta as a
  template. This is a single-process sequential chunked copy — deliberately not
  SpikeInterface's `write_binary_recording`, which is what OOM-killed the sidredge
  peaks job on this exact NWB file in [[session-0029-dredge-motion-primate]]; a
  plain raw copy needs no worker pool at all, so that failure mode doesn't apply
  here.
- `residuals/src/preprocessing/0030_nwb_to_spikeglx_macaque3.sbatch` — cpu_short,
  16G, runs the above. Submitted as job **17292635**.
- `residuals/src/preprocessing/0030_kilosort_iblsorter_macaque3.sbatch` — the same
  l40s_public/USR1-trap pattern as dataset1_p1's `run_kilosort_iblsorter.sbatch`,
  pointed at the new synthetic bin, with one fix applied from the start this time:
  `trap handle_stop USR1 TERM`, since 0025 already learned the hard way that NYU's
  low-GPU-utilization policy kill sends SIGTERM directly and bypasses a USR1-only
  trap. Takes a `RUN_SUFFIX` env var so a smoke run and the full run land in
  separate output directories, and a `STOP_AFTER` env var (passed straight through
  to `run_kilosort_iblsorter.py`'s existing `--stop-after` flag) for a cheap early
  validation stage. Submitted as a smoke run, `STOP_AFTER=whitening_matrix
  RUN_SUFFIX=smoke`, job **17292807**, gated on the conversion job via
  `--dependency=afterok:17292635` — this exists specifically to catch a bad
  geometry/meta parse before spending ~2 hours of GPU time on the full sort.
  Both the conversion job and the smoke run completed clean (6:59 and 2:44), so
  the full run (no `STOP_AFTER`) was submitted as job **17295617**, landing in
  `residuals/runs/primate/kilosort_iblsorter_macaque3/`.
- `residuals/src/preprocessing/0030_ks_overlap_census.py` — matches a Kilosort
  ALF spike train against a residual-pursuit run's consolidated event tables.
  Both sides already record events in raw sample-index units at the recording's
  own sample rate (`spikes.samples.npy` on the Kilosort side, `spike_times.npy` on
  the residual-pursuit side), so the ±0.5 ms tolerance window is a direct integer
  comparison with no unit conversion — the same design [[session-022-kilosort-baseline]]
  used for dataset1_p1, whose own census scripts were never checked into the repo
  and no longer exist on job scratch, so this is a clean rebuild rather than a
  port. Computes time-only and same-channel (via `channels.rawInd.npy`) precision
  and recall, against accepted events alone and against accepted+rejected
  proposals together, the same four-way split the original census reported.
  Sanity-tested against dataset1_p1's existing Kilosort ALF output plus the
  0019-default 20% bar run and it reproduces session-022's documented number
  almost exactly (0.4805 recall of good Kilosort spikes vs the recorded 0.48),
  so the matching logic is trusted; it just hasn't been pointed at macaque3 yet
  because that Kilosort run (17295617) hasn't finished.

## What's still needed for the literal residual-trace comparison

Worked out but not yet implemented (from a parallel investigation before this
card was written): the residual-pursuit side needs no new code at all —
`plot_0019_recording_replay.py`/`plot_0019_full_recording_replay.py` already
subtract each accepted event's saved reconstructed waveform patch from the raw
trace, so pointing them at a macaque3 run is enough. For Kilosort, the recipe is
`predicted = (amplitude * template) @ whitening_mat_inv`, subtracted from the raw
trace at each spike's time — using the **raw `iblsorter/` output folder, not
`alf/`** (`alf/whitening_mat_inv.npy` is a bogus 384×384 identity placeholder;
the real dense pseudo-inverse lives in `iblsorter/whitening_mat_inv.npy`), with
`iblsorter/amplitudes.npy` supplying real nonzero per-spike scale factors
(correcting an earlier note in [[session-022-kilosort-baseline]] that claimed
these were all zero — that must have been a different array). The one
unconfirmed detail is the exact sample offset of a spike's time within its
82-sample template window (~20 samples, derived from `nt0min` padding, not yet
checked against a real aligned spike). Worth remembering going in: Kilosort's own
preprocessing (IBL destriping) differs from the residual pipeline's (bandpass +
median-CAR), so the two residual traces will be qualitatively comparable side by
side, not pixel-identical.

## Next steps

- [x] Check conversion job 17292635 and smoke kilosort job 17292807 both succeed
      — both completed clean (6:59 and 2:44).
- [x] Submit the full macaque3 kilosort run (no `STOP_AFTER`) once the smoke run
      is clean — submitted as job **17295617**.
- [x] Sanity-test `0030_ks_overlap_census.py` against dataset1_p1's existing
      Kilosort ALF output and the 0019-default 20% bar run
      (`0019_allchannel_pass3_round1_fraction20_step10_fitted8`) — reproduces
      session-022's documented recall almost exactly.
- [ ] Once macaque3's Kilosort run (17295617) finishes, run the census against
      `0025_macaque3_0019_mean20_q32`.
- [ ] Implement the literal residual-trace comparison per the recipe above, and
      verify the template-offset empirically on a real aligned spike before
      trusting the subtraction pixel-by-pixel.

## Links

- [[session-0025-primate-nwb-peeling]]
- [[session-022-kilosort-baseline]]
- [[session-0029-dredge-motion-primate]]
- [[project_overview]]
