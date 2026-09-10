# Primate NWB Recordings (0025)
**Created:** 2026-09-02
**Last updated:** 2026-09-03
**Status:** all eight full runs COMPLETE (2026-09-08), every one `all_passes_complete` — human1 87,743/163,192, macaque1 119,831/422,379, macaque2 4,575,270/8,777,399, macaque3 6,134,349/9,165,599 (base/mean20, q32). Seven of eight galleries built (only macaque2-base owed). The TERM trap made the policy-kill requeues uneventful. The macaque2/3 event counts are enormous (mean20 ≈ 2× base on 41–43 min recordings) and need the same quality scrutiny as the mean20 runs at home.

## Why 0025 exists

The user supplied four new primate recordings and asked for the q32 model on all
four. The 0019 lineage reads raw SpikeGLX binaries through `spikeglx.Reader` and
has no NWB support, so an adapter must land before any run. This card owns the
plan; the runs it produces are the first out-of-family test of the peeling
stack (different species, different probes, different noise floors).

## What the pipeline actually needs from a reader (explored 2026-09-02)

The import chain is `0019 → 0016 → 0014 → residuals_0012` (all
`importlib`-loaded from `residuals/src/preprocessing/`; `0016.main()` is the real
entry point, `0019.main()` monkey-patches its namespace). The recording reader
is constructed in exactly one place — `0016_onehot_lattice_peeling.py:1189-1207`
(`import spikeglx` … `reader = spikeglx.Reader(args.recording_path)`) — and the
whole pipeline touches only five attributes:

1. `reader.fs` → float Hz (feeds `make_filter` and chunk math)
2. `reader.ns` → total samples
3. `reader[t0:t1, :n_channels]` → **(time, channels) float32 volts**; the
   `[:, :n_channels]` slice in `pursue()`/detect/fit paths is what drops the
   SpikeGLX sync column; `n_channels = len(reader.geometry["x"])`
4. `reader.geometry["x"]`, `reader.geometry["y"]` → µm arrays row-aligned to
   data columns (builds the 48 µm neighborhoods; `0016.validate_config`
   hardcodes `radius_um == 48.0 and merge_radius_um == 48.0`)
5. `reader.close()` (no-op is fine)

There are **no hardcoded 384-channel, 30 kHz, or 1958-chunk constants** in the
modules — `dataset1_p1` lives only in sbatch paths. `preprocess_voltage`
(`residuals_0012.py:109-118`) takes volts (bandpass 300–6000 Hz order-3, then
per-sample median across channels); `robust_channel_noise` is MAD-based and
unit-invariant. The int16→volts factor `2.34375e-06` exists only inside
`spikeglx.Reader` (and hand-rolled in two plot scripts, which produced the
documented traps: 385-column memmap misalignment, and volts-not-counts).

## What the four NWB files contain (inspected read-only via h5py)

All four: raw `data` is **int16, shape (T, 384)**, `conversion=2.34375e-06`,
`offset=0.0` — the same volts scale as dataset1_p1's NP1.0 AP band.

| file | series name(s) | rate | samples |
|---|---|---|---|
| `human1_sub-Pt01_ecephys.nwb` (8.4 G) | `ElectricalSeriesRaw` (single) | 30000.0 | (check) |
| `macaque1_sub-L11104_ecephys.nwb` (6.9 G) | `ElectricalSeriesAPImec` + `ElectricalSeriesLFImec` — use **APImec** | 30000.0 | 17,984,796 (≈10 min) |
| `macaque2_sub-C_20220121_ecephys.nwb` (28 G) | `ElectricalSeriesAP` (LF sibling unknown — enumerate) | 30000.380672 | (check) |
| `macaque3_sub-C_20220218_ecephys.nwb` (28 G) | `ElectricalSeriesAP` (LF sibling unknown — enumerate) | 30000.380672 | (check) |

- Series names differ per file — **never hardcode the series path** (the repo's
  one existing NWB loader, `detect_peaks.py:16-26`, hardcodes
  `ElectricalSeriesAP` and would already fail on these files).
- `general/extracellular_ephys/electrodes` has **768 rows (two probes per
  file)**; the series' `DynamicTableRegion` (`electrodes` link + `idxstart/idxcount`
  or region ref) selects which 384 rows feed the data columns, **in order**.
  The adapter must resolve region → electrode rows → geometry.
- Geometry columns present: `rel_x`/`rel_y` (µm; human1 spans x 0–48,
  y 0–3820 ≈ NP1.0; macaque2 x 0–68), absolute `x`/`y` absent, `location` all
  "unknown". Probe-frame µm is exactly what the pipeline wants; verify
  per-shank origin/flip against a known probe map before trusting depth.
- Rate lives on the series' `starting_time` attrs (`rate`); the non-integer
  30000.3807 Hz is fine — the pipeline only does float math on `fs`.
- `pynwb` is **not** installed in the ibl-sorter overlay; `h5py` is. The
  adapter should be h5py-only (pynwb would need an overlay write).

## Environment check (2026-09-02)

- **pynwb is installed in none of the four overlays** (pytorch, pytorch-plotly,
  ibl-sorter, cellpose all raise `ModuleNotFoundError`). h5py 3.16.0 and
  spikeinterface 0.104.1 are in pytorch.ext3; SI 0.104.8 + h5py in ibl-sorter.
- **SpikeInterface's `read_nwb_recording` needs no pynwb**: `use_pynwb=False` is
  the default, its pynwb imports are all function-local on the pynwb branch, and
  the h5py branch's dependency closure is numpy + h5py + SI-core
  (`nwbextractors.py`; reference tree is byte-identical to installed 0.104.1).
- **Verified empirically on macaque1 in pytorch.ext3** (read-only): extractor
  built, `fs=30000.0`, `17,984,796` samples (≈10 min), 384 channels int16,
  `location` property carries the rel_x/rel_y geometry as (384, 2) µm,
  `group='Imec.'`, channel_ids from the `channel_name` column ('AP0'…),
  `return_in_uV` traces are float32 µV (absmax ≈ 1.19 mV, sane AP amplitudes).
- **Series paths must be explicit per file**: macaque1 holds two
  ElectricalSeries (`acquisition/ElectricalSeriesAPImec`,
  `acquisition/ElectricalSeriesLFImec`) and auto-resolution raises; human1
  (`ElectricalSeriesRaw`) auto-resolves, macaque2/3 (`ElectricalSeriesAP`)
  must be enumerated before launch.
- fs note: macaque1 reads exactly 30000.0; macaque2/3 carry 30000.380672.
- **pytorch.ext3 runs the full peeling machinery.** The user worried that the
  missing spikeglx package would block the residual pass entirely — it does
  not, for two reasons. First, `spikeglx` is imported lazily inside functions
  only (`residuals_0012.py:1096` in `run_recording`, `0014:521`,
  `0016:1189` in `main`); every module-level import across the
  0012→0014→0016→0019 chain is stdlib + numpy + `scipy.signal` + torch, and
  pytorch.ext3 has scipy 1.17.1 and torch 2.11.0. Second, the standalone
  copy-based design replaces the reader-construction sites, so spikeglx is
  never imported at all. Empirical proof: the 0019 CPU self-test passes in
  pytorch.ext3 (`--self-test --device cpu`). GPU availability on the compute
  nodes follows from the standard `--nv` invocation.

## Implementation plan (revised after the environment check)

Per the user's direction: **standalone copy-based module, no monkey-patching.**
Build `residuals/src/preprocessing/0025_primate_peeling.py` by copying the
relevant machinery of 0016/0014/residuals_0012/0019 into one file (the same
derivation pattern the earlier sessions used, but self-contained), replacing
the `spikeglx.Reader` construction with an NWB reader. Run on the
**pytorch.ext3** overlay — the spikeglx/ibl-sorter dependency disappears
entirely for these runs.

Merge mechanics (the risky part, learned from the 024 bare-name lesson):
concatenate the four modules' definitions into ONE namespace in precedence
order 0012 → 0014 → 0016 → 0019 so later definitions shadow earlier ones —
this reproduces what 0019's runtime monkey-patching achieves, and it also
fixes 024's failure mode automatically (`pursue()` calling `process_chunk` by
bare name resolves to the single merged definition). Mechanically: drop each
file's importlib spec-loading header, rewrite every `PIPELINE.foo` /
`OLD.foo` / `BASE.foo` qualified reference to the merged bare name, and keep
explicit module-level aliases where a patch captured an older version on
purpose (read 0019.main() for the exact rebindings). A post-merge grep for
`PIPELINE\.|OLD\.|BASE\.` outside strings must come up empty.

Reader decision (user, 2026-09-02): **SI-wrapped.** Call
`read_nwb_recording(path, electrical_series_path=…)` (installed 0.104.1, zero
installs, verified end-to-end on macaque1) and wrap it in the duck-typed
adapter — `fs` from `rec.get_sampling_frequency()`, `ns` from
`get_num_samples()`, geometry from `rec.get_property("location")` (already
region-mapped 384×2 µm), traces as raw `int16` slices times the per-channel
`gain_to_uV`/1e6 to reach volts. The vendored ~100-line h5py lift remains the
fallback if SI's wrapper misbehaves (series discovery lines 319–345, region→rows
379–400, rate/t_start 735–744, conversion/offset 843–874, geometry 770–788 of
`nwbextractors.py`).

Either way the five-attribute contract holds (`fs`, `ns`,
`reader[t0:t1, :C]` in float32 volts, `geometry["x"/"y"]` µm, `close()`), and
`preprocess_voltage` is unchanged. SI's µV scaling (`gain_to_uV` =
conversion·1e6) must not be double-applied — prefer raw int16 × conversion.

## Original plan notes (superseded where revised above)

**Design: a duck-typed `NwbReader` plus a suffix dispatch at the single
construction site.** New module `residuals/src/preprocessing/0025_nwb_reader.py`
(h5py only), and a ~3-line change in `0016.main()` replacing
`reader = spikeglx.Reader(args.recording_path)` with a dispatch on suffix
(`.nwb` → `NwbReader`, else spikeglx). No behavior change for `.ap.bin` paths;
every 0019/023/024 run stays byte-identical.

`NwbReader(path)` must:

1. Open the file, enumerate `acquisition/*` groups, pick the electrical series:
   candidate groups with an int16 2-D `data` dataset and a `starting_time`
   group; prefer one whose region selects 384 electrodes; allow an explicit
   override (flag or constructor arg) since file layouts vary.
2. Resolve the DynamicTableRegion onto the electrodes table: read the region
   indices, index the electrodes table's `rel_x`/`rel_y` with them — the
   resulting 384-element x/y arrays are `geometry["x"]`/`geometry["y"]`,
   row-aligned to data columns.
3. Expose `fs` from `starting_time` attrs (`rate`), `ns = data.shape[0]`,
   and `__getitem__` returning `np.asarray(data[t0:t1], np.float32) *
   conversion - offset` (chunked h5py reads; never materialize the file).
   `close()` closes the handle.
4. Guard: assert the data is int16 with `conversion != 0`; assert 384 columns
   only as a warning (the pipeline itself is channel-count agnostic, but the
   48 µm neighborhoods and the 300–6000 Hz filter assume an NP1-like probe).

Config/CLI: `recording_path` is already a bare positional, so nothing changes
in argparse; run-dir naming stays in the sbatch.

## Validation sequence and launch (user decisions 2026-09-02)

The user chose **both** acceptance variants, so the run matrix is 8 jobs: four
recordings × {base 20% bar, mean20} at q=32. Launch is **auto** — no review
pause after smokes; submit the full runs as soon as the validation ladder is
green.

1. Reader unit check (stdin-piped, no scratch files): open each of the four
   NWBs, print fs/ns/geometry extents/conversion; read a 1 s slice, compare
   robust noise (~µV scale expected), verify x/y row-alignment and that the
   region mapping is a permutation (no electrode row used twice).
2. CPU self-test of the merged module in pytorch.ext3 (`--self-test --device
   cpu`): 0019's self-test must pass through the merge, plus a synthetic-NWB
   adapter test (tiny h5py file mirroring the real layout: series group with
   `neurodata_type` attr, electrodes table + region, `starting_time.rate`).
3. Merge-fidelity gate (GPU sbatch): synthetic SpikeGLX fixture reused from the
   024 self-test, the SAME int16 samples also written as a synthetic NWB, then
   0019 (ibl-sorter overlay, reads .bin) vs 0025 (pytorch overlay, reads .nwb)
   with identical flags — consolidated event arrays must match to ~1e-4. This
   is the only direct check that the copy-merge reproduces the 0019 chain; stop
   before full submissions if it fails.
4. 30 s `--duration-seconds` smoke per file through the full stack
   (calibration + fit + pursue) — catches unit, geometry, and rate surprises.
5. Full runs: one parameterized sbatch (`SUBJECT` × `VARIANT` env vars, USR1
   requeue trap, `--resume`, pytorch.ext3, mem 48G) into
   `residuals/runs/primate/0025_<subject>_0019_<variant>_q32/`. Flag sets are
   the q-sweep's: base `--all-channel-min-fraction 0.2 --pass-fraction-step 0.1`,
   mean20 adds `--all-channel-rule mean-channel`, both `--q 32`, threshold 5,
   fitted projection 8, 3 passes × 1 round, mean-channel-rmse objective.
   Plot suites after runs land, per [[feedback_plot_suite_completeness]].

## Execution note

Two implementer agents were killed by host failures before writing anything
(2026-09-02); the queue attempt failed because the module did not exist. The
user's instruction for the retry: dispatch **sequential scoped agents, each
turn writing ~500 LOC** — module first, then equivalence sbatch, then smoke
sbatch, then the run sbatch + submissions — so a host kill costs one small
step, not the whole implementation.

## Watch-items

- The XYZ source lattice is fixed at x/y ∈ [−150, 150] µm, z ∈ [1, 300] µm
  (`residuals_0012.py:19-20`) — fine for NP1-like rel_y spans, but primate
  probes with different pitches will stretch channel-to-source distances; if
  fits saturate at lattice edges, revisit before interpreting results.
- Two probes per file: confirm the region really selects 384 (not 768 with
  interleaving) — column count of `data` is the ground truth.
- The 300–6000 Hz filter requires `freq_max < fs/2` — satisfied at 30 kHz.
- GPU walltime is unknown per file; estimate from `ns` after the reader check,
  then budget 24 h walltime per run like the sweep jobs.

## Implementation state (2026-09-03)

The module is not yet written. Two subagent attempts returned empty results
(the same host-failure mode recorded in the execution note), so the merge was
taken over directly. The full merge design is now specified and a generator
script exists, but it has one bug left to fix before the module can land.

### The merge design (this is the hard-won part — do not re-derive it)

The four modules are concatenated into one namespace in precedence order
0012 → 0014 → 0016 → 0019, and every cross-level reference is rewritten to a
**level-suffixed final name** rather than relying on shadowing. A name defined
at level L whose last definition is at a later level gets the suffix
`_0012`/`_0014`/`_0016`; the last-defining level keeps the bare name. This is
more robust than the card's original "rewrite to bare name" plan because it
also fixes bare-name references *inside* each level's own code (e.g. 0012's
`process_chunk` calling its own `detect_events` must not silently resolve to
0019's). The reference-resolution rules, worked out from a full AST pass over
all four files:

- `BASE.x` (any level) → the 0012-level final name of `x`.
- `OLD.x` → the 0014-level final name, except `OLD.calibration_detect` → bare
  `calibration_detect` (0019 patches it onto OLD before main runs).
- `PIPELINE.x` (only in 0019) → bare `x` for the patch-list names
  {Config, output_metadata, detect_events, process_chunk, validate_config,
  self_test, alternating_fit, pursue, empty_chunk, concatenate_parts};
  `PIPELINE.orient_omega` → `preserve_omega_polarity`; otherwise the
  0016-level final name.
- Bare `x` inside level-L code → the level-L final name, with two exceptions:
  in 0016 code, bare patch-list names resolve to 0019's (post-patch) and bare
  `orient_omega` → `preserve_omega_polarity`; in 0019 code,
  `_output_metadata_0016`/`_validate_config_0016` → `output_metadata_0016`/
  `validate_config_0016` (the pre-patch captures, which become plain aliases
  defined after the 0016 section).
- The Config chain is four distinct classes: 0012's `Config` → `Config_0012`,
  0014's → `Config_0014`, 0016's `class Config(OLD.Config)` →
  `class Config_0016(Config_0014)`, 0019's `class Config(PIPELINE.Config)` →
  `class Config(Config_0016)`. 0014's `Config.base()` returns
  `BASE.Config(...)` → `Config_0012(...)`.

Structural edits: drop each file's module docstring, the importlib header
block (`HERE`/`SPEC`/`OLD`/`BASE`/`sys.modules`/`exec`/`EPS = ...`), and the
`import importlib.util` / `import sys` lines. Drop these dead entry points:
0012's `run_recording`, `self_test`, `parse_args`, `main`; 0014's
`parse_args`; 0016's `main`; 0019's `main` and its `__main__` guard. Keep
0016's `parse_args` (it is the surviving one; add a
`--electrical-series-path` flag) and 0019's `self_test` (the surviving one).
The merged `main()` is 0016's main body reworked: `import spikeglx` +
`spikeglx.Reader(...)` replaced by `build_nwb_reader(...)`, `OLD.atomic_*` →
`atomic_*`, `OLD.calibration_detect` → `calibration_detect`,
`OLD.calibration_paths` → `calibration_paths`, `orient_omega` →
`preserve_omega_polarity`, `BASE.build_neighborhoods`/`BASE.make_filter` →
bare. The resume-error string says "0025 run".

The NWB reader (already drafted in the generator): `NwbReader` duck-types the
five-attribute contract (`fs`, `ns`, `geometry["x"/"y"]`, 2-arg
`__getitem__` returning float32 volts, `close()`). `build_nwb_reader` tries
`spikeinterface.extractors.read_nwb_recording` for fs/ns/geometry (the
`location` property), falls back to pure h5py, and always reads traces via
h5py raw int16 × `conversion` (never SI's µV gain). Series paths come from a
per-filename dict (human1 → None auto, macaque1 → `acquisition/ElectricalSeriesAPImec`,
macaque2/3 → `acquisition/ElectricalSeriesAP`) plus the CLI override. The
h5py geometry fallback resolves `series["electrodes"]` onto the electrodes
table's `rel_x`/`rel_y` (mirrors SI's `_retrieve_electrodes_indices...` +
`_fetch_locations_and_groups` in
`/scratch/ap7151/_REFERENCE/spikeinterface/src/spikeinterface/extractors/nwbextractors.py`).
A synthetic-NWB round-trip section is appended to the self-test.

### The generator and its one remaining bug

The generator is at `/state/partition1/job-16842781/opencode/merge_0025.py`
(job state dir, not in the repo — copy it into the repo or re-derive before it
is purged). It reads the four source files, applies the structural drops, then
renames tokens per the rules above and writes
`residuals/src/preprocessing/0025_primate_peeling.py`.

**Blocker:** the token renamer uses `tokenize.untokenize`, which rewrites
whitespace across the whole file — `print("x", flush=True)` becomes
`print ("x",flush =True )`. This is why the self-test marker
`print("0019 self-test passed", flush=True)` is not found in the rendered
output (the generator asserts on it), and it would make the 5000-line module
unreadable even if the assert were removed. **Fix:** replace name tokens
in-place using their `(start, end)` byte offsets on the original text instead
of `untokenize` — splice `text[:start] + newname + text[end:]` for each
renamed token, working from the end of the file backwards so offsets stay
valid. Everything else in the generator (drop logic, the resolution maps, the
adapter and main text) is believed correct.

### What remains after the module lands

Run the validation ladder from the plan, in order, before any full run:
(1) reader unit check on the four real NWBs (fs/ns/geometry extents, 1 s slice
noise, region mapping is a permutation); (2) `--self-test --device cpu` in
pytorch.ext3; (3) the merge-fidelity gate (0019 on a synthetic .bin vs 0025 on
the same samples as .nwb, events match to ~1e-4); (4) 30 s smokes per file;
(5) the 8 full runs (4 recordings × {base, mean20} at q32) into
`residuals/runs/primate/0025_<subject>_0019_<variant>_q32/`, auto-launch after
the ladder is green. The four NWBs are still in `/scratch/ap7151/_RAW_DATA/primate/`.

## Module landed and ladder queued (2026-09-03, evening)

The generator bug was fixed and the module is live at
`residuals/src/preprocessing/0025_primate_peeling.py` (5018 lines). The fix
replaced `tokenize.untokenize` with in-place splicing of renamed tokens at
their absolute byte offsets (applied end-of-file backwards so offsets stay
valid); whitespace now renders byte-identical to the sources. Four smaller
generator bugs surfaced and were fixed in the same pass: the module docstring
lost its `"""` delimiters because they were the generator string's own
delimiters; the self-test SYNTH block's first line double-indented because
the marker replacement kept the original line's indent; 0014's dead `main`
entry point survived (and referenced the dropped `parse_args_0014`), so the
drop list was extended; and 0019's three module-level pre-patch captures
(`_output_metadata_0016`, `_validate_config_0016`, `_process_chunk_0016`)
broke as assignments in the static merge and were dropped — their call sites
resolve directly to the renamed 0016 defs. `--electrical-series-path` was
also added to the surviving parse_args via a generator PRE replacement.

Structural verification on the merged file: syntax parses, zero
PIPELINE/OLD/BASE/spikeglx/importlib references outside docstrings, no
duplicate top-level defs or module-level assigns, and the Config chain is
`Config_0012` → `Config_0014` (fresh class whose `base()` builds
`Config_0012`) → `Config_0016(Config_0014)` → `Config(Config_0016)`. The
never-referenced top-level defs are exactly the superseded pre-patch variants
the design intentionally keeps.

Reader unit check passed on all four NWBs (pytorch.ext3, singularity): all
384-column regions are permutations of the 768-row electrodes tables,
conversion is 2.34375e-06 everywhere, noise is 10–24 µV, and ns is
human1 25,014,692, macaque1 17,984,796, macaque2 74,430,627, macaque3
76,657,363 (14/10/41/43 min). One card correction: human1 holds TWO series
(`ElectricalSeriesLFP` and `ElectricalSeriesRaw`), so it no longer auto-
resolves — the per-filename map pins it to `acquisition/ElectricalSeriesRaw`.

CPU self-test passed in pytorch.ext3 (`0025 self-test passed`), including
0019's self-test through the merge plus the synthetic-NWB round-trip. Per the
user's standing instruction, all Python now runs through singularity —
including stdlib-only generator runs (a bare-metal `python3` slip was called
out and corrected).

The validation ladder is queued as one SLURM dependency chain
(`residuals/src/preprocessing/0025_fidelity_gate.sbatch` +
`0025_primate_run.sbatch` with SUBJECT/VARIANT/SMOKE env vars; fixture and
comparator scripts `0025_fidelity_fixture.py`/`0025_fidelity_compare.py`).
The fidelity gate (job 16905048) truncates the first 120 s of dataset1_p1,
writes the same int16 samples as an NWB (geometry mirrored byte-for-byte from
spikeglx's NP1.0 trace_header, since the meta carries no snsShankMap), runs
0019 on the .bin (ibl-sorter) vs 0025 on the .nwb (pytorch) with identical
q32 flags, and compares every consolidated top-level npy array (integer
arrays exactly, floats to 1e-4). The four 30 s smokes (16905049–52) gate on
the fidelity gate; the eight full runs (16905057–64, 4 subjects × {base,
mean20} at q32, 24 h walltime, USR1 requeue + --resume) gate on all four
smokes. Outputs go to `residuals/runs/primate/0025_<subject>_smoke_<variant>_q32`
and `..._0019_<variant>_q32`.

**Fidelity gate canceled by the user (2026-09-03, evening).** The user
questioned why it existed and had the whole first chain (16905048–64)
canceled about three minutes into the gate's run, then asked for the fixture
and comparator scripts to be removed from the repo. The plan is now simply:
smokes gate the full runs.

The smokes then caught a real merge bug, of exactly the class an equivalence
check would have caught. The generator built its rename table with
`ast.walk`, which includes **class methods**, so method names like `get`
were level-suffixed at the wrong levels and attribute calls were mangled
(`cache.get_0016(...)`, `config.base_0016()`, `pass_counts.get_0012(...)` on
dict methods, `cache.diagnostics_0016(...)`). In the original runtime only
module-level names were ever patched, so the table must come from
module-level defs only. Fixed in the generator (table from `tree.body`,
plus attribute tokens after a `.` are never renamed) and regenerated: the
diff against the broken module is exactly 8 call sites, all attribute-call
reverts, nothing else. The first smoke chain (16905159–65) failed 3-for-3
after ~2 min each (calibration shards completed, alternating_fit crashed on
the missing `get_0016`), which auto-canceled the first full-run set; the
failed smoke dirs were deleted and the chain resubmitted as smokes
16905946–49 gating full runs 16905950–57 (4 subjects × {base, mean20} at
q32) into the same `residuals/runs/primate/` directories. CPU self-test
passes on the fixed module.

## Full-run results and the SIGTERM requeue trap (2026-09-04)

The fixed chain ran overnight: all four smokes passed (6:56–17:10 each) and
the eight full runs auto-launched. Five completed cleanly (human1-base
1:39, macaque1-base 1:11, macaque1-mean20 1:16, macaque2-mean20 9:39,
macaque3-base 8:13 — all exit 0), and three were killed by the cluster's
low-GPU-utilization policy: `CANCELLED by 0 ... DUE to SIGNAL Terminated`,
i.e. SIGTERM from uid 0, at 2:02 (human1-mean20), 2:00 (macaque2-base), and
10:00 (macaque3-mean20, killed after 8 healthy hours, so the 2h-low-util
window closed mid-run).

The sbatch trap only handled USR1 (fired 60 s before the 24 h walltime —
the mechanism the 024 convolving jobs have survived on, since their kills
are walltime-related and their util stays healthy). SIGTERM bypassed it.
Fix: `trap handle_usr1 USR1 TERM` in `0025_primate_run.sbatch` so a policy
kill now triggers the same graceful stop + `scontrol requeue` + `--resume`
path, at the cost of losing only the in-flight chunk (atomic chunk writes).
The three killed runs were resubmitted with the fixed sbatch as 16939190
(human1-mean20), 16939191 (macaque2-base), 16939192 (macaque3-mean20),
resuming from their existing chunk dirs. Caveat worth watching: if the
policy monitor keeps flagging the same phases (e.g. CPU-bound calibration
or shard building), the jobs may ping-pong at ~2 h each requeue; cheap now,
but if it happens repeatedly the low-util phases themselves need looking at
(AGENTS.md mitigations: batching, torch.compile, keeping the CPU busy).

## Plot suites for the five completed runs (2026-09-04)

Queued `0025_primate_plots.sbatch` (SUBJECT/VARIANT env vars, cpu_short,
64G) — a copy of the 0019 q-sweep suite pointed at
`residuals/runs/primate/0025_<subject>_0019_<variant>_q32`, outputs to
`residuals/out/<run>/` with the offline gallery index.html. Jobs
16946284–88 cover human1-base, macaque1-base, macaque1-mean20,
macaque2-mean20, macaque3-base; the three still-running runs get suites
once they complete. To make the two replay panels NWB-aware,
`plot_0019_recording_replay.py` and `plot_0019_full_recording_replay.py`
now dispatch on the recording suffix: `.nwb` loads the 0025 module's
`build_nwb_reader` (which resolves the per-file electrical series) and
reads volts directly, `.bin` keeps the SpikeGLX meta+memmap path unchanged.

## All eight full runs complete (2026-09-08)

The three resubmitted runs (16939190–92) all finished cleanly, so the TERM
trap did its job — the policy kills cost one requeue each and nothing else.
human1-mean20 resumed and finished in 1:23 (163,192 events), macaque2-base
in 8:14 (4,575,270), and macaque3-mean20 in just 8:50 (9,165,599) — that one
was killed at 10 h after eight healthy hours, so the resume only had the
last chunk visits and consolidation left. Every run ends
`stopping_reason: all_passes_complete`. The full matrix (q32, base / mean20):

| subject | length | base | mean20 |
|---|---|---|---|
| human1 (14 min) | 25.0M samples | 87,743 | 163,192 |
| macaque1 (10 min) | 18.0M samples | 119,831 | 422,379 |
| macaque2 (41 min) | 74.4M samples | 4,575,270 | 8,777,399 |
| macaque3 (43 min) | 76.7M samples | 6,134,349 | 9,165,599 |

The mean rule roughly doubles the yield on every subject (1.5–3.5×), the
same ordering as at home. The macaque2/3 counts are the headline: 4.6–9.2M
events from 41–43 minute recordings is of order one accepted event per
sample-row of a dense recording, and with no per-channel floor on those
runs the sigma-mix / near-surface audit matters as much here as it does for
the mean20 q64 run in [[session-023-acceptance-rule-variants]].

Plot suites 16946284–88 for the first five runs all completed exit 0
(14 PNGs + index each); the human1-mean20 and macaque3-mean20 galleries are
also already on disk with the full panel set, so the only missing gallery is
**macaque2-base** — its run finished after the suites were queued.

## Next steps

- [ ] Queue the plot suite for macaque2-base (run complete, gallery owed).
- [ ] Accepted-event quality on the primate runs: sigma mix, near-surface
      share of sigma-2 events, and an ACG short-lag check — heaviest on
      macaque3-mean20 (9.17M) and macaque2-mean20 (8.78M).

## Links

- [[session-019-all-channel-error]]
- [[session-023-acceptance-rule-variants]]
- [[session-024-convolving-detection-peeling]]
- [[session-0029-dredge-motion-primate]]
- [[project_overview]]