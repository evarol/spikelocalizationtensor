"""Copy a primate NWB recording into a synthetic SpikeGLX .ap.bin/.meta pair.

iblsorter's run_spike_sorting_ibl always builds its own spikeglx.Reader(dat_path)
internally (iblsorter/main.py) with no way to hand it an in-memory recording or
override geometry directly, so an NWB file cannot be sorted as-is. SpikeGLX .bin
files are themselves raw int16 samples scaled by a separate meta conversion
factor -- the same representation NWB uses -- so the trace copy is a bit-exact
column copy, and the meta only needs to carry real geometry and sample rate.

Geometry goes through ibllib's spikeglx.py:_map_channels_from_meta, which
prefers ~snsShankMap over ~snsGeomMap when both are present, so the template's
~snsShankMap line is dropped and replaced with ~snsGeomMap (shank:x:y:flag,
integer um only -- the parser regex has no sign or decimal point). The reader
then applies fixed offsets (x = 70 - x, y += 20 for a "major_version == 1"
probe) meant to invert real Neuropixels manufacturing coordinates back to
metal-can-relative ones; on a non-Neuropixels probe those offsets just mirror
and shift the geometry, which preserves all pairwise channel distances and is
harmless for spike sorting (only the vs.-anatomy orientation is lost, and nothing
downstream of iblsorter's own output needs that here).
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "primate_peeling_0025", HERE / "0025_primate_peeling.py"
)
PEELING = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = PEELING
_SPEC.loader.exec_module(PEELING)

TEMPLATE_META = Path(
    "/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.meta"
)

# Fields the template carries that must reflect the new file rather than
# dataset1_p1's; everything else (imroTbl gain table, imDatPrb_type, ...) is
# left untouched since the 0025 investigation already confirmed the real
# meta's apGain=500 table yields the same 2.34375e-06 s2v as the NWB files.
DROP_PREFIXES = ("~snsShankMap=", "~snsChanMap=", "fileSHA1=")


def build_meta_text(template_text, n_channels, fs, ns, geometry_x, geometry_y, out_bin_name):
    lines = [
        line for line in template_text.splitlines()
        if not line.startswith(DROP_PREFIXES)
    ]
    overrides = {
        "fileName": out_bin_name,
        "fileSizeBytes": str(int(ns) * n_channels * 2),
        "fileTimeSecs": f"{ns / fs:.4f}",
        "imSampRate": repr(float(fs)),
        "nSavedChans": str(n_channels),
        "snsApLfSy": f"{n_channels},0,0",
        "snsSaveChanSubset": f"0:{n_channels - 1}",
    }
    out_lines = []
    seen = set()
    for line in lines:
        key = line.split("=", 1)[0] if "=" in line else None
        if key in overrides:
            out_lines.append(f"{key}={overrides[key]}")
            seen.add(key)
        else:
            out_lines.append(line)
    for key, value in overrides.items():
        if key not in seen:
            out_lines.append(f"{key}={value}")

    x_int = np.clip(np.round(geometry_x), 0, None).astype(np.int64)
    y_int = np.clip(np.round(geometry_y), 0, None).astype(np.int64)
    geom_entries = "".join(f"(0:{x}:{y}:1)" for x, y in zip(x_int, y_int))
    out_lines.append(f"~snsGeomMap=(0,1,1){geom_entries}")
    return "\n".join(out_lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nwb_path", type=Path)
    parser.add_argument("out_bin", type=Path)
    parser.add_argument("--electrical-series-path", default=None)
    parser.add_argument("--template-meta", type=Path, default=TEMPLATE_META)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    args = parser.parse_args()

    reader = PEELING.build_nwb_reader(args.nwb_path, args.electrical_series_path)
    n_channels = reader._n_channels
    fs, ns = reader.fs, reader.ns
    print(f"source: {args.nwb_path} fs={fs} ns={ns} n_channels={n_channels}", flush=True)

    out_bin = args.out_bin
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    tmp_bin = out_bin.with_suffix(out_bin.suffix + ".tmp")

    chunk_samples = max(1, int(args.chunk_seconds * fs))
    with open(tmp_bin, "wb") as fh:
        for start in range(0, ns, chunk_samples):
            stop = min(start + chunk_samples, ns)
            # raw int16 straight from the NWB dataset -- no unit conversion --
            # since SpikeGLX .bin files carry the same raw-int16-plus-separate-
            # conversion-factor representation.
            chunk = np.asarray(reader._data[start:stop], dtype=np.int16)
            chunk.tofile(fh)
            print(f"  wrote {stop}/{ns} samples", flush=True)
    tmp_bin.rename(out_bin)

    template_text = args.template_meta.read_text()
    meta_text = build_meta_text(
        template_text, n_channels, fs, ns,
        reader.geometry["x"], reader.geometry["y"], out_bin.name,
    )
    out_meta = out_bin.with_suffix(".meta")
    out_meta.write_text(meta_text)
    reader.close()
    print(f"wrote {out_bin} ({out_bin.stat().st_size} bytes) and {out_meta}", flush=True)


if __name__ == "__main__":
    main()
