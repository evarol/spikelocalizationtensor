"""Plot per-pass coordinate distributions for one 0014 pursuit chunk."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


COORDINATES = ((0, "x", "global lateral x"), (1, "y", "probe depth y"), (2, "z", "axial distance z"))


def coordinate_edges(index, positions, metadata, width):
    if index < 2:
        low = float(positions[:, index].min()) - 150.0
        high = float(positions[:, index].max()) + 150.0
    else:
        low, high = metadata["lattice_bounds_um"][0][2], metadata["lattice_bounds_um"][1][2]
    return np.arange(np.floor(low / width) * width, np.ceil(high / width) * width + width, width)


def plot_coordinate(values, passes, edges, label, chunk_index, out):
    n_passes = int(passes.max()) + 1
    counts = np.empty((n_passes, len(edges) - 1), dtype=np.int64)
    for pass_index in range(n_passes):
        counts[pass_index] = np.histogram(values[passes == pass_index], bins=edges)[0]
    fractions = counts / counts.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(13, 4.6), constrained_layout=True)
    image = axis.imshow(
        fractions, origin="upper", aspect="auto", cmap="inferno", vmin=0,
        vmax=max(0.01, float(fractions.max())), interpolation="nearest",
        extent=(edges[0], edges[-1], n_passes - 0.5, -0.5),
    )
    axis.set(
        xlabel=f"{label} (µm)", ylabel="residual pass",
        yticks=np.arange(n_passes),
        yticklabels=[f"pass {index + 1} · {counts[index].sum():,} fits" for index in range(n_passes)],
        title=f"{label} distribution by residual pass · chunk {chunk_index}",
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("fraction of accepted localizations within chunk/pass")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    figure.savefig(out, dpi=800, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--bin-width-um", type=float, default=10.0)
    args = parser.parse_args()
    if args.chunk_index < 0 or args.bin_width_um <= 0:
        raise ValueError("chunk index must be nonnegative and bin width must be positive")
    chunk_path = args.run / "chunks" / f"chunk_{args.chunk_index:06d}.npz"
    if not chunk_path.exists():
        raise FileNotFoundError(chunk_path)
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")

    with np.load(chunk_path, allow_pickle=False) as chunk:
        global_sources = np.asarray(chunk["global_sources"], dtype=np.float32)
        passes = np.asarray(chunk["residual_pass"], dtype=np.int64)
    if global_sources.ndim != 2 or global_sources.shape[1] != 3 or len(global_sources) != len(passes):
        raise ValueError("global_sources and residual_pass must be event-aligned")
    metadata = json.loads((args.run / "config.json").read_text())
    positions = np.load(args.run / "channel_positions.npy")
    args.out_dir.mkdir(parents=True)
    for index, suffix, label in COORDINATES:
        plot_coordinate(
            global_sources[:, index], passes,
            coordinate_edges(index, positions, metadata, args.bin_width_um),
            label, args.chunk_index, args.out_dir / f"{suffix}_by_residual_pass.png",
        )


if __name__ == "__main__":
    main()
