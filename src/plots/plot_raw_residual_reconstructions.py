"""Plot analytic reconstructions of saved raw-residual waveform fits."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from plot_reconstructions import profile_parameters, reconstruct


EPS = np.finfo(np.float32).tiny
PASS_COLORS = ("#482878", "#31688e", "#35b779", "#b5de2b")


def load_chunk(run, chunk_index):
    path = run / "chunks" / f"chunk_{chunk_index:06d}.npz"
    required = {
        "residual_waveforms",
        "local_coords",
        "neighbor_ids",
        "sources",
        "profile_idx",
        "temporal_idx",
        "alpha",
        "input_energy",
        "captured_fraction",
        "residual_pass",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"{path} is missing fields: {sorted(missing)}")
        return {key: archive[key] for key in required}


def choose_examples(chunk, n_passes):
    chosen = []
    for residual_pass in range(n_passes):
        rows = np.flatnonzero(chunk["residual_pass"] == residual_pass)
        if not len(rows):
            raise ValueError(f"chunk has no accepted fits in pass {residual_pass + 1}")
        strong = rows[
            chunk["input_energy"][rows] >= np.median(chunk["input_energy"][rows])
        ]
        target = np.median(chunk["captured_fraction"][rows])
        chosen.append(
            strong[np.argmin(np.abs(chunk["captured_fraction"][strong] - target))]
        )
    return np.asarray(chosen, dtype=np.int64)


def geometry_waveforms(axis, coordinates, values, scale, color, linestyle="-"):
    time_offset = (np.arange(values.shape[1]) - values.shape[1] / 2) * 0.34
    for channel, waveform in enumerate(values):
        axis.plot(
            coordinates[channel, 0] + time_offset,
            coordinates[channel, 1] + scale * waveform,
            color=color,
            linewidth=0.9,
            linestyle=linestyle,
        )


def spatial_width_image(source, rho, extent=165.0, n=241):
    grid = np.linspace(-extent, extent, n)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    distance2 = (xx - source[0]) ** 2 + (yy - source[1]) ** 2
    image = rho / np.sqrt(distance2 + rho ** 2)
    return grid, image


def plot_examples(run, chunk, metadata, output, chunk_index):
    config = metadata["config"]
    omega = np.load(run / "omega.npy")
    n_passes = int(np.max(chunk["residual_pass"])) + 1
    indices = choose_examples(chunk, n_passes)
    mask = chunk["neighbor_ids"][indices] >= 0
    measured = chunk["residual_waveforms"][indices] * mask[:, :, None]
    coordinates = chunk["local_coords"][indices]
    sources = chunk["sources"][indices]
    temporal_idx = chunk["temporal_idx"][indices]
    alpha = chunk["alpha"][indices]
    parameters = profile_parameters(
        config["kernel"], chunk["profile_idx"][indices], config["n_scales"]
    )
    footprint, predicted = reconstruct(
        coordinates,
        mask,
        sources,
        parameters,
        omega,
        temporal_idx,
        alpha,
        config["kernel"],
    )
    after = measured - predicted
    rho = np.sqrt(sources[:, 2] ** 2 + parameters[:, 0] ** 2)
    fs = float(metadata["fs"])

    figure, axes = plt.subplots(
        4,
        n_passes,
        figsize=(4.0 * n_passes, 13.0),
        constrained_layout=True,
        squeeze=False,
        gridspec_kw={"height_ratios": (1.05, 1.05, 0.7, 1.0)},
    )
    for column, index in enumerate(indices):
        valid = mask[column]
        coords = coordinates[column, valid]
        scale = 17.0 / max(float(np.abs(measured[column, valid]).max()), EPS)
        capture = float(chunk["captured_fraction"][index])

        axis = axes[0, column]
        geometry_waveforms(
            axis, coords, measured[column, valid], scale, "#e03131"
        )
        geometry_waveforms(
            axis, coords, predicted[column, valid], scale, "#2f9e44", "--"
        )
        axis.scatter(coords[:, 0], coords[:, 1], s=7, marker="s", color="0.55")
        axis.set_title(
            f"pass {column + 1} · median-quality fit\n"
            f"captured {100 * capture:.1f}% · relative residual {100 * (1-capture):.1f}%",
            fontsize=9,
        )
        if column == 0:
            axis.set_ylabel("input residual (red)\nreconstruction (green)")

        axis = axes[1, column]
        geometry_waveforms(axis, coords, after[column, valid], scale, "#3182bd")
        axis.scatter(coords[:, 0], coords[:, 1], s=7, marker="s", color="0.55")
        axis.set_title("post-subtraction local residual", fontsize=9)
        if column == 0:
            axis.set_ylabel("residual after fitted atom")

        axis = axes[2, column]
        projected = np.einsum(
            "c,ct->t", footprint[column, valid], measured[column, valid]
        )
        projected_after = np.einsum(
            "c,ct->t", footprint[column, valid], after[column, valid]
        )
        model_time = alpha[column] * omega[temporal_idx[column]]
        time_ms = 1000 * np.arange(len(projected)) / fs
        axis.plot(time_ms, projected, color="#e03131", label="input projection")
        axis.plot(
            time_ms,
            model_time,
            color="#2f9e44",
            linestyle="--",
            label="fitted atom",
        )
        axis.plot(time_ms, projected_after, color="#3182bd", label="after")
        axis.axhline(0, color="0.6", linewidth=0.6)
        axis.set(
            title=f"temporal row {temporal_idx[column]} · α={alpha[column]:.2e}",
            xlabel="waveform time (ms)",
        )
        if column == 0:
            axis.set_ylabel("spatial projection")
            axis.legend(fontsize=7)

        axis = axes[3, column]
        grid, image = spatial_width_image(sources[column], rho[column])
        axis.imshow(
            image,
            origin="lower",
            extent=(grid[0], grid[-1], grid[0], grid[-1]),
            cmap="magma",
            aspect="equal",
            interpolation="nearest",
        )
        axis.scatter(
            coords[:, 0],
            coords[:, 1],
            s=16,
            marker="s",
            facecolors="none",
            edgecolors="white",
            linewidths=0.7,
        )
        axis.plot(0, 0, "+", color="#00ffff", markersize=11, markeredgewidth=1.7)
        axis.plot(
            sources[column, 0],
            sources[column, 1],
            "o",
            markerfacecolor="none",
            markeredgecolor="#4c8dff",
            markersize=12,
            markeredgewidth=1.8,
        )
        axis.set_title(
            f"analytic footprint · identifiable ρ={rho[column]:.1f} µm",
            fontsize=9,
        )
        axis.set_xlabel("local lateral x (µm)")
        if column == 0:
            axis.set_ylabel("local depth y (µm)")

        for axis in axes[:, column]:
            axis.tick_params(labelsize=7)
            axis.grid(alpha=0.12)

    figure.suptitle(
        f"Continuous residual-smoke reconstruction examples · chunk {chunk_index} · "
        "one representative accepted fit per pass",
        fontsize=14,
    )
    figure.savefig(output / "reconstruction_examples.png", dpi=800)
    plt.close(figure)


def plot_diagnostics(run, metadata, output):
    residual_pass = np.load(run / "residual_pass.npy", mmap_mode="r")
    captured_fraction = np.load(run / "captured_fraction.npy", mmap_mode="r")
    input_energy = np.load(run / "input_energy.npy", mmap_mode="r")
    temporal_idx = np.load(run / "temporal_idx.npy", mmap_mode="r")
    omega = np.load(run / "omega.npy")
    n_passes = int(np.max(residual_pass)) + 1
    pass_numbers = np.arange(1, n_passes + 1)
    colors = PASS_COLORS[:n_passes]

    figure, axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
    samples = [captured_fraction[residual_pass == index] for index in range(n_passes)]
    violin = axes[0].violinplot(
        samples, positions=pass_numbers, showmedians=True, showextrema=False
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.78)
    violin["cmedians"].set_color("black")
    medians = [float(np.median(sample)) for sample in samples]
    axes[0].set(
        title="energy captured by one fitted atom",
        xlabel="residual pass",
        ylabel="fraction of local waveform energy",
        xticks=pass_numbers,
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))

    error_limits = np.linspace(0.45, 0.96, 350)
    for index, sample in enumerate(samples):
        error = 1 - np.asarray(sample)
        cdf = np.searchsorted(np.sort(error), error_limits, side="right") / len(error)
        axes[1].plot(
            error_limits,
            cdf,
            color=colors[index],
            label=f"pass {index + 1} · median {np.median(error):.3f}",
        )
    axes[1].set(
        title="post-subtraction relative energy",
        xlabel="relative squared error after fitted atom",
        ylabel="accepted fits at or below error",
        xlim=(error_limits[0], error_limits[-1]),
        ylim=(0, 1),
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].legend(fontsize=7)

    finite = (
        np.isfinite(input_energy)
        & (input_energy > 0)
        & np.isfinite(captured_fraction)
    )
    energy = np.log10(np.asarray(input_energy[finite]))
    image = axes[2].hexbin(
        energy,
        np.asarray(captured_fraction[finite]),
        gridsize=70,
        bins="log",
        mincnt=1,
        cmap="magma",
    )
    figure.colorbar(image, ax=axes[2], label="accepted fits")
    axes[2].set(
        title="capture versus presented residual",
        xlabel="log10 input waveform energy",
        ylabel="captured fraction",
        ylim=(0, min(1.0, float(np.quantile(captured_fraction, 0.999)))),
    )
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))

    usage = np.zeros((n_passes, len(omega)), dtype=np.float64)
    for residual_index in range(n_passes):
        for temporal_row in range(len(omega)):
            keep = (residual_pass == residual_index) & (temporal_idx == temporal_row)
            usage[residual_index, temporal_row] = np.median(captured_fraction[keep])
    image = axes[3].imshow(
        usage,
        aspect="auto",
        cmap="viridis",
        vmin=float(usage.min()),
        vmax=float(usage.max()),
        interpolation="nearest",
    )
    for row in range(n_passes):
        for column in range(len(omega)):
            axes[3].text(
                column,
                row,
                f"{100 * usage[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if usage[row, column] < np.median(usage) else "black",
            )
    axes[3].set(
        title="median captured energy (%)",
        xlabel="temporal row",
        ylabel="residual pass",
        xticks=np.arange(len(omega)),
        yticks=np.arange(n_passes),
        yticklabels=pass_numbers,
    )
    figure.colorbar(image, ax=axes[3], label="median captured fraction")

    figure.suptitle(
        f"Continuous Q8 residual-smoke reconstruction diagnostics · "
        f"{len(residual_pass):,} accepted fits · medians "
        + ", ".join(f"P{i + 1} {100 * value:.1f}%" for i, value in enumerate(medians)),
        fontsize=12,
    )
    figure.savefig(output / "reconstruction_diagnostics.png", dpi=800)
    plt.close(figure)
    return medians


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    args = parser.parse_args()

    metadata = json.loads((args.run / "config.json").read_text())
    if metadata["config"]["kernel"] != "monopole":
        raise ValueError("residual reconstruction plots currently require monopole fits")
    chunk = load_chunk(args.run, args.chunk_index)
    args.out.mkdir(parents=True, exist_ok=True)
    plot_examples(args.run, chunk, metadata, args.out, args.chunk_index)
    medians = plot_diagnostics(args.run, metadata, args.out)
    print(
        "median captured fraction by pass: "
        + ", ".join(f"{value:.6f}" for value in medians),
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
