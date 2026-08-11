"""Probe-global XY, YZ, and ZX histograms, alone and versus monopolar localization."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECTIONS = (("x", "y", "xy"), ("y", "z", "yz"), ("z", "x", "zx"))
AXIS_LABEL = {
    "x": "x — lateral (µm)",
    "y": "y — dist. from probe (µm)",
    "z": "z — depth (µm)",
}


def load_analytic(session, fit_path):
    centroids = np.load(session / "centroids.npy", mmap_mode="r")
    fit = np.load(fit_path)
    sources = fit["sources"]
    return {
        "x": np.asarray(centroids[:, 0]) + sources[:, 0],
        "y": sources[:, 2],
        "z": np.asarray(centroids[:, 1]) + sources[:, 1],
    }


def load_monopole(path):
    values = np.load(path, mmap_mode="r")
    return {
        "x": np.asarray(values[:, 0]),
        "y": np.asarray(values[:, 2]),
        "z": np.asarray(values[:, 1]),
    }


def shared_limits(analytic, monopole):
    limits = {}
    for coordinate in ("x", "y", "z"):
        values = np.concatenate((analytic[coordinate], monopole[coordinate]))
        values = values[np.isfinite(values)]
        low, high = np.quantile(values, (0.002, 0.998))
        margin = 0.03 * max(high - low, 1.0)
        limits[coordinate] = (low - margin, high + margin)
    return limits


def histogram(values, horizontal, vertical, limits):
    counts, x_edges, y_edges = np.histogram2d(
        values[horizontal], values[vertical], bins=300,
        range=(limits[horizontal], limits[vertical]))
    return np.log1p(counts.T), x_edges, y_edges


def panel(axis, image, x_edges, y_edges, horizontal, vertical, title, vmax):
    artist = axis.imshow(
        image, origin="lower", aspect="equal",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        cmap="magma", interpolation="nearest", vmin=0, vmax=vmax)
    axis.set_title(title, fontsize=15, loc="left", fontweight="bold")
    axis.set_xlabel(AXIS_LABEL[horizontal], fontsize=11)
    axis.set_ylabel(AXIS_LABEL[vertical], fontsize=11)
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(labelsize=9)
    axis.set_facecolor("white")
    axis.spines[["top", "right"]].set_visible(False)
    return artist


def figure_size(horizontal, vertical, panels):
    width = 8.0 if horizontal == "z" else 5.0
    height = 8.0 if vertical == "z" else 5.0
    return panels * width + 2.2, height + 1.8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--monopole", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("out/plots/gpu_fit/projections"))
    args = parser.parse_args()

    analytic = load_analytic(args.session, args.fit)
    monopole = load_monopole(args.monopole)
    limits = shared_limits(analytic, monopole)
    args.out.mkdir(parents=True, exist_ok=True)

    for horizontal, vertical, label in PROJECTIONS:
        analytic_image, x_edges, y_edges = histogram(
            analytic, horizontal, vertical, limits)
        monopole_image, mono_x, mono_y = histogram(
            monopole, horizontal, vertical, limits)
        vmax = max(float(analytic_image.max()), float(monopole_image.max()))

        figure, axis = plt.subplots(
            figsize=figure_size(horizontal, vertical, 1), constrained_layout=True)
        artist = panel(axis, analytic_image, x_edges, y_edges, horizontal, vertical,
                       f"Analytic SLT — {label.upper()}", vmax)
        colorbar = figure.colorbar(artist, ax=axis, pad=0.01, shrink=0.8)
        colorbar.set_label("log(1 + spike count per bin)")
        figure.suptitle(
            f"dataset1_p1 probe-global {label.upper()} localization histogram",
            fontsize=13)
        output = args.out / f"hist_{label}.png"
        figure.savefig(output, dpi=800)
        plt.close(figure)

        figure, axes = plt.subplots(
            1, 2, figsize=figure_size(horizontal, vertical, 2),
            constrained_layout=True)
        artist = panel(axes[0], analytic_image, x_edges, y_edges,
                       horizontal, vertical, "Analytic SLT", vmax)
        panel(axes[1], monopole_image, mono_x, mono_y,
              horizontal, vertical, "Monopolar", vmax)
        colorbar = figure.colorbar(artist, ax=axes, pad=0.01, shrink=0.8)
        colorbar.set_label("log(1 + spike count per bin)")
        figure.suptitle(
            f"dataset1_p1 probe-global {label.upper()} localizations",
            fontsize=13)
        output = args.out / f"compare_monopole_{label}.png"
        figure.savefig(output, dpi=800)
        plt.close(figure)
        print(f"saved {label.upper()} histogram and comparison", flush=True)


if __name__ == "__main__":
    main()
