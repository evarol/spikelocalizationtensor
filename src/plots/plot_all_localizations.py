"""Plot probe-global localization grids for kernels and reference fits."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


KERNELS = (
    "monopole", "exponential", "gauss", "lorentz", "power",
    "student", "yukawa", "dog", "gauss_aniso", "mono_aniso",
)
PROJECTIONS = (("x", "y", "xy"), ("y", "z", "yz"), ("z", "x", "zx"))
LABELS = {
    "x": "lateral x (µm)",
    "y": "distance from probe (µm)",
    "z": "probe depth (µm)",
}


def fit_path(session, kernel, masked):
    if masked:
        return session / f"gpu_fit_voxel_1um_masked_{kernel}.npz"
    suffix = "_optimized" if kernel.endswith("_aniso") else ""
    return session / f"gpu_fit_voxel_1um_{kernel}{suffix}.npz"


def analytic_localizations(session, kernel, centroids, masked):
    path = fit_path(session, kernel, masked)
    if not path.exists():
        raise FileNotFoundError(path)
    fit = np.load(path)
    sources = np.asarray(fit["sources"])
    return {
        "x": np.asarray(centroids[:, 0]) + sources[:, 0],
        "y": sources[:, 2],
        "z": np.asarray(centroids[:, 1]) + sources[:, 1],
        "title": f"{kernel} · nMSE {float(fit['nmse']):.4f}",
    }


def monopolar_localizations(path):
    values = np.load(path, mmap_mode="r")
    return {
        "x": np.asarray(values[:, 0]),
        "y": np.asarray(values[:, 2]),
        "z": np.asarray(values[:, 1]),
        "title": "monopolar reference",
    }


def continuous_localizations(path):
    with np.load(path) as values:
        locations = np.asarray(values["localizations_continuous"])
    with path.with_suffix(".json").open() as stream:
        summary = json.load(stream)
    return {
        "x": locations[:, 0],
        "y": locations[:, 1],
        "z": locations[:, 2],
        "title": (
            "monopole continuous · nMSE "
            f"{summary['nmse_continuous_for_processed_spikes']:.4f}"
        ),
    }


def limits(methods, coordinate):
    values = np.concatenate([method[coordinate] for method in methods])
    values = values[np.isfinite(values)]
    low, high = np.quantile(values, (0.002, 0.998))
    margin = 0.03 * max(high - low, 1.0)
    return low - margin, high + margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--monopole", type=Path, required=True)
    ap.add_argument("--continuous", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--masked", action="store_true")
    args = ap.parse_args()

    outputs = [args.out / f"localization_grid_{label}.png"
               for _, _, label in PROJECTIONS]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing}")

    centroids = np.load(args.session / "centroids.npy", mmap_mode="r")
    methods = [analytic_localizations(args.session, kernel, centroids, args.masked)
               for kernel in KERNELS]
    methods.append(monopolar_localizations(args.monopole))
    if args.continuous is not None:
        methods.append(continuous_localizations(args.continuous))
    axis_limits = {coordinate: limits(methods, coordinate)
                   for coordinate in ("x", "y", "z")}

    args.out.mkdir(parents=True, exist_ok=True)
    for horizontal, vertical, label in PROJECTIONS:
        images = []
        x_edges = y_edges = None
        for method in methods:
            counts, x_edges, y_edges = np.histogram2d(
                method[horizontal], method[vertical], bins=300,
                range=(axis_limits[horizontal], axis_limits[vertical]))
            images.append(np.log1p(counts.T))
        vmax = max(float(image.max()) for image in images)

        figure, axes = plt.subplots(
            3, 4, figsize=(18, 14), constrained_layout=True,
            sharex=True, sharey=True,
        )
        artist = None
        for axis, method, image in zip(axes.flat, methods, images):
            artist = axis.imshow(
                image, origin="lower", aspect="equal",
                extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
                cmap="magma", interpolation="nearest", vmin=0, vmax=vmax)
            axis.set_title(method["title"], fontsize=10, loc="left")
            axis.set_facecolor("white")
            axis.spines[["top", "right"]].set_visible(False)
        for axis in axes.flat[len(methods):]:
            axis.axis("off")
        for axis in axes[-1, :]:
            axis.set_xlabel(LABELS[horizontal])
        for axis in axes[:, 0]:
            axis.set_ylabel(LABELS[vertical])
        figure.colorbar(
            artist, ax=axes, pad=0.01, shrink=0.75,
            label="log(1 + spike count per bin)")
        qualifier = "masked-channel " if args.masked else ""
        figure.suptitle(
            f"dataset1_p1 · all {qualifier}localization methods · {label.upper()}")
        output = args.out / f"localization_grid_{label}.png"
        figure.savefig(output, dpi=800)
        plt.close(figure)
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
