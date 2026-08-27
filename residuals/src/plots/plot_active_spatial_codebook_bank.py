"""Render every active monopole scale on the eight-contact local geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONTACTS = np.array(
    [
        [0.0, -30.0], [16.0, -30.0], [0.0, -10.0], [16.0, -10.0],
        [0.0, 10.0], [16.0, 10.0], [0.0, 30.0], [16.0, 30.0],
    ]
)
SIGMAS = np.geomspace(2.0, 512.0, 9)
SOURCE = (16.0, 10.0, 1.0)


def normalized_footprint(sigma: float) -> np.ndarray:
    x, y, z = SOURCE
    distance2 = (CONTACTS[:, 0] - x) ** 2 + (CONTACTS[:, 1] - y) ** 2 + z ** 2
    footprint = sigma / np.sqrt(distance2 + sigma ** 2)
    return footprint / np.linalg.norm(footprint)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("out/plots/active_spatial_codebook_bank.png")
    )
    args = parser.parse_args()

    figure, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    for index, (axis, sigma) in enumerate(zip(axes.flat, SIGMAS)):
        footprint = normalized_footprint(float(sigma))
        relative_ptp = footprint / footprint.max()
        points = axis.scatter(
            CONTACTS[:, 0], CONTACTS[:, 1], c=relative_ptp, cmap="viridis",
            vmin=0, vmax=1, s=50 + 400 * relative_ptp, edgecolors="#183047", linewidths=0.6,
        )
        axis.scatter(SOURCE[0], SOURCE[1], marker="x", s=92, color="#b54a3f", linewidths=2.0)
        for channel, (x, y) in enumerate(CONTACTS):
            axis.text(x + 1.1, y + 1.3, str(channel), fontsize=8, color="#183047")
        axis.set(
            title=f"bank index {index}: sigma={sigma:.0f} um",
            xlim=(-9, 25), ylim=(-43, 43), aspect="equal",
            xticks=(0, 16), yticks=(-30, -10, 10, 30),
        )
        axis.grid(alpha=0.22)
        axis.text(
            0.03, 0.03,
            f"relative channel PTP\n{relative_ptp.min():.2f} to 1.00",
            transform=axis.transAxes, fontsize=8, va="bottom",
            bbox={"facecolor": "white", "edgecolor": "0.7", "pad": 2},
        )
        if index % 3 == 0:
            axis.set_ylabel("y (um)")
        if index >= 6:
            axis.set_xlabel("x (um)")

    colorbar = figure.colorbar(points, ax=axes, shrink=0.84, pad=0.02)
    colorbar.set_label("relative channel peak-to-peak amplitude")
    figure.suptitle(
        "Active spatial codebook: normalized monopole footprints at a source 1 um above contact 5",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=800, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
