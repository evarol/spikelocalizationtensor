"""Visualize synthetic monopole codebook candidates on a fixed 8-contact geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CHANNEL_XY = np.array(
    [
        [0.0, -30.0],
        [16.0, -30.0],
        [0.0, -10.0],
        [16.0, -10.0],
        [0.0, 10.0],
        [16.0, 10.0],
        [0.0, 30.0],
        [16.0, 30.0],
    ]
)

CASES = (
    ("near contact", (16.0, 10.0, 1.0), 1.0),
    ("near, wider", (16.0, 10.0, 8.0), 8.0),
    ("between contacts", (8.0, 0.0, 20.0), 16.0),
    ("deep, broad", (8.0, 0.0, 60.0), 64.0),
)


def temporal_shape(n_samples: int = 90) -> tuple[np.ndarray, np.ndarray]:
    time_ms = np.linspace(-1.5, 1.5, n_samples)
    trough = -np.exp(-0.5 * ((time_ms + 0.12) / 0.16) ** 2)
    rebound = 0.35 * np.exp(-0.5 * ((time_ms - 0.27) / 0.23) ** 2)
    shape = trough + rebound
    return time_ms, shape / np.linalg.norm(shape)


def normalized_monopole(source: tuple[float, float, float], sigma: float) -> np.ndarray:
    x, y, z = source
    distance2 = (CHANNEL_XY[:, 0] - x) ** 2 + (CHANNEL_XY[:, 1] - y) ** 2 + z**2
    footprint = sigma / np.sqrt(distance2 + sigma**2)
    return footprint / np.linalg.norm(footprint)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/plots/synthetic_lattice_footprints.png"),
    )
    args = parser.parse_args()

    time_ms, omega = temporal_shape()
    omega_ptp = np.ptp(omega)
    target_peak_ptp_uv = 50.0
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(CHANNEL_XY)))
    figure = plt.figure(figsize=(16, 7.4), constrained_layout=True)
    grid = figure.add_gridspec(2, len(CASES), height_ratios=(1.0, 1.25))

    for column, (name, source, sigma) in enumerate(CASES):
        footprint = normalized_monopole(source, sigma)
        alpha_uv = target_peak_ptp_uv / (footprint.max() * omega_ptp)
        channel_ptp_uv = alpha_uv * footprint * omega_ptp

        spatial = figure.add_subplot(grid[0, column])
        spatial.scatter(
            CHANNEL_XY[:, 0],
            CHANNEL_XY[:, 1],
            s=40 + 9 * channel_ptp_uv,
            c=channel_ptp_uv,
            cmap="viridis",
            vmin=0,
            vmax=target_peak_ptp_uv,
            edgecolors="black",
            linewidths=0.6,
            zorder=2,
        )
        spatial.scatter(source[0], source[1], marker="x", s=80, color="#d62728", linewidths=2.0, zorder=3)
        for channel, (x, y) in enumerate(CHANNEL_XY):
            spatial.text(x + 1.6, y + 1.6, f"{channel}", fontsize=8)
        spatial.set(
            title=(f"{name}\nsource=({source[0]:.0f}, {source[1]:.0f}, {source[2]:.0f}) um, "
                   f"sigma={sigma:.0f} um"),
            xlabel="x (um)",
            ylabel="y (um)",
            xlim=(-9, 25),
            ylim=(-43, 43),
            aspect="equal",
        )
        spatial.grid(alpha=0.2)
        spatial.text(
            0.02,
            0.02,
            f"alpha={alpha_uv:.1f} uV\nchannel PTP: {channel_ptp_uv.min():.1f}-{channel_ptp_uv.max():.1f} uV",
            transform=spatial.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "0.7", "pad": 2},
        )
        xz = spatial.inset_axes((0.54, 0.49, 0.42, 0.30))
        xz.scatter(CHANNEL_XY[:, 0], np.zeros(len(CHANNEL_XY)), s=12, color="0.25")
        xz.scatter(source[0], source[2], marker="x", s=42, color="#d62728", linewidths=1.5)
        xz.set(
            xlim=(-3, 20),
            ylim=(0, 70),
            xticks=(0, 16),
            yticks=(0, 30, 60),
            title="x-z",
        )
        xz.tick_params(labelsize=6)
        xz.grid(alpha=0.2)

        waveform = figure.add_subplot(grid[1, column])
        for channel, color in enumerate(colors):
            waveform.plot(
                time_ms,
                alpha_uv * footprint[channel] * omega,
                color=color,
                linewidth=1.25,
                label=f"ch {channel}: {channel_ptp_uv[channel]:.1f}",
            )
        waveform.axhline(0, color="0.35", linewidth=0.7)
        waveform.set(xlabel="time (ms)", ylabel="synthetic voltage (uV)", ylim=(-43, 18))
        waveform.grid(alpha=0.2)
        waveform.legend(fontsize=7, ncol=2, frameon=False, loc="lower right")

    figure.suptitle(
        "Synthetic normalized monopole codebook candidates: same peak PTP, different spatial spread",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=800, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
