"""Plot fitted source locations in full-probe coordinates."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from maths import load_channel_map


def density(ax, horizontal, depth, x_edges, depth_edges, title, xlabel):
    counts, _, _ = np.histogram2d(depth, horizontal, bins=(depth_edges, x_edges))
    nonzero = counts[counts > 0]
    vmax = max(1, float(np.percentile(nonzero, 99.7))) if nonzero.size else 1
    image = ax.imshow(
        counts,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], depth_edges[0], depth_edges[-1]],
        aspect="auto",
        cmap="magma",
        norm=LogNorm(vmin=1, vmax=vmax),
        interpolation="nearest",
    )
    ax.set(xlabel=xlabel, ylabel="probe depth (µm)", title=title)
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--recording", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out/plots/gpu_fit/probe_global.png"))
    args = ap.parse_args()

    fit = np.load(args.fit)
    centroids = np.load(args.session / "centroids.npy", mmap_mode="r")
    sources = fit["sources"]
    global_xy = np.asarray(centroids[:, :2]) + sources[:, :2]
    axial = sources[:, 2]

    probe = load_channel_map(args.recording)
    contacts = np.asarray(probe.contact_positions)
    x_pad = 170.0
    depth_pad = 170.0
    x_edges = np.linspace(contacts[:, 0].min() - x_pad,
                          contacts[:, 0].max() + x_pad, 321)
    depth_edges = np.linspace(contacts[:, 1].min() - depth_pad,
                              contacts[:, 1].max() + depth_pad, 769)
    axial_edges = np.geomspace(max(1.0, axial.min()), axial.max(), 241)

    fig, ax = plt.subplots(1, 3, figsize=(15, 10), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1.15, 0.65, 1.15]})
    im = density(ax[0], global_xy[:, 0], global_xy[:, 1], x_edges, depth_edges,
                 "all fitted sources on the probe", "global lateral x (µm)")
    ax[0].scatter(contacts[:, 0], contacts[:, 1], s=4, marker="s", facecolors="none",
                  edgecolors="#54d2d2", linewidths=.3, label="probe contacts")
    ax[0].legend(loc="upper right", fontsize=7)
    fig.colorbar(im, ax=ax[0], pad=.02, label="spikes per bin")

    ax[1].hist(global_xy[:, 1], bins=depth_edges, orientation="horizontal",
               color="#386cb0")
    ax[1].set(xlabel="spikes", ylabel="probe depth (µm)", title="depth distribution")
    ax[1].set_ylim(depth_edges[0], depth_edges[-1])
    ax[1].grid(alpha=.25, axis="x")

    im = density(ax[2], axial, global_xy[:, 1], axial_edges, depth_edges,
                 "distance from probe plane", "axial z (µm, log scale)")
    ax[2].set_xscale("log")
    fig.colorbar(im, ax=ax[2], pad=.02, label="spikes per bin")

    fig.suptitle(f"probe-global spike localizations · {len(sources):,} spikes")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=800, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
