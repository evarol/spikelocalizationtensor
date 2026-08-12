"""Plot diagnostics for a saved analytic GPU fit."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out/plots/gpu_fit"))
    ap.add_argument("--max-points", type=int, default=200_000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fit = np.load(args.fit, allow_pickle=True)
    centroids = np.load(args.session / "centroids.npy", mmap_mode="r")
    sources = fit["sources"]
    local_xy = sources[:, :2]
    global_xy = centroids[:, :2] + local_xy
    pick = fit["pick"]
    site = fit["site_idx"]
    profile = fit["profile_idx"]
    history = fit["history"].tolist()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    steps = [h["step"] for h in history]
    losses = [h["nmse"] for h in history]
    used = [h["used"] for h in history]
    ax[0].plot(steps, losses, "-o", color="#386cb0")
    ax[0].set(xlabel="iteration", ylabel="nMSE", title="fit convergence")
    ax[0].grid(alpha=.3)
    ax[1].plot(steps, used, "-o", color="#2ca25f")
    ax[1].set(xlabel="iteration", ylabel="unique candidates used",
              title="codebook usage")
    ax[1].grid(alpha=.3)
    fig.savefig(args.out / "convergence.png", dpi=800)
    plt.close(fig)

    rng = np.random.default_rng(0)
    keep = rng.choice(len(global_xy), min(args.max_points, len(global_xy)), replace=False)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    ax[0].scatter(global_xy[keep, 0], global_xy[keep, 1], s=0.2, c=fit["sigma"][keep],
                  cmap="viridis", rasterized=True)
    ax[0].set(xlabel="global x (µm)", ylabel="global y (µm)", title="localized sources")
    ax[1].scatter(local_xy[keep, 0], local_xy[keep, 1], s=0.2, c=site[keep],
                  cmap="turbo", rasterized=True)
    ax[1].set(xlabel="local x (µm)", ylabel="local y (µm)", title="sites relative to anchor")
    ax[2].hist(fit["sigma"], bins=30, color="#756bb1")
    ax[2].set(xlabel="selected profile scale (µm)", ylabel="spikes", title="profile scales")
    fig.savefig(args.out / "localization_usage.png", dpi=800)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].hist(np.bincount(site), bins=50, color="#e6550d")
    ax[0].set(xlabel="spikes per site", ylabel="number of sites", title="site concentration")
    ax[1].hist(np.bincount(profile), bins=max(1, int(profile.max()) + 1),
               color="#31a354")
    ax[1].set(xlabel="profile index", ylabel="spikes", title="profile usage")
    fig.savefig(args.out / "codebook_usage.png", dpi=800)
    plt.close(fig)

    if "alpha" in fit:
        alpha = np.asarray(fit["alpha"])
        magnitude = np.abs(alpha)
        positive = magnitude[magnitude > 0]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        if len(positive):
            ax[0].hist(np.log10(positive), bins=60, color="#3182bd")
        ax[0].set(xlabel=r"log10 $|\alpha_s|$", ylabel="spikes",
                  title="closed-form spike gains")
        q = np.asarray(fit["temporal_idx"])
        medians = [
            np.median(magnitude[q == i]) if np.any(q == i) else 0
            for i in range(len(fit["omega"]))
        ]
        ax[1].bar(np.arange(len(medians)), medians, color="#756bb1")
        ax[1].set(xlabel="temporal row", ylabel=r"median $|\alpha_s|$",
                  title="gain by temporal code")
        fig.savefig(args.out / "gain_usage.png", dpi=800)
        plt.close(fig)
    print(f"wrote plots to {args.out}")


if __name__ == "__main__":
    main()
