"""Compare the frozen 0019 20% model's fit quality on ap7151 events vs the am15577 spike sites."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REASON_LABELS = {
    0: "accepted (clean)",
    2: "rmse",
    16: "all-channel",
    18: "rmse+all-channel",
    24: "proj+all-channel",
    26: "rmse+proj+all-channel",
}

MINE_COLOR = "#1f77b4"
HIS_COLOR = "#ff7f0e"


def density_hist(ax, values, color, label, bins, log_y):
    ax.hist(
        values,
        bins=bins,
        histtype="step",
        linewidth=1.8,
        density=True,
        color=color,
        label=label,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mine",
        type=Path,
        default=Path(
            "residuals/runs/dataset1_p1/0019_allchannel_pass3_round1_fraction20_step10_fitted8"
        ),
    )
    parser.add_argument(
        "--his",
        type=Path,
        default=Path("residuals/runs/dataset1_p1/0019_crossfit_hisspikes_fraction20"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("residuals/out/0019_cross_comparison/cross_fit_comparison.png"),
    )
    args = parser.parse_args()

    import json

    with open(args.his / "summary.json") as handle:
        n_accepted = int(json.load(handle)["n_accepted"])

    mine = {
        name: np.load(args.mine / f"{name}.npy", mmap_mode="r")
        for name in (
            "captured_fraction",
            "min_channel_captured_fraction",
            "mean_channel_normalized_rmse",
            "rho",
        )
    }
    his = {
        name: np.load(args.his / f"{name}.npy", mmap_mode="r")
        for name in (
            "captured_fraction",
            "min_channel_captured_fraction",
            "mean_channel_normalized_rmse",
            "rho",
            "reasons",
        )
    }
    n_mine = mine["captured_fraction"].shape[0]
    n_his = his["captured_fraction"].shape[0]
    label_mine = f"ap7151 events (n={n_mine:,})"
    label_his = f"am15577 spike sites (n={n_his:,})"
    accepted = np.load(args.his / "all_ok.npy")

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]
    bins = np.linspace(-0.1, 1.0, 111)
    density_hist(ax, np.asarray(mine["captured_fraction"]), MINE_COLOR, label_mine, bins, False)
    density_hist(ax, np.asarray(his["captured_fraction"]), HIS_COLOR, label_his, bins, False)
    ax.axvline(
        np.median(np.asarray(mine["captured_fraction"])),
        color=MINE_COLOR,
        linestyle="--",
        linewidth=1.2,
    )
    ax.axvline(
        np.median(np.asarray(his["captured_fraction"])),
        color=HIS_COLOR,
        linestyle="--",
        linewidth=1.2,
    )
    ax.set_title("total captured fraction (same fitter, same model)")
    ax.set_xlabel("captured / input energy")
    ax.set_ylabel("density")
    ax.legend()

    ax = axes[0, 1]
    bins = np.linspace(-0.6, 1.0, 161)
    density_hist(
        ax, np.asarray(mine["min_channel_captured_fraction"]), MINE_COLOR, label_mine, bins, False
    )
    density_hist(
        ax, np.asarray(his["min_channel_captured_fraction"]), HIS_COLOR, label_his, bins, False
    )
    ax.axvline(0.2, color="k", linestyle=":", linewidth=1.5, label="20% acceptance bar")
    ax.set_title("worst-channel captured fraction")
    ax.set_xlabel("min channel captured / noise-normalized energy")
    ax.set_ylabel("density")
    ax.legend()

    ax = axes[0, 2]
    bins = np.linspace(0.4, 3.0, 131)
    density_hist(
        ax, np.asarray(mine["mean_channel_normalized_rmse"]), MINE_COLOR, label_mine, bins, False
    )
    density_hist(
        ax, np.asarray(his["mean_channel_normalized_rmse"]), HIS_COLOR, label_his, bins, False
    )
    ax.set_title("mean-channel normalized RMSE")
    ax.set_xlabel("mean-channel RMSE / robust noise")
    ax.set_ylabel("density")
    ax.legend()

    ax = axes[1, 0]
    subset = np.asarray(mine["captured_fraction"])[: n_mine]
    rmse = np.asarray(mine["mean_channel_normalized_rmse"])[:n_mine]
    hb = ax.hexbin(
        subset,
        rmse,
        gridsize=90,
        bins="log",
        cmap="Blues",
        extent=(-0.1, 1.0, 0.4, 3.0),
        mincnt=1,
    )
    fig.colorbar(hb, ax=ax, label="count")
    ax.set_title(f"{label_mine} — captured vs RMSE")
    ax.set_xlabel("captured fraction")
    ax.set_ylabel("mean-channel normalized RMSE")

    ax = axes[1, 1]
    his_captured = np.asarray(his["captured_fraction"])
    his_rmse = np.asarray(his["mean_channel_normalized_rmse"])
    his_min = np.asarray(his["min_channel_captured_fraction"])
    hb = ax.hexbin(
        his_captured,
        his_rmse,
        gridsize=90,
        bins="log",
        cmap="Oranges",
        extent=(-0.1, 1.0, 0.4, 3.0),
        mincnt=1,
    )
    fig.colorbar(hb, ax=ax, label="count")
    ax.set_title(
        f"{label_his} — {n_accepted:,} pass the 20% bar ({100 * n_accepted / n_his:.1f}%)"
    )
    ax.set_xlabel("captured fraction")
    ax.set_ylabel("mean-channel normalized RMSE")

    ax = axes[1, 2]
    reasons = np.asarray(his["reasons"]).astype(np.int64)
    accepted_mask = reasons == 0
    counts_rej = [
        int(np.count_nonzero((reasons == code) & ~accepted_mask)) for code in REASON_LABELS
    ]
    counts_acc = [
        int(np.count_nonzero((reasons == code) & accepted_mask)) for code in REASON_LABELS
    ]
    positions = np.arange(len(REASON_LABELS))
    ax.bar(
        positions - 0.2,
        counts_rej,
        width=0.4,
        color=HIS_COLOR,
        alpha=0.75,
        label="rejected",
    )
    ax.bar(
        positions + 0.2,
        counts_acc,
        width=0.4,
        color=MINE_COLOR,
        alpha=0.75,
        label="accepted",
    )
    ax.set_yscale("symlog")
    ax.set_xticks(positions, list(REASON_LABELS.values()), rotation=20, ha="right")
    ax.set_title("am15577 sites: rejection reasons at the ap7151 20% bar")
    ax.set_ylabel("sites")
    ax.legend()

    fig.suptitle(
        "0019 frozen 20% model cross-fit: ap7151 events vs am15577 spike sites",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=800, facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
