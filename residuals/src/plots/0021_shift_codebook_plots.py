"""Plot the 0021 shift-invariant four-prior master codebooks (Q32 and Q64).

Six-figure suite over the learned codebooks in residuals/runs/ibl_bwm/:
per-cone atom galleries (with prototypes and per-atom trough-to-peak widths),
atom width distributions against 026's two-prior reference, lag histograms
(the shift structure the codebook learned), fit convergence, and cone
occupancy. Cone order follows 0021's prototype_index = polarity + 2*speed:
0 pos-fast, 1 neg-fast, 2 pos-slow, 3 neg-slow.

Usage:
    singularity exec --nv --overlay /scratch/${USER}/_ENVS/pytorch.ext3:ro \
        /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
        /bin/bash -c "source /ext3/env.sh && python residuals/src/plots/0021_shift_codebook_plots.py"
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "residuals" / "runs" / "ibl_bwm"
OUT = REPO / "residuals" / "out" / "0021_shift_codebook"
REFERENCE_026 = "master_codebook_q32"
CONE_COLORS = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c", 3: "#9467bd"}
CONE_LABELS = {0: "pos · fast", 1: "neg · fast", 2: "pos · slow", 3: "neg · slow"}
FS = 30000.0


def trough_to_peak(omega):
    omega = np.asarray(omega, dtype=np.float32)
    rows = np.arange(len(omega))
    extremum = np.abs(omega).argmax(axis=1)
    sign = np.sign(omega[rows, extremum])
    sign[sign == 0] = 1.0
    flipped = omega * sign[:, None]
    columns = np.arange(omega.shape[1])[None, :]
    after = columns > extremum[:, None]
    opposite = np.where(after, -flipped, -np.inf)
    opposite_index = opposite.argmax(axis=1)
    has_opposite = after.any(axis=1)
    width = (opposite_index - extremum).clip(min=0).astype(np.float32)
    peak_value = flipped[rows, extremum]
    below = after & (flipped < 0.5 * peak_value[:, None])
    half_index = np.where(below, columns, omega.shape[1]).argmin(axis=1)
    has_half = below.any(axis=1)
    half_width = (half_index - extremum).clip(min=0).astype(np.float32)
    return np.where(has_opposite, width, half_width)


def load_run(q, tag):
    run = RUNS / f"master_codebook_0021_q{q}_{tag}"
    return {
        "q": q,
        "run": run,
        "tag": tag,
        "omega": np.load(run / "omega.npy").astype(np.float32),
        "prototypes": np.load(run / "prototypes.npy").astype(np.float32),
        "assignment": np.load(run / "atom_prototype.npy").astype(int),
        "history": json.loads((run / "alternating_history.json").read_text()),
        "init": json.loads((run / "prototype_initialization.json").read_text()),
    }


def atom_gallery(run):
    q = run["q"]
    tag = run["tag"]
    per_cone = q // 4
    order = np.argsort(
        np.array([trough_to_peak(run["omega"][[atom]])[0]
                  for atom in range(q)])
    )
    by_cone = [order[run["assignment"][order] == cone] for cone in range(4)]
    figure, axes = plt.subplots(
        per_cone + 1, 4,
        figsize=(4.8 * 4, 1.35 * (per_cone + 1)),
        constrained_layout=True,
    )
    samples = np.arange(run["omega"].shape[1]) / FS * 1000
    for cone in range(4):
        axes[0, cone].plot(
            samples, run["prototypes"][cone], color=CONE_COLORS[cone],
            linewidth=2.2,
        )
        axes[0, cone].set_title(CONE_LABELS[cone], color=CONE_COLORS[cone],
                                fontsize=11, fontweight="bold")
        for row, atom in enumerate(by_cone[cone], start=1):
            width_ms = trough_to_peak(run["omega"][[atom]])[0] / FS * 1000
            axes[row, cone].plot(samples, run["omega"][atom],
                                 color=CONE_COLORS[cone], linewidth=1.1)
            axes[row, cone].set_ylabel(
                rf"$\Omega_{{{atom}}}$" + "\n" + f"{width_ms:.2f} ms",
                fontsize=7,
            )
            axes[row, cone].tick_params(labelsize=6)
        for row in range(per_cone + 1):
            axes[row, cone].axhline(0, color="0.85", linewidth=0.5)
            axes[row, cone].set_xticks([])
            axes[row, cone].set_yticks([])
    figure.suptitle(
        f"0021 four-prior codebook Q{q} — atoms sorted by trough-to-peak "
        "width within each cone (prototype on top, width in ms)",
        fontsize=12,
    )
    path = OUT / f"atom_gallery_q{q}_{tag}.png"
    figure.savefig(path, dpi=800)
    plt.close(figure)
    return path


def atom_widths(runs, suffix):
    figure, axes = plt.subplots(
        1, len(runs) + 1,
        figsize=(4.6 * (len(runs) + 1), 3.6),
        constrained_layout=True,
    )
    for axis, run in zip(axes, runs):
        q = run["q"]
        widths = trough_to_peak(run["omega"]) / FS * 1000
        for cone in range(4):
            member = run["assignment"] == cone
            axis.scatter(np.flatnonzero(member), widths[member],
                         color=CONE_COLORS[cone], s=26,
                         label=CONE_LABELS[cone] if q == runs[0]["q"] else None)
        axis.set_title(f"Q{q} · {run['tag']}", fontsize=11)
        axis.set_xlabel("atom index")
        axis.set_ylabel("trough-to-peak width (ms)")
        axis.grid(alpha=0.2)
        axis.set_xlim(-1, run["q"])
    reference = np.load(RUNS / REFERENCE_026 / "omega.npy").astype(np.float32)
    reference_widths = trough_to_peak(reference) / FS * 1000
    for axis, run in zip(axes, runs):
        thresholds = run["init"]["speed_split"]
        for polarity in (0, 1):
            centers = thresholds[f"polarity_{polarity}"]["width_split_samples"]
            axis.axhline(centers[1] / FS * 1000, color=CONE_COLORS[2 * polarity],
                         linewidth=0.8, linestyle="--", alpha=0.6)
    axes[-1].hist(reference_widths, bins=24, color="0.55")
    axes[-1].set_title("026 two-prior reference", fontsize=11)
    axes[-1].set_xlabel("trough-to-peak width (ms)")
    axes[-1].set_ylabel("atoms")
    axes[-1].grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper right")
    figure.suptitle(
        "learned atom widths by cone (dashed: the init speed-split threshold "
        "for each polarity)", fontsize=12,
    )
    path = OUT / f"atom_widths_by_cone_{suffix}.png"
    figure.savefig(path, dpi=800)
    plt.close(figure)
    return path


def lag_histograms(runs, suffix):
    figure, axes = plt.subplots(
        1, len(runs) + 1, figsize=(5.2 * (len(runs) + 1), 3.6),
        constrained_layout=True,
    )
    for axis, run in zip(axes, runs):
        histogram = np.asarray(run["history"][-1]["lag_histogram"], dtype=float)
        offsets = np.arange(len(histogram)) - len(histogram) // 2
        total = histogram.sum()
        axis.bar(offsets, histogram / total * 100, color="#1f77b4", width=0.8)
        zero = histogram[len(histogram) // 2] / total * 100
        mean_abs = run["history"][-1]["mean_abs_lag"]
        axis.set_title(
            f"Q{run['q']} {run['tag']} final — {zero:.0f}% at zero lag, "
            f"mean |τ| = {mean_abs:.2f} samples", fontsize=10,
        )
        axis.set_xlabel("lag τ (samples, 30 kHz)")
        axis.set_ylabel("events (%)")
        axis.grid(alpha=0.2)
    first = np.asarray(runs[0]["history"][0]["lag_histogram"], dtype=float)
    offsets = np.arange(len(first)) - len(first) // 2
    axes[-1].bar(offsets, first / first.sum() * 100, color="0.6", width=0.8)
    axes[-1].set_title(f"Q{runs[0]['q']} {runs[0]['tag']} iteration 1",
                       fontsize=10)
    axes[-1].set_xlabel("lag τ (samples, 30 kHz)")
    axes[-1].set_ylabel("events (%)")
    axes[-1].grid(alpha=0.2)
    figure.suptitle(
        "per-event lag histograms from the shift-bank correlation "
        "(final alternating iteration)", fontsize=12,
    )
    path = OUT / f"lag_histograms_{suffix}.png"
    figure.savefig(path, dpi=800)
    plt.close(figure)
    return path


def fit_convergence(runs, suffix):
    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 3.9), constrained_layout=True,
    )
    for run in runs:
        iterations = [entry["iteration"] for entry in run["history"]]
        objectives = np.asarray(
            [entry["fixed_assignment_objective"] for entry in run["history"]]
        ) / 1e9
        changes = [entry["maximum_row_change"] for entry in run["history"]]
        axes[0].plot(iterations, objectives, marker="o", markersize=3.5,
                     label=f"Q{run['q']} {run['tag']}")
        axes[1].plot(iterations, changes, marker="o", markersize=3.5,
                     label=f"Q{run['q']} {run['tag']}")
    for axis, ylabel in ((axes[0], "fixed-assignment SSE (×1e9)"),
                         (axes[1], "max atom row change")):
        axis.set_xlabel("alternating iteration")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    figure.suptitle(
        "alternating-fit convergence (Q32 stopped when the proposal no "
        "longer improved; Q64 used all 10)", fontsize=12,
    )
    path = OUT / f"fit_convergence_{suffix}.png"
    figure.savefig(path, dpi=800)
    plt.close(figure)
    return path


def cone_occupancy(runs, suffix):
    figure, axes = plt.subplots(
        1, len(runs), figsize=(7.2 * len(runs), 3.8), constrained_layout=True,
    )
    for axis, run in zip(axes, runs):
        counts = np.asarray(run["history"][-1]["row_counts"], dtype=float) / 1e3
        colors = [CONE_COLORS[cone] for cone in run["assignment"]]
        axis.bar(np.arange(run["q"]), counts, color=colors)
        axis.set_title(f"Q{run['q']} {run['tag']} final counts", fontsize=11)
        axis.set_xlabel("atom index")
        axis.set_ylabel("events (×1000)")
        axis.grid(alpha=0.2, axis="y")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CONE_COLORS[cone])
               for cone in range(4)]
    axes[0].legend(handles, [CONE_LABELS[cone] for cone in range(4)],
                   fontsize=8, loc="upper left")
    figure.suptitle(
        "cone occupancy — events per atom in the final iteration "
        "(negative-polarity cones carry the pool's imbalance)", fontsize=12,
    )
    path = OUT / f"cone_occupancy_{suffix}.png"
    figure.savefig(path, dpi=800)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--tag", type=str, default="p4_shift",
                        help="run-dir tag, e.g. p4_shift or p4f20_shift")
    parser.add_argument("--suffix", type=str, default=None,
                        help="output filename suffix (default: the tag)")
    args = parser.parse_args()
    suffix = args.suffix or args.tag
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [load_run(q, args.tag) for q in args.q]
    paths = []
    for run in runs:
        paths.append(atom_gallery(run))
    paths.append(atom_widths(runs, suffix))
    paths.append(lag_histograms(runs, suffix))
    paths.append(fit_convergence(runs, suffix))
    paths.append(cone_occupancy(runs, suffix))
    for path in paths:
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
