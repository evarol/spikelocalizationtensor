"""Render representative first-fit waveform pages from an exhaustive diagnostic."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def choose_rows(path, quantiles):
    with np.load(path, allow_pickle=False) as archive:
        values = np.sum(archive["per_channel_delta_chi2"], axis=1)
    if not len(values):
        return []
    order = np.argsort(values, kind="stable")
    return np.unique(
        np.rint(np.asarray(quantiles) * (len(order) - 1)).astype(np.int64)
    ).tolist()


def plot_candidate(pdf, chunk_index, row_index, archive, fs):
    observed = archive["observed"][row_index]
    reconstructed = archive["reconstructed"][row_index]
    residual = archive["residual"][row_index]
    mask = archive["neighbor_mask"][row_index]
    delta = archive["per_channel_delta_chi2"][row_index]
    ids = archive["neighbor_ids"][row_index]
    time_ms = 1000 * (np.arange(observed.shape[1]) - observed.shape[1] // 2) / fs
    scale = max(float(np.abs(np.stack((observed, reconstructed, residual))).max()), 1e-6)
    figure, axes = plt.subplots(3, observed.shape[0], figsize=(16, 6.1), sharex=True)
    labels = (("observed $Y_c(t)$", observed, "#202020"), ("reconstruction $\\hat Y_c(t)$", reconstructed, "#2b8a3e"), ("residual $R_c(t)$", residual, "#1971c2"))
    for row, (label, values, color) in enumerate(labels):
        for channel, axis in enumerate(axes[row]):
            if mask[channel]:
                axis.plot(time_ms, values[channel], color=color, linewidth=1.0)
                if row == 0:
                    axis.set_title(
                        f"ch {ids[channel]}\\n$\\Delta\\chi_c^2={delta[channel]:.1f}$",
                        fontsize=8,
                    )
            axis.axhline(0, color="0.75", linewidth=0.55)
            axis.set_ylim(-1.05 * scale, 1.05 * scale)
            axis.grid(alpha=0.14)
            axis.tick_params(labelsize=7)
        axes[row, 0].set_ylabel(label, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("time (ms)", fontsize=8)
    total_delta = float(delta[mask].sum())
    figure.suptitle(
        f"First fit only | chunk {chunk_index} row {row_index} | "
        f"score {archive['detection_score'][row_index]:.2f} | "
        f"total $\\Delta\\chi^2={total_delta:.1f}$ | "
        f"temporal row {archive['temporal_idx'][row_index]}",
        fontsize=11,
    )
    pdf.savefig(figure, dpi=800, bbox_inches="tight")
    plt.close(figure)
    return total_delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quantiles", type=float, nargs="+", default=(0, 0.05, 0.2, 0.5, 0.8, 0.95, 1))
    args = parser.parse_args()

    metadata = json.loads((args.run / "metadata.json").read_text())
    paths = sorted((args.run / "chunks").glob("chunk_*.npz"))
    if not paths:
        raise FileNotFoundError("first-fit diagnostic has no chunk files")
    args.output.mkdir(parents=True, exist_ok=True)
    selected = []
    with PdfPages(args.output / "first_fit_waveform_gallery.pdf") as pdf:
        for chunk_index, path in enumerate(paths):
            rows = choose_rows(path, args.quantiles)
            with np.load(path, allow_pickle=False) as archive:
                for row_index in rows:
                    total_delta = plot_candidate(
                        pdf, chunk_index, row_index, archive, float(metadata["fs"])
                    )
                    selected.append(
                        {
                            "chunk_index": chunk_index,
                            "row_index": row_index,
                            "total_delta_chi2": total_delta,
                        }
                    )
    (args.output / "gallery_selection.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )
    print(f"wrote {len(selected)} first-fit pages to {args.output}", flush=True)


if __name__ == "__main__":
    main()
