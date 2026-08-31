"""Plot 0019 temporal prototypes, assigned atoms, and cone angles."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def normalized_rows(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True),
        np.finfo(np.float64).tiny,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    omega = normalized_rows(np.load(args.run / "omega.npy"))
    prototypes = normalized_rows(np.load(args.run / "prototypes.npy"))
    assignment = np.load(args.run / "atom_prototype.npy").astype(np.int64)
    metadata = json.loads((args.run / "metadata.json").read_text())
    config = metadata["config"]
    time_ms = np.linspace(
        -config["ms_before"],
        config["ms_after"],
        omega.shape[1],
        endpoint=False,
    )
    cosine = np.sum(omega * prototypes[assignment], axis=1)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(omega)))

    figure, axes = plt.subplots(
        len(prototypes),
        2,
        figsize=(12, 3.6 * len(prototypes)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for prototype_index in range(len(prototypes)):
        rows = np.flatnonzero(assignment == prototype_index)
        waveform_axis = axes[prototype_index, 0]
        angle_axis = axes[prototype_index, 1]
        waveform_axis.plot(
            time_ms,
            prototypes[prototype_index],
            color="black",
            linewidth=3,
            label=f"prototype {prototype_index}",
            zorder=10,
        )
        for row in rows:
            waveform_axis.plot(
                time_ms,
                omega[row],
                color=colors[row],
                linewidth=1.2,
                alpha=0.9,
                label=f"Omega {row}: {angle[row]:.1f}°",
            )
        waveform_axis.axhline(0, color="0.75", linewidth=0.6)
        waveform_axis.set_xlabel("Time relative to detected peak (ms)")
        waveform_axis.set_ylabel("Unit-norm amplitude")
        waveform_axis.set_title(
            f"Prototype {prototype_index}: "
            f"{'positive' if prototype_index == 0 else 'negative'} family"
        )
        waveform_axis.legend(fontsize=7, ncol=2)
        angle_axis.bar(
            [str(row) for row in rows],
            angle[rows],
            color=colors[rows],
        )
        angle_axis.axhline(
            config["prototype_cone_deg"],
            color="black",
            linestyle="--",
            linewidth=1,
            label="cone boundary",
        )
        angle_axis.set_xlabel("Temporal atom index")
        angle_axis.set_ylabel("Angle from prototype (degrees)")
        angle_axis.set_ylim(
            0,
            max(config["prototype_cone_deg"] * 1.15, angle[rows].max() * 1.1),
        )
        angle_axis.set_title("Assigned-atom cone angles")
        angle_axis.legend(fontsize=8)

    figure.suptitle(
        "0019 all-channel temporal codebook: two learned prototype cones",
        fontsize=14,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800)
    plt.close(figure)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
