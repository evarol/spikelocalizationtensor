import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "blue": "#DCEAF7",
    "orange": "#F9E2BE",
    "green": "#D7ECE5",
    "red": "#F4D2CE",
    "gray": "#E7EAED",
    "future": "#F5F6F7",
    "ink": "#183047",
    "muted": "#536878",
    "arrow": "#40576A",
    "feedback": "#B54A3F",
}


def panel(ax, y, height, title, color):
    patch = FancyBboxPatch(
        (0.03, y),
        0.94,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.2,
        edgecolor="#B9C4CC",
        facecolor=color,
        alpha=0.26,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        0.05,
        y + height - 0.030,
        title,
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=4,
    )


def node(
    ax,
    x,
    y,
    width,
    height,
    title,
    body,
    color,
    title_size=10,
    body_size=8,
    linestyle="solid",
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.009",
        linewidth=1.25,
        linestyle=linestyle,
        edgecolor=COLORS["ink"] if linestyle == "solid" else "#84939E",
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=3,
    )
    ax.text(
        x + width / 2,
        y + height * 0.31,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=COLORS["muted"],
        linespacing=1.22,
        zorder=3,
    )
    return patch


def edge(patch, side):
    x = patch.get_x() + patch.get_width() / 2
    y = patch.get_y() + patch.get_height() / 2
    if side == "left":
        x = patch.get_x()
    elif side == "right":
        x = patch.get_x() + patch.get_width()
    elif side == "top":
        y = patch.get_y() + patch.get_height()
    elif side == "bottom":
        y = patch.get_y()
    return x, y


def arrow(ax, start, end, color=None, linewidth=1.6):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=linewidth,
        color=color or COLORS["arrow"],
        shrinkA=2,
        shrinkB=2,
        zorder=1,
    )
    ax.add_patch(patch)


def build_figure():
    fig, ax = plt.subplots(figsize=(18, 11))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "XYZ-Sigma Residual Pursuit",
        ha="center",
        va="center",
        fontsize=23,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.5,
        0.930,
        "one frozen temporal codebook row × one discrete monopole footprint × one gain per event",
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["muted"],
    )

    panel(ax, 0.650, 0.225, "A  CALIBRATION: LEARN AND FREEZE ONE Q=8 TEMPORAL CODEBOOK", COLORS["orange"])
    recording = node(
        ax, 0.045, 0.700, 0.125, 0.105,
        "SpikeGLX recording", "AP data\n30 kHz × 384 channels", COLORS["blue"]
    )
    preprocess = node(
        ax, 0.195, 0.700, 0.145, 0.105,
        "Preprocess", "band-pass + common-median\nreference", COLORS["blue"]
    )
    detect = node(
        ax, 0.365, 0.700, 0.145, 0.105,
        "Calibration detection", "32 deterministic recording-wide\nchunks · negative peaks", COLORS["orange"]
    )
    sample = node(
        ax, 0.535, 0.700, 0.145, 0.105,
        "Isolated event shards", "≤100k events · 1 ms isolation\nlocal geometry + fixed seed", COLORS["orange"]
    )
    learn = node(
        ax, 0.705, 0.690, 0.250, 0.125,
        "Alternating discrete fit → frozen Omega", "geometry-grouped (x,y,z,sigma,q) assignment\nclosed-form row refit · 10 iterations · Q=8", COLORS["orange"], body_size=7.6
    )
    for left, right in zip(
        (recording, preprocess, detect, sample),
        (preprocess, detect, sample, learn),
    ):
        arrow(ax, edge(left, "right"), edge(right, "left"))

    panel(ax, 0.220, 0.365, "B  FRESH FOUR-PASS RESIDUAL PURSUIT IN ONE-SECOND CHUNKS", COLORS["green"])
    pursuit_input = node(
        ax, 0.045, 0.410, 0.145, 0.100,
        "Current residual", "fresh preprocessed chunk\nor prior accepted subtraction", COLORS["green"], body_size=7.4
    )
    pursuit_score = node(
        ax, 0.220, 0.410, 0.145, 0.100,
        "① Peak proposal + NMS", "negative peaks · score ≥ 8\nstable time/channel ordering", COLORS["green"], body_size=7.4
    )
    pursuit_detect = node(
        ax, 0.395, 0.410, 0.180, 0.100,
        "② Grouped discrete fit", "select q, x, y, z, sigma, alpha\nby local least squares", COLORS["green"], body_size=7.3
    )
    localize = node(
        ax, 0.605, 0.410, 0.150, 0.100,
        "③ Final-fit gates", "sqrt(captured energy) ≥ 8\nchannel RMSE ≤ 3", COLORS["red"], body_size=7.4
    )
    subtract = node(
        ax, 0.785, 0.410, 0.165, 0.100,
        "④ Lock out + subtract", "exclude prior-pass events within\n0.5 ms / 48 um", COLORS["red"], body_size=7.3
    )
    for left, right in zip(
        (pursuit_input, pursuit_score, pursuit_detect, localize),
        (pursuit_score, pursuit_detect, localize, subtract),
    ):
        arrow(ax, edge(left, "right"), edge(right, "left"))

    spatial_bank = node(
        ax, 0.395, 0.325, 0.360, 0.052,
        "Discrete monopole dictionary", "sigma = 2, 4, ..., 512 um · x,y ∈ [-150, 150] um · z ∈ [1, 300] um · no rho refinement",
        COLORS["orange"], title_size=8.4, body_size=6.7
    )
    arrow(ax, edge(spatial_bank, "top"), edge(pursuit_detect, "bottom"), COLORS["arrow"], 1.3)

    guard = node(
        ax, 0.610, 0.245, 0.340, 0.062,
        "Pass controller", "keep only positive energy reduction · advance through exactly four passes", COLORS["red"], title_size=8.8, body_size=7.1
    )
    arrow(ax, edge(subtract, "bottom"), edge(guard, "top"), COLORS["feedback"], 1.8)
    guard_left = edge(guard, "left")
    score_bottom = edge(pursuit_score, "bottom")
    feedback_y = 0.335
    ax.plot(
        [guard_left[0], score_bottom[0], score_bottom[0]],
        [guard_left[1], guard_left[1], feedback_y],
        color=COLORS["feedback"],
        linewidth=1.9,
        zorder=1,
    )
    arrow(ax, (score_bottom[0], feedback_y), score_bottom, COLORS["feedback"], 1.9)
    ax.text(
        0.385,
        0.248,
        "accepted fits → updated residual → next pass",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=COLORS["feedback"],
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none"},
        zorder=4,
    )

    node(
        ax, 0.030, 0.055, 0.650, 0.100,
        "OUTPUT", "atomic chunk checkpoints → consolidated event arrays: time · channel · xyz · sigma · q · alpha · fitted score · pass · diagnostics\nroot Omega, geometry/neighborhood metadata, and optional residual-waveform shards", COLORS["gray"], title_size=10.3, body_size=7.45
    )
    node(
        ax, 0.720, 0.055, 0.250, 0.100,
        "RESUME", "completed calibration shards and chunks\nare skipped by --resume after requeue", COLORS["future"], title_size=9.2, body_size=7.8, linestyle="dashed"
    )
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure()
    figure.savefig(args.out, dpi=800, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
