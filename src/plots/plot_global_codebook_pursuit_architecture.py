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
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "Peak-Channel Codebook Greedy Pursuit",
        ha="center",
        va="center",
        fontsize=23,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.5,
        0.930,
        "learn temporal shapes once → dense template scoring → IBL-style projection-threshold local maxima → fit and subtract",
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["muted"],
    )

    panel(ax, 0.640, 0.240, "A  LEARN AND FREEZE THE TEMPORAL CODEBOOKS", COLORS["orange"])
    recording = node(
        ax, 0.055, 0.690, 0.125, 0.110,
        "SpikeGLX recording", "AP data\n30 kHz × 384 channels", COLORS["blue"]
    )
    preprocess = node(
        ax, 0.210, 0.690, 0.145, 0.110,
        "Preprocess chunks", "300–6,000 Hz band-pass\nmedian subtraction + MAD", COLORS["blue"]
    )
    detect = node(
        ax, 0.385, 0.690, 0.145, 0.110,
        "Voltage-peak sweep", "−voltage / MAD ≥ 6\nlocal maxima + 1 ms isolation", COLORS["orange"]
    )
    sample = node(
        ax, 0.560, 0.690, 0.145, 0.110,
        "Peak-channel sample", "90 samples per event\n≤100k events · seed 42", COLORS["orange"]
    )
    learn = node(
        ax, 0.735, 0.680, 0.210, 0.130,
        "Cluster and freeze Omega_Q", "absolute-correlation assignment + signed row refit\nQ = 4, 8, 16, 32, 64 · no pursuit updates", COLORS["orange"], body_size=7.7
    )
    for left, right in zip(
        (recording, preprocess, detect, sample),
        (preprocess, detect, sample, learn),
    ):
        arrow(ax, edge(left, "right"), edge(right, "left"))

    ax.text(
        0.5,
        0.610,
        "The pursuit threshold is a post-convolution candidate gate, analogous to IBL sorter Th; there is no threshold-calibration stage.",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["muted"],
    )

    panel(ax, 0.205, 0.355, "B  RUN IBL-STYLE TEMPLATE PURSUIT FOR EACH Q", COLORS["green"])
    pursuit_input = node(
        ax, 0.055, 0.390, 0.145, 0.095,
        "Current residual", "preprocessed chunk\n+ frozen Omega_Q", COLORS["green"], body_size=7.5
    )
    pursuit_score = node(
        ax, 0.230, 0.390, 0.145, 0.095,
        "① Dense scoring", "convolve every window\nover rows × spatial scales", COLORS["green"], body_size=7.5
    )
    pursuit_detect = node(
        ax, 0.405, 0.390, 0.145, 0.095,
        "② Candidate gate", "projection score ≥ 6\nspace-time local maxima", COLORS["green"], body_size=7.5
    )
    localize = node(
        ax, 0.580, 0.390, 0.165, 0.095,
        "③ Localize + reconstruct", "fit source, scale, row, gain\non multichannel residual", COLORS["green"], body_size=7.5
    )
    subtract = node(
        ax, 0.775, 0.390, 0.165, 0.095,
        "④ Refit gain + subtract", "accept only positive reduction\nand ≥5% captured energy", COLORS["red"], body_size=7.5
    )
    for left, right in zip(
        (pursuit_input, pursuit_score, pursuit_detect, localize),
        (pursuit_score, pursuit_detect, localize, subtract),
    ):
        arrow(ax, edge(left, "right"), edge(right, "left"))

    guard = node(
        ax, 0.600, 0.245, 0.340, 0.064,
        "Round guard", "keep the round only if fractional core-energy drop ≥ 0.002 · stop by round 60", COLORS["red"], title_size=8.8, body_size=7.1
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
        0.455,
        0.277,
        "accepted round → updated residual → score again",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=COLORS["feedback"],
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none"},
        zorder=4,
    )

    node(
        ax, 0.030, 0.045, 0.650, 0.095,
        "OUTPUT", "accepted events: time · channel · xyz · scale · temporal row · gain · projection score · energy · round\ncompare the five independent fixed projection-threshold Q runs", COLORS["gray"], title_size=10.5, body_size=8.0
    )
    node(
        ax, 0.720, 0.045, 0.250, 0.095,
        "FUTURE (NOT IMPLEMENTED)", "optionally prune redundant codebook rows\nby absolute correlation / cosine similarity", COLORS["future"], title_size=9.2, body_size=7.8, linestyle="dashed"
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
