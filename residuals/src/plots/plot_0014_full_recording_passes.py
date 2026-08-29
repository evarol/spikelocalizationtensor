"""Render the complete 0014 recording through its saved residual passes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import torch

from preprocessing.residuals_0012 import make_filter, preprocess_voltage


BACKGROUND = "#0d0d0d"
FONT = "#d7d7d7"
GRID = "#292929"


def signed_block_extrema(values, samples_per_bin):
    """Keep the signed largest-magnitude sample in each time bin and channel."""
    n_samples, n_channels = values.shape
    if n_samples % samples_per_bin:
        raise ValueError("core length must be divisible by --samples-per-bin")
    blocks = values.reshape(-1, samples_per_bin, n_channels)
    index = blocks.abs().argmax(dim=1, keepdim=True)
    return blocks.gather(1, index).squeeze(1)


def subtract_saved_predictions(residual, times, ids, predictions, n_before, batch_size):
    """Exactly replay the accepted saved atoms with GPU accumulation."""
    device = residual.device
    offsets = torch.arange(predictions.shape[2], device=device) - n_before
    for start in range(0, len(times), batch_size):
        stop = min(start + batch_size, len(times))
        batch_ids = torch.as_tensor(ids[start:stop], device=device)
        mask = batch_ids >= 0
        batch_times = torch.as_tensor(times[start:stop], device=device)
        prediction = torch.as_tensor(predictions[start:stop], device=device)
        sample_index = batch_times[:, None, None] + offsets[None, None, :]
        sample_index = sample_index.expand(-1, batch_ids.shape[1], -1)
        channel_index = batch_ids[:, :, None].expand_as(sample_index)
        valid = mask[:, :, None].expand_as(prediction)
        residual.index_put_(
            (sample_index[valid], channel_index[valid]),
            -prediction[valid], accumulate=True,
        )


def amplitude_weights(alpha):
    amplitude = np.abs(np.asarray(alpha, dtype=np.float64))
    amplitude = np.where(np.isfinite(amplitude), amplitude, 0.0)
    positive = amplitude[amplitude > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    return (amplitude / max(scale, np.finfo(np.float64).tiny)).astype(np.float32), scale


def categorical_raster(x, y, labels, weights, palette, xlim, ylim, nx, ny):
    ix = np.floor((x - xlim[0]) * nx / (xlim[1] - xlim[0])).astype(np.int64)
    iy = np.floor((y - ylim[0]) * ny / (ylim[1] - ylim[0])).astype(np.int64)
    keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = iy[keep] * nx + ix[keep]
    mass = np.bincount(flat, weights=weights[keep], minlength=nx * ny).reshape(ny, nx)
    rgb = np.zeros((ny, nx, 3), dtype=np.float32)
    for channel in range(3):
        rgb[..., channel] = np.bincount(
            flat, weights=weights[keep] * palette[labels[keep], channel], minlength=nx * ny
        ).reshape(ny, nx)
    intensity = np.clip(1.35 * (1 - np.exp(-mass / 1.4)), 0, 1)
    color = rgb / np.maximum(mass[..., None], 1e-6)
    return np.asarray([0.05, 0.05, 0.05], dtype=np.float32) * (1 - intensity[..., None]) + color * intensity[..., None]


def style_axis(axis):
    axis.set_facecolor(BACKGROUND)
    axis.grid(color=GRID, alpha=0.8, linewidth=0.45)
    axis.set_axisbelow(True)
    axis.tick_params(colors=FONT, labelsize=7)
    axis.xaxis.label.set_color(FONT)
    axis.yaxis.label.set_color(FONT)
    axis.title.set_color("#eeeeee")
    for spine in axis.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-bin", type=int, default=10_000)
    parser.add_argument("--reconstruction-batch-size", type=int, default=2_048)
    parser.add_argument("--raster-width", type=int, default=1_750)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if args.samples_per_bin < 1 or args.reconstruction_batch_size < 1 or args.raster_width < 1:
        raise ValueError("bin, reconstruction batch, and raster width must be positive")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this full-recording replay requires a CUDA device")

    import spikeglx

    metadata = json.loads((args.run / "config.json").read_text())
    config = metadata["config"]
    fs = float(metadata["fs"])
    first = int(metadata["first_sample"])
    stop = int(metadata["stop_sample"])
    n_channels = int(metadata["n_channels"])
    chunk_samples = int(round(float(config["chunk_seconds"]) * fs))
    if chunk_samples % args.samples_per_bin:
        raise ValueError("--samples-per-bin must divide the configured chunk length")
    n_bins = int(np.ceil((stop - first) / args.samples_per_bin))
    n_passes = int(config["outer_passes"])
    positions = np.load(args.run / "channel_positions.npy")
    channel_order = np.lexsort((positions[:, 0], positions[:, 1]))
    n_before = int(round(float(config["ms_before"]) * fs / 1000))
    n_after = int(round(float(config["ms_after"]) * fs / 1000))
    margin = max(int(round(float(config["read_margin_ms"]) * fs / 1000)), n_before + n_after, 128)
    filter_config = type("FilterConfig", (), {
        "freq_min": 300.0, "freq_max": 6000.0, "filter_order": 3,
    })()
    sos = make_filter(fs, filter_config)
    stages = np.empty((n_passes + 1, n_channels, n_bins), dtype=np.float32)

    reader = spikeglx.Reader(metadata["recording_path"])
    try:
        for chunk_index, core_start in enumerate(range(first, stop, chunk_samples)):
            core_stop = min(core_start + chunk_samples, stop)
            if (core_stop - core_start) % args.samples_per_bin:
                raise ValueError("the final core length must be divisible by --samples-per-bin")
            read_start = max(0, core_start - margin)
            read_stop = min(reader.ns, core_stop + margin)
            data = preprocess_voltage(reader[read_start:read_stop, :n_channels], sos)
            chunk_path = args.run / "chunks" / f"chunk_{chunk_index:06d}.npz"
            with np.load(chunk_path, allow_pickle=False) as saved:
                needed = {"spike_times", "neighbor_ids", "predictions", "residual_pass", "noise"}
                missing = needed.difference(saved.files)
                if missing:
                    raise ValueError(f"{chunk_path} lacks exact replay arrays: {sorted(missing)}")
                residual = torch.as_tensor(data, dtype=torch.float32, device=args.device)
                noise = torch.as_tensor(saved["noise"], dtype=torch.float32, device=args.device)
                local_slice = slice(core_start - read_start, core_stop - read_start)
                first_bin = (core_start - first) // args.samples_per_bin
                n_local_bins = (core_stop - core_start) // args.samples_per_bin
                stages[0, :, first_bin:first_bin + n_local_bins] = (
                    signed_block_extrema(residual[local_slice] / noise, args.samples_per_bin)
                    .T.cpu().numpy()
                )
                passes = saved["residual_pass"]
                times = saved["spike_times"] - read_start
                ids = saved["neighbor_ids"]
                predictions = saved["predictions"]
                for residual_pass in range(n_passes):
                    rows = np.flatnonzero(passes == residual_pass)
                    if len(rows):
                        subtract_saved_predictions(
                            residual, times[rows], ids[rows], predictions[rows], n_before,
                            args.reconstruction_batch_size,
                        )
                    stages[residual_pass + 1, :, first_bin:first_bin + n_local_bins] = (
                        signed_block_extrema(residual[local_slice] / noise, args.samples_per_bin)
                        .T.cpu().numpy()
                    )
            if (chunk_index + 1) % 50 == 0 or core_stop == stop:
                print(f"replayed chunk {chunk_index + 1}: samples [{core_start}, {core_stop})", flush=True)
    finally:
        reader.close()

    spike_times = np.load(args.run / "spike_times.npy", mmap_mode="r")
    sources = np.load(args.run / "global_sources.npy", mmap_mode="r")
    temporal_idx = np.load(args.run / "temporal_idx.npy", mmap_mode="r")
    alpha = np.load(args.run / "alpha.npy", mmap_mode="r")
    residual_pass = np.load(args.run / "residual_pass.npy", mmap_mode="r")
    omega = np.load(args.run / "omega.npy")
    time_minutes = np.asarray(spike_times, dtype=np.float64) / (60 * fs)
    depth = np.asarray(sources[:, 1], dtype=np.float32)
    weights, alpha_scale = amplitude_weights(alpha)
    finite = np.isfinite(time_minutes) & np.isfinite(depth)
    time_limit = max(float(time_minutes[finite].max()), 1e-9)
    depth_low, depth_high = np.quantile(depth[finite], (0.002, 0.998))
    palette = plt.colormaps["rainbow"](np.linspace(0, 1, len(omega)))[:, :3]
    colormap = ListedColormap(palette, name=f"q{len(omega)}_rgb")
    boundaries = np.arange(len(omega) + 1, dtype=np.float64) - 0.5
    normalization = BoundaryNorm(boundaries, len(omega))
    raster_height = max(256, int((depth_high - depth_low) / 3))
    vlim = float(np.quantile(np.abs(stages[0]), 0.995))
    vlim = min(12.0, max(4.0, vlim))

    figure, axes = plt.subplots(
        n_passes + 1, 2, figsize=(17, 17), constrained_layout=True,
        facecolor=BACKGROUND, gridspec_kw={"width_ratios": (1.08, 1)},
    )
    image = None
    for stage in range(n_passes + 1):
        voltage_axis, raster_axis = axes[stage]
        image = voltage_axis.imshow(
            stages[stage, channel_order], origin="lower", aspect="auto", cmap="RdBu_r",
            vmin=-vlim, vmax=vlim, interpolation="nearest",
            extent=(0, (stop - first) / (60 * fs), 0, n_channels),
        )
        if stage == 0:
            voltage_axis.set_title("preprocessed recording")
            raster_axis.imshow(
                np.full((raster_height, args.raster_width, 3), 0.05, dtype=np.float32),
                origin="lower", aspect="auto", extent=(0, time_limit, depth_low, depth_high),
            )
            raster_axis.set_title("fitted spikes: none")
        else:
            pass_index = stage - 1
            voltage_axis.set_title(f"residual after pass {stage}")
            rows = np.flatnonzero(finite & (residual_pass <= pass_index))
            raster_axis.imshow(
                categorical_raster(
                    time_minutes[rows], depth[rows], np.asarray(temporal_idx[rows]), weights[rows],
                    palette, (0, time_limit), (depth_low, depth_high), args.raster_width, raster_height,
                ),
                origin="lower", aspect="auto", interpolation="nearest",
                extent=(0, time_limit, depth_low, depth_high),
            )
            raster_axis.set_title(
                f"cumulative fits through pass {stage} · {len(rows):,} spikes"
            )
        voltage_axis.set_ylabel("probe depth (µm)")
        voltage_axis.set_yticks(np.linspace(0, n_channels, 6), [
            f"{positions[channel_order[min(n_channels - 1, int(row))], 1]:.0f}"
            for row in np.linspace(0, n_channels - 1, 6)
        ])
        raster_axis.set_ylabel("localized depth (µm)")
        for axis in (voltage_axis, raster_axis):
            axis.set_xlim(0, time_limit)
            style_axis(axis)
        raster_axis.set_ylim(depth_low, depth_high)
    for axis in axes[-1]:
        axis.set_xlabel("recording time (min)")
    colorbar = figure.colorbar(image, ax=axes[:, 0], pad=0.012, fraction=0.018)
    colorbar.set_label("voltage / robust channel noise (signed bin extremum)")
    mappable = plt.cm.ScalarMappable(norm=normalization, cmap=colormap)
    qbar = figure.colorbar(mappable, ax=axes[:, 1], pad=0.012, fraction=0.018)
    qbar.set_label("selected temporal codebook row · density weighted by |α|")
    qbar.set_ticks(np.arange(len(omega)))
    qbar.set_ticklabels([rf"$\Omega_{{{row}}}$" for row in range(len(omega))])
    figure.suptitle(
        f"0014 full-recording residual replay · {n_passes} pursuit passes · "
        f"cumulative rasters weighted by |α| (median {alpha_scale:.3e})",
        color="#eeeeee", fontsize=13,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800, facecolor=BACKGROUND)
    plt.close(figure)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
