"""Plot spike reconstructions and sampled reconstruction-error diagnostics."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


EPS = 1e-12


def profile_parameters(kernel, profile_idx, n_scales=10):
    scales = np.geomspace(1.0, 512.0, n_scales)
    if kernel in ("gauss_aniso", "mono_aniso"):
        lateral, axial = np.meshgrid(scales, scales, indexing="ij")
        parameters = np.column_stack((lateral.ravel(), axial.ravel()))
    elif kernel in ("power", "student", "yukawa", "dog"):
        parameters = np.column_stack((scales, np.full(n_scales, 2.0)))
    else:
        parameters = scales[:, None]
    return parameters[np.asarray(profile_idx, dtype=np.int64)]


def kernel_values(kernel, dxy2, dz2, parameters):
    p0 = parameters[..., 0]
    d2 = dxy2 + dz2
    if kernel == "monopole":
        return p0 / np.sqrt(d2 + p0 ** 2)
    if kernel == "exponential":
        return np.exp(-np.sqrt(d2) / p0)
    if kernel == "gauss":
        return np.exp(-d2 / (2 * p0 ** 2))
    if kernel == "lorentz":
        return p0 ** 2 / (d2 + p0 ** 2)
    p1 = parameters[..., 1]
    if kernel == "power":
        return (p0 / np.sqrt(d2 + p0 ** 2)) ** p1
    if kernel == "student":
        return (1 + d2 / p0 ** 2) ** (-p1)
    if kernel == "yukawa":
        return (p0 / np.sqrt(d2 + p0 ** 2)) * np.exp(
            -np.sqrt(d2 + 1e-8) / p1)
    if kernel == "dog":
        return (
            np.exp(-d2 / (2 * p0 ** 2))
            - np.exp(-d2 / (2 * (p0 * p1) ** 2)) / p1 ** 2
        )
    if kernel == "gauss_aniso":
        return np.exp(-(dxy2 / (2 * p0 ** 2) + dz2 / (2 * p1 ** 2)))
    if kernel == "mono_aniso":
        return p0 / np.sqrt(dxy2 + dz2 * (p0 / p1) ** 2 + p0 ** 2)
    raise ValueError(f"unsupported kernel: {kernel}")


def reconstruct(off, mask, sources, parameters, omega, temporal_idx, alpha, kernel):
    dxy2 = (
        (off[..., 0] - sources[:, None, 0]) ** 2
        + (off[..., 1] - sources[:, None, 1]) ** 2
    )
    dz2 = sources[:, None, 2] ** 2
    footprint = kernel_values(
        kernel, dxy2, dz2, parameters[:, None, :]).astype(np.float32)
    footprint *= mask
    footprint /= np.maximum(
        np.linalg.norm(footprint, axis=1, keepdims=True), EPS)
    prediction = (
        alpha[:, None, None]
        * footprint[:, :, None]
        * omega[temporal_idx, None, :]
    )
    return footprint, prediction


def sampled_errors(session, fit, kernel, sample_size, seed, chunk=8192):
    waveforms = np.load(session / "neighborhood_waveforms.npy", mmap_mode="r")
    off = np.load(session / "local_coords.npy", mmap_mode="r")
    ids = np.load(session / "neighbor_ids.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(waveforms), min(sample_size, len(waveforms)),
                                 replace=False))
    errors = np.empty(len(indices), dtype=np.float32)
    energies = np.empty(len(indices), dtype=np.float32)
    parameters = profile_parameters(kernel, fit["profile_idx"])
    for start in range(0, len(indices), chunk):
        rows = indices[start:start + chunk]
        mask = np.asarray(ids[rows] >= 0)
        measured = np.asarray(waveforms[rows], dtype=np.float32) * mask[:, :, None]
        _, predicted = reconstruct(
            np.asarray(off[rows]), mask, np.asarray(fit["sources"])[rows],
            parameters[rows], np.asarray(fit["omega"]),
            np.asarray(fit["temporal_idx"])[rows],
            np.asarray(fit["alpha"])[rows], kernel,
        )
        energy = np.square(measured).sum((1, 2))
        energies[start:start + len(rows)] = energy
        errors[start:start + len(rows)] = (
            np.square(measured - predicted).sum((1, 2))
            / np.maximum(energy, EPS)
        )
    return indices, errors, energies


def choose_examples(indices, errors, energies, n_examples):
    usable = np.isfinite(errors) & (energies >= np.quantile(energies, 0.1))
    order = np.flatnonzero(usable)[np.argsort(errors[usable])]
    ranks = np.linspace(0.03, 0.97, n_examples)
    chosen = order[np.rint(ranks * (len(order) - 1)).astype(int)]
    return indices[chosen], errors[chosen], ranks


def spatial_image(kernel, source, parameters, extent=160.0, n=201):
    axis = np.linspace(-extent, extent, n)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    dxy2 = (xx - source[0]) ** 2 + (yy - source[1]) ** 2
    dz2 = np.full_like(dxy2, source[2] ** 2)
    image = kernel_values(kernel, dxy2, dz2, parameters)
    return axis, image


def plot_examples(session, fit, kernel, indices, errors, ranks, output):
    waveforms = np.load(session / "neighborhood_waveforms.npy", mmap_mode="r")
    off_all = np.load(session / "local_coords.npy", mmap_mode="r")
    ids = np.load(session / "neighbor_ids.npy", mmap_mode="r")
    mask = np.asarray(ids[indices] >= 0)
    measured = np.asarray(waveforms[indices], dtype=np.float32) * mask[:, :, None]
    off = np.asarray(off_all[indices])
    sources = np.asarray(fit["sources"])[indices]
    temporal_idx = np.asarray(fit["temporal_idx"])[indices]
    alpha = np.asarray(fit["alpha"])[indices]
    parameters = profile_parameters(kernel, fit["profile_idx"])[indices]
    footprint, predicted = reconstruct(
        off, mask, sources, parameters, np.asarray(fit["omega"]),
        temporal_idx, alpha, kernel)

    n_examples = len(indices)
    figure, axes = plt.subplots(
        4, n_examples, figsize=(3.1 * n_examples, 12), constrained_layout=True,
        squeeze=False, gridspec_kw={"height_ratios": (1.3, 0.65, 1.2, 0.65)},
    )
    for column in range(n_examples):
        valid = mask[column]
        scale = 16 / max(float(np.abs(measured[column, valid]).max()), EPS)
        time_offset = np.arange(measured.shape[2]) * 0.32
        axis = axes[0, column]
        for channel in np.flatnonzero(valid):
            axis.plot(
                off[column, channel, 0] + time_offset,
                off[column, channel, 1] + measured[column, channel] * scale,
                color="#e03131", linewidth=0.9,
                label="measured" if channel == np.flatnonzero(valid)[0] else None,
            )
            axis.plot(
                off[column, channel, 0] + time_offset,
                off[column, channel, 1] + predicted[column, channel] * scale,
                color="#2f9e44", linewidth=1.05, linestyle="--",
                label="model" if channel == np.flatnonzero(valid)[0] else None,
            )
            axis.plot(off[column, channel, 0], off[column, channel, 1],
                      "s", markersize=2.8, color="#969696")
        axis.set_title(
            f"spike {indices[column]:,} · sampled p{100*ranks[column]:.0f}\n"
            f"relative squared error {errors[column]:.3f}", fontsize=8)
        axis.tick_params(labelsize=6)
        if column == 0:
            axis.set_ylabel("measured vs. model\non contact geometry", fontsize=8)
            axis.legend(fontsize=6)

        axis = axes[1, column]
        x = np.arange(valid.sum())
        measured_ptp = np.ptp(measured[column, valid], axis=1)
        predicted_ptp = np.ptp(predicted[column, valid], axis=1)
        axis.bar(x - 0.19, measured_ptp, 0.38, color="#e03131")
        axis.bar(x + 0.19, predicted_ptp, 0.38, color="#2f9e44")
        ptp_error = np.abs(measured_ptp - predicted_ptp).sum() / max(
            measured_ptp.sum(), EPS)
        axis.set_title(f"peak-to-peak relative error {ptp_error:.3f}", fontsize=7.5)
        axis.set_xlabel("valid channel", fontsize=7)
        axis.tick_params(labelsize=6)
        if column == 0:
            axis.set_ylabel("peak-to-peak", fontsize=8)

        axis = axes[2, column]
        grid, image = spatial_image(kernel, sources[column], parameters[column])
        if image.min() < 0 < image.max():
            bound = float(np.abs(image).max())
            norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
            cmap = "coolwarm"
        else:
            norm = None
            cmap = "magma"
        axis.imshow(image, origin="lower", extent=(grid[0], grid[-1], grid[0], grid[-1]),
                    cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
        axis.scatter(off[column, valid, 0], off[column, valid, 1], s=16, marker="s",
                     facecolors="none", edgecolors="white", linewidths=0.7)
        axis.plot(0, 0, "+", color="#00ffff", markersize=11, markeredgewidth=1.7)
        axis.plot(sources[column, 0], sources[column, 1], "o", markerfacecolor="none",
                  markeredgecolor="#4c8dff", markersize=13, markeredgewidth=2)
        parameter_text = ", ".join(f"{value:.1f}" for value in parameters[column])
        axis.set_title(
            f"site ({sources[column,0]:+.0f}, {sources[column,1]:+.0f}, "
            f"{sources[column,2]:.0f}) µm · p=({parameter_text})", fontsize=7.5)
        axis.tick_params(labelsize=6)
        if column == 0:
            axis.set_ylabel("selected spatial footprint", fontsize=8)

        axis = axes[3, column]
        projected = np.einsum("c,ct->t", footprint[column], measured[column])
        model_time = alpha[column] * np.asarray(fit["omega"])[temporal_idx[column]]
        time_ms = np.arange(len(projected)) / 30
        axis.plot(time_ms, projected, color="#e03131", linewidth=1,
                  label="spatial projection" if column == 0 else None)
        axis.plot(time_ms, model_time, color="#2f9e44", linewidth=1.1,
                  linestyle="--", label="alpha × Omega[q]" if column == 0 else None)
        axis.axhline(0, color="0.7", linewidth=0.5)
        axis.set_title(f"temporal row q={temporal_idx[column]} · alpha={alpha[column]:.2f}",
                       fontsize=7.5)
        axis.set_xlabel("time (ms)", fontsize=7)
        axis.tick_params(labelsize=6)
        if column == 0:
            axis.set_ylabel("projected waveform", fontsize=8)
            axis.legend(fontsize=6)

    figure.suptitle(
        f"Masked {kernel} reconstruction examples · measured red, model green dashed",
        fontsize=12)
    figure.savefig(output / "reconstruction_examples.png", dpi=800,
                   bbox_inches="tight")
    plt.close(figure)


def plot_diagnostics(session, fit, kernel, indices, errors, energies, output):
    counts_all = np.load(session / "neighbor_counts.npy", mmap_mode="r")
    counts = np.asarray(counts_all[indices])
    temporal_idx = np.asarray(fit["temporal_idx"])[indices]
    finite = np.isfinite(errors)
    clipped = errors[finite & (errors <= np.quantile(errors[finite], 0.995))]

    figure, axes = plt.subplots(1, 4, figsize=(18, 4.3), constrained_layout=True)
    axes[0].hist(clipped, bins=80, color="#3182bd")
    axes[0].axvline(np.median(errors[finite]), color="#cb181d", linestyle="--",
                    label=f"median {np.median(errors[finite]):.3f}")
    axes[0].axvline(np.quantile(errors[finite], 0.9), color="#756bb1", linestyle=":",
                    label=f"p90 {np.quantile(errors[finite], 0.9):.3f}")
    axes[0].set(xlabel="per-spike relative squared error", ylabel="sampled spikes",
                title="reconstruction-error distribution")
    axes[0].legend(fontsize=7)

    channel_values = np.arange(4, 9)
    medians = np.array([
        np.median(errors[counts == count]) if np.any(counts == count) else np.nan
        for count in channel_values
    ])
    lower = np.array([
        np.quantile(errors[counts == count], 0.25) if np.any(counts == count) else np.nan
        for count in channel_values
    ])
    upper = np.array([
        np.quantile(errors[counts == count], 0.75) if np.any(counts == count) else np.nan
        for count in channel_values
    ])
    axes[1].errorbar(channel_values, medians, yerr=(medians - lower, upper - medians),
                     fmt="o-", color="#2ca25f", capsize=3)
    axes[1].set(xlabel="real channels in patch", ylabel="relative squared error",
                title="median and IQR by padding", xticks=channel_values)
    axes[1].grid(alpha=0.25)

    plot_energy = np.log10(np.maximum(energies[finite], EPS))
    plot_error = np.minimum(errors[finite], np.quantile(errors[finite], 0.995))
    image = axes[2].hexbin(plot_energy, plot_error, gridsize=55, bins="log", cmap="magma")
    figure.colorbar(image, ax=axes[2], label="sampled spikes")
    axes[2].set(xlabel="log10 measured waveform energy",
                ylabel="relative squared error", title="error vs. signal energy")

    q_values = np.arange(len(fit["omega"]))
    q_error = np.array([
        np.median(errors[temporal_idx == q]) if np.any(temporal_idx == q) else np.nan
        for q in q_values
    ])
    q_fraction = np.array([(temporal_idx == q).mean() for q in q_values])
    axes[3].bar(q_values - 0.2, q_error, 0.4, color="#756bb1",
                label="median error")
    twin = axes[3].twinx()
    twin.bar(q_values + 0.2, q_fraction, 0.4, color="#fd8d3c", alpha=0.7,
             label="spike fraction")
    axes[3].set(xlabel="temporal row q", ylabel="median relative squared error",
                title="error and usage by temporal row", xticks=q_values)
    twin.set_ylabel("fraction of sampled spikes")
    handles_a, labels_a = axes[3].get_legend_handles_labels()
    handles_b, labels_b = twin.get_legend_handles_labels()
    axes[3].legend(handles_a + handles_b, labels_a + labels_b, fontsize=7)

    figure.suptitle(
        f"Masked {kernel} reconstruction diagnostics · {len(indices):,}-spike sample · "
        f"full-fit nMSE {float(fit['nmse']):.4f}", fontsize=11)
    figure.savefig(output / "reconstruction_diagnostics.png", dpi=800,
                   bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--n-examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    archive = np.load(args.fit, allow_pickle=True)
    required = {"sources", "profile_idx", "omega", "temporal_idx", "alpha", "nmse"}
    missing = required.difference(archive.files)
    if missing:
        raise KeyError(f"fit is missing fields: {sorted(missing)}")
    fit = {key: archive[key] for key in required}
    archive.close()
    indices, errors, energies = sampled_errors(
        args.session, fit, args.kernel, args.sample_size, args.seed)
    example_indices, example_errors, ranks = choose_examples(
        indices, errors, energies, args.n_examples)
    args.out.mkdir(parents=True)
    plot_examples(args.session, fit, args.kernel, example_indices,
                  example_errors, ranks, args.out)
    plot_diagnostics(args.session, fit, args.kernel, indices, errors, energies, args.out)
    print(f"sample median error={np.median(errors):.6f}", flush=True)
    print(f"sample p90 error={np.quantile(errors, 0.9):.6f}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
