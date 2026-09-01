"""Cross evaluation: run the 0019 fitter on the other pipeline's spike list.

Takes the external extraction's spike times and peak channels as the proposal
list, fits the frozen 0019 model (run codebook, neighborhoods, filter) on raw
recording windows at those sites, and records the full 0019 fit-metric table:
detection score, mean/max channel-normalized RMSE, captured fraction, the
all-channel rule, and the same reason bitmask as the peeling acceptance. No
detection, no peeling, no subtraction: every proposal is fit once on the raw
segment it lands in.

A built-in self-check re-fits a sample of the run's own accepted pass-0 events
through the identical harness; pass-0 events were fit on raw windows, so the
metrics must reproduce the run's consolidated values.
"""

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "allchannel_0019_for_cross", HERE / "0019_allchannel_peeling.py"
)
mod19 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod19
SPEC.loader.exec_module(mod19)
PIPELINE = mod19.PIPELINE
OLD = PIPELINE.OLD
BASE = PIPELINE.BASE
EPS = PIPELINE.EPS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--his-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--fit-batch-size", type=int, default=2048)
    parser.add_argument("--all-channel-min-fraction", type=float, default=None)
    parser.add_argument("--check-events", type=int, default=20000)
    parser.add_argument("--max-segments", type=int, default=0)
    return parser.parse_args()


def load_run_config(run_dir):
    metadata = json.loads((Path(run_dir) / "config.json").read_text())
    fields = {
        key: value
        for key, value in metadata["config"].items()
        if key in mod19.Config.__dataclass_fields__
    }
    return mod19.Config(**fields), metadata


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def fit_batch(
    data_t,
    times,
    channels,
    noise_t,
    safe_fit_ids,
    fit_mask,
    fit_offsets_t,
    omega_t,
    sites,
    axes,
    sigmas,
    config,
    cache,
    channel_fraction,
    n_before,
    n_after,
):
    waveforms, ids, local_offsets, mask, local_noise = BASE.extract_waveforms_torch(
        data_t,
        times,
        channels,
        safe_fit_ids,
        fit_mask,
        fit_offsets_t,
        noise_t,
        n_before,
        n_after,
    )
    fit = PIPELINE.fit_grouped(
        waveforms,
        local_offsets,
        mask,
        local_noise,
        omega_t,
        sites,
        axes,
        sigmas,
        config,
        cache,
    )
    normalized_waveform = waveforms / local_noise[:, :, None]
    channel_input = normalized_waveform.square().sum(dim=2) * mask
    channel_floor = channel_input * channel_fraction
    all_ok = ((fit["channel_improvement"] >= channel_floor - 1e-6) | ~mask).all(dim=1)
    min_channel_fraction = torch.where(
        channel_input > 1e-8,
        fit["channel_improvement"] / channel_input.clamp_min(1e-8),
        torch.full_like(channel_input, float("inf")),
    ).amin(dim=1)
    accepted = (
        torch.isfinite(fit["alpha"])
        & (fit["alpha"] > 0)
        & torch.isfinite(fit["maximum_channel_normalized_rmse"])
        & (fit["maximum_channel_normalized_rmse"] <= config.max_channel_normalized_rmse)
        & (fit["captured_fraction"] >= config.min_captured_fraction)
        & (fit["fitted_projection_score"] >= config.min_fitted_projection)
        & (fit["raw_energy_drop"] > config.min_raw_energy_drop)
        & all_ok
    )
    reasons = torch.zeros(len(times), dtype=torch.int32, device=data_t.device)
    reasons += (~(torch.isfinite(fit["alpha"]) & (fit["alpha"] > 0))).to(torch.int32) * 1
    reasons += (
        (~torch.isfinite(fit["maximum_channel_normalized_rmse"]))
        | (fit["maximum_channel_normalized_rmse"] > config.max_channel_normalized_rmse)
    ).to(torch.int32) * 2
    reasons += (fit["captured_fraction"] < config.min_captured_fraction).to(torch.int32) * 4
    reasons += (fit["fitted_projection_score"] < config.min_fitted_projection).to(torch.int32) * 8
    reasons += (~all_ok).to(torch.int32) * 16
    reasons += (fit["raw_energy_drop"] <= config.min_raw_energy_drop).to(torch.int32) * 32
    anchor_noise = noise_t[channels].clamp_min(EPS)
    at_t = data_t[times, channels] / anchor_noise
    window_offsets = torch.arange(-n_before, n_after, device=data_t.device)
    window = data_t[
        (times[:, None] + window_offsets[None, :]).clamp(0, len(data_t) - 1),
        channels[:, None],
    ]
    peak = window.abs().argmax(dim=1)
    arange = torch.arange(len(times), device=data_t.device)
    extreme = window[arange, peak] / anchor_noise
    fit = dict(fit)
    fit.pop("prediction", None)
    fit.update(
        {
            "all_ok": all_ok,
            "min_channel_captured_fraction": min_channel_fraction,
            "reasons": reasons,
            "accepted": accepted,
            "detection_score_at_t": at_t,
            "extreme_score": extreme,
        }
    )
    return fit


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {}
    return {
        key: float(value)
        for key, value in (
            ("n", len(array)),
            ("mean", array.mean()),
            ("median", np.median(array)),
            ("p10", np.percentile(array, 10)),
            ("p90", np.percentile(array, 90)),
        )
    }


def summarize(merged, channel_fraction, extra):
    accepted = merged["accepted"].astype(bool)
    hist = {
        str(int(value)): int(count)
        for value, count in zip(
            *np.unique(merged["reasons"][merged["reasons"] != 0], return_counts=True)
        )
    }
    return {
        "n_proposed": int(len(merged["spike_times"])),
        "n_alpha_positive": int((merged["alpha"] > 0).sum()),
        "n_pass_detection_5": int((np.abs(merged["extreme_score"]) >= 5.0).sum()),
        "n_pass_rmse_3": int((merged["maximum_channel_normalized_rmse"] <= 3.0).sum()),
        "n_pass_projection_8": int((merged["fitted_projection_score"] >= 8.0).sum()),
        "n_pass_all_channel": int(merged["all_ok"].astype(bool).sum()),
        "n_accepted": int(accepted.sum()),
        "acceptance_rate": float(accepted.mean()) if len(accepted) else 0.0,
        "reason_histogram": hist,
        "all_channel_min_fraction": float(channel_fraction),
        "mean_channel_normalized_rmse": quantiles(
            merged["mean_channel_normalized_rmse"]
        ),
        "maximum_channel_normalized_rmse": quantiles(
            merged["maximum_channel_normalized_rmse"]
        ),
        "captured_fraction": quantiles(merged["captured_fraction"]),
        "min_channel_captured_fraction": quantiles(
            merged["min_channel_captured_fraction"]
        ),
        "extreme_score": quantiles(merged["extreme_score"]),
        **extra,
    }


def consolidate(parts, output):
    merged = {
        key: np.concatenate([part[key] for part in parts]) for key in parts[0]
    }
    for key, value in merged.items():
        OLD.atomic_npy(output / f"{key}.npy", value)
    return merged


def load_segment_parts(segments_dir):
    parts = []
    for path in sorted(segments_dir.glob("seg_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if not len(archive["spike_times"]):
                continue
            if "accepted" not in archive.files:
                raise RuntimeError(f"{path} predates the current schema; delete it")
            parts.append({key: archive[key] for key in archive.files})
    return parts


def run_self_check(args, config, reader, run_dir, check_dir, context):
    (
        sos,
        n_channels,
        n_before,
        n_after,
        margin,
        omega_t,
        sites,
        axes,
        sigmas,
        cache,
        safe_fit_ids,
        fit_mask,
        fit_offsets_t,
    ) = context
    check_dir = Path(check_dir)
    check_dir.mkdir(parents=True, exist_ok=True)
    my_times = np.load(run_dir / "spike_times.npy")
    my_pass = np.load(run_dir / "recording_pass.npy")
    my_channels = np.load(run_dir / "spike_channels.npy")
    fs = float(reader.fs)
    chunk_samples = max(1, int(round(config.chunk_seconds * fs)))
    first_chunk = 100
    n_check_chunks = max(1, int(np.ceil(args.check_events / 200)))
    block_start = first_chunk * chunk_samples
    block_stop = (first_chunk + n_check_chunks) * chunk_samples
    pool = np.nonzero(
        (my_pass == 0) & (my_times >= block_start) & (my_times < block_stop)
    )[0]
    if len(pool) > args.check_events:
        rng = np.random.default_rng(0)
        pool = rng.choice(pool, args.check_events, replace=False)
    by_chunk = {}
    for row in pool:
        by_chunk.setdefault(int(my_times[row] // chunk_samples), []).append(row)
    parts = []
    with torch.inference_mode():
        for chunk_number, rows in sorted(by_chunk.items()):
            rows = np.asarray(rows)
            core_start = chunk_number * chunk_samples
            core_stop = min(core_start + chunk_samples, reader.ns)
            read_start = max(0, core_start - margin)
            read_stop = min(reader.ns, core_stop + margin)
            data_np = BASE.preprocess_voltage(
                reader[read_start:read_stop, :n_channels], sos
            )
            data_t = torch.as_tensor(data_np, dtype=torch.float32, device=config.device)
            noise_t = torch.as_tensor(
                BASE.robust_channel_noise(data_np), device=config.device
            )
            times = my_times[rows].astype(np.int64)
            channels = my_channels[rows].astype(np.int64)
            fit = fit_batch(
                data_t,
                torch.as_tensor(times - read_start, device=config.device),
                torch.as_tensor(channels, device=config.device),
                noise_t,
                safe_fit_ids,
                fit_mask,
                fit_offsets_t,
                omega_t,
                sites,
                axes,
                sigmas,
                config,
                cache,
                config.all_channel_min_fraction,
                n_before,
                n_after,
            )
            part = {key: to_numpy(value) for key, value in fit.items()}
            part["spike_times"] = times
            part["spike_channels"] = channels
            part["my_run_row"] = rows
            parts.append(part)
    if not parts:
        raise RuntimeError("self-check produced no events")
    merged = {
        key: np.concatenate([part[key] for part in parts]) for key in parts[0]
    }
    reference = {}
    for field in (
        "mean_channel_normalized_rmse",
        "maximum_channel_normalized_rmse",
        "captured_fraction",
        "fitted_projection_score",
        "alpha",
    ):
        saved = np.load(run_dir / f"{field}.npy")[merged["my_run_row"]]
        mine = merged[field]
        both = np.isfinite(saved) & np.isfinite(mine)
        reference[field] = {
            "n_compared": int(both.sum()),
            "max_abs_diff": float(np.max(np.abs(saved[both] - mine[both])))
            if both.any()
            else None,
            "mean_abs_diff": float(np.mean(np.abs(saved[both] - mine[both])))
            if both.any()
            else None,
        }
    summary = summarize(
        merged, config.all_channel_min_fraction, {"reference_check": reference}
    )
    OLD.atomic_json(check_dir / "summary.json", summary)
    print("self-check:", json.dumps(summary), flush=True)


def main():
    args = parse_args()
    config, metadata = load_run_config(args.run_dir)
    if args.all_channel_min_fraction is not None:
        config = replace(config, all_channel_min_fraction=args.all_channel_min_fraction)
    config = replace(config, device=args.device)
    run_dir = Path(args.run_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    import spikeglx

    reader = spikeglx.Reader(metadata["recording_path"])
    fs = float(reader.fs)
    n_channels = metadata["n_channels"]
    fit_ids = np.load(run_dir / "fit_neighborhood_ids.npy")
    fit_offsets = np.load(run_dir / "fit_neighborhood_offsets.npy")
    sos = BASE.make_filter(fs, config.base())
    omega_t = mod19.preserve_omega_polarity(
        torch.as_tensor(np.load(run_dir / "omega.npy").astype(np.float32), device=config.device)
    )
    sites_np, axes_np = BASE.coarse_lattice(config.base())
    sites = torch.as_tensor(sites_np, device=config.device)
    axes = [torch.as_tensor(axis, device=config.device) for axis in axes_np]
    sigmas = torch.as_tensor(BASE.sigma_bank(config.base()), device=config.device)
    cache = OLD.FootprintCache(sites, sigmas, config.device)
    safe_fit_ids, fit_mask = BASE.gpu_neighborhood(fit_ids, config.device)
    fit_offsets_t = torch.as_tensor(fit_offsets, device=config.device)
    n_before = int(round(config.ms_before * fs / 1000))
    n_after = int(round(config.ms_after * fs / 1000))
    margin = max(
        int(round(config.read_margin_ms * fs / 1000)),
        2 * (n_before + n_after),
        128,
    )
    context = (
        sos,
        n_channels,
        n_before,
        n_after,
        margin,
        omega_t,
        sites,
        axes,
        sigmas,
        cache,
        safe_fit_ids,
        fit_mask,
        fit_offsets_t,
    )

    if args.check_events:
        run_self_check(
            args, config, reader, run_dir, output / "self_check", context
        )

    his_times = np.load(Path(args.his_results) / "spike_times.npy").astype(np.int64)
    his_channels = np.load(Path(args.his_results) / "spike_channels.npy").astype(np.int64)
    order = np.argsort(his_times, kind="stable")
    his_times, his_channels, his_rows = his_times[order], his_channels[order], order

    segment_samples = max(1, int(round(args.segment_seconds * fs)))
    segments_dir = output / "segments"
    segments_dir.mkdir(exist_ok=True)
    starts = list(range(0, reader.ns, segment_samples))
    if args.max_segments:
        starts = starts[: args.max_segments]
    total = 0
    for number, segment_start in enumerate(starts):
        segment_stop = min(segment_start + segment_samples, reader.ns)
        path = segments_dir / f"seg_{number:06d}.npz"
        if path.exists():
            with np.load(path) as archive:
                total += len(archive["spike_times"])
            continue
        first = int(np.searchsorted(his_times, segment_start, side="left"))
        last = int(np.searchsorted(his_times, segment_stop, side="left"))
        if first >= last:
            OLD.atomic_npz(
                path,
                {
                    "spike_times": np.empty(0, dtype=np.int64),
                    "reason": np.empty(0, dtype=np.int32),
                },
            )
            continue
        read_start = max(0, segment_start - margin)
        read_stop = min(reader.ns, segment_stop + margin)
        data_np = BASE.preprocess_voltage(
            reader[read_start:read_stop, :n_channels], sos
        )
        data_t = torch.as_tensor(data_np, dtype=torch.float32, device=config.device)
        noise_t = torch.as_tensor(
            BASE.robust_channel_noise(data_np), device=config.device
        )
        parts = []
        with torch.inference_mode():
            for offset in range(first, last, args.fit_batch_size):
                stop = min(offset + args.fit_batch_size, last)
                local = his_times[offset:stop] - read_start
                valid = (local >= n_before) & (local < (read_stop - read_start) - n_after)
                if not valid.all():
                    local = local[valid]
                if not len(local):
                    continue
                times = torch.as_tensor(local, device=config.device)
                channels = torch.as_tensor(
                    his_channels[offset:stop][valid], device=config.device
                )
                fit = fit_batch(
                    data_t,
                    times,
                    channels,
                    noise_t,
                    safe_fit_ids,
                    fit_mask,
                    fit_offsets_t,
                    omega_t,
                    sites,
                    axes,
                    sigmas,
                    config,
                    cache,
                    config.all_channel_min_fraction,
                    n_before,
                    n_after,
                )
                part = {key: to_numpy(value) for key, value in fit.items()}
                part["spike_times"] = (local + read_start).astype(np.int64)
                part["spike_channels"] = his_channels[offset:stop][valid].astype(np.int64)
                part["his_row"] = his_rows[offset:stop][valid].astype(np.int64)
                parts.append(part)
        if not parts:
            OLD.atomic_npz(
                path,
                {
                    "spike_times": np.empty(0, dtype=np.int64),
                    "reason": np.empty(0, dtype=np.int32),
                },
            )
            continue
        fields = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        OLD.atomic_npz(path, fields)
        total += len(fields["spike_times"])
        print(
            f"segment {number + 1}/{len(starts)} proposed={last - first} "
            f"cumulative={total}",
            flush=True,
        )

    parts = load_segment_parts(segments_dir)
    if not parts:
        raise RuntimeError("no segments produced events")
    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    for key, value in merged.items():
        OLD.atomic_npy(output / f"{key}.npy", value)
    summary = summarize(merged, config.all_channel_min_fraction, {})
    OLD.atomic_json(output / "summary.json", summary)
    print("cross-fit summary:", json.dumps(summary), flush=True)
    reader.close()


if __name__ == "__main__":
    main()
