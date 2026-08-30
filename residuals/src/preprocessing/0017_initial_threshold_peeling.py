"""Initial negative-threshold discovery with 0016 localization and peeling."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "onehot_0016", HERE / "0016_onehot_lattice_peeling.py"
)
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)
BASE = PIPELINE.BASE
_output_metadata_0016 = PIPELINE.output_metadata
_validate_config_0016 = PIPELINE.validate_config


@dataclass(frozen=True)
class Config(PIPELINE.Config):
    threshold: float = 6.0
    discovery_temporal_radius_samples: int = 5
    discovery_isolation_radius_samples: int = 0


def output_metadata(config, recording_path, fs, n_channels, first, stop):
    metadata = _output_metadata_0016(
        config, recording_path, fs, n_channels, first, stop
    )
    metadata["detector"] = (
        "negative local minima on per-channel MAD-standardized voltage; "
        "spatiotemporal NMS; subtract and rescore"
    )
    metadata["discovery_score"] = "-voltage / per-channel robust noise"
    metadata["discovery_threshold_units"] = "per-channel robust-noise standard deviations"
    metadata["discovery_template_search"] = False
    if config.discovery_isolation_radius_samples == 0:
        metadata["config"].pop("discovery_isolation_radius_samples")
    return metadata


def detect_events(
    residual,
    noise,
    omega,
    footprints,
    safe_detection_ids,
    config,
    fs,
    valid_start,
    valid_stop,
):
    del omega, fs
    scores = -residual / noise[None]
    valid_neighbors = footprints.abs().sum(dim=1) > 0
    maximum = (
        None
        if config.discovery_isolation_radius_samples
        else config.max_events_per_pass
    )
    times, channels, selected_scores = BASE.spatiotemporal_nms(
        scores,
        (safe_detection_ids, valid_neighbors),
        config.threshold,
        config.discovery_temporal_radius_samples,
        valid_start,
        valid_stop,
        maximum,
    )
    before_isolation = len(times)
    if config.discovery_isolation_radius_samples and len(times):
        occupancy = torch.zeros_like(scores, dtype=torch.bool)
        occupancy[times, channels] = True
        prefix = torch.cat(
            (
                torch.zeros(
                    (1, scores.shape[1]), dtype=torch.int32, device=scores.device
                ),
                occupancy.cumsum(dim=0, dtype=torch.int32),
            )
        )
        radius = config.discovery_isolation_radius_samples
        left = (times - radius).clamp_min(0)
        right = (times + radius + 1).clamp_max(len(scores))
        neighbors = safe_detection_ids[channels]
        valid = valid_neighbors[channels]
        counts = (
            prefix[right[:, None], neighbors]
            - prefix[left[:, None], neighbors]
        ).masked_fill(~valid, 0).sum(dim=1)
        keep = counts == 1
        times = times[keep]
        channels = channels[keep]
        selected_scores = selected_scores[keep]
    if config.max_events_per_pass is not None and len(times) > config.max_events_per_pass:
        selected_scores, keep = torch.topk(
            selected_scores,
            config.max_events_per_pass,
            largest=True,
            sorted=False,
        )
        times = times[keep]
        channels = channels[keep]
        order = torch.argsort(times, stable=True)
        times = times[order]
        channels = channels[order]
        selected_scores = selected_scores[order]
    unavailable = torch.full_like(times, -1)
    return (
        times,
        channels,
        selected_scores,
        unavailable,
        unavailable,
        {
            "channel_samples_above_threshold": int(
                (scores >= config.threshold).sum().item()
            ),
            "time_samples_above_threshold": int(
                (scores.amax(dim=1) >= config.threshold).sum().item()
            ),
            "local_maxima_before_isolation": before_isolation,
            "isolated_proposals": len(times),
        },
    )


def validate_config(config):
    _validate_config_0016(config)
    if config.threshold <= 0:
        raise ValueError("initial spike threshold must be positive")
    if config.discovery_temporal_radius_samples < 0:
        raise ValueError("discovery temporal radius must be nonnegative")
    if config.discovery_isolation_radius_samples < 0:
        raise ValueError("discovery isolation radius must be nonnegative")


def self_test(device):
    config = Config(device=device, discovery_temporal_radius_samples=5)
    residual = torch.zeros(64, 3, device=device)
    residual[20, 1] = -7
    residual[22, 2] = -6.5
    residual[45, 0] = 12
    noise = torch.ones(3, device=device)
    safe_ids = torch.tensor(
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=torch.long, device=device
    )
    footprints = torch.ones(3, 1, 3, device=device)
    detected = detect_events(
        residual,
        noise,
        torch.empty(0, device=device),
        footprints,
        safe_ids,
        config,
        1000,
        0,
        len(residual),
    )
    times, channels, scores, initial_sigma, initial_temporal, counts = detected
    if times.tolist() != [20] or channels.tolist() != [1] or scores.tolist() != [7.0]:
        raise AssertionError("negative-threshold discovery or NMS is incorrect")
    if initial_sigma.tolist() != [-1] or initial_temporal.tolist() != [-1]:
        raise AssertionError("template-free discoveries must use unavailable sentinels")
    if counts["channel_samples_above_threshold"] != 2:
        raise AssertionError("threshold crossing count is incorrect")
    isolated_config = Config(
        device=device,
        discovery_temporal_radius_samples=5,
        discovery_isolation_radius_samples=30,
    )
    isolated_residual = torch.zeros(64, 3, device=device)
    isolated_residual[20, 1] = -7
    isolated_residual[30, 2] = -6.5
    isolated = detect_events(
        isolated_residual,
        noise,
        torch.empty(0, device=device),
        footprints,
        safe_ids,
        isolated_config,
        1000,
        0,
        len(isolated_residual),
    )
    if len(isolated[0]) or isolated[-1]["local_maxima_before_isolation"] != 2:
        raise AssertionError("wide discovery isolation is incorrect")
    boundary_config = Config(device=device, discovery_temporal_radius_samples=0)
    boundary_residual = torch.zeros(256, 1, device=device)
    boundary_residual[44, 0] = -7
    boundary_residual[45, 0] = -8
    boundary_residual[211, 0] = -9
    boundary_residual[212, 0] = -10
    boundary = detect_events(
        boundary_residual,
        torch.ones(1, device=device),
        torch.empty(0, device=device),
        torch.ones(1, 1, 1, device=device),
        torch.zeros(1, 1, dtype=torch.long, device=device),
        boundary_config,
        30000,
        45,
        212,
    )
    if boundary[0].tolist() != [45, 211]:
        raise AssertionError("discovery must preserve center-aligned valid bounds")
    print("0017 self-test passed", flush=True)


def main():
    PIPELINE.Config = Config
    PIPELINE.output_metadata = output_metadata
    PIPELINE.detect_events = detect_events
    PIPELINE.validate_config = validate_config
    PIPELINE.self_test = self_test
    PIPELINE.main()


if __name__ == "__main__":
    main()
