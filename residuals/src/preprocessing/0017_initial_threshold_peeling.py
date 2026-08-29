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
    del omega
    scores = -residual / noise[None]
    valid_neighbors = footprints.abs().sum(dim=1) > 0
    n_before = int(round(config.ms_before * fs / 1000))
    peak_start = min(len(scores), valid_start + n_before)
    peak_stop = min(len(scores), valid_stop + n_before)
    times, channels, selected_scores = BASE.spatiotemporal_nms(
        scores,
        (safe_detection_ids, valid_neighbors),
        config.threshold,
        config.discovery_temporal_radius_samples,
        peak_start,
        peak_stop,
        config.max_events_per_pass,
    )
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
        },
    )


def validate_config(config):
    _validate_config_0016(config)
    if config.threshold <= 0:
        raise ValueError("initial spike threshold must be positive")
    if config.discovery_temporal_radius_samples < 0:
        raise ValueError("discovery temporal radius must be nonnegative")


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
