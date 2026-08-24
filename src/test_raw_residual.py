from dataclasses import replace

import numpy as np
import torch

from maths import (
    build_codebook_detection_footprints,
    localize_spikes_fixed_codebook,
)
from preprocessing.raw_residual import (
    ResidualConfig,
    channel_neighborhoods,
    detect_residual_peaks,
    peel_preprocessed_chunk,
    select_template_peaks,
    select_template_peaks_torch,
    subtract_predictions,
    subtract_predictions_monotone,
)


def test_channel_neighborhoods_are_centered_and_padded():
    positions = np.array([[0, 0], [0, 20], [0, 40], [100, 0]], dtype=np.float32)
    ids, local, centroids, counts, _ = channel_neighborhoods(positions, 25)
    assert counts.tolist() == [2, 3, 2, 1]
    assert ids.shape == (4, 3)
    assert ids[0].tolist() == [0, 1, -1]
    np.testing.assert_allclose(local[1, :3].mean(axis=0), 0)
    np.testing.assert_allclose(centroids[3], positions[3])


def test_residual_detector_is_temporally_and_spatially_exclusive():
    residual = np.zeros((40, 3), dtype=np.float32)
    residual[20, 0] = -7
    residual[20, 1] = -9
    residual[21, 2] = -8
    neighbors = [np.array([0, 1]), np.array([0, 1, 2]), np.array([1, 2])]
    times, channels, scores = detect_residual_peaks(
        residual,
        np.ones(3, dtype=np.float32),
        neighbors,
        threshold=6,
        temporal_radius=2,
        valid_start=3,
        valid_stop=37,
    )
    assert times.tolist() == [20]
    assert channels.tolist() == [1]
    np.testing.assert_allclose(scores, [9])


def test_torch_template_peak_selection_matches_numpy():
    scores = np.zeros((20, 4), dtype=np.float32)
    scores[7, 0] = 7
    scores[8, 1] = 9
    scores[14, 2] = 8
    scores[14, 3] = 8
    neighbors = [
        np.array([0, 1]),
        np.array([0, 1, 2]),
        np.array([1, 2, 3]),
        np.array([2, 3]),
    ]
    neighborhood_ids = np.array(
        [[0, 1, -1], [0, 1, 2], [1, 2, 3], [2, 3, -1]], dtype=np.int32
    )
    expected = select_template_peaks(
        scores, neighbors, threshold=6, temporal_radius=2, n_before=5
    )
    actual = select_template_peaks_torch(
        torch.as_tensor(scores), neighborhood_ids,
        threshold=6, temporal_radius=2, n_before=5,
    )
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_array_equal(actual_value, expected_value)


def test_fixed_codebook_recovers_an_exact_model():
    off = np.array(
        [[[-20, -20], [20, -20], [-20, 20], [20, 20]]], dtype=np.float32
    )
    source = np.array([0, 0, 20], dtype=np.float32)
    d2 = np.square(off[0] - source[:2]).sum(axis=1) + source[2] ** 2
    footprint = 1 / np.sqrt(d2 + 1)
    footprint /= np.linalg.norm(footprint)
    time = np.arange(30)
    omega = -np.exp(-np.square(time - 15) / 8)[None].astype(np.float32)
    omega /= np.linalg.norm(omega, axis=1, keepdims=True)
    waveform = (4 * footprint[:, None] * omega[0]).astype(np.float32)[None]
    fit = localize_spikes_fixed_codebook(
        off,
        waveform,
        omega,
        kernels=("monopole",),
        n_scales=1,
        n_sites=7,
        refine_levels=6,
        device="cpu",
    )
    np.testing.assert_allclose(fit["prediction"], waveform, atol=2e-3)
    assert fit["captured_energy"][0] / fit["input_energy"][0] > 0.999
    assert abs(fit["alpha"][0] - 4) < 2e-3

    cache = {}
    cached_fit = localize_spikes_fixed_codebook(
        off,
        waveform,
        omega,
        kernels=("monopole",),
        n_scales=1,
        n_sites=7,
        refine_levels=6,
        device="cpu",
        coarse_footprint_cache=cache,
    )
    assert len(cache) == 1
    np.testing.assert_allclose(cached_fit["prediction"], fit["prediction"])


def test_configuration_batching_matches_serial_configuration_groups():
    base = np.array(
        [[-20, -20], [20, -20], [-20, 20], [20, 20]], dtype=np.float32
    )
    off = np.stack((base, base + [5, -3], base + [-4, 7], base), axis=0)
    time = np.arange(30)
    omega = np.stack(
        (
            -np.exp(-np.square(time - 14) / 8),
            np.exp(-np.square(time - 17) / 12),
        )
    ).astype(np.float32)
    omega /= np.linalg.norm(omega, axis=1, keepdims=True)
    rng = np.random.default_rng(3)
    waveforms = rng.normal(size=(len(off), 4, 30)).astype(np.float32)
    serial = localize_spikes_fixed_codebook(
        off, waveforms, omega, n_scales=2, n_sites=5,
        refine_levels=2, device="cpu", config_batch_size=1,
    )
    batched = localize_spikes_fixed_codebook(
        off, waveforms, omega, n_scales=2, n_sites=5,
        refine_levels=2, device="cpu", config_batch_size=4,
    )
    for key in ("coarse_pick", "profile_idx", "temporal_idx"):
        np.testing.assert_array_equal(batched[key], serial[key])
    for key in ("alpha", "captured_energy", "prediction"):
        np.testing.assert_allclose(batched[key], serial[key], rtol=1e-6, atol=1e-6)


def test_fixed_codebook_continuously_refines_an_off_grid_source():
    off = np.array(
        [[
            [-40, -20],
            [-20, 0],
            [0, -30],
            [20, 10],
            [40, 30],
            [-10, 40],
            [30, -40],
            [50, -10],
        ]],
        dtype=np.float32,
    )
    source = np.array([3.35, -7.2, 20.4], dtype=np.float32)
    d2 = np.square(off[0] - source[:2]).sum(axis=1) + source[2] ** 2
    footprint = 1 / np.sqrt(d2 + 1)
    footprint /= np.linalg.norm(footprint)
    time = np.arange(30)
    omega = -np.exp(-np.square(time - 15) / 8)[None].astype(np.float32)
    omega /= np.linalg.norm(omega, axis=1, keepdims=True)
    waveform = (4 * footprint[:, None] * omega[0]).astype(np.float32)[None]
    fit = localize_spikes_fixed_codebook(
        off,
        waveform,
        omega,
        kernels=("monopole",),
        n_scales=1,
        n_sites=7,
        refine_levels=6,
        continuous=True,
        device="cpu",
    )
    np.testing.assert_allclose(fit["sources"][0, :2], source[:2], atol=0.05)
    np.testing.assert_allclose(fit["sources"][0, 2], source[2], atol=0.15)
    assert np.all(fit["sources_grid"][0] == np.rint(fit["sources_grid"][0]))
    assert fit["continuous_displacement_um"][0] > 0
    assert fit["continuous_energy_gain"][0] >= 0
    np.testing.assert_allclose(fit["prediction"], waveform, atol=2e-3)


def test_subtraction_updates_only_the_selected_channels():
    residual = np.zeros((20, 4), dtype=np.float32)
    residual[7:11, 1] = 3
    residual[7:11, 3] = 5
    prediction = np.array([[[3, 3, 3, 3], [5, 5, 5, 5]]], dtype=np.float32)
    subtract_predictions(
        residual,
        np.array([9]),
        np.array([[1, 3]], dtype=np.int32),
        np.array([[True, True]]),
        prediction,
        n_before=2,
        n_after=2,
    )
    np.testing.assert_allclose(residual, 0)


def test_monotone_subtraction_refits_overlapping_gains():
    residual = np.zeros((12, 1), dtype=np.float32)
    residual[5:7, 0] = [2, -1]
    prediction = np.array([[[2, -1]], [[2, -1]]], dtype=np.float32)
    result = subtract_predictions_monotone(
        residual,
        np.array([6, 6]),
        np.array([[0], [0]], dtype=np.int32),
        np.ones((2, 1), dtype=bool),
        prediction,
        n_before=1,
        n_after=1,
    )
    assert result["accepted"].tolist() == [True, False]
    np.testing.assert_allclose(residual, 0)


def test_chunk_peeling_emits_localization_and_reconstruction_metrics():
    fs = 30000.0
    positions = np.array(
        [[-20, -20], [20, -20], [-20, 20], [20, 20]], dtype=np.float32
    )
    ids, local, centroids, counts, neighbors = channel_neighborhoods(positions, 60)
    length = 30
    time = np.arange(length)
    omega = -np.exp(-np.square(time - 15) / 8)[None].astype(np.float32)
    omega /= np.linalg.norm(omega, axis=1, keepdims=True)
    source = np.array([local[0, 0, 0], local[0, 0, 1], 0], dtype=np.float32)
    d2 = np.square(local[0, :4] - source[:2]).sum(axis=1) + source[2] ** 2
    footprint = 1 / np.sqrt(d2 + 1)
    footprint /= np.linalg.norm(footprint)
    data = np.zeros((120, 4), dtype=np.float32)
    data[45:75] += (20 * footprint[:, None] * omega[0]).T
    detection_footprints, _ = build_codebook_detection_footprints(
        local,
        ids >= 0,
        positions - centroids,
        kernels=("monopole",),
        n_scales=1,
        device="cpu",
    )
    config = ResidualConfig(
        threshold=3,
        radius_um=60,
        ms_before=0.5,
        ms_after=0.5,
        temporal_radius_ms=0.1,
        max_residual_passes=2,
        min_captured_fraction=0.9,
        n_scales=1,
        n_sites=7,
        refine_levels=6,
        fit_batch_size=16,
        device="cpu",
        save_waveforms=True,
    )
    result = peel_preprocessed_chunk(
        data,
        global_start=1000,
        core_start=20,
        core_stop=100,
        channel_positions=positions,
        neighborhood_ids=ids,
        channel_local_coords=local,
        channel_centroids=centroids,
        neighbor_counts=counts,
        spatial_neighbors=neighbors,
        detection_footprints=detection_footprints,
        omega=omega,
        fs=fs,
        config=config,
    )
    assert len(result["spike_times"]) == 1
    assert result["spike_times"][0] == 1060
    assert result["captured_fraction"][0] > 0.99
    assert result["residual_pass"][0] == 0
    assert result["residual_waveforms"].shape == (1, 4, 30)
    assert result["pass_energy_drop_fraction"][0] > 0.99

    rollback = peel_preprocessed_chunk(
        data,
        global_start=1000,
        core_start=20,
        core_stop=100,
        channel_positions=positions,
        neighborhood_ids=ids,
        channel_local_coords=local,
        channel_centroids=centroids,
        neighbor_counts=counts,
        spatial_neighbors=neighbors,
        detection_footprints=detection_footprints,
        omega=omega,
        fs=fs,
        config=replace(config, min_pass_energy_drop_fraction=1.01),
    )
    assert len(rollback["spike_times"]) == 0
