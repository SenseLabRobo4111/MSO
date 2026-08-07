"""Tests for the ROS-independent occupancy prediction contracts."""

import numpy as np
import pytest

from sensemap.occupancy_contracts import (
    DEFAULT_CROP_SIZE,
    MODEL_INPUT_SIZE,
    accumulate_provisional_occupancy,
    aligned_grid_offset,
    extract_centered_occupancy_crop,
    overlay_measured_occupancy,
    overlay_measurements_at_offset,
    resize_occupancy_channels,
    validate_crop_size,
)


def test_crop_size_default_matches_model_input_and_is_validated():
    """The default has one model pixel per crop cell and rejects bad values."""
    assert DEFAULT_CROP_SIZE == MODEL_INPUT_SIZE == 256
    assert validate_crop_size(np.int64(32)) == 32
    for invalid in (True, 0, -1, 2.5, "256"):
        with pytest.raises(ValueError):
            validate_crop_size(invalid)


def test_centered_crop_has_exact_requested_size_and_unknown_padding():
    """Odd crop sizes and map-edge padding retain their cell alignment."""
    occupancy = np.arange(12, dtype=np.int8).reshape(3, 4)
    crop = extract_centered_occupancy_crop(occupancy, 0, 0, 3)
    expected = np.array([
        [-1, -1, -1],
        [-1, 0, 1],
        [-1, 4, 5],
    ], dtype=np.int8)
    np.testing.assert_array_equal(crop, expected)


def test_channel_resize_uses_nearest_neighbour_labels():
    """A categorical resize repeats labels without mixed channel values."""
    channels = np.array([
        [[255, 0, 0], [0, 255, 0]],
        [[0, 0, 255], [0, 0, 0]],
    ], dtype=np.uint8)
    resized = resize_occupancy_channels(channels, 4)
    expected = np.repeat(np.repeat(channels, 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(resized, expected)
    assert set(np.unique(resized)).issubset({0, 255})


def test_measured_cells_replace_provisional_values_exactly():
    """Every known measurement wins while unknown cells leave predictions."""
    provisional = np.array([[100, 0, 100], [0, 100, 0]], dtype=np.int8)
    measured = np.array([[0, 100, -1], [-1, 42, -1]], dtype=np.int8)
    output = overlay_measured_occupancy(provisional, measured)
    expected = np.array([[0, 100, 100], [0, 42, 0]], dtype=np.int8)
    np.testing.assert_array_equal(output, expected)


def test_accumulation_never_uses_unknown_as_a_numeric_value():
    """Known-only and unknown-only cases bypass weighted arithmetic."""
    existing = np.array([[-1, 100, 0, -1]], dtype=np.int8)
    provisional = np.array([[100, -1, 100, -1]], dtype=np.int8)
    output = accumulate_provisional_occupancy(existing, provisional)
    np.testing.assert_array_equal(
        output, np.array([[100, 100, 25, -1]], dtype=np.int8))


def test_known_cell_accumulation_preserves_historical_termwise_truncation():
    """Unknown safety must not silently change retained known-cell arithmetic."""
    existing = np.array([[25]], dtype=np.int8)
    provisional = np.array([[100]], dtype=np.int8)
    output = accumulate_provisional_occupancy(existing, provisional)
    np.testing.assert_array_equal(output, np.array([[43]], dtype=np.int8))


def test_accumulation_reapplies_measurements_after_blending():
    """Learned history cannot overwrite current measured occupancy."""
    existing = np.array([[100, 0, 100]], dtype=np.int8)
    provisional = np.array([[100, 0, 0]], dtype=np.int8)
    measured = np.array([[0, 100, -1]], dtype=np.int8)
    output = accumulate_provisional_occupancy(
        existing, provisional, measured)
    np.testing.assert_array_equal(
        output, np.array([[0, 100, 75]], dtype=np.int8))


def test_aligned_overlay_clips_edges_and_preserves_unknown_targets():
    """A partial measured map overlay is bounds-safe and unknown-safe."""
    target = np.full((3, 3), 50, dtype=np.int8)
    measured = np.array([[0, 100], [-1, 0]], dtype=np.int8)
    output = overlay_measurements_at_offset(target, measured, -1, 1)
    expected = target.copy()
    expected[1, 0] = 100
    expected[2, 0] = 0
    np.testing.assert_array_equal(output, expected)


def test_grid_offset_rejects_fractional_origin_drift():
    """Global overlays must never round incompatible map origins silently."""
    assert aligned_grid_offset(1.0, -0.5, 0.0, 0.0, 0.05) == (20, -10)
    with pytest.raises(ValueError, match="do not share"):
        aligned_grid_offset(1.024, -0.5, 0.0, 0.0, 0.05)
