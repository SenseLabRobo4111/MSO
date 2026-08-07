"""Pure occupancy-grid contracts used by the public prediction node."""

from numbers import Integral

import cv2
import numpy as np


UNKNOWN = -1
MODEL_INPUT_SIZE = 256
DEFAULT_CROP_SIZE = MODEL_INPUT_SIZE


def validate_crop_size(value):
    """Return a positive integer crop size or reject the configuration."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("crop_size must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError("crop_size must be a positive integer")
    return value


def aligned_grid_offset(
        source_origin_x, source_origin_y, target_origin_x, target_origin_y,
        resolution, tolerance_cells=1e-6):
    """Return a source-to-target cell offset only for a shared grid lattice."""
    values = (
        source_origin_x, source_origin_y, target_origin_x, target_origin_y,
        resolution, tolerance_cells,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("grid geometry values must be finite")
    if resolution <= 0.0 or tolerance_cells < 0.0:
        raise ValueError("resolution must be positive and tolerance non-negative")

    offset_x = (source_origin_x - target_origin_x) / resolution
    offset_y = (source_origin_y - target_origin_y) / resolution
    rounded_x = int(round(offset_x))
    rounded_y = int(round(offset_y))
    if (abs(offset_x - rounded_x) > tolerance_cells
            or abs(offset_y - rounded_y) > tolerance_cells):
        raise ValueError("occupancy-grid origins do not share one cell lattice")
    return rounded_x, rounded_y


def extract_centered_occupancy_crop(occupancy, center_x, center_y, crop_size):
    """Extract a fixed square crop, padding cells outside the map as unknown."""
    occupancy = np.asarray(occupancy)
    if occupancy.ndim != 2:
        raise ValueError("occupancy must be a two-dimensional array")
    crop_size = validate_crop_size(crop_size)
    if not isinstance(center_x, Integral) or not isinstance(center_y, Integral):
        raise ValueError("crop center coordinates must be integers")

    start_x = int(center_x) - crop_size // 2
    start_y = int(center_y) - crop_size // 2
    end_x = start_x + crop_size
    end_y = start_y + crop_size

    valid_start_x = max(start_x, 0)
    valid_start_y = max(start_y, 0)
    valid_end_x = min(end_x, occupancy.shape[1])
    valid_end_y = min(end_y, occupancy.shape[0])

    crop = np.full((crop_size, crop_size), UNKNOWN, dtype=occupancy.dtype)
    if valid_start_x >= valid_end_x or valid_start_y >= valid_end_y:
        return crop

    local_start_x = valid_start_x - start_x
    local_start_y = valid_start_y - start_y
    local_end_x = local_start_x + valid_end_x - valid_start_x
    local_end_y = local_start_y + valid_end_y - valid_start_y
    crop[local_start_y:local_end_y, local_start_x:local_end_x] = occupancy[
        valid_start_y:valid_end_y, valid_start_x:valid_end_x
    ]
    return crop


def occupancy_to_channels(occupancy):
    """Encode occupied, unknown, and free masks as discrete model channels."""
    occupancy = np.asarray(occupancy)
    if occupancy.ndim != 2:
        raise ValueError("occupancy must be a two-dimensional array")
    channels = np.zeros((*occupancy.shape, 3), dtype=np.uint8)
    channels[:, :, 0] = (occupancy == 100).astype(np.uint8) * 255
    channels[:, :, 1] = (occupancy == UNKNOWN).astype(np.uint8) * 255
    channels[:, :, 2] = (occupancy == 0).astype(np.uint8) * 255
    return channels


def resize_occupancy_channels(channels, output_size=MODEL_INPUT_SIZE):
    """Resize discrete occupancy channels without inventing mixed labels."""
    channels = np.asarray(channels)
    if channels.ndim != 3:
        raise ValueError("channels must be a three-dimensional array")
    output_size = validate_crop_size(output_size)
    return cv2.resize(
        channels,
        (output_size, output_size),
        interpolation=cv2.INTER_NEAREST,
    )


def overlay_measured_occupancy(provisional, measured):
    """Copy every known measured cell over a provisional occupancy grid."""
    provisional = np.asarray(provisional)
    measured = np.asarray(measured)
    if provisional.shape != measured.shape:
        raise ValueError("provisional and measured grids must have equal shapes")
    if provisional.ndim != 2:
        raise ValueError("occupancy grids must be two-dimensional")

    output = provisional.copy()
    measured_mask = measured != UNKNOWN
    output[measured_mask] = measured[measured_mask]
    return output


def accumulate_provisional_occupancy(
        existing, provisional, measured=None, existing_weight=0.75):
    """Blend known predictions safely, then restore measured occupancy."""
    existing = np.asarray(existing)
    provisional = np.asarray(provisional)
    if existing.shape != provisional.shape or existing.ndim != 2:
        raise ValueError("existing and provisional grids must be equal 2-D shapes")
    if not 0.0 <= existing_weight <= 1.0:
        raise ValueError("existing_weight must be between zero and one")

    output = np.full(existing.shape, UNKNOWN, dtype=np.int8)
    existing_known = existing != UNKNOWN
    provisional_known = provisional != UNKNOWN
    only_existing = existing_known & ~provisional_known
    only_provisional = ~existing_known & provisional_known
    both_known = existing_known & provisional_known

    output[only_existing] = existing[only_existing]
    output[only_provisional] = provisional[only_provisional]
    if np.any(both_known):
        # Preserve the historical known-cell arithmetic: each weighted term is
        # truncated separately before addition. The contract change here is
        # limited to keeping UNKNOWN out of numeric blending.
        historical_existing = (
            existing_weight * existing[both_known].astype(np.float64)
        ).astype(np.int8)
        historical_provisional = (
            (1.0 - existing_weight)
            * provisional[both_known].astype(np.float64)
        ).astype(np.int8)
        output[both_known] = historical_existing + historical_provisional

    if measured is not None:
        output = overlay_measured_occupancy(output, measured)
    return output


def overlay_measurements_at_offset(target, measured, start_x, start_y):
    """Overlay the in-bounds part of a measured grid on a target grid."""
    target = np.asarray(target)
    measured = np.asarray(measured)
    if target.ndim != 2 or measured.ndim != 2:
        raise ValueError("target and measured grids must be two-dimensional")
    if not isinstance(start_x, Integral) or not isinstance(start_y, Integral):
        raise ValueError("grid offsets must be integers")

    start_x = int(start_x)
    start_y = int(start_y)
    end_x = start_x + measured.shape[1]
    end_y = start_y + measured.shape[0]
    valid_start_x = max(start_x, 0)
    valid_start_y = max(start_y, 0)
    valid_end_x = min(end_x, target.shape[1])
    valid_end_y = min(end_y, target.shape[0])
    output = target.copy()
    if valid_start_x >= valid_end_x or valid_start_y >= valid_end_y:
        return output

    local_start_x = valid_start_x - start_x
    local_start_y = valid_start_y - start_y
    local_end_x = local_start_x + valid_end_x - valid_start_x
    local_end_y = local_start_y + valid_end_y - valid_start_y
    output[valid_start_y:valid_end_y, valid_start_x:valid_end_x] = (
        overlay_measured_occupancy(
            output[valid_start_y:valid_end_y, valid_start_x:valid_end_x],
            measured[local_start_y:local_end_y, local_start_x:local_end_x],
        )
    )
    return output
