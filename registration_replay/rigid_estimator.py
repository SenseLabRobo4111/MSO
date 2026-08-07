"""Deterministic, unit-scale SE(2) estimation from 2-D feature correspondences."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class RigidEstimate:
    se2_image: Optional[np.ndarray]
    input_correspondences: int
    inlier_count: int
    inlier_ratio: float
    median_inlier_residual_px: float
    p90_inlier_residual_px: float
    source_minor_spread_px: float
    target_minor_spread_px: float
    iterations: int
    reject_reason: str = ""


def _proper_kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_mean - rotation @ source_mean
    result = np.eye(3, dtype=float)
    result[:2, :2] = rotation
    result[:2, 2] = translation
    return result


def _minor_spread(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    covariance = np.cov(points.T, bias=True)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(np.sqrt(max(0.0, eigenvalues[0])))


def _empty(reason: str, count: int = 0, iterations: int = 0) -> RigidEstimate:
    return RigidEstimate(
        se2_image=None,
        input_correspondences=count,
        inlier_count=0,
        inlier_ratio=0.0,
        median_inlier_residual_px=float("inf"),
        p90_inlier_residual_px=float("inf"),
        source_minor_spread_px=0.0,
        target_minor_spread_px=0.0,
        iterations=iterations,
        reject_reason=reason,
    )


def correspondences_from_matches(
    keypoints_source: Sequence[cv2.KeyPoint],
    keypoints_target: Sequence[cv2.KeyPoint],
    matches: Iterable[cv2.DMatch],
    max_correspondences: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a distance-ordered, one-to-one correspondence set."""

    ordered = sorted(matches, key=lambda match: float(match.distance))
    source_seen = set()
    target_seen = set()
    source_points = []
    target_points = []
    for match in ordered:
        if match.queryIdx in source_seen or match.trainIdx in target_seen:
            continue
        source_seen.add(match.queryIdx)
        target_seen.add(match.trainIdx)
        source_points.append(keypoints_source[match.queryIdx].pt)
        target_points.append(keypoints_target[match.trainIdx].pt)
        if len(source_points) >= max_correspondences:
            break
    return np.asarray(source_points, dtype=float), np.asarray(target_points, dtype=float)


def estimate_se2_ransac(
    source: np.ndarray,
    target: np.ndarray,
    *,
    residual_threshold_px: float = 3.0,
    iterations: int = 384,
    minimum_sample_baseline_px: float = 8.0,
    random_seed: Optional[int] = None,
) -> RigidEstimate:
    """Estimate a proper unit-scale rigid transform with two-point RANSAC.

    Scale and reflection are absent from the model, not merely checked after an
    affine fit.  A Kabsch refinement is performed on the winning consensus set.
    """

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        return _empty("wrong_correspondence_shape")
    count = int(len(source))
    if count < 2:
        return _empty("too_few_correspondences", count=count)
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        return _empty("nonfinite_correspondence", count=count)

    if random_seed is None:
        digest = hashlib.blake2b(
            np.ascontiguousarray(np.column_stack([source, target]), dtype=np.float32).tobytes(),
            digest_size=8,
        ).digest()
        random_seed = int.from_bytes(digest, "little", signed=False)
    rng = np.random.default_rng(random_seed)

    best_mask = None
    best_count = -1
    best_median = float("inf")
    valid_iterations = 0
    for _ in range(iterations):
        first, second = rng.choice(count, size=2, replace=False)
        delta_source = source[second] - source[first]
        delta_target = target[second] - target[first]
        if (
            np.linalg.norm(delta_source) < minimum_sample_baseline_px
            or np.linalg.norm(delta_target) < minimum_sample_baseline_px
        ):
            continue
        valid_iterations += 1
        angle = math_atan2(delta_target) - math_atan2(delta_source)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
        translation = 0.5 * (
            target[first] - rotation @ source[first]
            + target[second] - rotation @ source[second]
        )
        residuals = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        mask = residuals <= residual_threshold_px
        consensus = int(np.count_nonzero(mask))
        median = float(np.median(residuals[mask])) if consensus else float("inf")
        if consensus > best_count or (consensus == best_count and median < best_median):
            best_mask, best_count, best_median = mask, consensus, median

    if best_mask is None or best_count < 2:
        return _empty("no_nondegenerate_sample", count=count, iterations=valid_iterations)

    transform = _proper_kabsch(source[best_mask], target[best_mask])
    for _ in range(3):
        residuals = np.linalg.norm(
            source @ transform[:2, :2].T + transform[:2, 2] - target,
            axis=1,
        )
        refined_mask = residuals <= residual_threshold_px
        if np.count_nonzero(refined_mask) < 2 or np.array_equal(refined_mask, best_mask):
            best_mask = refined_mask
            break
        best_mask = refined_mask
        transform = _proper_kabsch(source[best_mask], target[best_mask])

    residuals = np.linalg.norm(
        source @ transform[:2, :2].T + transform[:2, 2] - target,
        axis=1,
    )
    inlier_residuals = residuals[best_mask]
    inlier_count = int(inlier_residuals.size)
    return RigidEstimate(
        se2_image=transform,
        input_correspondences=count,
        inlier_count=inlier_count,
        inlier_ratio=float(inlier_count / count),
        median_inlier_residual_px=(
            float(np.median(inlier_residuals)) if inlier_count else float("inf")
        ),
        p90_inlier_residual_px=(
            float(np.quantile(inlier_residuals, 0.9)) if inlier_count else float("inf")
        ),
        source_minor_spread_px=_minor_spread(source[best_mask]),
        target_minor_spread_px=_minor_spread(target[best_mask]),
        iterations=valid_iterations,
    )


def math_atan2(vector: np.ndarray) -> float:
    return float(np.arctan2(vector[1], vector[0]))


def estimate_from_keypoint_matches(
    keypoints_source: Sequence[cv2.KeyPoint],
    keypoints_target: Sequence[cv2.KeyPoint],
    matches: Iterable[cv2.DMatch],
    **kwargs,
) -> RigidEstimate:
    source, target = correspondences_from_matches(
        keypoints_source, keypoints_target, matches
    )
    return estimate_se2_ransac(source, target, **kwargs)
