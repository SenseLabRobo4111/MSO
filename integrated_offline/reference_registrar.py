#!/usr/bin/env python3
"""Reference registrar reconstructed only from the manuscript description.

This is a new, independent implementation for controlled offline diagnostics.
It is not the historical MSO online implementation and must not be used to relabel the
paper's original pairwise or system results.  The implementation choices that
the manuscript leaves unspecified are documented in
``REFERENCE_REGISTRAR.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


SCALES = (1.0, 0.5, 0.25)
LOWE_RATIO = 0.9
AMBIGUITY_KEEP_FACTOR = 0.8
MATCH_WEIGHT_MIN = 0.5
RANSAC_STAGES = ((3.0, 0.30, "3px_30pct"),
                 (5.0, 0.20, "5px_20pct"),
                 (10.0, 0.03, "10px_3pct"))
MAX_ITERS = 5000
UNIT_SCALE_TOLERANCE = 0.02
MIN_GATE_INLIER_RATIO = 0.20
MIN_OBSERVED_SUPPORT_OVERLAP = 0.20


@dataclass
class FeaturePool:
    points: np.ndarray
    descriptors: np.ndarray
    ambiguity: np.ndarray
    angles: np.ndarray
    sizes: np.ndarray
    families: np.ndarray

    @classmethod
    def empty(cls) -> "FeaturePool":
        return cls(
            np.empty((0, 2), np.float32),
            np.empty((0, 32), np.uint8),
            np.empty((0,), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), np.int16),
        )


def _read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _observed_mask(path: str | Path | None, shape: tuple[int, int]) -> np.ndarray:
    if not path:
        return np.zeros(shape, dtype=bool)
    observed = _read_gray(path)
    if observed.shape != shape:
        observed = cv2.resize(observed, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    # The archived occupancy grids use 100 for unknown and {0,255} for observed
    # free/occupied cells.  A tolerance makes the adapter robust to file codecs.
    return np.abs(observed.astype(np.int16) - 100) > 3


def _ambiguity(image: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability = image.astype(np.float32) / 255.0
    edges = cv2.Canny(image, 50, 150).astype(np.float32) / 255.0
    score = 0.7 * probability * (1.0 - probability) + 0.3 * edges
    score[observed] = 0.0
    return score, edges


def _morphological_skeleton(binary: np.ndarray, max_iterations: int = 32) -> np.ndarray:
    work = (binary > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max_iterations):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = cv2.erode(work, element)
        if cv2.countNonZero(work) == 0:
            break
    return skeleton


def _points_from_response(response: np.ndarray, max_points: int) -> list[cv2.KeyPoint]:
    corners = cv2.goodFeaturesToTrack(
        response,
        maxCorners=max_points,
        qualityLevel=0.005,
        minDistance=4,
        blockSize=5,
        useHarrisDetector=False,
    )
    if corners is None:
        return []
    return [cv2.KeyPoint(float(p[0][0]), float(p[0][1]), 15.0) for p in corners]


def _sample(score: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0,), np.float32)
    x = np.clip(np.rint(points[:, 0]).astype(int), 0, score.shape[1] - 1)
    y = np.clip(np.rint(points[:, 1]).astype(int), 0, score.shape[0] - 1)
    return score[y, x].astype(np.float32)


def _extract_family(
    gray: np.ndarray,
    score: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    orb: cv2.ORB,
    scale: float,
    family_id: int,
) -> FeaturePool:
    keypoints, descriptors = orb.compute(gray, keypoints)
    if descriptors is None or not keypoints:
        return FeaturePool.empty()
    points_low = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
    ambiguity = _sample(score, points_low)
    mean_score = float(np.mean(score))
    if mean_score <= 1e-12:
        keep = ambiguity <= 1e-12
    else:
        keep = ambiguity < AMBIGUITY_KEEP_FACTOR * mean_score
    if not np.any(keep):
        return FeaturePool.empty()
    inv = 1.0 / scale
    return FeaturePool(
        points=points_low[keep] * inv,
        descriptors=descriptors[keep],
        ambiguity=ambiguity[keep],
        angles=np.asarray([kp.angle for kp in keypoints], np.float32)[keep],
        sizes=np.asarray([kp.size for kp in keypoints], np.float32)[keep] * inv,
        families=np.full(int(np.count_nonzero(keep)), family_id, np.int16),
    )


def _combine(pools: list[FeaturePool]) -> FeaturePool:
    nonempty = [pool for pool in pools if len(pool.points)]
    if not nonempty:
        return FeaturePool.empty()
    return FeaturePool(*(
        np.concatenate([getattr(pool, field) for pool in nonempty], axis=0)
        for field in ("points", "descriptors", "ambiguity", "angles", "sizes", "families")
    ))


def _extract(image: np.ndarray, observed: np.ndarray) -> FeaturePool:
    all_pools: list[FeaturePool] = []
    for level, scale in enumerate(SCALES):
        if scale == 1.0:
            gray = image
            obs = observed
        else:
            size = (max(16, round(image.shape[1] * scale)), max(16, round(image.shape[0] * scale)))
            gray = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            obs = cv2.resize(observed.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        score, edges_float = _ambiguity(gray, obs)
        edges = (edges_float * 255).astype(np.uint8)
        occupied = (gray >= 166).astype(np.uint8) * 255
        skeleton = _morphological_skeleton(occupied)
        nfeatures = max(250, round(1100 * scale))
        orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=12,
            fastThreshold=7,
        )
        orb_keypoints = orb.detect(gray, None)
        family_keypoints = (
            orb_keypoints,
            _points_from_response(edges, max(150, round(500 * scale))),
            _points_from_response(skeleton, max(120, round(400 * scale))),
        )
        for family_id, keypoints in enumerate(family_keypoints):
            pool = _extract_family(gray, score, keypoints, orb, scale, family_id)
            if len(pool.points):
                # Encode the pyramid level with the family.  Matching remains
                # family-specific but permits cross-level correspondences.
                pool.families[:] = family_id
                all_pools.append(pool)
    return _combine(all_pools)


def _candidate_matches(source: FeaturePool, target: FeaturePool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_indices: list[int] = []
    target_indices: list[int] = []
    distances: list[float] = []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    for family in (0, 1, 2):
        s_idx = np.flatnonzero(source.families == family)
        t_idx = np.flatnonzero(target.families == family)
        if not len(s_idx) or not len(t_idx):
            continue
        knn_survivors: list[cv2.DMatch] = []
        if len(t_idx) >= 2:
            for pair in matcher.knnMatch(source.descriptors[s_idx], target.descriptors[t_idx], k=2):
                if len(pair) == 2 and pair[0].distance < LOWE_RATIO * pair[1].distance:
                    knn_survivors.append(pair[0])
        if len(knn_survivors) < 8:
            direct = sorted(
                matcher.match(source.descriptors[s_idx], target.descriptors[t_idx]),
                key=lambda item: item.distance,
            )
            keep_n = max(1, int(math.ceil(0.8 * len(direct))))
            chosen = direct[:keep_n]
        else:
            chosen = knn_survivors
        for match in chosen:
            source_indices.append(int(s_idx[match.queryIdx]))
            target_indices.append(int(t_idx[match.trainIdx]))
            distances.append(float(match.distance))
    if not source_indices:
        return (np.empty((0,), int), np.empty((0,), int), np.empty((0,), float))
    s = np.asarray(source_indices, int)
    t = np.asarray(target_indices, int)
    d = np.asarray(distances, float)

    endpoint_sum = source.ambiguity[s] + target.ambiguity[t]
    variance = float(np.var(np.concatenate((source.ambiguity[s], target.ambiguity[t]))))
    weights = np.ones_like(endpoint_sum) if variance <= 1e-12 else np.exp(-endpoint_sum / (2.0 * variance))
    keep = weights > MATCH_WEIGHT_MIN
    return s[keep], t[keep], d[keep]


def _angular_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + 180.0) % 360.0 - 180.0


def _robust_keep(values: np.ndarray, floor: float, multiplier: float = 4.0) -> np.ndarray:
    if len(values) < 4:
        return np.ones(len(values), dtype=bool)
    median = float(np.median(values))
    residual = np.abs(values - median)
    mad = float(np.median(residual))
    limit = max(floor, multiplier * 1.4826 * mad)
    return residual <= limit


def _geometric_filter(
    source: FeaturePool, target: FeaturePool, s: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(s) < 4:
        return s, t
    angle_delta = _angular_delta(target.angles[t], source.angles[s])
    log_scale = np.log(np.maximum(target.sizes[t], 1e-3) / np.maximum(source.sizes[s], 1e-3))
    displacement = target.points[t] - source.points[s]
    centre = np.median(displacement, axis=0)
    radial = np.linalg.norm(displacement - centre, axis=1)
    keep = _robust_keep(angle_delta, 45.0)
    keep &= _robust_keep(log_scale, math.log(1.8))
    keep &= radial <= max(160.0, 4.0 * 1.4826 * float(np.median(np.abs(radial - np.median(radial)))))
    return s[keep], t[keep]


def _staged_ransac(
    source_points: np.ndarray, target_points: np.ndarray, seed: int
) -> tuple[np.ndarray | None, np.ndarray | None, str, float]:
    if len(source_points) < 3:
        return None, None, "none", 0.0
    last_matrix: np.ndarray | None = None
    last_mask: np.ndarray | None = None
    last_stage = "none"
    last_ratio = 0.0
    for threshold, required_ratio, stage_name in RANSAC_STAGES:
        cv2.setRNGSeed(seed)
        matrix, mask = cv2.estimateAffinePartial2D(
            source_points,
            target_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=threshold,
            maxIters=MAX_ITERS,
            confidence=0.99,
            refineIters=10,
        )
        if matrix is None or mask is None:
            continue
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        last_matrix, last_mask, last_stage, last_ratio = matrix, mask, stage_name, ratio
        if ratio >= required_ratio:
            break
    return last_matrix, last_mask, last_stage, last_ratio


def _nearest_rotation(linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, singular, vt = np.linalg.svd(linear)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation, singular


def _observed_occupied_overlap(
    h_source_to_target: np.ndarray,
    target_observed_path: str | Path | None,
    source_observed_path: str | Path | None,
    output_shape: tuple[int, int],
) -> float:
    if not target_observed_path or not source_observed_path:
        return 0.0
    target = _read_gray(target_observed_path)
    source = _read_gray(source_observed_path)
    target_occ = (target >= 200).astype(np.uint8)
    source_occ = (source >= 200).astype(np.uint8)
    warped = cv2.warpAffine(
        source_occ,
        h_source_to_target[:2],
        (output_shape[1], output_shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    intersection = int(np.count_nonzero(warped & target_occ))
    denominator = min(int(np.count_nonzero(warped)), int(np.count_nonzero(target_occ)))
    return intersection / denominator if denominator else 0.0


def register(*, target_path: str, source_path: str, context: dict) -> Mapping[str, Any]:
    """Estimate ``H_i_from_j_pixel`` and apply a GT-free pre-commit gate."""
    target = _read_gray(target_path)
    source = _read_gray(source_path)
    event = context.get("event", {})
    target_observed_path = event.get("target_observed_path")
    source_observed_path = event.get("source_observed_path")
    target_observed = _observed_mask(target_observed_path, target.shape)
    source_observed = _observed_mask(source_observed_path, source.shape)

    target_pool = _extract(target, target_observed)
    source_pool = _extract(source, source_observed)
    s_idx, t_idx, _distances = _candidate_matches(source_pool, target_pool)
    candidate_matches = len(s_idx)
    s_idx, t_idx = _geometric_filter(source_pool, target_pool, s_idx, t_idx)
    filtered_matches = len(s_idx)
    seed_bytes = hashlib.sha256(str(event.get("event_id", "reference")).encode("utf-8")).digest()[:4]
    seed = int.from_bytes(seed_bytes, "little") & 0x7FFFFFFF
    matrix, inlier_mask, stage, inlier_ratio = _staged_ransac(
        source_pool.points[s_idx], target_pool.points[t_idx], seed
    )
    if matrix is None or inlier_mask is None:
        return {
            "candidate_returned": False,
            "accepted": False,
            "committed": False,
            "rigid_valid": False,
            "reject_reason": "no_candidate",
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "ransac_stage": "none",
            "gate_score": 0.0,
            "gate_threshold": MIN_OBSERVED_SUPPORT_OVERLAP,
            "reference_diagnostics": {
                "target_features": len(target_pool.points),
                "source_features": len(source_pool.points),
                "weighted_matches": candidate_matches,
                "geometric_matches": filtered_matches,
            },
        }

    raw_h = np.eye(3, dtype=float)
    raw_h[:2] = matrix
    rotation, singular = _nearest_rotation(raw_h[:2, :2])
    determinant = float(np.linalg.det(raw_h[:2, :2]))
    scale = float(math.sqrt(abs(determinant)))
    shear = float(abs(singular[0] - singular[1]))
    unit_se2 = bool(
        determinant > 0.0
        and np.max(np.abs(singular - 1.0)) <= UNIT_SCALE_TOLERANCE
        and shear <= UNIT_SCALE_TOLERANCE
    )
    rigid_h = np.eye(3, dtype=float)
    rigid_h[:2, :2] = rotation
    rigid_h[:2, 2] = raw_h[:2, 2]
    inlier_count = int(np.count_nonzero(inlier_mask))
    overlap = _observed_occupied_overlap(
        rigid_h, target_observed_path, source_observed_path, target.shape
    )
    accepted = bool(
        unit_se2
        and inlier_ratio >= MIN_GATE_INLIER_RATIO
        and overlap >= MIN_OBSERVED_SUPPORT_OVERLAP
    )
    if not unit_se2:
        reason = "non_unit_se2_candidate"
    elif inlier_ratio < MIN_GATE_INLIER_RATIO:
        reason = "low_inlier_ratio"
    elif overlap < MIN_OBSERVED_SUPPORT_OVERLAP:
        reason = "low_observed_occupied_support_overlap"
    else:
        reason = "accepted"

    response: dict[str, Any] = {
        "candidate_returned": True,
        "accepted": accepted,
        # This adapter is offline.  It cannot establish an atomic persistent-map commit.
        "committed": False,
        "rigid_valid": unit_se2,
        "reject_reason": reason,
        "raw_H_i_from_j_pixel": raw_h.tolist(),
        "raw_scale": scale,
        "raw_shear": shear,
        "raw_determinant": determinant,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "ransac_stage": stage,
        "gate_score": overlap,
        "gate_threshold": MIN_OBSERVED_SUPPORT_OVERLAP,
        "reference_diagnostics": {
            "target_features": len(target_pool.points),
            "source_features": len(source_pool.points),
            "weighted_matches": candidate_matches,
            "geometric_matches": filtered_matches,
            "unit_scale_tolerance": UNIT_SCALE_TOLERANCE,
        },
    }
    if unit_se2:
        response["H_i_from_j_pixel"] = rigid_h.tolist()
    return response
