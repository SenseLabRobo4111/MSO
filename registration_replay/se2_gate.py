"""Conservative validation and SE(2) projection for map-registration candidates.

The matcher is deliberately kept outside this module.  A candidate must pass
all structural and evidence checks before the returned rigid transform may be
committed.  The reference pose used for evaluation must never be passed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CandidateDiagnostics:
    """Matcher and map-agreement evidence associated with one affine candidate."""

    inlier_count: int = 0
    inlier_ratio: float = 0.0
    median_residual_px: float = float("inf")
    occupancy_iou: float = 0.0
    occupancy_chamfer_median_px: float = float("inf")
    source_minor_spread_px: float = 0.0
    target_minor_spread_px: float = 0.0


@dataclass(frozen=True)
class GateThresholds:
    """Safety-oriented defaults expressed only in observable matcher quantities."""

    scale_tolerance: float = 0.02
    anisotropy_tolerance: float = 0.01
    orthogonality_tolerance: float = 0.03
    min_inliers: int = 13
    min_inlier_ratio: float = 0.025
    max_median_residual_px: float = 3.5
    min_occupancy_iou: float = 0.25
    max_occupancy_chamfer_median_px: float = 2.0
    min_minor_spread_px: float = 5.0


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reasons: Tuple[str, ...]
    se2_image: Optional[np.ndarray]
    determinant: float
    singular_value_min: float
    singular_value_max: float
    isotropic_scale: float
    anisotropy: float
    orthogonality_error: float


def _reject(reason: str) -> GateDecision:
    return GateDecision(
        accepted=False,
        reasons=(reason,),
        se2_image=None,
        determinant=float("nan"),
        singular_value_min=float("nan"),
        singular_value_max=float("nan"),
        isotropic_scale=float("nan"),
        anisotropy=float("nan"),
        orthogonality_error=float("nan"),
    )


def gate_se2_candidate(
    affine: Optional[np.ndarray],
    diagnostics: CandidateDiagnostics,
    thresholds: GateThresholds = GateThresholds(),
) -> GateDecision:
    """Validate an affine candidate, then project it to an SE(2) transform.

    Translation is retained from the candidate.  Rotation is the closest proper
    rotation under the Frobenius norm.  No projection is returned after a failed
    gate, preventing callers from accidentally committing a rejected estimate.
    """

    if affine is None:
        return _reject("no_candidate")

    h = np.asarray(affine, dtype=float)
    if h.shape == (2, 3):
        h = np.vstack([h, [0.0, 0.0, 1.0]])
    if h.shape != (3, 3) or not np.all(np.isfinite(h)):
        return _reject("nonfinite_or_wrong_shape")

    a = h[:2, :2]
    determinant = float(np.linalg.det(a))
    u, singular_values, vt = np.linalg.svd(a)
    smax = float(singular_values[0])
    smin = float(singular_values[-1])
    scale = float(np.sqrt(max(determinant, 0.0))) if determinant > 0.0 else float("nan")
    anisotropy = float((smax - smin) / max(smax + smin, 1e-12))
    normalised = a / scale if np.isfinite(scale) and scale > 0.0 else a
    orthogonality_error = float(np.linalg.norm(normalised.T @ normalised - np.eye(2), ord="fro"))

    reasons = []
    if determinant <= 0.0:
        reasons.append("reflection_or_singular")
    if not np.isfinite(scale) or abs(scale - 1.0) > thresholds.scale_tolerance:
        reasons.append("scale")
    if anisotropy > thresholds.anisotropy_tolerance:
        reasons.append("anisotropy_or_shear")
    if orthogonality_error > thresholds.orthogonality_tolerance:
        reasons.append("nonorthogonal")
    if diagnostics.inlier_count < thresholds.min_inliers:
        reasons.append("too_few_inliers")
    if diagnostics.inlier_ratio < thresholds.min_inlier_ratio:
        reasons.append("low_inlier_ratio")
    if diagnostics.median_residual_px > thresholds.max_median_residual_px:
        reasons.append("large_feature_residual")
    if diagnostics.occupancy_iou < thresholds.min_occupancy_iou:
        reasons.append("low_occupancy_iou")
    if diagnostics.occupancy_chamfer_median_px > thresholds.max_occupancy_chamfer_median_px:
        reasons.append("large_occupancy_chamfer")
    if (
        diagnostics.source_minor_spread_px < thresholds.min_minor_spread_px
        or diagnostics.target_minor_spread_px < thresholds.min_minor_spread_px
    ):
        reasons.append("low_spatial_spread")

    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt

    se2 = np.eye(3, dtype=float)
    se2[:2, :2] = rotation
    se2[:2, 2] = h[:2, 2]
    return GateDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        se2_image=se2 if not reasons else None,
        determinant=determinant,
        singular_value_min=smin,
        singular_value_max=smax,
        isotropic_scale=scale,
        anisotropy=anisotropy,
        orthogonality_error=orthogonality_error,
    )
