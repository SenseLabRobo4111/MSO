"""File-isolated adapter around the reconstructed reference registrar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np

from integrated_offline import reference_registrar

from .hashing import array_sha256
from .mapping import full_transform_from_crop_estimate


def validate_se2_transform(value: Any, *, atol: float = 1e-6) -> np.ndarray:
    """Return a finite, invertible, right-handed rigid SE(2) matrix."""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (2, 3):
        matrix = np.vstack((matrix, [0.0, 0.0, 1.0]))
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("registrar transform is not a finite 3x3 matrix")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=atol, rtol=0.0):
        raise ValueError("registrar transform has a non-SE(2) homogeneous row")
    rotation = matrix[:2, :2]
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0 or not np.isfinite(determinant) or not np.allclose(
            rotation.T @ rotation, np.eye(2), atol=atol, rtol=0.0) or not (
                np.isclose(determinant, 1.0, atol=atol, rtol=0.0)):
        raise ValueError("registrar transform is not an invertible rigid SE(2) pose")
    if abs(float(np.linalg.det(matrix))) <= atol:
        raise ValueError("registrar transform is singular")
    return matrix


@dataclass(frozen=True)
class RegistrationAttempt:
    candidate_returned: bool
    gate_accepted: bool
    reject_reason: str
    crop_transform_target_from_source: np.ndarray | None
    full_transform_target_from_source: np.ndarray | None
    diagnostics: dict[str, Any]
    input_hashes: dict[str, str]


class ReferenceRegistrarAdapter:
    name = "reconstructed_reference_registrar"

    def register(
        self,
        *,
        event_id: str,
        target_input: np.ndarray,
        source_input: np.ndarray,
        target_observed: np.ndarray,
        source_observed: np.ndarray,
        target_origin_xy: tuple[int, int],
        source_origin_xy: tuple[int, int],
    ) -> RegistrationAttempt:
        arrays = {
            "target_registrar_input": np.asarray(target_input, dtype=np.uint8),
            "source_registrar_input": np.asarray(source_input, dtype=np.uint8),
            "target_observed_gate_input": np.asarray(target_observed, dtype=np.uint8),
            "source_observed_gate_input": np.asarray(source_observed, dtype=np.uint8),
        }
        shapes = {value.shape for value in arrays.values()}
        if len(shapes) != 1:
            raise ValueError(
                f"registration inputs have unequal shapes: {sorted(shapes)}"
            )
        input_hashes = {name: array_sha256(value) for name, value in arrays.items()}
        with tempfile.TemporaryDirectory(prefix="mso_registration_") as temporary:
            root = Path(temporary)
            paths: dict[str, Path] = {}
            for name, value in arrays.items():
                path = root / f"{name}.png"
                if not cv2.imwrite(str(path), value):
                    raise OSError(f"could not write registration input {path}")
                paths[name] = path
            raw_result = reference_registrar.register(
                    target_path=str(paths["target_registrar_input"]),
                    source_path=str(paths["source_registrar_input"]),
                    context={
                        "event": {
                            "event_id": event_id,
                            "target_observed_path": str(
                                paths["target_observed_gate_input"]
                            ),
                            "source_observed_path": str(
                                paths["source_observed_gate_input"]
                            ),
                        }
                    },
                )
        if raw_result is None:
            return RegistrationAttempt(
                False, False, "registrar_returned_none", None, None,
                {"contract_valid": False, "contract_error": "none_result"},
                input_hashes)
        if not isinstance(raw_result, dict):
            return RegistrationAttempt(
                False, False, "registrar_returned_non_mapping", None, None,
                {"contract_valid": False, "contract_error": "non_mapping_result"},
                input_hashes)
        result = dict(raw_result)
        crop_transform = result.pop("H_i_from_j_pixel", None)
        crop_matrix: np.ndarray | None = None
        full_matrix: np.ndarray | None = None
        contract_error: str | None = None
        if crop_transform is not None:
            try:
                crop_matrix = validate_se2_transform(crop_transform)
                full_matrix = validate_se2_transform(
                    full_transform_from_crop_estimate(
                        crop_matrix, target_origin_xy, source_origin_xy))
            except (TypeError, ValueError, np.linalg.LinAlgError) as error:
                crop_matrix = None
                full_matrix = None
                contract_error = str(error)
        candidate_claimed = result.get("candidate_returned") is True
        accepted_claimed = result.get("accepted") is True
        rigid_valid = result.get("rigid_valid") is True
        if not candidate_claimed:
            crop_matrix = None
            full_matrix = None
        candidate_valid = bool(
            candidate_claimed and crop_matrix is not None
            and full_matrix is not None and contract_error is None)
        gate_accepted = bool(candidate_valid and accepted_claimed and rigid_valid)
        if not candidate_claimed:
            reject_reason = str(result.get("reject_reason", "no_candidate"))
        elif contract_error is not None or crop_matrix is None:
            reject_reason = "invalid_transform_contract"
        elif not rigid_valid:
            reject_reason = "rigid_valid_not_true"
        elif not accepted_claimed:
            reject_reason = str(result.get("reject_reason", "registrar_rejected"))
        else:
            reject_reason = "accepted"
        result["contract_valid"] = candidate_valid
        result["contract_error"] = contract_error
        result["rigid_valid"] = rigid_valid
        return RegistrationAttempt(
            candidate_returned=candidate_valid,
            gate_accepted=gate_accepted,
            reject_reason=reject_reason,
            crop_transform_target_from_source=crop_matrix,
            full_transform_target_from_source=full_matrix,
            diagnostics=result,
            input_hashes=input_hashes,
        )


class IdentityFixtureRegistrar:
    """Deterministic identity fixture for contract tests only."""

    name = "identity_fixture_registrar"

    def register(
        self,
        *,
        event_id: str,
        target_input: np.ndarray,
        source_input: np.ndarray,
        target_observed: np.ndarray,
        source_observed: np.ndarray,
        target_origin_xy: tuple[int, int],
        source_origin_xy: tuple[int, int],
    ) -> RegistrationAttempt:
        del event_id
        inputs = {
            "target_registrar_input": array_sha256(target_input),
            "source_registrar_input": array_sha256(source_input),
            "target_observed_gate_input": array_sha256(target_observed),
            "source_observed_gate_input": array_sha256(source_observed),
        }
        crop = np.eye(3, dtype=float)
        full = full_transform_from_crop_estimate(
            crop, target_origin_xy, source_origin_xy
        )
        return RegistrationAttempt(
            True,
            True,
            "accepted",
            crop,
            full,
            {
                "candidate_returned": True,
                "accepted": True,
                "reject_reason": "accepted",
                "inlier_count": 100,
                "inlier_ratio": 1.0,
                "gate_score": 1.0,
                "gate_threshold": 0.2,
                "rigid_valid": True,
            },
            inputs,
        )
