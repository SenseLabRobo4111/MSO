"""Predictor adapters with a deterministic fixture and a frozen checkpoint path."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .hashing import sha256_file
from .environment_lock import normalize_device


UNKNOWN_VALUE = 100
FREE_MAX_VALUE = 10
OCCUPIED_MIN_VALUE = 200


class Predictor(Protocol):
    name: str
    artifact_sha256: str

    def predict(self, measured_crop: np.ndarray) -> np.ndarray:
        """Return probability in ``[0, 1]`` at the input world-crop shape."""


def _validate_crop(measured_crop: np.ndarray) -> np.ndarray:
    crop = np.asarray(measured_crop, dtype=np.uint8)
    if crop.ndim != 2 or min(crop.shape) < 1:
        raise ValueError(
            f"predictor crop must be a non-empty 2-D array, got {crop.shape}"
        )
    return crop


def _resize_semantic(crop: np.ndarray, side: int) -> np.ndarray:
    """Resize categorical occupancy without inventing intermediate labels."""

    if crop.shape == (side, side):
        return crop
    return cv2.resize(crop, (side, side), interpolation=cv2.INTER_NEAREST)


def _restore_probability(probability: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Map model output explicitly back to the simulator world crop."""

    if probability.shape == shape:
        return probability.astype(np.float32, copy=False)
    height, width = shape
    return cv2.resize(
        probability.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)


def validate_probability_output(
    value: np.ndarray, expected_shape: tuple[int, int]
) -> np.ndarray:
    """Return a finite probability raster with the exact requested shape."""

    probability = np.asarray(value, dtype=np.float32)
    if probability.shape != expected_shape:
        raise ValueError(
            "predictor output shape differs from the world-crop shape: "
            f"expected {expected_shape}, got {probability.shape}"
        )
    if not np.isfinite(probability).all():
        raise ValueError("predictor output contains a non-finite value")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("predictor output must contain probabilities in [0, 1]")
    return probability


@dataclass
class ObservedControl:
    """Neutral unknown prior with measured cells kept authoritative."""

    name: str = "observed_control"
    artifact_sha256: str = "none"

    def predict(self, measured_crop: np.ndarray) -> np.ndarray:
        crop = _validate_crop(measured_crop)
        probability = np.full(crop.shape, 0.5, dtype=np.float32)
        probability[crop <= FREE_MAX_VALUE] = 0.0
        probability[crop >= OCCUPIED_MIN_VALUE] = 1.0
        return probability


@dataclass
class DeterministicFixturePredictor:
    """Non-learned fixture used only for train/validation smoke tests.

    The fixture extrapolates from measured free and occupied support using
    distance fields.  It never reads the simulator truth map and must not be
    used for test-split claims.
    """

    name: str = "deterministic_fixture"
    artifact_sha256: str = "fixture-v1"

    def predict(self, measured_crop: np.ndarray) -> np.ndarray:
        crop = _validate_crop(measured_crop)
        free = crop <= FREE_MAX_VALUE
        occupied = crop >= OCCUPIED_MIN_VALUE
        if not np.any(free) or not np.any(occupied):
            return ObservedControl().predict(crop)
        distance_to_occupied = cv2.distanceTransform(
            (~occupied).astype(np.uint8), cv2.DIST_L2, 3
        )
        distance_to_free = cv2.distanceTransform(
            (~free).astype(np.uint8), cv2.DIST_L2, 3
        )
        logits = np.clip(
            (distance_to_free - distance_to_occupied) / 6.0, -12.0, 12.0
        )
        probability = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        probability[free] = 0.0
        probability[occupied] = 1.0
        return probability


class FrozenTorchPredictor:
    """Adapter for a validation-selected replacement-study checkpoint."""

    name = "validation_selected_deconv_student"

    def __init__(
        self,
        checkpoint: Path,
        device: str = "cuda",
        model_input_px: int = 256,
    ) -> None:
        import torch

        from sensemap.explore_model.SenseMapNet import DistillMapNetDeconv

        self._torch = torch
        normalized_device = normalize_device(device)
        if normalized_device.startswith("cuda:") and not torch.cuda.is_available():
            raise RuntimeError("the locked CUDA execution device is unavailable")
        self.device = torch.device(normalized_device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        required = {
            "schema",
            "status",
            "v2_lock_sha256",
            "dataset_manifest_sha256",
            "source_split_sha256",
            "training_configuration_sha256",
            "stage",
            "seed",
            "selected_validation_metric",
            "selected_epoch",
            "model_state_dict",
        }
        missing = required.difference(payload)
        if (
            missing
            or payload.get("schema") != "mso.prospective_v2.checkpoint/1"
            or payload.get("stage") != "student"
            or payload.get("status")
            != "prospective_v2_not_historical_recovery"
        ):
            raise ValueError(
                "checkpoint is not a selected replacement student; "
                f"missing={sorted(missing)}"
            )
        for field in (
            "v2_lock_sha256",
            "dataset_manifest_sha256",
            "source_split_sha256",
            "training_configuration_sha256",
        ):
            digest = payload.get(field)
            if not isinstance(digest, str) or len(digest) != 64 or not all(
                character in "0123456789abcdef" for character in digest):
                raise ValueError(f"checkpoint {field} is malformed")
        metric = payload.get("selected_validation_metric")
        if not isinstance(metric, (int, float)) or isinstance(
                metric, bool) or not math.isfinite(metric):
            raise ValueError("checkpoint validation metric is non-finite")
        self.model_input_px = int(model_input_px)
        if self.model_input_px != 256:
            raise ValueError("runtime model input must match the locked 256 px model")
        model = DistillMapNetDeconv(image_size=self.model_input_px, dim=4)
        if sum(parameter.numel() for parameter in model.parameters()) != 342771:
            raise ValueError("runtime predictor parameter count is incompatible")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model = model.to(self.device).eval()
        self.artifact_sha256 = sha256_file(checkpoint)
        self.metadata = {
            "seed": int(payload["seed"]),
            "selected_validation_unknown_bce": float(metric),
            "dataset_manifest_sha256": str(
                payload["dataset_manifest_sha256"]
            ),
            "v2_lock_sha256": str(payload["v2_lock_sha256"]),
        }

    def predict(self, measured_crop: np.ndarray) -> np.ndarray:
        world_crop = _validate_crop(measured_crop)
        crop = _resize_semantic(world_crop, self.model_input_px)
        occupied = crop >= OCCUPIED_MIN_VALUE
        unknown = np.abs(crop.astype(np.int16) - UNKNOWN_VALUE) <= 3
        free = crop <= FREE_MAX_VALUE
        features = np.stack((occupied, unknown, free), axis=0).astype(np.float32)
        tensor = self._torch.from_numpy(features).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(tensor)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
            raise ValueError("unexpected checkpoint output structure")
        output = outputs[0]
        prediction = output.detach().float().cpu().numpy().squeeze()
        if prediction.shape != (self.model_input_px, self.model_input_px):
            raise ValueError(f"unexpected checkpoint output shape {prediction.shape}")
        if not np.isfinite(prediction).all() or np.any(
            (prediction < 0.0) | (prediction > 1.0)
        ):
            raise ValueError("checkpoint output is not a finite probability map")
        prediction = prediction.astype(np.float32, copy=False)
        prediction[free] = 0.0
        prediction[occupied] = 1.0
        restored = _restore_probability(prediction, world_crop.shape)
        world_free = world_crop <= FREE_MAX_VALUE
        world_occupied = world_crop >= OCCUPIED_MIN_VALUE
        restored[world_free] = 0.0
        restored[world_occupied] = 1.0
        return restored


def make_predictor(
    arm: str,
    kind: str,
    checkpoint: Path | None = None,
    device: str = "cuda",
    model_input_px: int = 256,
) -> Predictor:
    control_arms = {"observed_only", "oracle_completion"}
    model_arms = {
        "prediction_on",
        "prediction_registration_only",
        "prediction_frontier_only",
        "prediction_both",
        "adversarial_prediction",
    }
    if arm in control_arms:
        return ObservedControl()
    if arm not in model_arms:
        raise ValueError(f"unknown arm {arm!r}")
    if kind == "fixture":
        return DeterministicFixturePredictor()
    if kind == "checkpoint":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for checkpoint predictor")
        return FrozenTorchPredictor(
            checkpoint,
            device=device,
            model_input_px=model_input_px,
        )
    raise ValueError(f"unknown predictor kind {kind!r}")
