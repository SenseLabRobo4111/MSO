from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("evaluate_predictions.py")
SPEC = importlib.util.spec_from_file_location("evaluate_predictions", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_confusion_metrics_are_exact() -> None:
    target = np.asarray([[True, True], [False, False]])
    prediction = np.asarray([[True, False], [True, False]])
    mask = np.ones_like(target, dtype=bool)
    result = MODULE.confusion_metrics(target, prediction, mask)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["iou"] == 1.0 / 3.0


def test_calibration_error_is_zero_for_perfect_probabilities() -> None:
    target = np.asarray([[True, False], [True, False]])
    probability = target.astype(np.float32)
    mask = np.ones_like(target, dtype=bool)
    assert MODULE.expected_calibration_error(target, probability, mask) == 0.0


def test_boundary_metrics_are_identity_safe() -> None:
    target = np.zeros((32, 32), dtype=bool)
    target[8:24, 10:22] = True
    mask = np.ones_like(target, dtype=bool)
    result = MODULE.boundary_metrics(target, target.copy(), mask)
    assert result["boundary_f1"] == 1.0
    assert result["boundary_chamfer_px"] == 0.0


def test_topology_detects_a_split_free_component() -> None:
    occupied = np.ones((24, 24), dtype=bool)
    occupied[2:22, 2:22] = False
    one_component, _ = MODULE.free_topology(occupied)
    split = occupied.copy()
    split[:, 12] = True
    two_components, _ = MODULE.free_topology(split)
    assert one_component == 1
    assert two_components == 2
