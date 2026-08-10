#!/usr/bin/env python3
"""Evaluate the single SHA-bound positive control on v2 validation only."""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from mso_recovery.objective import BinaryAccumulator  # noqa: E402
from mso_recovery.v2_data import V2ManifestDataset  # noqa: E402
from mso_recovery.v2_lock import (  # noqa: E402
    atomic_write_json,
    load_lock,
    sha256_file,
    verify_hashed_artifact,
)
from sensemap.explore_model.SenseMapNet import DistillMapNetDeconv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def unwrap_state(payload: object) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("Positive-control payload must be a mapping")
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping):
        raise TypeError("Positive-control state must be a mapping")
    if any(str(key).startswith("gen.") for key in state):
        tensors = OrderedDict(
            (str(key)[4:], value.detach().cpu())
            for key, value in state.items()
            if str(key).startswith("gen.") and isinstance(value, torch.Tensor)
        )
    else:
        tensors = OrderedDict(
            (str(key), value.detach().cpu())
            for key, value in state.items()
            if isinstance(value, torch.Tensor)
        )
    if not tensors:
        raise ValueError("Positive control contains no generator tensors")
    return tensors


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("Positive-control output already exists")
    lock, lock_sha = load_lock(args.lock_root)
    if not torch.cuda.is_available():
        raise RuntimeError("The locked v2 environment requires CUDA")
    weights = verify_hashed_artifact(args.lock_root.resolve(), lock["positive_control"])
    manifest = args.dataset_root / "manifest.csv"
    dataset = V2ManifestDataset(args.dataset_root, manifest, "val")
    loader = DataLoader(
        dataset,
        batch_size=int(lock["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(lock["training"]["num_workers"]),
        pin_memory=True,
    )
    state = unwrap_state(torch.load(weights, map_location="cpu", weights_only=True))
    model = DistillMapNetDeconv(image_size=256, dim=4).cuda()
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != int(
        lock["positive_control"]["parameter_count"]
    ):
        raise ValueError("Positive-control parameter count changed")
    model.eval()
    accumulator = BinaryAccumulator()
    seen = 0
    with torch.inference_mode():
        for features, labels, unknown, _, _ in loader:
            labels = labels.cuda(non_blocking=True)
            unknown = unknown.cuda(non_blocking=True)
            prediction = model(features.cuda(non_blocking=True))[0]
            accumulator.update(prediction, labels, unknown)
            seen += int(labels.shape[0])
    if seen != len(dataset):
        raise RuntimeError("Positive-control validation was incomplete")
    metrics = accumulator.metrics()
    result = {
        "schema": "mso.prospective_v2.positive_control/1",
        "status": "validation_only_positive_control_not_manuscript_checkpoint",
        "v2_lock_sha256": lock_sha,
        "candidate_logical_name": lock["positive_control"]["logical_name"],
        "candidate_sha256": sha256_file(weights),
        "dataset_manifest_sha256": sha256_file(manifest),
        "validation_samples": seen,
        "decision_threshold": 0.5,
        "validation_unknown_bce": float(metrics["unknown_bce"]),
        "validation_unknown_f1": float(metrics["unknown_f1"]),
        "validation_unknown_iou": float(metrics["unknown_iou"]),
        "validation_unknown_precision": float(metrics["unknown_precision"]),
        "validation_unknown_recall": float(metrics["unknown_recall"]),
        "test_images_decoded": 0,
        "test_outcomes_accessed": False,
    }
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
