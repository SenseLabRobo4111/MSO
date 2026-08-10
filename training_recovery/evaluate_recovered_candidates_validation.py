#!/usr/bin/env python3
"""Validation-only comparison of explicitly labelled recovered candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from mso_recovery.common import sha256_file, write_json  # noqa: E402
from mso_recovery.data import FloorplanManifestDataset  # noqa: E402
from mso_recovery.objective import BinaryAccumulator  # noqa: E402
from sensemap.explore_model.SenseMapNet import DistillMapNetDeconv  # noqa: E402


STATUS = "recovered_candidate_validation_not_manuscript_checkpoint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def unwrap_generator_state(payload: object) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("Candidate payload must be a mapping")
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping):
        raise TypeError("Candidate state_dict must be a mapping")
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
        if len(tensors) != len(state):
            raise ValueError("Plain candidate state contains non-tensor values")
    if not tensors:
        raise ValueError("Candidate contains no generator tensors")
    return tensors


def full_image_scores(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[list[float], list[float]]:
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    for index in range(prediction.shape[0]):
        predicted_image = prediction[index, 0]
        target_image = target[index, 0]
        mse = float(np.mean((predicted_image - target_image) ** 2))
        psnr_values.append(float(10.0 * np.log10(1.0 / max(mse, 1e-20))))
        ssim_values.append(
            float(
                structural_similarity(
                    target_image,
                    predicted_image,
                    data_range=1.0,
                )
            )
        )
    return psnr_values, ssim_values


def evaluate_candidate(
    path: Path,
    loader: DataLoader,
    dataset: FloorplanManifestDataset,
    device: str,
    manifest_sha256: str,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = unwrap_generator_state(payload)
    model = DistillMapNetDeconv(image_size=256, dim=4)
    model.load_state_dict(state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 342_771:
        raise ValueError(f"Unexpected candidate parameter count: {parameter_count}")
    model.eval().to(device)

    accumulator = BinaryAccumulator()
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    unknown_mse_sum = 0.0
    unknown_cell_count = 0
    seen = 0
    started = time.time()
    with torch.inference_mode():
        for features, labels, unknown, _, _ in loader:
            labels = labels.to(device, non_blocking=device.startswith("cuda"))
            unknown = unknown.to(device, non_blocking=device.startswith("cuda"))
            prediction = model(
                features.to(device, non_blocking=device.startswith("cuda"))
            )[0]
            accumulator.update(prediction, labels, unknown)
            squared_error = (prediction.float() - labels.float()).square()
            unknown_mse_sum += float((squared_error * unknown).sum().cpu())
            unknown_cell_count += int(unknown.sum().cpu())
            batch_psnr, batch_ssim = full_image_scores(
                prediction.detach().float().cpu().numpy(),
                labels.detach().float().cpu().numpy(),
            )
            psnr_values.extend(batch_psnr)
            ssim_values.extend(batch_ssim)
            seen += int(labels.shape[0])

    if seen != len(dataset):
        raise RuntimeError(f"Incomplete validation evaluation: {seen} of {len(dataset)}")
    metrics = accumulator.metrics()
    unknown_mse = unknown_mse_sum / max(1, unknown_cell_count)
    return {
        "status": STATUS,
        "candidate_filename": path.name,
        "candidate_sha256": sha256_file(path),
        "manifest_sha256": manifest_sha256,
        "split": "val",
        "test_split_accessed": False,
        "validation_samples": seen,
        "dataset_integrity": dataset.integrity,
        "parameter_count": parameter_count,
        "state_tensor_count": len(state),
        "decision_threshold": 0.5,
        "selection_metric": "validation_unknown_bce",
        "validation_unknown_bce": float(metrics["unknown_bce"]),
        "validation_unknown_f1": float(metrics["unknown_f1"]),
        "validation_unknown_iou": float(metrics["unknown_iou"]),
        "validation_unknown_precision": float(metrics["unknown_precision"]),
        "validation_unknown_recall": float(metrics["unknown_recall"]),
        "validation_unknown_mse": unknown_mse,
        "validation_unknown_psnr": float(
            10.0 * np.log10(1.0 / max(unknown_mse, 1e-20))
        ),
        "validation_full_image_psnr_mean": float(np.mean(psnr_values)),
        "validation_full_image_psnr_std": float(np.std(psnr_values)),
        "validation_full_image_ssim_mean": float(np.mean(ssim_values)),
        "validation_full_image_ssim_std": float(np.std(ssim_values)),
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset = FloorplanManifestDataset(
        args.dataset_root,
        args.manifest,
        "val",
        verify_hashes=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    manifest_sha = sha256_file(args.manifest)
    results = [
        evaluate_candidate(path, loader, dataset, args.device, manifest_sha)
        for path in args.weights
    ]
    ranked = sorted(results, key=lambda row: float(row["validation_unknown_bce"]))
    comparison = {
        "status": STATUS,
        "protocol": (
            "complete building-disjoint validation split; primary selection metric "
            "minimum unknown-region BCE; fixed 0.5 threshold for F1/IoU; full-image "
            "PSNR and skimage SSIM with data_range=1"
        ),
        "manifest_sha256": manifest_sha,
        "split": "val",
        "test_split_accessed": False,
        "candidate_count": len(results),
        "ranking_by_predeclared_validation_unknown_bce": [
            row["candidate_filename"] for row in ranked
        ],
        "suggested_candidate_by_predeclared_metric": ranked[0]["candidate_filename"],
        "suggestion_is_not_test_authorization": True,
        "results": results,
    }
    for row in results:
        write_json(
            args.output_dir / f"{Path(str(row['candidate_filename'])).stem}.json",
            row,
        )
    write_json(args.output_dir / "validation_candidate_comparison.json", comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
