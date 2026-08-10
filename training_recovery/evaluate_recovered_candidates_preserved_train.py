#!/usr/bin/env python3
"""Positive-control evaluation on a frozen subset of preserved train only."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from mso_recovery.common import sha256_file, write_json  # noqa: E402
from mso_recovery.objective import BinaryAccumulator  # noqa: E402
from sensemap.explore_model.SenseMapNet import DistillMapNetDeconv  # noqa: E402


STATUS = "preserved_train_positive_control_not_manuscript_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=20260807)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def safe_child(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if root not in candidate.parents:
        raise ValueError(f"Path escapes dataset root: {relative}")
    return candidate


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


class PreservedTrainSubset(Dataset):
    def __init__(
        self,
        root: Path,
        inventory: Path,
        sample_count: int,
        sample_seed: int,
    ) -> None:
        self.root = root.resolve()
        with inventory.open("r", encoding="utf-8", newline="") as stream:
            all_rows = list(csv.DictReader(stream))
        train_rows = [row for row in all_rows if row["archive_partition"] == "train"]
        if len(train_rows) != 6841:
            raise ValueError(f"Unexpected preserved train count: {len(train_rows)}")
        rng = np.random.default_rng(sample_seed)
        indices = np.sort(
            rng.choice(len(train_rows), size=min(sample_count, len(train_rows)), replace=False)
        )
        self.rows = [train_rows[int(index)] for index in indices]
        digest = hashlib.sha256()
        for row in self.rows:
            digest.update(row["sample_id"].encode("utf-8"))
            digest.update(b"\n")
            observation = safe_child(self.root, row["obs_relpath"])
            target = safe_child(self.root, row["target_relpath"])
            if sha256_file(observation) != row["obs_sha256"]:
                raise ValueError(f"Observation hash mismatch: {row['sample_id']}")
            if sha256_file(target) != row["target_sha256"]:
                raise ValueError(f"Target hash mismatch: {row['sample_id']}")
        self.sample_ids_sha256 = digest.hexdigest()
        self.integrity = "selected_raw_files_verified_sha256"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        resampling = Image.Resampling.NEAREST
        observation = np.asarray(
            Image.open(safe_child(self.root, row["obs_relpath"]))
            .convert("RGB")
            .resize((256, 256), resampling),
            dtype=np.uint8,
        ).copy()
        target = np.asarray(
            Image.open(safe_child(self.root, row["target_relpath"]))
            .convert("L")
            .resize((256, 256), resampling),
            dtype=np.uint8,
        ).copy()
        if not np.all(((observation == 255).sum(axis=2)) == 1):
            raise ValueError(f"Observation is not one-hot: {row['sample_id']}")
        features = torch.from_numpy((observation == 255).astype(np.float32)).permute(2, 0, 1)
        labels = torch.from_numpy((target == 255).astype(np.float32)).unsqueeze(0)
        unknown = torch.from_numpy((observation[:, :, 1] == 255).astype(np.float32)).unsqueeze(0)
        return features, labels, unknown, row["sample_id"]


def evaluate(path: Path, loader: DataLoader, dataset: PreservedTrainSubset, device: str) -> dict[str, object]:
    state = unwrap_generator_state(torch.load(path, map_location="cpu", weights_only=True))
    model = DistillMapNetDeconv(image_size=256, dim=4)
    model.load_state_dict(state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 342_771:
        raise ValueError(f"Unexpected parameter count: {parameter_count}")
    model.eval().to(device)
    accumulator = BinaryAccumulator()
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    seen = 0
    started = time.time()
    with torch.inference_mode():
        for features, labels, unknown, _ in loader:
            labels = labels.to(device)
            unknown = unknown.to(device)
            prediction = model(features.to(device))[0]
            accumulator.update(prediction, labels, unknown)
            predicted = prediction.detach().float().cpu().numpy()
            truth = labels.detach().float().cpu().numpy()
            for index in range(predicted.shape[0]):
                mse = float(np.mean((predicted[index, 0] - truth[index, 0]) ** 2))
                psnr_values.append(float(10.0 * np.log10(1.0 / max(mse, 1e-20))))
                ssim_values.append(
                    float(
                        structural_similarity(
                            truth[index, 0], predicted[index, 0], data_range=1.0
                        )
                    )
                )
            seen += int(labels.shape[0])
    metrics = accumulator.metrics()
    return {
        "status": STATUS,
        "candidate_filename": path.name,
        "candidate_sha256": sha256_file(path),
        "parameter_count": parameter_count,
        "archive_partition_accessed": "train_only",
        "test_partition_accessed": False,
        "scientific_role": "distribution_match_positive_control_only",
        "sample_count": seen,
        "decision_threshold": 0.5,
        "unknown_bce": float(metrics["unknown_bce"]),
        "unknown_precision": float(metrics["unknown_precision"]),
        "unknown_recall": float(metrics["unknown_recall"]),
        "unknown_f1": float(metrics["unknown_f1"]),
        "unknown_iou": float(metrics["unknown_iou"]),
        "full_image_psnr_mean": float(np.mean(psnr_values)),
        "full_image_psnr_std": float(np.std(psnr_values)),
        "full_image_ssim_mean": float(np.mean(ssim_values)),
        "full_image_ssim_std": float(np.std(ssim_values)),
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    dataset = PreservedTrainSubset(
        args.dataset_root, args.inventory, args.sample_count, args.sample_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    results = [evaluate(path, loader, dataset, args.device) for path in args.weights]
    comparison = {
        "status": STATUS,
        "inventory_sha256": sha256_file(args.inventory),
        "archive_partition_accessed": "train_only",
        "test_partition_accessed": False,
        "sample_seed": args.sample_seed,
        "sample_count": len(dataset),
        "sample_ids_sha256": dataset.sample_ids_sha256,
        "dataset_integrity": dataset.integrity,
        "results": results,
    }
    write_json(args.output_dir / "preserved_train_positive_control.json", comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
