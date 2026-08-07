#!/usr/bin/env python3
"""Pinned PSNR/SSIM evaluation for a declared candidate or reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

from mso_reconstructed.artifacts import sha256_file, unwrap_generator_state  # noqa: E402
from mso_reconstructed.data import ManifestDataset  # noqa: E402
from sensemap.explore_model.SenseMapNet import DistillMapNetDeconv  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--skip-raw-hash-verification",
        action="store_true",
        help="Diagnostic-only bypass; output is marked as input hashes not verified.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ManifestDataset(
        args.dataset_root,
        args.manifest,
        args.split,
        verify_raw_hashes=not args.skip_raw_hash_verification,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    payload = torch.load(args.weights, map_location="cpu", weights_only=True)
    state = unwrap_generator_state(payload)
    model = DistillMapNetDeconv(image_size=256, dim=4)
    model.load_state_dict(state, strict=True)
    model.eval().to(args.device)

    psnr_values, ssim_values = [], []
    started = time.time()
    seen = 0
    with torch.inference_mode():
        for features, labels, _, _ in loader:
            if args.max_samples is not None and seen >= args.max_samples:
                break
            output = model(features.to(args.device))[0].cpu().numpy()
            target = labels.numpy()
            for index in range(output.shape[0]):
                if args.max_samples is not None and seen >= args.max_samples:
                    break
                prediction = output[index, 0]
                truth = target[index, 0]
                mse = float(np.mean((prediction - truth) ** 2))
                psnr_values.append(float(10.0 * np.log10(1.0 / max(mse, 1e-20))))
                ssim_values.append(
                    float(structural_similarity(truth, prediction, data_range=1.0))
                )
                seen += 1

    result = {
        "status": (
            "declared_artifact_evaluation"
            if dataset.verify_raw_hashes
            else "declared_artifact_evaluation_unverified_input"
        ),
        "protocol": "processed-pair-v1; full-image PSNR and skimage SSIM, data_range=1",
        "weights": str(args.weights),
        "weights_sha256": sha256_file(args.weights),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "dataset_integrity": dataset.integrity_status,
        "split": args.split,
        "n": seen,
        "psnr_mean": float(np.mean(psnr_values)),
        "psnr_std": float(np.std(psnr_values)),
        "ssim_mean": float(np.mean(ssim_values)),
        "ssim_std": float(np.std(ssim_values)),
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
