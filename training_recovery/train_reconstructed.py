#!/usr/bin/env python3
"""Train a validation-selected teacher or student for the replacement study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from mso_recovery import RECONSTRUCTION_STATUS  # noqa: E402
from mso_recovery.common import sha256_file, write_json  # noqa: E402
from mso_recovery.data import FloorplanManifestDataset  # noqa: E402
from mso_recovery.objective import (  # noqa: E402
    BinaryAccumulator,
    feature_distillation,
    full_bce,
    unknown_bce,
    unknown_soft_dice_loss,
)
from sensemap.explore_model.SenseMapNet import (  # noqa: E402
    DistillMapNetDeconv,
    TeacherMapNet2,
)


METRIC_FIELDS = (
    "epoch",
    "train_total_loss",
    "train_unknown_bce",
    "validation_unknown_bce",
    "validation_unknown_precision",
    "validation_unknown_recall",
    "validation_unknown_f1",
    "validation_unknown_iou",
    "epoch_seconds",
    "selected_best",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("teacher", "student"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "replacement_protocol.json")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--maximum-epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.maximum_epochs is not None:
        config["maximum_epochs"] = args.maximum_epochs
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    default_seed = config["teacher_seed"] if args.stage == "teacher" else config["student_seeds"][0]
    config["seed"] = int(default_seed if args.seed is None else args.seed)
    config["stage"] = args.stage
    return config


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def worker_seed(worker_id: int) -> None:
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


def load_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Not a replacement-study checkpoint: {path}")
    if payload.get("status") != RECONSTRUCTION_STATUS:
        raise ValueError(f"Checkpoint has an incompatible provenance status: {path}")
    return payload


def amp_context(device: str, config: dict):
    enabled = device.startswith("cuda") and config["mixed_precision"] == "bfloat16"
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    manifest_sha256: str,
    integrity: str,
) -> dict:
    return {
        "status": RECONSTRUCTION_STATUS,
        "stage": config["stage"],
        "seed": config["seed"],
        "epoch": epoch,
        "best_validation_unknown_bce": best_metric,
        "best_epoch": best_epoch,
        "configuration": config,
        "manifest_sha256": manifest_sha256,
        "dataset_integrity": integrity,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def write_metric(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = load_config(args)
    seed = int(config["seed"])
    set_determinism(seed)
    if args.stage == "student" and args.teacher_checkpoint is None:
        raise SystemExit("--teacher-checkpoint is required for student training")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        raise SystemExit(f"Refusing to overwrite non-empty run: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_data = FloorplanManifestDataset(
        args.dataset_root,
        args.manifest,
        "train",
        verify_hashes=not args.skip_hash_verification,
    )
    validation_data = FloorplanManifestDataset(
        args.dataset_root,
        args.manifest,
        "val",
        verify_hashes=not args.skip_hash_verification,
    )
    train_generator = torch.Generator()
    train_loader = DataLoader(
        train_data,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=train_generator,
        num_workers=int(config["num_workers"]),
        pin_memory=args.device.startswith("cuda"),
        worker_init_fn=worker_seed,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=args.device.startswith("cuda"),
        worker_init_fn=worker_seed,
    )
    manifest_sha = sha256_file(args.manifest)

    if args.stage == "teacher":
        model = TeacherMapNet2(image_size=256, dim=4).to(args.device)
        teacher = None
    else:
        model = DistillMapNetDeconv(image_size=256, dim=4).to(args.device)
        teacher = TeacherMapNet2(image_size=256, dim=4).to(args.device)
        teacher_payload = load_checkpoint(args.teacher_checkpoint)
        if teacher_payload.get("stage") != "teacher":
            raise ValueError("Student training requires a teacher-stage checkpoint")
        if teacher_payload.get("manifest_sha256") != manifest_sha:
            raise ValueError("Teacher and student dataset manifests differ")
        teacher.load_state_dict(teacher_payload["model_state_dict"], strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != 342_771:
            raise ValueError(f"Student parameter count changed: {parameter_count}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config["adam_betas"]),
        weight_decay=float(config["weight_decay"]),
    )
    start_epoch = 0
    best_metric = float("inf")
    best_epoch = -1
    if args.resume is not None:
        payload = load_checkpoint(args.resume)
        if payload.get("configuration") != config:
            raise ValueError("Resume configuration differs from the locked run configuration")
        if payload.get("manifest_sha256") != manifest_sha:
            raise ValueError("Resume checkpoint and dataset manifests differ")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_metric = float(payload["best_validation_unknown_bce"])
        best_epoch = int(payload["best_epoch"])

    run_manifest = {
        "status": RECONSTRUCTION_STATUS,
        "configuration": config,
        "manifest_sha256": manifest_sha,
        "training_samples": len(train_data),
        "validation_samples": len(validation_data),
        "dataset_integrity": train_data.integrity,
        "teacher_checkpoint_sha256": (
            sha256_file(args.teacher_checkpoint) if args.teacher_checkpoint else None
        ),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "hardware": {
            "device": args.device,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None
            ),
        },
    }
    write_json(args.output_dir / "run_manifest.json", run_manifest)

    if args.dry_run:
        features, labels, unknown, sample_ids, floorplans = next(iter(train_loader))
        with torch.inference_mode(), amp_context(args.device, config):
            output = model(features.to(args.device))[0]
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "features": list(features.shape),
                    "labels": list(labels.shape),
                    "unknown": list(unknown.shape),
                    "output": list(output.shape),
                    "first_sample": sample_ids[0],
                    "first_floorplan": floorplans[0],
                },
                indent=2,
            )
        )
        return

    weights = config["loss_weights"]
    for epoch in range(start_epoch, int(config["maximum_epochs"])):
        epoch_start = time.perf_counter()
        train_generator.manual_seed(seed + epoch)
        model.train()
        train_total = 0.0
        train_unknown = 0.0
        train_batches = 0
        for features, labels, unknown, _, _ in train_loader:
            features = features.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            unknown = unknown.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args.device, config):
                prediction, *student_features = model(features)
                loss_unknown = unknown_bce(prediction, labels, unknown)
                loss = (
                    float(weights["unknown_bce"]) * loss_unknown
                    + float(weights["unknown_soft_dice"])
                    * unknown_soft_dice_loss(prediction, labels, unknown)
                    + float(weights["full_bce"]) * full_bce(prediction, labels)
                )
                if teacher is not None:
                    with torch.no_grad():
                        _, *teacher_features = teacher(features)
                    loss = loss + float(weights["feature_distillation"]) * feature_distillation(
                        tuple(student_features), tuple(teacher_features)
                    )
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach().cpu())
            train_unknown += float(loss_unknown.detach().cpu())
            train_batches += 1

        model.eval()
        accumulator = BinaryAccumulator()
        with torch.inference_mode():
            for features, labels, unknown, _, _ in validation_loader:
                features = features.to(args.device, non_blocking=True)
                labels = labels.to(args.device, non_blocking=True)
                unknown = unknown.to(args.device, non_blocking=True)
                with amp_context(args.device, config):
                    prediction = model(features)[0]
                accumulator.update(prediction, labels, unknown)
        validation = accumulator.metrics()
        current_metric = float(validation["unknown_bce"])
        selected = current_metric < best_metric
        if selected:
            best_metric = current_metric
            best_epoch = epoch
            torch.save(
                checkpoint_payload(
                    model,
                    optimizer,
                    config,
                    epoch,
                    best_metric,
                    best_epoch,
                    manifest_sha,
                    train_data.integrity,
                ),
                args.output_dir / "best.pt",
            )
        torch.save(
            checkpoint_payload(
                model,
                optimizer,
                config,
                epoch,
                best_metric,
                best_epoch,
                manifest_sha,
                train_data.integrity,
            ),
            args.output_dir / "last.pt",
        )
        row = {
            "epoch": epoch,
            "train_total_loss": train_total / max(1, train_batches),
            "train_unknown_bce": train_unknown / max(1, train_batches),
            "validation_unknown_bce": validation["unknown_bce"],
            "validation_unknown_precision": validation["unknown_precision"],
            "validation_unknown_recall": validation["unknown_recall"],
            "validation_unknown_f1": validation["unknown_f1"],
            "validation_unknown_iou": validation["unknown_iou"],
            "epoch_seconds": time.perf_counter() - epoch_start,
            "selected_best": int(selected),
        }
        write_metric(args.output_dir / "metrics.csv", row)
        print(json.dumps(row, sort_keys=True), flush=True)
        epochs_without_improvement = epoch - best_epoch
        if (
            epoch + 1 >= int(config["minimum_epochs_before_stopping"])
            and epochs_without_improvement >= int(config["early_stopping_patience"])
        ):
            break

    completion = {
        "status": RECONSTRUCTION_STATUS,
        "completed": True,
        "last_epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_unknown_bce": best_metric,
        "best_checkpoint_sha256": sha256_file(args.output_dir / "best.pt"),
        "last_checkpoint_sha256": sha256_file(args.output_dir / "last.pt"),
    }
    write_json(args.output_dir / "completion.json", completion)


if __name__ == "__main__":
    main()
