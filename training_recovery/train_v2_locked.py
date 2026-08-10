#!/usr/bin/env python3
"""Train only one lock-authorized v2 job; no methodological CLI overrides."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from mso_recovery.objective import (  # noqa: E402
    BinaryAccumulator,
    feature_distillation,
    full_bce,
    unknown_bce,
    unknown_soft_dice_loss,
)
from mso_recovery.v2_data import V2ManifestDataset  # noqa: E402
from mso_recovery.v2_lock import (  # noqa: E402
    atomic_write_json,
    load_lock,
    sha256_file,
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
    parser.add_argument("--lock-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def authorized_job(lock: dict, job_id: str) -> dict:
    teacher_seed = int(lock["training"]["teacher_seed"])
    if job_id == f"smoke_teacher_{teacher_seed}":
        return {
            "stage": "teacher",
            "seed": teacher_seed,
            "dataset_kind": "smoke",
            "maximum_epochs": int(lock["training"]["smoke_teacher"]["maximum_epochs"]),
        }
    if job_id == f"teacher_{teacher_seed}":
        return {
            "stage": "teacher",
            "seed": teacher_seed,
            "dataset_kind": "full",
            "maximum_epochs": int(lock["training"]["maximum_epochs"]),
        }
    for seed in lock["training"]["student_seeds"]:
        if job_id == f"student_{int(seed)}":
            return {
                "stage": "student",
                "seed": int(seed),
                "dataset_kind": "full",
                "maximum_epochs": int(lock["training"]["maximum_epochs"]),
            }
    raise ValueError(f"Job is not authorized by the v2 lock: {job_id}")


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def worker_seed(worker_id: int) -> None:
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_metric(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lock: dict,
    lock_sha: str,
    job: dict,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    manifest_sha: str,
    teacher_sha: str | None,
) -> dict:
    return {
        "schema": "mso.prospective_v2.checkpoint/1",
        "status": "prospective_v2_not_historical_recovery",
        "v2_lock_sha256": lock_sha,
        "dataset_manifest_sha256": manifest_sha,
        "source_split_sha256": lock["artifacts"]["source_split"]["sha256"],
        "training_configuration_sha256": lock["configuration_sha256"],
        "stage": job["stage"],
        "seed": job["seed"],
        "epoch": epoch,
        "selected_validation_metric": best_metric,
        "selected_epoch": best_epoch,
        "parent_teacher_checkpoint_sha256": teacher_sha,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def main() -> None:
    args = parse_args()
    lock, lock_sha = load_lock(args.lock_root)
    job = authorized_job(lock, args.job_id)
    if not torch.cuda.is_available():
        raise RuntimeError("Locked v2 training requires CUDA")
    dataset_root = args.campaign_root / "datasets" / job["dataset_kind"]
    manifest = dataset_root / "manifest.csv"
    validation_record = json.loads(
        (args.campaign_root / "validation" / f"{job['dataset_kind']}.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_sha = sha256_file(manifest)
    if validation_record.get("v2_lock_sha256") != lock_sha:
        raise ValueError("Dataset validation is not bound to the lock")
    if validation_record.get("dataset_manifest_sha256") != manifest_sha:
        raise ValueError("Dataset validation is not bound to this manifest")
    if validation_record.get("dataset_kind") != job["dataset_kind"]:
        raise ValueError("Dataset validation kind differs from the locked job")
    if validation_record.get("test_images_decoded") != 0 or validation_record.get(
        "test_outcomes_accessed"
    ) is not False:
        raise ValueError("Dataset validation lacks the no-test binding")
    output_dir = args.campaign_root / "runs" / args.job_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty locked run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(job["seed"])
    set_determinism(seed)
    train_data = V2ManifestDataset(dataset_root, manifest, "train")
    val_data = V2ManifestDataset(dataset_root, manifest, "val")
    generator = torch.Generator()
    train_loader = DataLoader(
        train_data,
        batch_size=int(lock["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(lock["training"]["num_workers"]),
        pin_memory=True,
        worker_init_fn=worker_seed,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=int(lock["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(lock["training"]["num_workers"]),
        pin_memory=True,
        worker_init_fn=worker_seed,
    )
    teacher = None
    teacher_sha = None
    if job["stage"] == "teacher":
        model = TeacherMapNet2(image_size=256, dim=4).cuda()
    else:
        model = DistillMapNetDeconv(image_size=256, dim=4).cuda()
        if sum(parameter.numel() for parameter in model.parameters()) != 342_771:
            raise ValueError("Student parameter count changed")
        teacher_path = (
            args.campaign_root
            / "runs"
            / f"teacher_{int(lock['training']['teacher_seed'])}"
            / "best.pt"
        )
        teacher_sha = sha256_file(teacher_path)
        teacher_completion = json.loads(
            (teacher_path.parent / "completion.json").read_text(encoding="utf-8")
        )
        if teacher_completion.get("best_checkpoint_sha256") != teacher_sha:
            raise ValueError("Selected teacher differs from teacher completion")
        teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
        required = {
            "v2_lock_sha256": lock_sha,
            "dataset_manifest_sha256": manifest_sha,
            "source_split_sha256": lock["artifacts"]["source_split"]["sha256"],
            "training_configuration_sha256": lock["configuration_sha256"],
            "stage": "teacher",
            "seed": int(lock["training"]["teacher_seed"]),
            "parent_teacher_checkpoint_sha256": None,
        }
        for key, expected in required.items():
            if teacher_payload.get(key) != expected:
                raise ValueError(f"Teacher checkpoint binding failed: {key}")
        teacher = TeacherMapNet2(image_size=256, dim=4).cuda()
        teacher.load_state_dict(teacher_payload["model_state_dict"], strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(lock["training"]["learning_rate"]),
        betas=tuple(float(value) for value in lock["training"]["adam_betas"]),
        eps=float(lock["training"]["adam_epsilon"]),
        weight_decay=float(lock["training"]["weight_decay"]),
        amsgrad=bool(lock["training"]["adam_amsgrad"]),
        foreach=bool(lock["training"]["adam_foreach"]),
        fused=bool(lock["training"]["adam_fused"]),
        capturable=bool(lock["training"]["adam_capturable"]),
        differentiable=bool(lock["training"]["adam_differentiable"]),
    )
    run_manifest = {
        "schema": "mso.prospective_v2.run/1",
        "status": "prospective_v2_not_historical_recovery",
        "job_id": args.job_id,
        "job": job,
        "v2_lock_sha256": lock_sha,
        "dataset_manifest_sha256": manifest_sha,
        "source_split_sha256": lock["artifacts"]["source_split"]["sha256"],
        "training_configuration_sha256": lock["configuration_sha256"],
        "parent_teacher_checkpoint_sha256": teacher_sha,
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "test_outcomes_accessed": False,
    }
    atomic_write_json(output_dir / "run_manifest.json", run_manifest)
    weights = lock["training"]["loss_weights"]
    best_metric = float("inf")
    best_epoch = -1
    maximum_epochs = int(job["maximum_epochs"])
    for epoch in range(1, maximum_epochs + 1):
        started = time.perf_counter()
        generator.manual_seed(seed + epoch)
        model.train()
        train_total = 0.0
        train_unknown = 0.0
        batches = 0
        for features, labels, unknown, _, _ in train_loader:
            features = features.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            unknown = unknown.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
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
            batches += 1
        model.eval()
        accumulator = BinaryAccumulator()
        with torch.inference_mode():
            for features, labels, unknown, _, _ in val_loader:
                prediction = model(features.cuda(non_blocking=True))[0]
                accumulator.update(
                    prediction,
                    labels.cuda(non_blocking=True),
                    unknown.cuda(non_blocking=True),
                )
        validation = accumulator.metrics()
        finite_values = [
            train_total,
            train_unknown,
            *(
                float(validation[key])
                for key in (
                    "unknown_bce",
                    "unknown_precision",
                    "unknown_recall",
                    "unknown_f1",
                    "unknown_iou",
                )
            ),
        ]
        if not all(math.isfinite(value) for value in finite_values):
            raise FloatingPointError(f"Non-finite training metric at epoch {epoch}")
        current = float(validation["unknown_bce"])
        selected = current < best_metric
        if selected:
            best_metric = current
            best_epoch = epoch
            atomic_torch_save(
                checkpoint_payload(
                    model,
                    optimizer,
                    lock,
                    lock_sha,
                    job,
                    epoch,
                    best_metric,
                    best_epoch,
                    manifest_sha,
                    teacher_sha,
                ),
                output_dir / "best.pt",
            )
        atomic_torch_save(
            checkpoint_payload(
                model,
                optimizer,
                lock,
                lock_sha,
                job,
                epoch,
                best_metric,
                best_epoch,
                manifest_sha,
                teacher_sha,
            ),
            output_dir / "last.pt",
        )
        row = {
            "epoch": epoch,
            "train_total_loss": train_total / batches,
            "train_unknown_bce": train_unknown / batches,
            "validation_unknown_bce": validation["unknown_bce"],
            "validation_unknown_precision": validation["unknown_precision"],
            "validation_unknown_recall": validation["unknown_recall"],
            "validation_unknown_f1": validation["unknown_f1"],
            "validation_unknown_iou": validation["unknown_iou"],
            "epoch_seconds": time.perf_counter() - started,
            "selected_best": int(selected),
        }
        write_metric(output_dir / "metrics.csv", row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if (
            job["dataset_kind"] == "full"
            and epoch >= int(lock["training"]["minimum_epochs_before_stopping"])
            and epoch - best_epoch
            >= int(lock["training"]["early_stopping_patience"])
        ):
            break
    completion = {
        "schema": "mso.prospective_v2.run_completion/1",
        "status": "prospective_v2_not_historical_recovery",
        "v2_lock_sha256": lock_sha,
        "dataset_manifest_sha256": manifest_sha,
        "source_split_sha256": lock["artifacts"]["source_split"]["sha256"],
        "training_configuration_sha256": lock["configuration_sha256"],
        "job_id": args.job_id,
        "stage": job["stage"],
        "seed": seed,
        "last_epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_unknown_bce": best_metric,
        "best_checkpoint_sha256": sha256_file(output_dir / "best.pt"),
        "last_checkpoint_sha256": sha256_file(output_dir / "last.pt"),
        "metrics_sha256": sha256_file(output_dir / "metrics.csv"),
        "parent_teacher_checkpoint_sha256": teacher_sha,
        "test_outcomes_accessed": False,
    }
    atomic_write_json(output_dir / "completion.json", completion)


if __name__ == "__main__":
    main()
