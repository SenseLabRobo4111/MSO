#!/usr/bin/env python3
"""Run the prospective V4 MSO reconstruction protocol.

V4 deliberately reuses the hash-bound V3 data, networks, optimiser, batch
augmentation and training objective.  It changes only validation accounting,
checkpoint selection and checkpoint retention.  It is prospective and is not
an authenticated reconstruction of the unavailable historical trainer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mso_full_components import (
    ProjectedTeacher,
    ReconstructedFFCPatchCritic,
    trainable_parameter_count,
)
from train_unified_adapter import CachedDataset, worker_seed
from train_paper_mso_reconstructed import (
    STATUS as V3_STATUS,
    TOPOLOGY_STATUS,
    EXPECTED_RESNET_COMMON_TENSORS,
    atomic_checkpoint,
    augment_batch,
    latency_profile,
    measured_authoritative_target,
    projection_fingerprint,
    reconcile_metrics,
    restore_rng_state,
    run_batch,
    set_determinism,
    sha256,
    validate_config as validate_v3_config,
    verify_dependencies,
    verify_resnet_loaded,
    write_json,
    write_metric,
)


STATUS = "prospective_paper_equation_reconstruction_v4_not_historical"
TEACHER_SEED = 101
STUDENT_SEEDS = (11, 23, 37, 53, 71)
METHOD_TEACHER = "MSO-Paper-Equation-Teacher-Reconstructed-V4"
METHOD_STUDENT = "MSO-Paper-Equation-Reconstructed-V4"
PRIMARY_KIND = "validation_best_primary_generator"
CALIBRATION_KIND = "validation_best_calibration_generator"
SNAPSHOT_KIND = "periodic_generator_snapshot"
RESUME_KIND = "epoch_boundary_resume_state_v4"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
VALIDATION_RECORD_KEYS = (
    "unknown_count",
    "unknown_tp",
    "unknown_fp",
    "unknown_fn",
    "unknown_tn",
    "unknown_precision",
    "unknown_recall",
    "unknown_f1",
    "unknown_iou",
    "unknown_bce",
    "unknown_brier",
    "unknown_ece15",
)


def validate_config(config: dict[str, object]) -> None:
    """Validate V4 and prove that the V3 training contract is unchanged."""
    v3_view = dict(config)
    v3_view.update(
        {
            "protocol_id": "mso_paper_equation_reconstruction_v3",
            "status": V3_STATUS,
            "selection_metric": "validation_unknown_bce",
            "selection_direction": "minimise",
            "selection_policy": (
                "retain_validation_best_and_epoch_499_last_without_early_stopping"
            ),
        }
    )
    validate_v3_config(v3_view)
    expected = {
        "protocol_id": "mso_paper_equation_reconstruction_v4",
        "status": STATUS,
        "teacher_checkpoint_for_student": (
            "teacher_validation_best_primary_unknown_occupied_micro_f1_at_0.5_"
            "bound_by_explicit_sha256"
        ),
        "selection_metric": "validation_unknown_occupied_micro_f1_at_0.5",
        "selection_direction": "maximise",
        "selection_tiebreaker": (
            "lower_validation_unknown_brier_then_earlier_epoch"
        ),
        "calibration_checkpoint_metric": "validation_unknown_brier",
        "calibration_checkpoint_direction": "minimise",
        "calibration_checkpoint_tiebreaker": (
            "lower_validation_unknown_bce_then_earlier_epoch"
        ),
        "selection_policy": (
            "retain_best_primary_best_calibration_epoch_499_last_and_every_10_"
            "completed_epoch_generator_snapshot_without_early_stopping"
        ),
        "checkpoint_policy": (
            "atomic_epoch_boundary_checkpoint_with_automatic_resume"
        ),
        "periodic_snapshot_interval_epochs": 10,
        "decision_threshold": 0.5,
        "ece_bins": 15,
        "shared_v3_training_source_sha256": (
            "7aad05cdfc63fe394b6f90d3c3e971f785daa61b4bcd52299def7a425b54f1c6"
        ),
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ValueError(f"V4 configuration contract differs for {key}")
    expected_metrics = [
        "integer_tp",
        "integer_fp",
        "integer_fn",
        "integer_tn",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "micro_iou",
        "bce",
        "brier",
        "ece15",
    ]
    if config.get("validation_metrics") != expected_metrics:
        raise ValueError("V4 validation metric contract differs")


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def validation_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float = 0.5,
    ece_bins: int = 15,
) -> dict[str, float | int]:
    """Return unknown-region occupied-class micro metrics.

    ECE uses 15 equal-width bins over the occupied probability.  Integer
    confusion counts are accumulated before any derived metric is calculated.
    """
    if threshold != 0.5:
        raise ValueError("V4 primary selection requires threshold 0.5")
    if ece_bins != 15:
        raise ValueError("V4 calibration audit requires exactly 15 bins")
    tp = fp = fn = tn = 0
    bce_sum = 0.0
    brier_sum = 0.0
    count = 0
    bin_count = np.zeros(ece_bins, dtype=np.int64)
    bin_probability_sum = np.zeros(ece_bins, dtype=np.float64)
    bin_target_sum = np.zeros(ece_bins, dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for features, target, unknown, _ in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            unknown = unknown.to(device, non_blocking=True)
            target = measured_authoritative_target(features, target, unknown)
            probability = model(features)[0].float().clamp(0.0, 1.0)
            mask = unknown >= 0.5
            probability_values = probability[mask].detach().cpu().numpy().astype(
                np.float64, copy=False
            )
            target_values = target[mask].detach().cpu().numpy().astype(
                np.float64, copy=False
            )
            if probability_values.size == 0:
                continue
            truth = target_values >= 0.5
            predicted = probability_values >= threshold
            tp += int(np.logical_and(predicted, truth).sum())
            fp += int(np.logical_and(predicted, ~truth).sum())
            fn += int(np.logical_and(~predicted, truth).sum())
            tn += int(np.logical_and(~predicted, ~truth).sum())
            clipped = np.clip(probability_values, 1e-6, 1.0 - 1e-6)
            bce_sum += float(
                (-target_values * np.log(clipped)
                 - (1.0 - target_values) * np.log1p(-clipped)).sum()
            )
            brier_sum += float(np.square(probability_values - target_values).sum())
            count += int(probability_values.size)
            indices = np.minimum(
                (probability_values * ece_bins).astype(np.int64), ece_bins - 1
            )
            bin_count += np.bincount(indices, minlength=ece_bins)
            bin_probability_sum += np.bincount(
                indices, weights=probability_values, minlength=ece_bins
            )
            bin_target_sum += np.bincount(
                indices, weights=target_values, minlength=ece_bins
            )
    if count == 0 or tp + fp + fn + tn != count:
        raise ValueError("unknown-region confusion accounting is incomplete")
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2 * tp, 2 * tp + fp + fn)
    iou = safe_ratio(tp, tp + fp + fn)
    occupied_bins = bin_count > 0
    calibration_gap = np.zeros(ece_bins, dtype=np.float64)
    calibration_gap[occupied_bins] = np.abs(
        bin_probability_sum[occupied_bins] / bin_count[occupied_bins]
        - bin_target_sum[occupied_bins] / bin_count[occupied_bins]
    )
    ece = float((bin_count * calibration_gap).sum() / count)
    return {
        "unknown_count": count,
        "unknown_tp": tp,
        "unknown_fp": fp,
        "unknown_fn": fn,
        "unknown_tn": tn,
        "unknown_precision": precision,
        "unknown_recall": recall,
        "unknown_f1": f1,
        "unknown_iou": iou,
        "unknown_bce": bce_sum / count,
        "unknown_brier": brier_sum / count,
        "unknown_ece15": ece,
    }


def metric_record(epoch: int, metrics: dict[str, float | int]) -> dict[str, Any]:
    return {"epoch": int(epoch), **metrics}


def _f1_fraction(record: dict[str, Any]) -> tuple[int, int]:
    numerator = 2 * int(record["unknown_tp"])
    denominator = (
        2 * int(record["unknown_tp"])
        + int(record["unknown_fp"])
        + int(record["unknown_fn"])
    )
    return (numerator, denominator if denominator else 1)


def primary_is_better(
    candidate: dict[str, Any], incumbent: dict[str, Any] | None
) -> bool:
    """Apply exact micro-F1, lower-Brier, earlier-epoch ordering."""
    if incumbent is None:
        return True
    candidate_numerator, candidate_denominator = _f1_fraction(candidate)
    incumbent_numerator, incumbent_denominator = _f1_fraction(incumbent)
    left = candidate_numerator * incumbent_denominator
    right = incumbent_numerator * candidate_denominator
    if left != right:
        return left > right
    candidate_brier = float(candidate["unknown_brier"])
    incumbent_brier = float(incumbent["unknown_brier"])
    if candidate_brier != incumbent_brier:
        return candidate_brier < incumbent_brier
    return int(candidate["epoch"]) < int(incumbent["epoch"])


def calibration_is_better(
    candidate: dict[str, Any], incumbent: dict[str, Any] | None
) -> bool:
    """Select minimum Brier, then minimum BCE, then the earlier epoch."""
    if incumbent is None:
        return True
    for key in ("unknown_brier", "unknown_bce"):
        candidate_value = float(candidate[key])
        incumbent_value = float(incumbent[key])
        if candidate_value != incumbent_value:
            return candidate_value < incumbent_value
    return int(candidate["epoch"]) < int(incumbent["epoch"])


def validate_parent_checkpoint_payload(
    payload: dict[str, object], protocol_id: str
) -> None:
    """Reject every student parent except the V4 teacher primary checkpoint."""
    if payload.get("checkpoint_kind") != PRIMARY_KIND:
        raise ValueError("student parent is not teacher best_primary.pt")
    if payload.get("stage") != "teacher":
        raise ValueError("student parent checkpoint is not a teacher checkpoint")
    if int(payload.get("seed", -1)) != TEACHER_SEED:
        raise ValueError("student parent checkpoint has the wrong teacher seed")
    if payload.get("protocol_id") != protocol_id:
        raise ValueError("student parent teacher protocol differs")


def validate_parent_checkpoint_sha(path: Path, declared_sha256: str | None) -> str:
    """Require a canonical explicit digest and equality with the parent file."""
    if not declared_sha256 or not SHA256_PATTERN.fullmatch(declared_sha256):
        raise ValueError("student stage requires an explicit lowercase SHA-256")
    actual = sha256(path)
    if actual != declared_sha256:
        raise ValueError("teacher best_primary checkpoint SHA-256 differs")
    return actual


def record_from_metric_row(row: dict[str, str]) -> dict[str, Any]:
    integer_keys = {
        "unknown_count",
        "unknown_tp",
        "unknown_fp",
        "unknown_fn",
        "unknown_tn",
    }
    record: dict[str, Any] = {"epoch": int(row["epoch"])}
    for key in VALIDATION_RECORD_KEYS:
        table_key = f"validation_{key}"
        if table_key not in row:
            raise ValueError(f"metrics table is missing {table_key}")
        record[key] = int(row[table_key]) if key in integer_keys else float(row[table_key])
    return record


def recompute_selection_from_metrics(
    metrics_path: Path, expected_epochs: int
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Independently recompute both final argbest records from the TSV."""
    with metrics_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != expected_epochs:
        raise ValueError("selection audit requires one row for every epoch")
    records = [record_from_metric_row(row) for row in rows]
    if [record["epoch"] for record in records] != list(range(expected_epochs)):
        raise ValueError("selection audit metrics are not contiguous")
    primary: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    for record in records:
        if primary_is_better(record, primary):
            primary = record
        if calibration_is_better(record, calibration):
            calibration = record
    if primary is None or calibration is None:
        raise ValueError("selection audit found no validation records")
    return primary, calibration, len(records)


def write_selection_manifest(
    metrics_path: Path,
    best_primary_path: Path,
    best_calibration_path: Path,
    output_path: Path,
    protocol_id: str,
    expected_epochs: int,
    stage: str,
    seed: int,
    metadata: dict[str, object],
    expected_projection_fingerprint: str | None,
) -> dict[str, object]:
    """Recompute argbest, verify checkpoint records and freeze the audit."""
    if metadata.get("protocol_id") != protocol_id:
        raise ValueError("selection-audit metadata protocol differs")
    primary, calibration, row_count = recompute_selection_from_metrics(
        metrics_path, expected_epochs
    )
    entries: dict[str, dict[str, object]] = {}
    for label, path, kind, expected_record in (
        ("primary", best_primary_path, PRIMARY_KIND, primary),
        (
            "calibration",
            best_calibration_path,
            CALIBRATION_KIND,
            calibration,
        ),
    ):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"{label} checkpoint payload is not a mapping")
        expected_payload = {
            **metadata,
            "checkpoint_kind": kind,
            "stage": stage,
            "seed": seed,
            "epoch": int(expected_record["epoch"]),
            "selection_record": expected_record,
            "projection_state_fingerprint": expected_projection_fingerprint,
        }
        for key, expected_value in expected_payload.items():
            if payload.get(key) != expected_value:
                raise ValueError(
                    f"{label} checkpoint provenance differs for {key}"
                )
        entries[label] = {
            "checkpoint": path.name,
            "checkpoint_kind": kind,
            "checkpoint_sha256": sha256(path),
            "epoch": int(expected_record["epoch"]),
            "validation_metrics": expected_record,
        }
        del payload
    manifest: dict[str, object] = {
        "protocol_id": protocol_id,
        "stage": stage,
        "seed": seed,
        "source": "independent_full_metrics_tsv_recomputation",
        "run_provenance": metadata,
        "projection_state_fingerprint": expected_projection_fingerprint,
        "metrics_by_epoch_sha256": sha256(metrics_path),
        "metrics_row_count": row_count,
        "primary_rule": (
            "maximum exact unknown occupied micro-F1 at 0.5; lower Brier; "
            "earlier epoch"
        ),
        "calibration_rule": "lower Brier; lower BCE; earlier epoch",
        "primary": entries["primary"],
        "calibration": entries["calibration"],
    }
    write_json(output_path, manifest)
    return manifest


def generator_checkpoint_payload(
    kind: str,
    stage: str,
    seed: int,
    record: dict[str, Any],
    model: nn.Module,
    projection_teacher: ProjectedTeacher | None,
    metadata: dict[str, object],
) -> dict[str, object]:
    if kind not in {PRIMARY_KIND, CALIBRATION_KIND}:
        raise ValueError("invalid V4 selected-checkpoint kind")
    return {
        **metadata,
        "checkpoint_kind": kind,
        "stage": stage,
        "seed": seed,
        "epoch": int(record["epoch"]),
        "selection_record": record,
        "model_state_dict": model.state_dict(),
        "projection_state_fingerprint": projection_fingerprint(projection_teacher),
    }


def periodic_snapshot_payload(
    stage: str,
    seed: int,
    epoch: int,
    model: nn.Module,
    projection_teacher: ProjectedTeacher | None,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        **metadata,
        "checkpoint_kind": SNAPSHOT_KIND,
        "stage": stage,
        "seed": seed,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "projection_state_fingerprint": projection_fingerprint(projection_teacher),
    }


def resume_payload(
    stage: str,
    seed: int,
    epoch: int,
    best_primary: dict[str, Any] | None,
    best_calibration: dict[str, Any] | None,
    elapsed_seconds: float,
    model: nn.Module,
    critic: nn.Module,
    projection_teacher: ProjectedTeacher | None,
    model_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    metadata: dict[str, object],
) -> dict[str, object]:
    from train_paper_mso_reconstructed import capture_rng_state

    return {
        **metadata,
        "checkpoint_kind": RESUME_KIND,
        "stage": stage,
        "seed": seed,
        "epoch": epoch,
        "best_primary_record": best_primary,
        "best_calibration_record": best_calibration,
        "elapsed_training_seconds": elapsed_seconds,
        "model_state_dict": model.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "projection_state_fingerprint": projection_fingerprint(projection_teacher),
        "model_optimizer_state_dict": model_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "rng_state": capture_rng_state(),
    }


def snapshot_relative_path(epoch: int) -> Path:
    return Path("snapshots") / f"epoch_{epoch:03d}.pt"


def verify_or_rebuild_selected(
    path: Path,
    kind: str,
    record: dict[str, Any] | None,
    current_epoch: int,
    stage: str,
    seed: int,
    model: nn.Module,
    projection_teacher: ProjectedTeacher | None,
    metadata: dict[str, object],
) -> None:
    if record is None:
        if path.exists():
            raise ValueError(f"{path.name} exists without a selection record")
        return
    rebuild = not path.exists()
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            **metadata,
            "checkpoint_kind": kind,
            "stage": stage,
            "seed": seed,
            "epoch": int(record["epoch"]),
            "selection_record": record,
            "projection_state_fingerprint": projection_fingerprint(
                projection_teacher
            ),
        }
        rebuild = any(existing.get(key) != value for key, value in expected.items())
    if rebuild:
        if current_epoch != int(record["epoch"]):
            raise ValueError(
                f"{path.name} differs and cannot be reconstructed from last.pt"
            )
        atomic_checkpoint(
            path,
            generator_checkpoint_payload(
                kind, stage, seed, record, model, projection_teacher, metadata
            ),
        )


def reconcile_periodic_snapshots(
    output: Path,
    current_epoch: int,
    interval: int,
    stage: str,
    seed: int,
    model: nn.Module,
    projection_teacher: ProjectedTeacher | None,
    metadata: dict[str, object],
) -> None:
    if interval <= 0:
        raise ValueError("snapshot interval must be positive")
    snapshot_root = output / "snapshots"
    snapshot_root.mkdir(exist_ok=True)
    for epoch in range(interval - 1, current_epoch + 1, interval):
        path = output / snapshot_relative_path(epoch)
        if not path.exists():
            if epoch != current_epoch:
                raise ValueError(f"missing immutable periodic snapshot: {path.name}")
            atomic_checkpoint(
                path,
                periodic_snapshot_payload(
                    stage, seed, epoch, model, projection_teacher, metadata
                ),
            )
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            **metadata,
            "checkpoint_kind": SNAPSHOT_KIND,
            "stage": stage,
            "seed": seed,
            "epoch": epoch,
            "projection_state_fingerprint": projection_fingerprint(
                projection_teacher
            ),
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError(f"periodic snapshot contract differs: {path.name}")


def export_predictions(
    model: nn.Module,
    dataset: CachedDataset,
    seed: int,
    split: str,
    config: dict[str, object],
    device: str,
    output: Path,
) -> Path:
    manifest = output / f"predictions_{split}.tsv"
    if manifest.is_file():
        return manifest
    probability_root = output / "probabilities" / split
    temporary_root = output / "probabilities" / f"{split}.tmp"
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    if probability_root.exists():
        shutil.rmtree(probability_root)
    temporary_root.mkdir(parents=True)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=str(device).startswith("cuda"),
        worker_init_fn=worker_seed,
    )
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for features, _, _, sample_ids in loader:
            features = features.to(device, non_blocking=True)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            probability = model(features)[0].float().clamp(0.0, 1.0)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0 / len(sample_ids)
            for sample_id, values in zip(sample_ids, probability[:, 0].cpu().numpy()):
                temporary = temporary_root / f"{sample_id}.npy"
                with temporary.open("wb") as stream:
                    np.save(stream, values.astype(np.float32), allow_pickle=False)
                final_path = probability_root / temporary.name
                rows.append(
                    {
                        "method": METHOD_STUDENT,
                        "seed": seed,
                        "split": split,
                        "sample_id": sample_id,
                        "probability_path": final_path.relative_to(output).as_posix(),
                        "probability_sha256": sha256(temporary),
                        "inference_ms": elapsed,
                    }
                )
    temporary_root.replace(probability_root)
    temporary_manifest = manifest.with_suffix(manifest.suffix + ".tmp")
    with temporary_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary_manifest.replace(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("teacher", "student"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-data-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--topology-contract", type=Path, required=True)
    parser.add_argument("--dependency-manifest", type=Path, required=True)
    parser.add_argument("--lama-root", type=Path, required=True)
    parser.add_argument("--resnet-weights-root", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--teacher-checkpoint-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.smoke_only and not args.resume:
        raise ValueError("formal training requires automatic-resume mode")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    config_sha = sha256(args.config)
    shared_training_path = Path(__file__).with_name(
        "train_paper_mso_reconstructed.py"
    )
    if sha256(shared_training_path) != config["shared_v3_training_source_sha256"]:
        raise ValueError("shared frozen training implementation hash differs")
    if args.stage == "teacher" and args.seed != TEACHER_SEED:
        raise ValueError("teacher seed differs from the frozen contract")
    if args.stage == "student" and args.seed not in STUDENT_SEEDS:
        raise ValueError("student seed differs from the frozen contract")
    if args.stage == "student":
        if args.teacher_checkpoint is None:
            raise ValueError("student stage requires teacher best_primary.pt")
        validate_parent_checkpoint_sha(
            args.teacher_checkpoint, args.teacher_checkpoint_sha256
        )
    elif (
        args.teacher_checkpoint is not None
        or args.teacher_checkpoint_sha256 is not None
    ):
        raise ValueError("teacher stage must not receive a parent checkpoint")
    if sha256(args.model_manifest) != config["model_manifest_sha256"]:
        raise ValueError("model manifest hash differs from the frozen config")
    verify_dependencies(
        args.lama_root,
        args.dependency_manifest,
        config["feature_reconstruction"]["dependency_manifest_sha256"],
    )
    resnet_weights = (
        args.resnet_weights_root
        / "ade20k"
        / "ade20k-resnet50dilated-ppm_deepsup"
        / "encoder_epoch_20.pth"
    )
    if sha256(resnet_weights) != config["feature_reconstruction"]["weights_sha256"]:
        raise ValueError("ADE20K ResNet weight hash differs from the frozen config")
    topology = json.loads(args.topology_contract.read_text(encoding="utf-8"))
    topology_sha = sha256(args.topology_contract)
    if topology.get("status") != TOPOLOGY_STATUS:
        raise ValueError("recovered topology contract status is invalid")
    if topology.get("source_checkpoint_sha256") != config["recovered_checkpoint_sha256"]:
        raise ValueError("recovered topology source differs from the frozen config")
    if topology.get("historical_weights_reused_for_initialisation") is not False:
        raise ValueError("historical weights must not initialise prospective training")

    set_determinism(args.seed)
    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.insert(0, str(args.lama_root.resolve()))
    from sensemap.explore_model.SenseMapNet import (
        DistillMapNetDeconv,
        FFCBlock,
        TeacherMapNet2,
    )
    from saicinpainting.training.losses.adversarial import make_discrim_loss
    from saicinpainting.training.losses.feature_matching import masked_l2_loss
    from saicinpainting.training.losses.perceptual import ResNetPL

    network_path = args.source_root / "sensemap" / "explore_model" / "SenseMapNet.py"
    ffc_path = args.source_root / "sensemap" / "explore_model" / "ffc.py"
    if sha256(network_path) != config["network_source_sha256"]:
        raise ValueError("SenseMap network source hash differs")
    if sha256(ffc_path) != config["ffc_source_sha256"]:
        raise ValueError("FFC source hash differs")
    if topology.get("network_source_sha256") != config["network_source_sha256"]:
        raise ValueError("topology and training network sources differ")
    if topology.get("ffc_source_sha256") != config["ffc_source_sha256"]:
        raise ValueError("topology and training FFC sources differ")

    teacher_payload: dict[str, object] | None = None
    if args.stage == "teacher":
        model = TeacherMapNet2(image_size=256, dim=32).to(args.device)
        if trainable_parameter_count(model) != 51_602_560:
            raise ValueError("teacher parameter count differs")
        projection_teacher = None
        method = METHOD_TEACHER
    else:
        model = DistillMapNetDeconv(image_size=256, dim=4).to(args.device)
        if trainable_parameter_count(model) != 342_771:
            raise ValueError("student parameter count differs")
        teacher_payload = torch.load(
            args.teacher_checkpoint, map_location="cpu", weights_only=False
        )
        validate_parent_checkpoint_payload(teacher_payload, str(config["protocol_id"]))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(config["projection_seed"]))
            projection_teacher = ProjectedTeacher(TeacherMapNet2)
        projection_teacher.core.load_state_dict(
            teacher_payload["model_state_dict"], strict=True
        )
        projection_teacher.freeze_all()
        projection_teacher.to(args.device)
        method = METHOD_STUDENT

    critic = ReconstructedFFCPatchCritic(FFCBlock).to(args.device)
    if trainable_parameter_count(critic) != 21_936_000:
        raise ValueError("critic parameter count differs from the recovered topology")
    extractor = ResNetPL(
        weight=1,
        weights_path=str(args.resnet_weights_root),
        arch_encoder="resnet50dilated",
        segmentation=True,
    )
    verify_resnet_loaded(extractor.impl, resnet_weights)
    extractor.to(args.device)
    extractor.eval()
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)

    discriminator_config = config["discriminator"]
    adversarial_objective = make_discrim_loss(
        "r1",
        gp_coef=float(discriminator_config["gp_coef"]),
        weight=float(config["loss_weights"]["adversarial"]),
        mask_as_fake_target=bool(discriminator_config["mask_as_fake_target"]),
        allow_scale_mask=bool(discriminator_config["allow_scale_mask"]),
        mask_scale_mode=str(discriminator_config["mask_scale_mode"]),
        use_unmasked_for_gen=bool(
            discriminator_config["use_unmasked_for_generator"]
        ),
        use_unmasked_for_discr=bool(
            discriminator_config["use_unmasked_for_discriminator"]
        ),
    )

    train_data = CachedDataset(args.model_data_root, args.model_manifest, "train")
    validation_data = CachedDataset(
        args.model_data_root, args.model_manifest, "validation"
    )
    expected_counts = config["split_counts"]
    if len(train_data) != expected_counts["train"]:
        raise ValueError("train sample count differs from the frozen contract")
    if len(validation_data) != expected_counts["validation"]:
        raise ValueError("validation sample count differs from the frozen contract")
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    augmentation_generator = torch.Generator().manual_seed(args.seed + 1_000_000)
    train_loader = DataLoader(
        train_data,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=shuffle_generator,
        num_workers=int(config["num_workers"]),
        pin_memory=str(args.device).startswith("cuda"),
        worker_init_fn=worker_seed,
        drop_last=bool(config["drop_last"]),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=str(args.device).startswith("cuda"),
        worker_init_fn=worker_seed,
        drop_last=False,
    )
    betas = tuple(float(value) for value in config["adam_betas"])
    model_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=betas,
        weight_decay=float(config["weight_decay"]),
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=float(config["learning_rate"]),
        betas=betas,
        weight_decay=float(config["weight_decay"]),
    )

    container_image_id = os.environ.get("MSO_CONTAINER_IMAGE_ID")
    bundle_sha = os.environ.get("MSO_BUNDLE_SHA256")
    if not container_image_id or not bundle_sha:
        raise ValueError("container image ID and bundle SHA must be supplied")
    teacher_sha = (
        sha256(args.teacher_checkpoint) if args.teacher_checkpoint is not None else None
    )
    source_hashes = {
        "trainer_v4": sha256(Path(__file__)),
        "shared_frozen_training_implementation": sha256(shared_training_path),
        "components": sha256(Path(__file__).with_name("mso_full_components.py")),
        "topology_verifier": sha256(
            Path(__file__).with_name("verify_recovered_components.py")
        ),
        "data_adapter": sha256(
            Path(__file__).with_name("train_unified_adapter.py")
        ),
        "data_adapter_import_dependency": sha256(
            Path(__file__).with_name("unified_models.py")
        ),
        "configuration": config_sha,
        "dependency_manifest": sha256(args.dependency_manifest),
        "sensemap_network": sha256(network_path),
        "ffc": sha256(ffc_path),
        "lama_adversarial": sha256(
            args.lama_root / "saicinpainting/training/losses/adversarial.py"
        ),
        "lama_feature_matching": sha256(
            args.lama_root / "saicinpainting/training/losses/feature_matching.py"
        ),
        "lama_perceptual": sha256(
            args.lama_root / "saicinpainting/training/losses/perceptual.py"
        ),
    }
    if topology.get("verifier_sha256") != source_hashes["topology_verifier"]:
        raise ValueError("topology contract was produced by a different verifier")
    metadata = {
        "status": STATUS,
        "protocol_id": config["protocol_id"],
        "configuration_sha256": config_sha,
        "model_manifest_sha256": sha256(args.model_manifest),
        "source_hashes": source_hashes,
        "topology_contract_sha256": topology_sha,
        "parent_teacher_checkpoint_kind": (
            PRIMARY_KIND if args.stage == "student" else None
        ),
        "parent_teacher_checkpoint_sha256": teacher_sha,
        "resnet50dilated_ade20k_weights_sha256": sha256(resnet_weights),
        "container_image_id": container_image_id,
        "bundle_sha256": bundle_sha,
    }
    if teacher_payload is not None:
        parent_contract = {
            "configuration_sha256": metadata["configuration_sha256"],
            "model_manifest_sha256": metadata["model_manifest_sha256"],
            "source_hashes": metadata["source_hashes"],
            "topology_contract_sha256": metadata["topology_contract_sha256"],
            "container_image_id": metadata["container_image_id"],
            "bundle_sha256": metadata["bundle_sha256"],
            "parent_teacher_checkpoint_kind": None,
            "parent_teacher_checkpoint_sha256": None,
        }
        for key, expected_value in parent_contract.items():
            if teacher_payload.get(key) != expected_value:
                raise ValueError(f"parent teacher checkpoint differs for {key}")
    run_manifest = {
        **metadata,
        "method": method,
        "stage": args.stage,
        "seed": args.seed,
        "configuration": config,
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "historical_limits": {
            "historical_trainer_recovered": False,
            "historical_critic_forward_recovered": False,
            "historical_weights_used_for_initialisation": False,
            "reported_adam_betas": config["adam_betas"],
            "recovered_candidate_adam_betas": config[
                "recovered_candidate_observed_adam_betas"
            ],
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "container_image_id": container_image_id,
        },
        "hardware": {
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }

    if args.output.exists() and any(args.output.iterdir()):
        if not args.resume:
            raise SystemExit(f"refusing non-empty output without --resume: {args.output}")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != run_manifest:
            raise ValueError("resume run manifest differs from the current contract")
    else:
        write_json(manifest_path, run_manifest)

    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    if args.smoke_only:
        model.train()
        features, target, unknown, sample_ids = next(iter(train_loader))
        features = features.to(args.device)
        target = target.to(args.device)
        unknown = unknown.to(args.device)
        features, target, unknown = augment_batch(
            features, target, unknown, augmentation_generator
        )
        target = measured_authoritative_target(features, target, unknown)
        losses = run_batch(
            args.stage,
            model,
            critic,
            projection_teacher,
            extractor,
            adversarial_objective,
            masked_l2_loss,
            model_optimizer,
            critic_optimizer,
            features,
            target,
            unknown,
            config,
        )
        if not all(np.isfinite(value) for value in losses.values()):
            raise ValueError("smoke losses are non-finite")
        write_json(
            args.output / "smoke.json",
            {
                "status": "v4_exact_formal_batch_path_forward_backward_smoke_only",
                "stage": args.stage,
                "seed": args.seed,
                "batch_size": len(sample_ids),
                "first_sample": sample_ids[0],
                "losses": losses,
                "peak_vram_mib": (
                    torch.cuda.max_memory_allocated() / (1024**2)
                    if torch.cuda.is_available()
                    else None
                ),
            },
        )
        return

    last_path = args.output / "last.pt"
    best_primary_path = args.output / "best_primary.pt"
    best_calibration_path = args.output / "best_calibration.pt"
    metrics_path = args.output / "metrics_by_epoch.tsv"
    snapshot_interval = int(config["periodic_snapshot_interval_epochs"])
    start_epoch = 0
    best_primary: dict[str, Any] | None = None
    best_calibration: dict[str, Any] | None = None
    elapsed_training = 0.0
    if not last_path.exists() and not metrics_path.exists():
        atomic_checkpoint(
            last_path,
            resume_payload(
                args.stage,
                args.seed,
                -1,
                best_primary,
                best_calibration,
                elapsed_training,
                model,
                critic,
                projection_teacher,
                model_optimizer,
                critic_optimizer,
                metadata,
            ),
        )
    if last_path.exists():
        if not args.resume:
            raise ValueError("resume checkpoint exists but --resume was not provided")
        resume = torch.load(last_path, map_location="cpu", weights_only=False)
        for key, value in metadata.items():
            if resume.get(key) != value:
                raise ValueError(f"resume checkpoint differs for {key}")
        if resume.get("checkpoint_kind") != RESUME_KIND:
            raise ValueError("resume checkpoint kind differs")
        if resume.get("stage") != args.stage or resume.get("seed") != args.seed:
            raise ValueError("resume checkpoint stage or seed differs")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        critic.load_state_dict(resume["critic_state_dict"], strict=True)
        if projection_teacher is not None:
            expected_projection = resume.get("projection_state_fingerprint")
            if projection_fingerprint(projection_teacher) != expected_projection:
                raise ValueError("fixed projection state differs on resume")
            projection_teacher.freeze_all()
        model_optimizer.load_state_dict(resume["model_optimizer_state_dict"])
        critic_optimizer.load_state_dict(resume["critic_optimizer_state_dict"])
        restore_rng_state(resume["rng_state"])
        current_epoch = int(resume["epoch"])
        start_epoch = current_epoch + 1
        best_primary = resume.get("best_primary_record")
        best_calibration = resume.get("best_calibration_record")
        if best_primary is not None and not isinstance(best_primary, dict):
            raise ValueError("best-primary resume record is invalid")
        if best_calibration is not None and not isinstance(best_calibration, dict):
            raise ValueError("best-calibration resume record is invalid")
        elapsed_training = float(resume["elapsed_training_seconds"])
        reconcile_metrics(metrics_path, current_epoch, args.output)
        verify_or_rebuild_selected(
            best_primary_path,
            PRIMARY_KIND,
            best_primary,
            current_epoch,
            args.stage,
            args.seed,
            model,
            projection_teacher,
            metadata,
        )
        verify_or_rebuild_selected(
            best_calibration_path,
            CALIBRATION_KIND,
            best_calibration,
            current_epoch,
            args.stage,
            args.seed,
            model,
            projection_teacher,
            metadata,
        )
        reconcile_periodic_snapshots(
            args.output,
            current_epoch,
            snapshot_interval,
            args.stage,
            args.seed,
            model,
            projection_teacher,
            metadata,
        )
        with (args.output / "resume_events.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps(
                    {
                        "resumed_from_epoch": current_epoch,
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    elif metrics_path.exists():
        raise ValueError("epoch metrics exist without a resume checkpoint")

    training_started = time.perf_counter()
    for epoch in range(start_epoch, int(config["maximum_epochs"])):
        epoch_started = time.perf_counter()
        shuffle_generator.manual_seed(args.seed + epoch)
        augmentation_generator.manual_seed(args.seed + 1_000_000 + epoch)
        model.train()
        if projection_teacher is not None:
            projection_teacher.freeze_all()
        totals = {
            "generator": 0.0,
            "critic": 0.0,
            "adversarial": 0.0,
            "feature": 0.0,
            "pixel": 0.0,
            "distill": 0.0,
        }
        batches = 0
        for features, target, unknown, _ in train_loader:
            features = features.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            unknown = unknown.to(args.device, non_blocking=True)
            features, target, unknown = augment_batch(
                features, target, unknown, augmentation_generator
            )
            target = measured_authoritative_target(features, target, unknown)
            values = run_batch(
                args.stage,
                model,
                critic,
                projection_teacher,
                extractor,
                adversarial_objective,
                masked_l2_loss,
                model_optimizer,
                critic_optimizer,
                features,
                target,
                unknown,
                config,
            )
            for key, value in values.items():
                totals[key] += value
            batches += 1
        if batches == 0:
            raise ValueError("training loader produced no batches")

        validation = validation_metrics(
            model,
            validation_loader,
            args.device,
            threshold=float(config["decision_threshold"]),
            ece_bins=int(config["ece_bins"]),
        )
        candidate = metric_record(epoch, validation)
        selected_primary = primary_is_better(candidate, best_primary)
        selected_calibration = calibration_is_better(candidate, best_calibration)
        if selected_primary:
            best_primary = candidate
        if selected_calibration:
            best_calibration = candidate
        row = {
            "epoch": epoch,
            "train_generator_loss": totals["generator"] / batches,
            "train_critic_loss": totals["critic"] / batches,
            "train_adversarial": totals["adversarial"] / batches,
            "train_feature_reconstruction": totals["feature"] / batches,
            "train_masked_pixel_l2": totals["pixel"] / batches,
            "train_feature_distillation": totals["distill"] / batches,
            "validation_unknown_count": validation["unknown_count"],
            "validation_unknown_tp": validation["unknown_tp"],
            "validation_unknown_fp": validation["unknown_fp"],
            "validation_unknown_fn": validation["unknown_fn"],
            "validation_unknown_tn": validation["unknown_tn"],
            "validation_unknown_precision": validation["unknown_precision"],
            "validation_unknown_recall": validation["unknown_recall"],
            "validation_unknown_f1": validation["unknown_f1"],
            "validation_unknown_iou": validation["unknown_iou"],
            "validation_unknown_bce": validation["unknown_bce"],
            "validation_unknown_brier": validation["unknown_brier"],
            "validation_unknown_ece15": validation["unknown_ece15"],
            "epoch_seconds": time.perf_counter() - epoch_started,
            "selected_best_primary": int(selected_primary),
            "selected_best_calibration": int(selected_calibration),
        }
        write_metric(metrics_path, row)
        elapsed_now = elapsed_training + (time.perf_counter() - training_started)
        atomic_checkpoint(
            last_path,
            resume_payload(
                args.stage,
                args.seed,
                epoch,
                best_primary,
                best_calibration,
                elapsed_now,
                model,
                critic,
                projection_teacher,
                model_optimizer,
                critic_optimizer,
                metadata,
            ),
        )
        if selected_primary:
            atomic_checkpoint(
                best_primary_path,
                generator_checkpoint_payload(
                    PRIMARY_KIND,
                    args.stage,
                    args.seed,
                    candidate,
                    model,
                    projection_teacher,
                    metadata,
                ),
            )
        if selected_calibration:
            atomic_checkpoint(
                best_calibration_path,
                generator_checkpoint_payload(
                    CALIBRATION_KIND,
                    args.stage,
                    args.seed,
                    candidate,
                    model,
                    projection_teacher,
                    metadata,
                ),
            )
        if (epoch + 1) % snapshot_interval == 0:
            snapshot_path = args.output / snapshot_relative_path(epoch)
            snapshot_path.parent.mkdir(exist_ok=True)
            if snapshot_path.exists():
                raise ValueError(f"refusing to overwrite snapshot {snapshot_path.name}")
            atomic_checkpoint(
                snapshot_path,
                periodic_snapshot_payload(
                    args.stage,
                    args.seed,
                    epoch,
                    model,
                    projection_teacher,
                    metadata,
                ),
            )
        print(json.dumps(row, sort_keys=True), flush=True)

    if best_primary is None or best_calibration is None:
        raise ValueError("formal run completed without selected checkpoints")
    selection_manifest_path = args.output / "selection_manifest.json"
    selection_manifest = write_selection_manifest(
        metrics_path,
        best_primary_path,
        best_calibration_path,
        selection_manifest_path,
        str(config["protocol_id"]),
        int(config["maximum_epochs"]),
        args.stage,
        args.seed,
        metadata,
        projection_fingerprint(projection_teacher),
    )
    audited_primary = selection_manifest["primary"]["validation_metrics"]
    audited_calibration = selection_manifest["calibration"]["validation_metrics"]
    if best_primary != audited_primary or best_calibration != audited_calibration:
        raise ValueError("resume-state selection differs from independent TSV argbest")
    best_primary = audited_primary
    best_calibration = audited_calibration
    selected = torch.load(
        best_primary_path, map_location="cpu", weights_only=False
    )
    if selected.get("checkpoint_kind") != PRIMARY_KIND:
        raise ValueError("deployment checkpoint is not best_primary.pt")
    model.load_state_dict(selected["model_state_dict"], strict=True)
    model.to(args.device).eval()
    total_training_seconds = elapsed_training + (
        time.perf_counter() - training_started
    )
    latency = latency_profile(model, validation_data, config, args.device)
    resource = {
        "deployed_parameter_count": trainable_parameter_count(model),
        "deployed_checkpoint": "best_primary.pt",
        "deployed_checkpoint_sha256": sha256(best_primary_path),
        "training_time_s": total_training_seconds,
        "latency_batch1": latency,
        "peak_vram_current_process_mib": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if torch.cuda.is_available()
            else None
        ),
    }
    write_json(args.output / "resource_profile.json", resource)

    snapshot_paths = [
        snapshot_relative_path(epoch)
        for epoch in range(
            snapshot_interval - 1,
            int(config["maximum_epochs"]),
            snapshot_interval,
        )
    ]
    if not all((args.output / path).is_file() for path in snapshot_paths):
        raise ValueError("formal run is missing periodic snapshots")
    output_names: list[str] = [
        "run_manifest.json",
        "metrics_by_epoch.tsv",
        "best_primary.pt",
        "best_calibration.pt",
        "last.pt",
        "resource_profile.json",
        "selection_manifest.json",
        *(path.as_posix() for path in snapshot_paths),
    ]
    completion: dict[str, object] = {
        "status": STATUS,
        "protocol_id": config["protocol_id"],
        "completed": True,
        "method": method,
        "stage": args.stage,
        "seed": args.seed,
        "last_epoch": int(config["maximum_epochs"]) - 1,
        "selection_metric": config["selection_metric"],
        "selection_tiebreaker": config["selection_tiebreaker"],
        "best_primary_epoch": int(best_primary["epoch"]),
        "best_primary_record": best_primary,
        "best_primary_checkpoint_sha256": sha256(best_primary_path),
        "best_calibration_epoch": int(best_calibration["epoch"]),
        "best_calibration_record": best_calibration,
        "best_calibration_checkpoint_sha256": sha256(best_calibration_path),
        "last_checkpoint_sha256": sha256(last_path),
        "periodic_snapshot_interval_epochs": snapshot_interval,
        "periodic_snapshot_sha256": {
            path.as_posix(): sha256(args.output / path) for path in snapshot_paths
        },
        "selection_manifest_sha256": sha256(selection_manifest_path),
        "topology_contract_sha256": topology_sha,
        "parent_teacher_checkpoint_kind": metadata[
            "parent_teacher_checkpoint_kind"
        ],
        "parent_teacher_checkpoint_sha256": teacher_sha,
        "best_primary_validation_unknown_bce": float(
            best_primary["unknown_bce"]
        ),
    }
    if args.stage == "student":
        test_data = CachedDataset(args.model_data_root, args.model_manifest, "test")
        if len(test_data) != expected_counts["test"]:
            raise ValueError("test sample count differs from the frozen contract")
        validation_predictions = export_predictions(
            model,
            validation_data,
            args.seed,
            "validation",
            config,
            args.device,
            args.output,
        )
        test_predictions = export_predictions(
            model,
            test_data,
            args.seed,
            "test",
            config,
            args.device,
            args.output,
        )
        completion.update(
            {
                "validation_predictions_sha256": sha256(validation_predictions),
                "test_predictions_sha256": sha256(test_predictions),
            }
        )
        output_names.extend(
            ["predictions_validation.tsv", "predictions_test.tsv"]
        )
    write_json(args.output / "completion.json", completion)
    output_names.append("completion.json")
    sha_manifest = args.output / "SHA256SUMS"
    sha_manifest.write_text(
        "".join(
            f"{sha256(args.output / name)}  {name}\n" for name in output_names
        ),
        encoding="utf-8",
    )
    write_json(
        args.output / "RUN_COMPLETE.json",
        {
            "completed": True,
            "completion_json_sha256": sha256(args.output / "completion.json"),
            "sha256_manifest_sha256": sha256(sha_manifest),
        },
    )


if __name__ == "__main__":
    main()
