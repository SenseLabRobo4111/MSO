#!/usr/bin/env python3
"""Train the prospective MSO paper-equation reconstruction.

This is a new, hash-bound teacher-to-student experiment on the preserved
benchmark split.  It implements the reported objectives but does not claim to
be the unavailable historical trainer or manuscript checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from mso_full_components import (
    ProjectedTeacher,
    ReconstructedFFCPatchCritic,
    trainable_parameter_count,
)
from train_unified_adapter import CachedDataset, worker_seed


STATUS = "prospective_paper_equation_reconstruction_not_historical"
TOPOLOGY_STATUS = "recovered_topology_verified_weights_not_reused"
TEACHER_SEED = 101
STUDENT_SEEDS = (11, 23, 37, 53, 71)
METHOD_TEACHER = "MSO-Paper-Equation-Teacher-Reconstructed"
METHOD_STUDENT = "MSO-Paper-Equation-Reconstructed"
EXPECTED_RESNET_COMMON_TENSORS = 275


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_checkpoint(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(value: dict[str, object]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def validate_config(config: dict[str, object]) -> None:
    expected = {
        "protocol_id": "mso_paper_equation_reconstruction_v3",
        "status": STATUS,
        "historical_trainer_recovered": False,
        "teacher_seed": TEACHER_SEED,
        "projection_seed": 20260809,
        "image_size": 256,
        "batch_size": 16,
        "num_workers": 8,
        "maximum_epochs": 500,
        "learning_rate": 0.001,
        "adam_betas": [0.5, 0.999],
        "weight_decay": 0.0,
        "mixed_precision": "float32",
        "student_seeds": list(STUDENT_SEEDS),
        "teacher_base_width": 32,
        "student_base_width": 4,
        "drop_last": False,
        "decision_threshold": 0.5,
        "latency_warmup": 20,
        "latency_repetitions": 100,
        "selection_metric": "validation_unknown_bce",
        "selection_direction": "minimise",
        "selection_policy": (
            "retain_validation_best_and_epoch_499_last_without_early_stopping"
        ),
        "checkpoint_policy": (
            "atomic_epoch_boundary_checkpoint_with_automatic_resume"
        ),
        "projection_policy": (
            "fixed_seed_random_initialisation_frozen_and_shared_across_students"
        ),
        "teacher_training": (
            "deterministic_random_initialisation_on_current_train_split"
        ),
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ValueError(f"configuration contract differs for {key}")
    if config.get("loss_weights") != {
        "adversarial": 10.0,
        "feature_reconstruction": 30.0,
        "masked_pixel_l2": 1.0,
        "feature_distillation": 5.0,
    }:
        raise ValueError("loss-weight contract differs from the reported method")
    if config.get("augmentation") != [
        "random_horizontal_flip",
        "random_vertical_flip",
        "random_rot90",
    ]:
        raise ValueError("augmentation contract differs")
    if config.get("augmentation_probabilities") != {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.5,
        "rot90_quarter_turns": [0, 1, 2, 3],
    }:
        raise ValueError("augmentation probability contract differs")
    discriminator = config.get("discriminator")
    if not isinstance(discriminator, dict) or discriminator != {
        "implementation": (
            "recovered_four_stage_ffc_state_tree_with_declared_channel_mean_readout"
        ),
        "initialization": "deterministic_random_per_training_stage",
        "adversarial_source_sha256": (
            "813364230c4a91af29a4cc866f57b75a10da5240bfadfa49ade2bbecb77178ca"
        ),
        "gp_coef": 0.01,
        "mask_as_fake_target": False,
        "allow_scale_mask": True,
        "mask_scale_mode": "maxpool",
        "mask_normalization": "mean_over_mask_intersecting_patches",
        "r1_patch_scope": "all_real_patches_archived_lama_regularizer",
        "use_unmasked_for_generator": False,
        "use_unmasked_for_discriminator": False,
        "historical_ffc_discriminator_recovered": False,
    }:
        raise ValueError("discriminator contract differs")


def measured_authoritative_target(
    features: torch.Tensor, target: torch.Tensor, unknown: torch.Tensor
) -> torch.Tensor:
    return torch.where(unknown >= 0.5, target, features[:, 0:1])


def augment_batch(
    features: torch.Tensor,
    target: torch.Tensor,
    unknown: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    transformed_features = []
    transformed_targets = []
    transformed_unknown = []
    for index in range(features.shape[0]):
        values = [features[index], target[index], unknown[index]]
        quarter_turns = int(torch.randint(0, 4, (1,), generator=generator).item())
        if quarter_turns:
            values = [torch.rot90(value, quarter_turns, (-2, -1)) for value in values]
        if bool(torch.randint(0, 2, (1,), generator=generator).item()):
            values = [torch.flip(value, (-1,)) for value in values]
        if bool(torch.randint(0, 2, (1,), generator=generator).item()):
            values = [torch.flip(value, (-2,)) for value in values]
        transformed_features.append(values[0])
        transformed_targets.append(values[1])
        transformed_unknown.append(values[2])
    return (
        torch.stack(transformed_features),
        torch.stack(transformed_targets),
        torch.stack(transformed_unknown),
    )


def distillation_loss(
    student_features: tuple[torch.Tensor, ...],
    teacher_features: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if len(student_features) != 4 or len(teacher_features) != 4:
        raise ValueError("four aligned feature taps are required")
    losses = []
    for student, teacher in zip(student_features, teacher_features):
        if student.shape != teacher.shape:
            raise ValueError(
                f"distillation shape mismatch: {student.shape} vs {teacher.shape}"
            )
        difference = student.float() - teacher.detach().float()
        losses.append(torch.linalg.vector_norm(difference) / difference.numel())
    return torch.stack(losses).sum()


def validation_metrics(
    model: nn.Module, loader: DataLoader, device: str
) -> dict[str, float]:
    bce_sum = 0.0
    count = 0
    tp = fp = fn = 0
    model.eval()
    with torch.inference_mode():
        for features, target, unknown, _ in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            unknown = unknown.to(device, non_blocking=True)
            target = measured_authoritative_target(features, target, unknown)
            probability = model(features)[0].float().clamp(1e-6, 1.0 - 1e-6)
            values = functional.binary_cross_entropy(
                probability, target, reduction="none"
            )
            bce_sum += float((values * unknown).sum().cpu())
            count += int(unknown.sum().cpu())
            predicted = probability >= 0.5
            truth = target >= 0.5
            mask = unknown >= 0.5
            tp += int((predicted & truth & mask).sum().cpu())
            fp += int((predicted & ~truth & mask).sum().cpu())
            fn += int((~predicted & truth & mask).sum().cpu())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "unknown_bce": bce_sum / max(1, count),
        "unknown_precision": precision,
        "unknown_recall": recall,
        "unknown_f1": 2 * precision * recall / max(1e-12, precision + recall),
        "unknown_iou": tp / max(1, tp + fp + fn),
    }


def require_finite_gradient_norm(parameters: Iterable[nn.Parameter], name: str) -> None:
    values = [parameter for parameter in parameters if parameter.grad is not None]
    if not values:
        raise ValueError(f"{name} produced no gradients")
    torch.nn.utils.clip_grad_norm_(
        values, max_norm=float("inf"), error_if_nonfinite=True
    )


def masked_patch_mean(values: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
    if values.shape != patch_mask.shape:
        raise ValueError("patch values and mask must have identical shapes")
    denominator = patch_mask.sum()
    return (values * patch_mask).sum() / denominator.clamp_min(1.0)


def r1_gradient_penalty(
    real_logits: torch.Tensor, real_batch: torch.Tensor
) -> torch.Tensor:
    gradient = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_batch,
        create_graph=True,
    )[0]
    return gradient.reshape(gradient.shape[0], -1).norm(2, dim=1).square().mean()


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(f"{name}\t{value.dtype}\t{tuple(value.shape)}\n".encode())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def projection_fingerprint(module: ProjectedTeacher | None) -> str | None:
    if module is None:
        return None
    state = {
        name: value
        for name, value in module.state_dict().items()
        if not name.startswith("core.")
    }
    return state_fingerprint(state)


def verify_dependencies(root: Path, manifest: Path, expected_sha: str) -> None:
    if sha256(manifest) != expected_sha:
        raise ValueError("LaMa dependency manifest hash differs from the frozen config")
    with manifest.open("r", encoding="utf-8") as stream:
        rows = [line.rstrip("\n").split("  ", 1) for line in stream if line.strip()]
    if len(rows) != 21:
        raise ValueError("LaMa dependency manifest must contain exactly 21 files")
    for expected, relative in rows:
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"archived dependency hash differs for {relative}")


def verify_resnet_loaded(module: nn.Module, weights_path: Path) -> None:
    source = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(source, dict):
        raise ValueError("ADE20K encoder weights are not a state mapping")
    loaded = module.state_dict()
    common = {
        name: value
        for name, value in source.items()
        if name in loaded and tuple(value.shape) == tuple(loaded[name].shape)
    }
    if len(common) != EXPECTED_RESNET_COMMON_TENSORS:
        raise ValueError("ADE20K encoder common-tensor count differs")
    if not all(torch.equal(value.cpu(), loaded[name].cpu()) for name, value in common.items()):
        raise ValueError("ADE20K encoder did not load the archived tensors exactly")
    with torch.inference_mode():
        outputs = module(torch.zeros(1, 3, 256, 256), return_feature_maps=True)
    expected = (
        (1, 256, 64, 64),
        (1, 512, 32, 32),
        (1, 1024, 32, 32),
        (1, 2048, 32, 32),
    )
    if tuple(tuple(value.shape) for value in outputs) != expected:
        raise ValueError("ADE20K encoder feature geometry differs")


def write_metric(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(row), delimiter="\t", lineterminator="\n"
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def reconcile_metrics(path: Path, last_epoch: int, output: Path) -> None:
    if not path.exists():
        if last_epoch >= 0:
            raise ValueError("resume checkpoint exists but epoch metrics are missing")
        return
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
        fields = reader.fieldnames
    if fields is None:
        raise ValueError("epoch metrics have no header")
    retained = [row for row in rows if int(row["epoch"]) <= last_epoch]
    discarded = [row for row in rows if int(row["epoch"]) > last_epoch]
    if len(retained) != last_epoch + 1:
        raise ValueError("epoch metrics are not contiguous through the resume checkpoint")
    if discarded:
        audit = output / "discarded_uncheckpointed_metrics.tsv"
        with audit.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
            )
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerows(discarded)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(retained)
        temporary.replace(path)


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


def latency_profile(
    model: nn.Module,
    dataset: CachedDataset,
    config: dict[str, object],
    device: str,
) -> dict[str, object]:
    features = dataset[0][0].unsqueeze(0).to(device)
    warmup = int(config["latency_warmup"])
    repetitions = int(config["latency_repetitions"])
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(features)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        values = []
        for _ in range(repetitions):
            started = time.perf_counter()
            model(features)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            values.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(values, dtype=float)
    return {
        "batch_size": 1,
        "warmup_repetitions": warmup,
        "timed_repetitions": repetitions,
        "median_ms": float(np.median(array)),
        "p10_ms": float(np.percentile(array, 10)),
        "p90_ms": float(np.percentile(array, 90)),
        "minimum_ms": float(array.min()),
        "maximum_ms": float(array.max()),
    }


def best_payload(
    stage: str,
    seed: int,
    epoch: int,
    metric: float,
    model: nn.Module,
    projection_teacher: ProjectedTeacher | None,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        **metadata,
        "checkpoint_kind": "validation_best_generator",
        "stage": stage,
        "seed": seed,
        "epoch": epoch,
        "best_validation_unknown_bce": metric,
        "model_state_dict": model.state_dict(),
        "projection_state_fingerprint": projection_fingerprint(projection_teacher),
    }


def resume_payload(
    stage: str,
    seed: int,
    epoch: int,
    best_epoch: int,
    best_metric: float,
    elapsed_seconds: float,
    model: nn.Module,
    critic: nn.Module,
    projection_teacher: ProjectedTeacher | None,
    model_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        **metadata,
        "checkpoint_kind": "epoch_boundary_resume_state",
        "stage": stage,
        "seed": seed,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_unknown_bce": best_metric,
        "elapsed_training_seconds": elapsed_seconds,
        "model_state_dict": model.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "projection_state_fingerprint": projection_fingerprint(projection_teacher),
        "model_optimizer_state_dict": model_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "rng_state": capture_rng_state(),
    }


def run_batch(
    stage: str,
    model: nn.Module,
    critic: nn.Module,
    projection_teacher: ProjectedTeacher | None,
    extractor: nn.Module,
    adversarial_objective: object,
    masked_l2_loss: object,
    model_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    target: torch.Tensor,
    unknown: torch.Tensor,
    config: dict[str, object],
) -> dict[str, float]:
    probability, *student_features = model(features)
    completed = probability * unknown + features[:, 0:1] * (1.0 - unknown)

    critic.train()
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    critic_optimizer.zero_grad(set_to_none=True)
    real_batch = target.detach().clone()
    fake_for_critic = completed.detach()
    real_batch.requires_grad_(True)
    real_logits = critic(real_batch, unknown)
    fake_logits = critic(fake_for_critic, unknown)
    patch_mask = adversarial_objective.interpolate_mask(
        unknown, fake_logits.shape[-2:]
    )
    critic_loss = (
        masked_patch_mean(functional.softplus(-real_logits), patch_mask)
        + masked_patch_mean(functional.softplus(fake_logits), patch_mask)
        + float(config["discriminator"]["gp_coef"])
        * r1_gradient_penalty(real_logits, real_batch)
    )
    real_batch.requires_grad_(False)
    if not bool(torch.isfinite(critic_loss)):
        raise ValueError("non-finite critic loss")
    critic_loss.backward()
    require_finite_gradient_norm(critic.parameters(), "critic")
    critic_optimizer.step()

    model_optimizer.zero_grad(set_to_none=True)
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    critic.eval()
    fake_logits_for_generator = critic(completed, unknown)
    generator_patch_mask = adversarial_objective.interpolate_mask(
        unknown, fake_logits_for_generator.shape[-2:]
    )
    adversarial = float(config["loss_weights"]["adversarial"]) * masked_patch_mean(
        functional.softplus(-fake_logits_for_generator), generator_patch_mask
    )
    feature = extractor(probability, target)
    pixel = masked_l2_loss(
        probability, target, unknown, weight_known=0.0, weight_missing=1.0
    ) / unknown.mean().clamp_min(1e-8)
    distill = probability.new_zeros(())
    if stage == "student":
        if projection_teacher is None:
            raise ValueError("student stage requires the fixed projected teacher")
        with torch.no_grad():
            _, *teacher_features = projection_teacher(features)
        distill = distillation_loss(
            tuple(student_features), tuple(teacher_features)
        )
    total = (
        adversarial
        + float(config["loss_weights"]["feature_reconstruction"]) * feature
        + float(config["loss_weights"]["masked_pixel_l2"]) * pixel
    )
    if stage == "student":
        total = total + float(config["loss_weights"]["feature_distillation"]) * distill
    if not bool(torch.isfinite(total)):
        raise ValueError("non-finite generator loss")
    total.backward()
    require_finite_gradient_norm(model.parameters(), "generator")
    model_optimizer.step()
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    critic.train()
    return {
        "generator": float(total.detach().cpu()),
        "critic": float(critic_loss.detach().cpu()),
        "adversarial": float(adversarial.detach().cpu()),
        "feature": float(feature.detach().cpu()),
        "pixel": float(pixel.detach().cpu()),
        "distill": float(distill.detach().cpu()),
    }


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
    expected_seed = TEACHER_SEED if args.stage == "teacher" else STUDENT_SEEDS
    if args.stage == "teacher" and args.seed != expected_seed:
        raise ValueError("teacher seed differs from the frozen contract")
    if args.stage == "student" and args.seed not in expected_seed:
        raise ValueError("student seed differs from the frozen contract")
    if args.stage == "student" and args.teacher_checkpoint is None:
        raise ValueError("student stage requires the prospective teacher checkpoint")
    if args.stage == "teacher" and args.teacher_checkpoint is not None:
        raise ValueError("teacher stage must start from deterministic random weights")
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
        if teacher_payload.get("checkpoint_kind") != "validation_best_generator":
            raise ValueError("student parent is not a validation-best checkpoint")
        if teacher_payload.get("stage") != "teacher":
            raise ValueError("student parent checkpoint is not a teacher checkpoint")
        if int(teacher_payload.get("seed", -1)) != TEACHER_SEED:
            raise ValueError("student parent checkpoint has the wrong teacher seed")
        if teacher_payload.get("protocol_id") != config["protocol_id"]:
            raise ValueError("student parent teacher protocol differs")
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
        "trainer": sha256(Path(__file__)),
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
                "status": "exact_formal_batch_path_forward_backward_smoke_only",
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
    best_path = args.output / "best.pt"
    metrics_path = args.output / "metrics_by_epoch.tsv"
    start_epoch = 0
    best_metric = float("inf")
    best_epoch = -1
    elapsed_training = 0.0
    if not last_path.exists() and not metrics_path.exists():
        atomic_checkpoint(
            last_path,
            resume_payload(
                args.stage,
                args.seed,
                -1,
                best_epoch,
                best_metric,
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
        start_epoch = int(resume["epoch"]) + 1
        best_epoch = int(resume["best_epoch"])
        best_metric = float(resume["best_validation_unknown_bce"])
        elapsed_training = float(resume["elapsed_training_seconds"])
        reconcile_metrics(metrics_path, start_epoch - 1, args.output)
        with (args.output / "resume_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "resumed_from_epoch": start_epoch - 1,
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    elif metrics_path.exists():
        raise ValueError("epoch metrics exist without a resume checkpoint")

    if best_epoch >= 0:
        rebuild_best = not best_path.exists()
        if best_path.exists():
            existing_best = torch.load(
                best_path, map_location="cpu", weights_only=False
            )
            best_contract = {
                **metadata,
                "checkpoint_kind": "validation_best_generator",
                "stage": args.stage,
                "seed": args.seed,
                "epoch": best_epoch,
                "best_validation_unknown_bce": best_metric,
                "projection_state_fingerprint": projection_fingerprint(
                    projection_teacher
                ),
            }
            rebuild_best = any(
                existing_best.get(key) != expected_value
                for key, expected_value in best_contract.items()
            )
        if rebuild_best:
            if start_epoch - 1 != best_epoch:
                raise ValueError(
                    "best checkpoint differs and cannot be reconstructed from last.pt"
                )
            atomic_checkpoint(
                best_path,
                best_payload(
                    args.stage,
                    args.seed,
                    best_epoch,
                    best_metric,
                    model,
                    projection_teacher,
                    metadata,
                ),
            )

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

        validation = validation_metrics(model, validation_loader, args.device)
        selected = validation["unknown_bce"] < best_metric
        if selected:
            best_metric = validation["unknown_bce"]
            best_epoch = epoch
        row = {
            "epoch": epoch,
            "train_generator_loss": totals["generator"] / batches,
            "train_critic_loss": totals["critic"] / batches,
            "train_adversarial": totals["adversarial"] / batches,
            "train_feature_reconstruction": totals["feature"] / batches,
            "train_masked_pixel_l2": totals["pixel"] / batches,
            "train_feature_distillation": totals["distill"] / batches,
            "validation_unknown_bce": validation["unknown_bce"],
            "validation_unknown_precision": validation["unknown_precision"],
            "validation_unknown_recall": validation["unknown_recall"],
            "validation_unknown_f1": validation["unknown_f1"],
            "validation_unknown_iou": validation["unknown_iou"],
            "epoch_seconds": time.perf_counter() - epoch_started,
            "selected_best": int(selected),
        }
        write_metric(metrics_path, row)
        elapsed_now = elapsed_training + (time.perf_counter() - training_started)
        atomic_checkpoint(
            last_path,
            resume_payload(
                args.stage,
                args.seed,
                epoch,
                best_epoch,
                best_metric,
                elapsed_now,
                model,
                critic,
                projection_teacher,
                model_optimizer,
                critic_optimizer,
                metadata,
            ),
        )
        if selected:
            atomic_checkpoint(
                best_path,
                best_payload(
                    args.stage,
                    args.seed,
                    epoch,
                    best_metric,
                    model,
                    projection_teacher,
                    metadata,
                ),
            )
        print(json.dumps(row, sort_keys=True), flush=True)

    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    model.to(args.device).eval()
    total_training_seconds = elapsed_training + (time.perf_counter() - training_started)
    latency = latency_profile(model, validation_data, config, args.device)
    resource = {
        "deployed_parameter_count": trainable_parameter_count(model),
        "training_time_s": total_training_seconds,
        "latency_batch1": latency,
        "peak_vram_current_process_mib": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if torch.cuda.is_available()
            else None
        ),
    }
    write_json(args.output / "resource_profile.json", resource)

    output_names = [
        "run_manifest.json",
        "metrics_by_epoch.tsv",
        "best.pt",
        "last.pt",
        "resource_profile.json",
    ]
    completion: dict[str, object] = {
        "status": STATUS,
        "completed": True,
        "method": method,
        "stage": args.stage,
        "seed": args.seed,
        "last_epoch": int(config["maximum_epochs"]) - 1,
        "best_epoch": best_epoch,
        "best_validation_unknown_bce": best_metric,
        "best_checkpoint_sha256": sha256(best_path),
        "last_checkpoint_sha256": sha256(last_path),
        "topology_contract_sha256": topology_sha,
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
                "parent_teacher_checkpoint_sha256": teacher_sha,
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
        "".join(f"{sha256(args.output / name)}  {name}\n" for name in output_names),
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
