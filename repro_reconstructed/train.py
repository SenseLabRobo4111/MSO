#!/usr/bin/env python3
"""Canonical entry for the explicitly reconstructed MSO training path."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

from mso_reconstructed.artifacts import sha256_file, unwrap_generator_state  # noqa: E402
from mso_reconstructed.data import ManifestDataset  # noqa: E402
from mso_reconstructed.objective import (  # noqa: E402
    feature_distillation,
    masked_pixel_l2,
    multiscale_reconstruction,
)
from sensemap.explore_model.SenseMapNet import (  # noqa: E402
    DistillMapNetDeconv,
    TeacherMapNet2,
)
from sensemap.explore_model.critic_model import CriticModel  # noqa: E402


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args():
    default_config = PACKAGE_ROOT / "config" / "reconstructed_default.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("teacher", "student"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--skip-raw-hash-verification",
        action="store_true",
        help="Diagnostic-only bypass; outputs are marked as input hashes not verified.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolved_config(args) -> dict:
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.beta1 is not None:
        config["adam_betas"][0] = args.beta1
    if args.beta2 is not None:
        config["adam_betas"][1] = args.beta2
    if args.seed is not None:
        config["seed"] = args.seed
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_teacher(path: Path, device: str, dim: int) -> TeacherMapNet2:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = unwrap_generator_state(payload)
    teacher = TeacherMapNet2(image_size=256, dim=dim)
    teacher.load_state_dict(state, strict=True)
    teacher.eval().to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def save_model(
    path: Path,
    model,
    stage: str,
    epoch: int,
    config: dict,
    manifest: Path,
    dataset_integrity: str,
):
    payload = {
        "status": "reconstructed",
        "stage": stage,
        "epoch": epoch,
        "config": config,
        "manifest_sha256": sha256_file(manifest),
        "dataset_integrity": dataset_integrity,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    config = resolved_config(args)
    set_seed(int(config["seed"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ManifestDataset(
        args.dataset_root,
        args.manifest,
        "train",
        image_size=int(config["image_size"]),
        verify_raw_hashes=not args.skip_raw_hash_verification,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=args.device.startswith("cuda"),
    )

    if args.stage == "teacher":
        model = TeacherMapNet2(
            image_size=int(config["image_size"]), dim=int(config["teacher_dim"])
        ).to(args.device)
        teacher = None
    else:
        if args.teacher_weights is None:
            raise SystemExit("--teacher-weights is required for student reconstruction")
        model = DistillMapNetDeconv(
            image_size=int(config["image_size"]), dim=int(config["student_dim"])
        ).to(args.device)
        teacher = load_teacher(
            args.teacher_weights, args.device, int(config["teacher_dim"])
        )
    critic = CriticModel().to(args.device)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if args.stage == "student" and trainable != 342_771:
        raise ValueError(f"Student parameter count changed: {trainable}")

    if args.dry_run:
        features, labels, masks, sample_ids = next(iter(loader))
        with torch.inference_mode():
            output = model(features.to(args.device))[0]
        report = {
            "status": "dry_run_only",
            "stage": args.stage,
            "dataset_samples": len(dataset),
            "batch_shape": list(features.shape),
            "output_shape": list(output.shape),
            "mask_shape": list(masks.shape),
            "first_sample": sample_ids[0],
            "trainable_parameter_count": trainable,
            "manifest_sha256": sha256_file(args.manifest),
            "dataset_integrity": dataset.integrity_status,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return

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
    weights = config["loss_weights"]

    for epoch in range(int(config["epochs"])):
        model.train()
        running = {"generator": 0.0, "critic": 0.0, "batches": 0}
        for features, labels, masks, _ in loader:
            features = features.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            masks = masks.to(args.device, non_blocking=True)

            critic_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_for_critic = model(features)[0]
            real_logits = critic(labels, masks)
            fake_logits = critic(fake_for_critic, masks)
            critic_loss = functional.binary_cross_entropy_with_logits(
                real_logits, torch.ones_like(real_logits)
            ) + functional.binary_cross_entropy_with_logits(
                fake_logits, torch.zeros_like(fake_logits)
            )
            critic_loss.backward()
            critic_optimizer.step()

            model_optimizer.zero_grad(set_to_none=True)
            prediction, *student_features = model(features)
            adversarial = functional.binary_cross_entropy_with_logits(
                critic(prediction, masks), torch.ones_like(fake_logits)
            )
            feature_reconstruction = multiscale_reconstruction(prediction, labels, masks)
            pixel = masked_pixel_l2(prediction, labels, masks)
            distillation = prediction.new_zeros(())
            if teacher is not None:
                with torch.no_grad():
                    _, *teacher_features = teacher(features)
                distillation = feature_distillation(
                    tuple(student_features), tuple(teacher_features)
                )
            generator_loss = (
                float(weights["adversarial"]) * adversarial
                + float(weights["feature_reconstruction"]) * feature_reconstruction
                + float(weights["pixel_l2"]) * pixel
                + float(weights["distillation"]) * distillation
            )
            generator_loss.backward()
            model_optimizer.step()

            running["generator"] += float(generator_loss.detach())
            running["critic"] += float(critic_loss.detach())
            running["batches"] += 1

        summary = {
            "epoch": epoch,
            "generator_loss": running["generator"] / running["batches"],
            "critic_loss": running["critic"] / running["batches"],
            "batches": running["batches"],
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        name = "teacher_last.pt" if args.stage == "teacher" else "student_last.pt"
        save_model(
            args.output_dir / name,
            model,
            args.stage,
            epoch,
            config,
            args.manifest,
            dataset.integrity_status,
        )


if __name__ == "__main__":
    main()
