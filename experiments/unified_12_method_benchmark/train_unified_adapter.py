#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from unified_models import build_model


SEEDS = (11, 23, 37, 53, 71)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def worker_seed(_: int) -> None:
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


class CachedDataset(Dataset):
    def __init__(self, root: Path, manifest: Path, split: str) -> None:
        self.root = root.resolve()
        with manifest.open("r", encoding="utf-8", newline="") as stream:
            all_rows = list(csv.DictReader(stream, delimiter="\t"))
        self.rows = [row for row in all_rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows for split {split!r}")
        seen: set[str] = set()
        for row in self.rows:
            if not row["sample_id"] or row["sample_id"] in seen:
                raise ValueError(f"missing or duplicate sample ID: {row['sample_id']!r}")
            seen.add(row["sample_id"])
            for key in ("observation_relpath", "target_relpath"):
                path = (self.root / row[key]).resolve()
                if not path.is_relative_to(self.root):
                    raise ValueError(f"path escapes model-data root: {row[key]}")
            observation = self.root / row["observation_relpath"]
            target = self.root / row["target_relpath"]
            if sha256(observation) != row["observation_sha256"]:
                raise ValueError(f"observation hash mismatch: {row['sample_id']}")
            if sha256(target) != row["target_sha256"]:
                raise ValueError(f"target hash mismatch: {row['sample_id']}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.root / row["observation_relpath"]) as image:
            observation = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(self.root / row["target_relpath"]) as image:
            target = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        if observation.shape != (256, 256, 3) or target.shape != (256, 256):
            raise ValueError(f"invalid cached shape: {row['sample_id']}")
        occupied = np.all(observation == [255, 0, 0], axis=2)
        unknown = np.all(observation == [0, 255, 0], axis=2)
        free = np.all(observation == [0, 0, 255], axis=2)
        if not np.all(occupied | unknown | free):
            raise ValueError(f"invalid cached categories: {row['sample_id']}")
        features = np.stack((occupied, unknown, free)).astype(np.float32)
        labels = (target >= 128).astype(np.float32)[None]
        return (
            torch.from_numpy(features),
            torch.from_numpy(labels),
            torch.from_numpy(unknown.astype(np.float32)[None]),
            row["sample_id"],
        )


def amp_context(config: dict[str, object], device: str):
    if device.startswith("cuda") and config["mixed_precision"] == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    unknown: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    bce = functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    unknown_bce = masked_mean(bce, unknown)
    probability = torch.sigmoid(logits)
    intersection = (probability * labels * unknown).sum()
    denominator = ((probability + labels) * unknown).sum().clamp_min(1.0)
    unknown_dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    full_bce = bce.mean()
    total = (
        float(weights["unknown_bce"]) * unknown_bce
        + float(weights["unknown_soft_dice"]) * unknown_dice
        + float(weights["full_bce"]) * full_bce
    )
    return total, unknown_bce


def validation_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    config: dict[str, object],
    device: str,
) -> dict[str, float]:
    bce_sum = 0.0
    unknown_count = 0
    tp = fp = fn = tn = 0
    model.eval()
    with torch.inference_mode():
        for features, labels, unknown, _ in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            unknown = unknown.to(device, non_blocking=True)
            with amp_context(config, device):
                logits = model(features)
            probability = torch.sigmoid(logits.float())
            bce = functional.binary_cross_entropy(probability, labels, reduction="none")
            bce_sum += float((bce * unknown).sum().cpu())
            unknown_count += int(unknown.sum().cpu())
            predicted = probability >= float(config["decision_threshold"])
            target = labels >= 0.5
            mask = unknown >= 0.5
            tp += int((predicted & target & mask).sum().cpu())
            fp += int((predicted & ~target & mask).sum().cpu())
            fn += int((~predicted & target & mask).sum().cpu())
            tn += int((~predicted & ~target & mask).sum().cpu())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "unknown_bce": bce_sum / max(1, unknown_count),
        "unknown_precision": precision,
        "unknown_recall": recall,
        "unknown_f1": 2.0 * precision * recall / max(1e-12, precision + recall),
        "unknown_iou": tp / max(1, tp + fp + fn),
        "unknown_cells": unknown_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def atomic_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    method: str,
    seed: int,
    epoch: int,
    best_epoch: int,
    best_metric: float,
    config: dict[str, object],
    model_manifest_sha256: str,
    source_hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "status": "prospective_uniform_adapter",
        "protocol_id": config["protocol_id"],
        "method": method,
        "seed": seed,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_unknown_bce": best_metric,
        "configuration": config,
        "model_manifest_sha256": model_manifest_sha256,
        "source_hashes": source_hashes,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def write_metric(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(row), delimiter="\t", lineterminator="\n"
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def profile_latency(
    model: torch.nn.Module,
    example: torch.Tensor,
    config: dict[str, object],
    device: str,
) -> dict[str, object]:
    model.eval()
    example = example.unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(int(config["latency_warmup"])):
            with amp_context(config, device):
                model(example)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings: list[float] = []
        for _ in range(int(config["latency_repetitions"])):
            start = time.perf_counter()
            with amp_context(config, device):
                model(example)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "n": len(timings),
        "median_ms": float(np.median(timings)),
        "p90_ms": float(np.percentile(timings, 90)),
        "minimum_ms": float(np.min(timings)),
    }


def export_predictions(
    model: torch.nn.Module,
    dataset: CachedDataset,
    method: str,
    seed: int,
    split: str,
    config: dict[str, object],
    device: str,
    output: Path,
) -> Path:
    probability_root = output / "probabilities" / split
    probability_root.mkdir(parents=True, exist_ok=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.startswith("cuda"),
        worker_init_fn=worker_seed,
    )
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for features, _, _, sample_ids in loader:
            features = features.to(device, non_blocking=True)
            start = time.perf_counter()
            with amp_context(config, device):
                probability = torch.sigmoid(model(features).float())
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed_each = (time.perf_counter() - start) * 1000.0 / len(sample_ids)
            for sample_id, values in zip(sample_ids, probability[:, 0].cpu().numpy()):
                path = probability_root / f"{sample_id}.npy"
                with path.open("wb") as stream:
                    np.save(stream, values.astype(np.float32), allow_pickle=False)
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "split": split,
                        "sample_id": sample_id,
                        "probability_path": path.relative_to(output).as_posix(),
                        "probability_sha256": sha256(path),
                        "inference_ms": elapsed_each,
                    }
                )
    manifest = output / f"predictions_{split}.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--model-data-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.method == "MSO":
        # The public FFC implementation executes its FFT path in float32.
        config = dict(config)
        config["mixed_precision"] = "float32"
    set_determinism(args.seed)
    source_hashes = {
        "trainer": sha256(Path(__file__)),
        "models": sha256(Path(__file__).with_name("unified_models.py")),
        "configuration": sha256(args.config),
    }
    manifest_hash = sha256(args.model_manifest)
    train_data = CachedDataset(args.model_data_root, args.model_manifest, "train")
    validation_data = CachedDataset(
        args.model_data_root, args.model_manifest, "validation"
    )
    train_generator = torch.Generator().manual_seed(args.seed)
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
    model = build_model(args.method).to(args.device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config["adam_betas"]),
        weight_decay=float(config["weight_decay"]),
    )
    run_manifest = {
        "status": "prospective_uniform_adapter",
        "method": args.method,
        "seed": args.seed,
        "configuration": config,
        "model_manifest_sha256": manifest_hash,
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "source_hashes": source_hashes,
        "parameter_count": parameter_count,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "hardware": {
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        },
    }
    write_json(args.output / "run_manifest.json", run_manifest)
    features, labels, unknown, sample_ids = next(iter(train_loader))
    if args.smoke_only:
        features = features.to(args.device)
        labels = labels.to(args.device)
        unknown = unknown.to(args.device)
        with amp_context(config, args.device):
            logits = model(features)
            loss, _ = training_loss(logits, labels, unknown, config["loss_weights"])
        loss.backward()
        write_json(
            args.output / "smoke.json",
            {
                "status": "forward_backward_smoke_only",
                "first_sample": sample_ids[0],
                "input_shape": list(features.shape),
                "output_shape": list(logits.shape),
                "loss": float(loss.detach().cpu()),
                "finite": bool(torch.isfinite(loss)),
                "parameter_count": parameter_count,
            },
        )
        return
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    best_metric = float("inf")
    best_epoch = -1
    training_start = time.perf_counter()
    for epoch in range(int(config["maximum_epochs"])):
        epoch_start = time.perf_counter()
        train_generator.manual_seed(args.seed + epoch)
        model.train()
        loss_sum = 0.0
        unknown_bce_sum = 0.0
        batches = 0
        for features, labels, unknown, _ in train_loader:
            features = features.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            unknown = unknown.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(config, args.device):
                logits = model(features)
                loss, unknown_bce = training_loss(
                    logits, labels, unknown, config["loss_weights"]
                )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            unknown_bce_sum += float(unknown_bce.detach().cpu())
            batches += 1
        validation = validation_metrics(model, validation_loader, config, args.device)
        selected = validation["unknown_bce"] < best_metric
        if selected:
            best_metric = validation["unknown_bce"]
            best_epoch = epoch
            atomic_checkpoint(
                args.output / "best.pt",
                checkpoint_payload(
                    model,
                    optimizer,
                    args.method,
                    args.seed,
                    epoch,
                    best_epoch,
                    best_metric,
                    config,
                    manifest_hash,
                    source_hashes,
                ),
            )
        atomic_checkpoint(
            args.output / "last.pt",
            checkpoint_payload(
                model,
                optimizer,
                args.method,
                args.seed,
                epoch,
                best_epoch,
                best_metric,
                config,
                manifest_hash,
                source_hashes,
            ),
        )
        metric_row = {
            "epoch": epoch,
            "train_total_loss": loss_sum / max(1, batches),
            "train_unknown_bce": unknown_bce_sum / max(1, batches),
            "validation_unknown_bce": validation["unknown_bce"],
            "validation_unknown_precision": validation["unknown_precision"],
            "validation_unknown_recall": validation["unknown_recall"],
            "validation_unknown_f1": validation["unknown_f1"],
            "validation_unknown_iou": validation["unknown_iou"],
            "epoch_seconds": time.perf_counter() - epoch_start,
            "selected_best": int(selected),
        }
        write_metric(args.output / "metrics_by_epoch.tsv", metric_row)
        print(json.dumps(metric_row, sort_keys=True), flush=True)
        if (
            epoch + 1 >= int(config["minimum_epochs_before_stopping"])
            and epoch - best_epoch >= int(config["early_stopping_patience"])
        ):
            break
    training_seconds = time.perf_counter() - training_start
    payload = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(args.device)
    latency = profile_latency(model, train_data[0][0], config, args.device)
    test_data = CachedDataset(args.model_data_root, args.model_manifest, "test")
    validation_predictions = export_predictions(
        model,
        validation_data,
        args.method,
        args.seed,
        "validation",
        config,
        args.device,
        args.output,
    )
    test_predictions = export_predictions(
        model,
        test_data,
        args.method,
        args.seed,
        "test",
        config,
        args.device,
        args.output,
    )
    resource = {
        "parameter_count": parameter_count,
        "training_time_s": training_seconds,
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if args.device.startswith("cuda")
            else None
        ),
        "latency_batch1": latency,
    }
    write_json(args.output / "resource_profile.json", resource)
    completion = {
        "status": "prospective_uniform_adapter",
        "completed": True,
        "method": args.method,
        "seed": args.seed,
        "last_epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_unknown_bce": best_metric,
        "best_checkpoint_sha256": sha256(args.output / "best.pt"),
        "last_checkpoint_sha256": sha256(args.output / "last.pt"),
        "validation_predictions_sha256": sha256(validation_predictions),
        "test_predictions_sha256": sha256(test_predictions),
    }
    write_json(args.output / "completion.json", completion)
    checksum_files = [
        "run_manifest.json",
        "metrics_by_epoch.tsv",
        "best.pt",
        "last.pt",
        "predictions_validation.tsv",
        "predictions_test.tsv",
        "resource_profile.json",
        "completion.json",
    ]
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha256(args.output / name)}  {name}\n" for name in checksum_files),
        encoding="utf-8",
    )
    print(json.dumps(completion, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
