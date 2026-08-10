"""Declared replacement objective and validation accumulators."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def unknown_bce(
    prediction: torch.Tensor, target: torch.Tensor, unknown: torch.Tensor
) -> torch.Tensor:
    loss = functional.binary_cross_entropy(
        prediction.float().clamp(1e-6, 1.0 - 1e-6), target.float(), reduction="none"
    )
    return masked_mean(loss, unknown.float())


def unknown_soft_dice_loss(
    prediction: torch.Tensor, target: torch.Tensor, unknown: torch.Tensor
) -> torch.Tensor:
    prediction = prediction.float() * unknown.float()
    target = target.float() * unknown.float()
    intersection = (prediction * target).sum()
    denominator = prediction.sum() + target.sum()
    return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)


def full_bce(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return functional.binary_cross_entropy(
        prediction.float().clamp(1e-6, 1.0 - 1e-6), target.float()
    )


def feature_distillation(
    student_features: tuple[torch.Tensor, ...],
    teacher_features: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if len(student_features) != 4 or len(teacher_features) != 4:
        raise ValueError("Teacher and student must expose four feature tensors")
    losses = []
    for student, teacher in zip(student_features, teacher_features):
        if student.shape != teacher.shape:
            raise ValueError(
                f"Distillation shape mismatch: {student.shape} vs {teacher.shape}"
            )
        losses.append(functional.mse_loss(student.float(), teacher.detach().float()))
    return torch.stack(losses).mean()


@dataclass
class BinaryAccumulator:
    bce_sum: float = 0.0
    cell_count: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> None:
        prediction = prediction.detach().float().clamp(1e-6, 1.0 - 1e-6)
        target = target.detach().bool()
        selected = mask.detach().bool()
        if not bool(selected.any()):
            return
        values = prediction[selected]
        truth = target[selected]
        self.bce_sum += float(
            functional.binary_cross_entropy(
                values, truth.float(), reduction="sum"
            ).cpu()
        )
        self.cell_count += int(selected.sum().cpu())
        decision = values >= 0.5
        self.true_positive += int((decision & truth).sum().cpu())
        self.false_positive += int((decision & ~truth).sum().cpu())
        self.false_negative += int((~decision & truth).sum().cpu())
        self.true_negative += int((~decision & ~truth).sum().cpu())

    def metrics(self) -> dict[str, float | int]:
        precision = self.true_positive / max(1, self.true_positive + self.false_positive)
        recall = self.true_positive / max(1, self.true_positive + self.false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        iou = self.true_positive / max(
            1, self.true_positive + self.false_positive + self.false_negative
        )
        return {
            "unknown_bce": self.bce_sum / max(1, self.cell_count),
            "unknown_precision": precision,
            "unknown_recall": recall,
            "unknown_f1": f1,
            "unknown_iou": iou,
            "unknown_cells": self.cell_count,
        }
