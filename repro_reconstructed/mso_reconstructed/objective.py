"""Explicit replacement losses for the reconstructed training path."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def masked_pixel_l2(
    prediction: torch.Tensor, target: torch.Tensor, unknown_mask: torch.Tensor
) -> torch.Tensor:
    return masked_mean((prediction - target).square(), unknown_mask)


def multiscale_reconstruction(
    prediction: torch.Tensor, target: torch.Tensor, unknown_mask: torch.Tensor
) -> torch.Tensor:
    total = prediction.new_zeros(())
    pred_level, target_level, mask_level = prediction, target, unknown_mask
    for level in range(4):
        total = total + masked_mean((pred_level - target_level).abs(), mask_level)
        if level != 3:
            pred_level = functional.avg_pool2d(pred_level, 2)
            target_level = functional.avg_pool2d(target_level, 2)
            mask_level = functional.max_pool2d(mask_level, 2)
    return total / 4.0


def feature_distillation(
    student_features: tuple[torch.Tensor, ...],
    teacher_features: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if len(student_features) != 4 or len(teacher_features) != 4:
        raise ValueError("The reconstructed objective requires four feature taps")
    losses = []
    for student, teacher in zip(student_features, teacher_features, strict=True):
        if student.shape != teacher.shape:
            raise ValueError(
                f"Feature shapes differ: student={student.shape}, teacher={teacher.shape}"
            )
        losses.append(functional.mse_loss(student, teacher.detach()))
    return torch.stack(losses).mean()
