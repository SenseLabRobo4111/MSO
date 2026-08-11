#!/usr/bin/env python3
"""Recovered MSO components used by the complete-objective reconstruction.

The teacher topology and the critic state tree are recoverable from the
archived checkpoint.  The historical critic forward function is not.  The
critic below is therefore an explicit, reviewable reconstruction of that
state tree, not an authenticated historical implementation.
"""

from __future__ import annotations

from typing import Type

import torch
from torch import nn
import torch.nn.functional as functional


class ProjectedTeacher(nn.Module):
    """Dim-32 teacher plus the four recovered teacher-to-student projections."""

    def __init__(self, teacher_class: Type[nn.Module]) -> None:
        super().__init__()
        self.core = teacher_class(image_size=256, dim=32)
        self.proj_conv2 = nn.Conv2d(64, 8, 1)
        self.proj_conv3 = nn.Conv2d(128, 16, 1)
        self.proj_decode2 = nn.Conv2d(128, 16, 1)
        self.proj_decode3 = nn.Conv2d(64, 8, 1)

    def forward(self, values: torch.Tensor):
        prediction, conv2, conv3, decode2, decode3 = self.core(values)
        return (
            prediction,
            self.proj_conv2(conv2),
            self.proj_conv3(conv3),
            self.proj_decode2(decode2),
            self.proj_decode3(decode3),
        )

    def freeze_all(self) -> None:
        """Freeze the teacher core and fixed projection contract."""
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class ReconstructedFFCPatchCritic(nn.Module):
    """FFC critic reconstructed from the complete archived parameter tree.

    The four convolution and FFC stages strict-load all 440 archived state
    entries.  Channel averaging is the declared parameter-free readout used
    to expose a 16-by-16 patch-response field to the recovered LaMa loss.
    """

    def __init__(self, ffc_block_class: Type[nn.Module]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1)
        self.fft1 = ffc_block_class(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fft2 = ffc_block_class(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.fft3 = ffc_block_class(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.fft4 = ffc_block_class(512)

    def forward(self, raster: torch.Tensor, unknown: torch.Tensor) -> torch.Tensor:
        if raster.ndim != 4 or unknown.ndim != 4:
            raise ValueError("critic inputs must be BCHW tensors")
        raster_geometry = (raster.shape[0], raster.shape[-2], raster.shape[-1])
        mask_geometry = (unknown.shape[0], unknown.shape[-2], unknown.shape[-1])
        if raster_geometry != mask_geometry:
            raise ValueError("critic raster and mask geometry differ")
        values = torch.cat((raster, unknown), dim=1)
        values = self.fft1(functional.leaky_relu(self.conv1(values), 0.2, inplace=False))
        values = self.fft2(functional.leaky_relu(self.conv2(values), 0.2, inplace=False))
        values = self.fft3(functional.leaky_relu(self.conv3(values), 0.2, inplace=False))
        values = self.fft4(functional.leaky_relu(self.conv4(values), 0.2, inplace=False))
        return values.mean(dim=1, keepdim=True)


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
