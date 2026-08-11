from __future__ import annotations

import os
import sys

import torch
from torch import nn
import torch.nn.functional as functional


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = self.up(values)
        if values.shape[-2:] != skip.shape[-2:]:
            values = functional.interpolate(
                values, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.conv(torch.cat([skip, values], dim=1))


class UNet(nn.Module):
    def __init__(self, base: int = 64) -> None:
        super().__init__()
        self.input = DoubleConv(3, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16)
        self.up1 = Up(base * 16, base * 8, base * 8)
        self.up2 = Up(base * 8, base * 4, base * 4)
        self.up3 = Up(base * 4, base * 2, base * 2)
        self.up4 = Up(base * 2, base, base)
        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        level0 = self.input(values)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        level3 = self.down3(level2)
        level4 = self.down4(level3)
        values = self.up1(level4, level3)
        values = self.up2(values, level2)
        values = self.up3(values, level1)
        values = self.up4(values, level0)
        return self.output(values)


class FourierUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, 1, bias=False),
            nn.GroupNorm(8, channels * 2),
            nn.SiLU(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        input_dtype = values.dtype
        spectrum = torch.fft.rfft2(values.float(), norm="ortho")
        stacked = torch.cat((spectrum.real, spectrum.imag), dim=1)
        stacked = self.mix(stacked)
        real, imaginary = stacked.chunk(2, dim=1)
        reconstructed = torch.fft.irfft2(
            torch.complex(real, imaginary), s=values.shape[-2:], norm="ortho"
        )
        return reconstructed.to(input_dtype)


class FourierUNet(UNet):
    def __init__(self, base: int = 64) -> None:
        super().__init__(base=base)
        self.fourier = FourierUnit(base * 16)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        level0 = self.input(values)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        level3 = self.down3(level2)
        level4 = self.down4(level3)
        level4 = level4 + self.fourier(level4)
        values = self.up1(level4, level3)
        values = self.up2(values, level2)
        values = self.up3(values, level1)
        values = self.up4(values, level0)
        return self.output(values)


class SeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(max(1, min(8, out_channels // 4)), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class MobileCompletionNet(nn.Module):
    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.enc1 = SeparableBlock(3, base)
        self.enc2 = SeparableBlock(base, base * 2, stride=2)
        self.enc3 = SeparableBlock(base * 2, base * 4, stride=2)
        self.enc4 = SeparableBlock(base * 4, base * 8, stride=2)
        self.middle = nn.Sequential(
            SeparableBlock(base * 8, base * 8),
            SeparableBlock(base * 8, base * 8),
            SeparableBlock(base * 8, base * 8),
        )
        self.dec3 = SeparableBlock(base * 8 + base * 4, base * 4)
        self.dec2 = SeparableBlock(base * 4 + base * 2, base * 2)
        self.dec1 = SeparableBlock(base * 2 + base, base)
        self.output = nn.Conv2d(base, 1, 1)

    @staticmethod
    def up(values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return functional.interpolate(
            values, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        level1 = self.enc1(values)
        level2 = self.enc2(level1)
        level3 = self.enc3(level2)
        level4 = self.middle(self.enc4(level3))
        values = self.dec3(torch.cat((self.up(level4, level3), level3), dim=1))
        values = self.dec2(torch.cat((self.up(values, level2), level2), dim=1))
        values = self.dec1(torch.cat((self.up(values, level1), level1), dim=1))
        return self.output(values)


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        kernel_height, kernel_width = self.kernel_size
        self.slide_size = self.in_channels * kernel_height * kernel_width

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask.shape[1] == 1:
            mask = mask.expand(-1, values.shape[1], -1, -1)
        masked = values * mask
        output = super().forward(masked)
        with torch.no_grad():
            kernel = torch.ones(
                (1, mask.shape[1], *self.kernel_size),
                device=mask.device,
                dtype=mask.dtype,
            )
            update = functional.conv2d(
                mask,
                kernel,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            ratio = self.slide_size / update.clamp_min(1.0)
            update = (update > 0).to(mask.dtype)
            ratio = ratio * update
        return output * ratio, update


class PartialConvAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.partial1 = PartialConv2d(3, 64, 7, padding=3, bias=False)
        self.partial2 = PartialConv2d(64, 64, 5, padding=2, bias=False)
        self.core = UNet(base=64)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        known = 1.0 - values[:, 1:2]
        values, known = self.partial1(values, known)
        values = functional.silu(values)
        values, _ = self.partial2(values, known)
        reduced = torch.stack(
            (
                values.mean(dim=1),
                1.0 - known[:, 0],
                values.std(dim=1),
            ),
            dim=1,
        )
        return self.core(reduced)


class GatedConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.feature = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.gate = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.norm = nn.GroupNorm(8, out_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return functional.silu(self.norm(self.feature(values))) * torch.sigmoid(
            self.gate(values)
        )


class GatedCompletionNet(nn.Module):
    def __init__(self, base: int = 48) -> None:
        super().__init__()
        self.enc1 = GatedConv(3, base)
        self.enc2 = GatedConv(base, base * 2, stride=2)
        self.enc3 = GatedConv(base * 2, base * 4, stride=2)
        self.enc4 = GatedConv(base * 4, base * 8, stride=2)
        self.middle = nn.Sequential(
            GatedConv(base * 8, base * 8),
            GatedConv(base * 8, base * 8),
            GatedConv(base * 8, base * 8),
        )
        self.dec3 = GatedConv(base * 8 + base * 4, base * 4)
        self.dec2 = GatedConv(base * 4 + base * 2, base * 2)
        self.dec1 = GatedConv(base * 2 + base, base)
        self.output = nn.Conv2d(base, 1, 1)

    @staticmethod
    def up(values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return functional.interpolate(values, size=skip.shape[-2:], mode="nearest")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        level1 = self.enc1(values)
        level2 = self.enc2(level1)
        level3 = self.enc3(level2)
        level4 = self.middle(self.enc4(level3))
        values = self.dec3(torch.cat((self.up(level4, level3), level3), dim=1))
        values = self.dec2(torch.cat((self.up(values, level2), level2), dim=1))
        values = self.dec1(torch.cat((self.up(values, level1), level1), dim=1))
        return self.output(values)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return functional.relu(values + self.block(values), inplace=True)


def occupancy_edges(values: torch.Tensor) -> torch.Tensor:
    measured_occupied = values[:, 0:1]
    kernel_x = values.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    kernel_y = kernel_x.transpose(-1, -2)
    grad_x = functional.conv2d(measured_occupied, kernel_x, padding=1)
    grad_y = functional.conv2d(measured_occupied, kernel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6).clamp(0.0, 1.0)


class ResNetCompletion(nn.Module):
    def __init__(self, aot: bool = False) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        if aot:
            self.middle = nn.Sequential(
                *[AOTBlock(256, (1, 2, 4, 8)) for _ in range(8)]
            )
        else:
            self.middle = nn.Sequential(*[ResidualBlock(256, 2) for _ in range(8)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, 7),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.cat((values, occupancy_edges(values)), dim=1)
        return self.decoder(self.middle(self.encoder(values)))


class AOTBlock(nn.Module):
    def __init__(self, channels: int, rates: tuple[int, ...]) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(
                        channels,
                        channels // len(rates),
                        3,
                        dilation=rate,
                    ),
                    nn.ReLU(inplace=True),
                )
                for rate in rates
            ]
        )
        self.fuse = nn.Conv2d(channels, channels, 3, padding=1)
        self.gate = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        candidate = self.fuse(torch.cat([branch(values) for branch in self.branches], 1))
        gate = torch.sigmoid(self.gate(values))
        return values * (1.0 - gate) + candidate * gate


class TransformerCompletion(nn.Module):
    def __init__(self, edge_branch: bool = False, gated_input: bool = False) -> None:
        super().__init__()
        self.edge_branch = edge_branch
        input_channels = 4 if edge_branch else 3
        if gated_input:
            self.stem = nn.Sequential(GatedConv(input_channels, 64), GatedConv(64, 64))
        else:
            self.stem = DoubleConv(input_channels, 64)
        self.down = nn.Sequential(
            Down(64, 128), Down(128, 256), Down(256, 384), Down(384, 512)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=6)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.edge_branch:
            values = torch.cat((values, occupancy_edges(values)), dim=1)
        values = self.down(self.stem(values))
        batch, channels, height, width = values.shape
        tokens = values.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        values = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return self.decoder(values)


class MSOAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        source_root = os.environ.get("MSO_ADAPTER_ROOT")
        if not source_root:
            raise RuntimeError("MSO_ADAPTER_ROOT is required for the MSO adapter")
        sys.path.insert(0, source_root)
        from sensemap.explore_model.SenseMapNet import DistillMapNetDeconv

        self.model = DistillMapNetDeconv(image_size=256, dim=4)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        probability = self.model(values)[0]
        return torch.logit(probability.clamp(1e-5, 1.0 - 1e-5))


def build_model(method: str) -> nn.Module:
    if method == "MSO":
        return MSOAdapter()
    if method == "U-Net":
        return UNet(base=64)
    if method == "LaMa-Fourier":
        return FourierUNet(base=64)
    if method == "MI-GAN":
        return MobileCompletionNet(base=32)
    if method == "PartialConv":
        return PartialConvAdapter()
    if method == "GatedConv":
        return GatedCompletionNet(base=48)
    if method == "EdgeConnect":
        return ResNetCompletion(aot=False)
    if method == "AOT-GAN":
        return ResNetCompletion(aot=True)
    if method == "MAT":
        return TransformerCompletion(edge_branch=False, gated_input=False)
    if method == "ZITS++":
        return TransformerCompletion(edge_branch=True, gated_input=False)
    if method == "HINT":
        return TransformerCompletion(edge_branch=False, gated_input=True)
    raise ValueError(f"No locked adapter is implemented for {method!r}")
