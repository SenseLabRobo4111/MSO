from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from unified_models import build_model  # noqa: E402


@pytest.mark.parametrize(
    "method",
    [
        "U-Net",
        "LaMa-Fourier",
        "MI-GAN",
        "PartialConv",
        "GatedConv",
        "EdgeConnect",
        "AOT-GAN",
        "MAT",
        "ZITS++",
        "HINT",
    ],
)
def test_adapter_forward_contract(method: str) -> None:
    model = build_model(method).eval()
    values = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        output = model(values)
    assert output.shape == (1, 1, 64, 64)
    assert torch.isfinite(output).all()


def test_unet_parameter_count_matches_registered_adapter() -> None:
    model = build_model("U-Net")
    assert sum(parameter.numel() for parameter in model.parameters()) == 31_037_633


def test_training_loss_is_finite() -> None:
    spec = importlib.util.spec_from_file_location(
        "train_unified_adapter", ROOT / "train_unified_adapter.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
    labels = torch.randint(0, 2, logits.shape).float()
    unknown = torch.ones_like(labels)
    loss, bce = module.training_loss(
        logits,
        labels,
        unknown,
        {"unknown_bce": 1.0, "unknown_soft_dice": 0.5, "full_bce": 0.0},
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(bce)
    assert logits.grad is not None
