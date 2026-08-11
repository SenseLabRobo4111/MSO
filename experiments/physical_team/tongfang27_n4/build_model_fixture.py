#!/usr/bin/env python3
"""Build the deterministic one-hot fixture used by the deployment lock."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_fixture() -> np.ndarray:
    """Return a fixed occupied/unknown/free raster with shape 3 x 256 x 256."""
    value = np.zeros((3, 256, 256), dtype=np.uint8)
    value[1, :, :] = 1

    free = np.zeros((256, 256), dtype=bool)
    free[40:216, 36:220] = True
    free[112:144, 12:244] = True
    occupied = np.zeros((256, 256), dtype=bool)
    occupied[:8, :] = True
    occupied[-8:, :] = True
    occupied[:, :8] = True
    occupied[:, -8:] = True
    occupied[72:80, 40:180] = True
    occupied[176:184, 76:220] = True
    occupied[80:176, 124:132] = True
    for start in range(24, 224, 32):
        occupied[start:start + 4, start:start + 20] = True

    free &= ~occupied
    value[:, occupied | free] = 0
    value[0, occupied] = 1
    value[2, free] = 1
    if not np.all(value.sum(axis=0) == 1):
        raise RuntimeError("fixture construction violated the one-hot contract")
    return value


def main() -> int:
    """Write the fixture without allowing implicit pickle data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, build_fixture(), allow_pickle=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
