"""Strict manifest-backed data access for the replacement experiment."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .common import sha256_file


REQUIRED_COLUMNS = {
    "sample_id",
    "split",
    "group_key",
    "group_kind",
    "floorplan_id",
    "obs_relpath",
    "target_relpath",
    "obs_sha256",
    "target_sha256",
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    missing = REQUIRED_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    seen: set[str] = set()
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate sample_id: {sample_id!r}")
        seen.add(sample_id)
        if row["group_kind"] != "building":
            raise ValueError(f"Non-building group for {sample_id}")
        if not row["group_key"]:
            raise ValueError(f"Missing building group for {sample_id}")
        split = row["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {split!r} for {sample_id}")
        group_splits.setdefault(row["group_key"], set()).add(split)
    leakage = {key: values for key, values in group_splits.items() if len(values) != 1}
    if leakage:
        raise ValueError(f"Building leakage across splits: {list(leakage.items())[:5]}")

    floorplan_splits: dict[str, set[str]] = {}
    for row in rows:
        floorplan_splits.setdefault(row["floorplan_id"], set()).add(row["split"])
    floorplan_leakage = {
        key: values for key, values in floorplan_splits.items() if len(values) != 1
    }
    if floorplan_leakage:
        raise ValueError(
            f"Floorplan leakage across splits: {list(floorplan_leakage.items())[:5]}"
        )
    return rows


def safe_child(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"Path escapes dataset root: {relative}")
    return candidate


class FloorplanManifestDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        manifest_path: Path,
        split: str,
        verify_hashes: bool = True,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        rows = read_manifest(manifest_path)
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No samples in split {split!r}")
        self.split = split
        self.integrity = "verified_sha256" if verify_hashes else "not_verified"
        if verify_hashes:
            for row in self.rows:
                observation = safe_child(self.dataset_root, row["obs_relpath"])
                target = safe_child(self.dataset_root, row["target_relpath"])
                if sha256_file(observation) != row["obs_sha256"]:
                    raise ValueError(f"Observation hash mismatch: {row['sample_id']}")
                if sha256_file(target) != row["target_sha256"]:
                    raise ValueError(f"Target hash mismatch: {row['sample_id']}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        observation_path = safe_child(self.dataset_root, row["obs_relpath"])
        target_path = safe_child(self.dataset_root, row["target_relpath"])
        observation = np.asarray(
            Image.open(observation_path).convert("RGB"), dtype=np.uint8
        ).copy()
        target = np.asarray(Image.open(target_path).convert("L"), dtype=np.uint8).copy()
        if observation.shape != (256, 256, 3) or target.shape != (256, 256):
            raise ValueError(
                f"Unexpected image shape for {row['sample_id']}: "
                f"{observation.shape}, {target.shape}"
            )
        observation_binary = observation == 255
        if not np.all(observation_binary.sum(axis=2) == 1):
            raise ValueError(f"Observation is not one-hot: {row['sample_id']}")
        target_binary = target == 255
        known = ~observation_binary[:, :, 1]
        measured_occupied = observation_binary[:, :, 0]
        if not np.array_equal(measured_occupied[known], target_binary[known]):
            raise ValueError(f"Known occupancy disagrees with target: {row['sample_id']}")
        features = torch.from_numpy(observation_binary.astype(np.float32)).permute(2, 0, 1)
        labels = torch.from_numpy(target_binary.astype(np.float32)).unsqueeze(0)
        unknown = torch.from_numpy(observation_binary[:, :, 1].astype(np.float32)).unsqueeze(0)
        return features, labels, unknown, row["sample_id"], row["floorplan_id"]
