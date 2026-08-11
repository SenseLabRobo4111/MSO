"""Locked manifest access for prospective v2; test decoding is impossible."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .v2_lock import sha256_file


ALLOWED_DECODE_SPLITS = {"train", "val"}
REQUIRED_COLUMNS = {
    "sample_id",
    "split",
    "group_key",
    "group_kind",
    "floorplan_id",
    "source_sha256",
    "obs_relpath",
    "target_relpath",
    "obs_sha256",
    "target_sha256",
    "raw_obs_relpath",
    "raw_target_relpath",
    "raw_obs_sha256",
    "raw_target_sha256",
}


def read_v2_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty v2 manifest: {path}")
    missing = REQUIRED_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"Missing v2 manifest columns: {sorted(missing)}")
    seen: set[str] = set()
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate sample ID: {sample_id!r}")
        seen.add(sample_id)
        if row["split"] not in ALLOWED_DECODE_SPLITS:
            raise ValueError(
                f"v2 manifests may contain train/val samples only: {sample_id}"
            )
        if row["group_kind"] != "operational_prefix_group":
            raise ValueError(f"Unapproved group semantics for {sample_id}")
        group_splits.setdefault(row["group_key"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("Operational-group leakage across train and validation")
    return rows


def safe_child(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError(f"Absolute dataset path is forbidden: {relative}")
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if resolved_root not in candidate.parents:
        raise ValueError(f"Dataset path escapes root: {relative}")
    return candidate


class V2ManifestDataset(Dataset):
    def __init__(self, dataset_root: Path, manifest_path: Path, split: str) -> None:
        if split not in ALLOWED_DECODE_SPLITS:
            raise ValueError("Prospective v2 test decoding is disabled")
        self.dataset_root = dataset_root.resolve()
        self.rows = [
            row for row in read_v2_manifest(manifest_path) if row["split"] == split
        ]
        self.integrity = "verified_sha256"
        if not self.rows:
            raise ValueError(f"No samples for v2 split {split}")
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
        observation = np.asarray(
            Image.open(safe_child(self.dataset_root, row["obs_relpath"])).convert("RGB"),
            dtype=np.uint8,
        ).copy()
        target = np.asarray(
            Image.open(safe_child(self.dataset_root, row["target_relpath"])).convert("L"),
            dtype=np.uint8,
        ).copy()
        if observation.shape != (256, 256, 3) or target.shape != (256, 256):
            raise ValueError(f"Unexpected v2 model shape: {row['sample_id']}")
        one_hot = observation == 255
        if not np.all(one_hot.sum(axis=2) == 1):
            raise ValueError(f"Non-one-hot v2 observation: {row['sample_id']}")
        target_binary = target == 255
        known = ~one_hot[:, :, 1]
        if not np.array_equal(one_hot[:, :, 0][known], target_binary[known]):
            raise ValueError(f"Known-target disagreement: {row['sample_id']}")
        features = torch.from_numpy(one_hot.astype(np.float32)).permute(2, 0, 1)
        labels = torch.from_numpy(target_binary.astype(np.float32)).unsqueeze(0)
        unknown = torch.from_numpy(one_hot[:, :, 1].astype(np.float32)).unsqueeze(0)
        return features, labels, unknown, row["sample_id"], row["floorplan_id"]
