"""Manifest-backed occupancy-map dataset with explicit split validation."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ALLOWED_GROUP_KINDS = {"building", "floorplan", "scene", "environment"}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    required = {
        "sample_id",
        "split",
        "group_key",
        "group_kind",
        "obs_relpath",
        "target_relpath",
        "obs_sha256",
        "target_sha256",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Manifest lacks columns: {sorted(missing)}")
    return rows


def validate_canonical_rows(rows: Iterable[dict[str, str]]) -> None:
    seen_samples: set[str] = set()
    group_splits: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        sample_id = row["sample_id"].strip()
        if not sample_id or sample_id in seen_samples:
            raise ValueError(f"Missing or duplicate sample_id: {sample_id!r}")
        seen_samples.add(sample_id)
        kind = row["group_kind"].strip().lower()
        key = row["group_key"].strip()
        if kind not in ALLOWED_GROUP_KINDS or not key:
            raise ValueError(
                f"Sample {sample_id} needs a semantic group_kind/group_key; "
                f"allowed kinds are {sorted(ALLOWED_GROUP_KINDS)}"
            )
        split = row["split"].strip().lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {split!r} for {sample_id}")
        group_splits.setdefault((kind, key), set()).add(split)
    leaks = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaks:
        preview = list(leaks.items())[:5]
        raise ValueError(f"Semantic groups cross splits: {preview}")


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"Path escapes dataset root: {relative}")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_pair(obs_path: Path, target_path: Path, image_size: int = 256):
    resampling = Image.Resampling.NEAREST
    obs_image = Image.open(obs_path).convert("RGB").resize((image_size, image_size), resampling)
    target_image = Image.open(target_path).convert("L").resize((image_size, image_size), resampling)

    obs = np.asarray(obs_image, dtype=np.uint8).copy()
    obs[obs == 255] = 1
    obs = obs.astype(np.float32, copy=False)
    target = (np.asarray(target_image, dtype=np.uint8) == 255).astype(np.float32)
    mask = obs[:, :, 1].copy()

    x = torch.from_numpy(obs).permute(2, 0, 1).contiguous()
    y = torch.from_numpy(target).unsqueeze(0).contiguous()
    m = torch.from_numpy(mask).unsqueeze(0).contiguous()
    return x, y, m


class ManifestDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        manifest: Path,
        split: str,
        image_size: int = 256,
        verify_raw_hashes: bool = True,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        all_rows = read_manifest(manifest)
        validate_canonical_rows(all_rows)
        self.rows = [row for row in all_rows if row["split"].lower() == split.lower()]
        if not self.rows:
            raise ValueError(f"No rows for split {split!r}")
        self.image_size = image_size
        self.verify_raw_hashes = verify_raw_hashes
        self.integrity_status = (
            "verified_manifest_raw_sha256"
            if verify_raw_hashes
            else "not_verified_diagnostic_bypass"
        )
        if self.verify_raw_hashes:
            self._verify_raw_hashes()

    def _verify_raw_hashes(self) -> None:
        for row in self.rows:
            obs_path = _safe_child(self.dataset_root, row["obs_relpath"])
            target_path = _safe_child(self.dataset_root, row["target_relpath"])
            if _sha256(obs_path) != row["obs_sha256"]:
                raise ValueError(f"Observation hash mismatch: {row['sample_id']}")
            if _sha256(target_path) != row["target_sha256"]:
                raise ValueError(f"Target hash mismatch: {row['sample_id']}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        obs_path = _safe_child(self.dataset_root, row["obs_relpath"])
        target_path = _safe_child(self.dataset_root, row["target_relpath"])
        x, y, mask = preprocess_pair(obs_path, target_path, self.image_size)
        return x, y, mask, row["sample_id"]
