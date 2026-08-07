#!/usr/bin/env python3
"""Inventory a preserved image archive without assigning semantic groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


FIELDS = [
    "sample_id",
    "split",
    "split_status",
    "archive_partition",
    "source_worker_prefix",
    "obs_relpath",
    "target_relpath",
    "obs_sha256",
    "target_sha256",
    "processed_pair_sha256",
    "obs_shape",
    "target_shape",
    "processed_obs_shape",
    "processed_target_shape",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def shape_text(shape) -> str:
    return "x".join(str(item) for item in shape)


def processed_pair(obs_path: Path, target_path: Path, image_size: int):
    resampling = Image.Resampling.NEAREST
    obs_raw = np.asarray(Image.open(obs_path).convert("RGB"), dtype=np.uint8)
    target_raw = np.asarray(Image.open(target_path).convert("L"), dtype=np.uint8)

    obs_image = Image.fromarray(obs_raw, mode="RGB").resize(
        (image_size, image_size), resampling
    )
    target_image = Image.fromarray(target_raw, mode="L").resize(
        (image_size, image_size), resampling
    )
    obs = np.asarray(obs_image, dtype=np.uint8).copy()
    obs[obs == 255] = 1
    obs = np.asarray(obs.transpose(2, 0, 1), dtype="<f4", order="C")
    target = (np.asarray(target_image, dtype=np.uint8) == 255).astype("<f4")
    target = np.asarray(target[None, :, :], dtype="<f4", order="C")

    digest = hashlib.sha256()
    digest.update(b"MSO-PROCESSED-PAIR-v1\0")
    digest.update(obs.tobytes(order="C"))
    digest.update(target.tobytes(order="C"))
    return (
        digest.hexdigest(),
        shape_text(obs_raw.shape),
        shape_text(target_raw.shape),
        shape_text(obs.shape),
        shape_text(target.shape),
    )


def worker_prefix(sample_dir_name: str) -> str:
    return sample_dir_name.split("_", 1)[0] if "_" in sample_dir_name else ""


def collect_rows(root: Path, image_size: int) -> list[dict[str, str]]:
    rows = []
    for partition in ("train", "test"):
        partition_dir = root / partition
        if not partition_dir.is_dir():
            continue
        for sample_dir in sorted(
            (item for item in partition_dir.iterdir() if item.is_dir()),
            key=lambda item: item.name,
        ):
            obs = sample_dir / "local_obs_0.png"
            target = sample_dir / "local_map_0.png"
            if not obs.is_file() or not target.is_file():
                raise FileNotFoundError(f"Incomplete sample directory: {sample_dir}")
            pair_hash, obs_shape, target_shape, proc_obs, proc_target = processed_pair(
                obs, target, image_size
            )
            rows.append(
                {
                    "sample_id": f"{partition}/{sample_dir.name}",
                    "split": partition,
                    "split_status": "preserved_archive_label_not_manuscript_split",
                    "archive_partition": partition,
                    "source_worker_prefix": worker_prefix(sample_dir.name),
                    "obs_relpath": obs.relative_to(root).as_posix(),
                    "target_relpath": target.relative_to(root).as_posix(),
                    "obs_sha256": sha256_file(obs),
                    "target_sha256": sha256_file(target),
                    "processed_pair_sha256": pair_hash,
                    "obs_shape": obs_shape,
                    "target_shape": target_shape,
                    "processed_obs_shape": proc_obs,
                    "processed_target_shape": proc_target,
                }
            )
    if not rows:
        raise ValueError(f"No train/test samples found under {root}")
    return rows


def write_inventory(
    dataset_root: Path, output: Path, summary: Path, image_size: int = 256
) -> dict[str, object]:
    rows = collect_rows(dataset_root.resolve(), image_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    aggregate = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        aggregate.update(row["sample_id"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(row["processed_pair_sha256"].encode("ascii"))
        aggregate.update(b"\n")
    counts = {
        partition: sum(row["archive_partition"] == partition for row in rows)
        for partition in sorted({row["archive_partition"] for row in rows})
    }
    payload = {
        "schema": "preserved_archive_inventory_v2",
        "status": "preserved_archive_not_manuscript_split",
        "sample_count": len(rows),
        "partition_counts": counts,
        "image_size": image_size,
        "processed_pair_protocol": (
            "RGB nearest resize; uint8 255 replaced by 1 then CHW little-endian "
            "float32; grayscale target nearest resize and ==255 to 1xHW little-endian "
            "float32; SHA256 over MSO-PROCESSED-PAIR-v1\\0 + observation bytes + target bytes"
        ),
        "aggregate_sample_processed_sha256": aggregate.hexdigest(),
        "inventory_sha256": sha256_file(output),
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_inventory(args.dataset_root, args.output, args.summary, args.image_size)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
