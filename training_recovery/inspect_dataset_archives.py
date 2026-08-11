#!/usr/bin/env python3
"""Summarize immutable dataset archives without extracting them."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(path: Path) -> dict[str, object]:
    samples: dict[str, set[str]] = defaultdict(set)
    files_by_split: Counter[str] = Counter()
    metadata_members: list[str] = []
    member_count = 0
    uncompressed_bytes = 0
    timestamps: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            member_count += 1
            uncompressed_bytes += info.file_size
            timestamps.append("%04d-%02d-%02dT%02d:%02d:%02d" % info.date_time)
            normalized = info.filename.strip("/")
            parts = normalized.split("/")
            if len(parts) >= 3 and parts[-1]:
                split = parts[-3]
                sample_id = parts[-2]
                if split in {"train", "test", "val", "validation"}:
                    samples[split].add(sample_id)
                    files_by_split[split] += 1
            lower = normalized.lower()
            if any(
                token in lower
                for token in (
                    "manifest",
                    "split",
                    "map_id",
                    "readme",
                    "train.py",
                    "trainer",
                    "lightning",
                    "checkpoint",
                    ".ckpt",
                    ".log",
                )
            ):
                metadata_members.append(info.filename)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "member_count": member_count,
        "uncompressed_bytes": uncompressed_bytes,
        "sample_counts": {key: len(value) for key, value in sorted(samples.items())},
        "file_counts": dict(sorted(files_by_split.items())),
        "earliest_member_time": min(timestamps) if timestamps else None,
        "latest_member_time": max(timestamps) if timestamps else None,
        "metadata_members": metadata_members[:500],
    }


if __name__ == "__main__":
    print(json.dumps([inspect(Path(name)) for name in sys.argv[1:]], indent=2))
