"""Fail-closed loading and hashing for the prospective v2 protocol."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


LOCK_FILENAME = "v2_lock.json"
LOCK_DIGEST_FILENAME = "v2_lock.sha256"
LOCK_SCHEMA = "mso.prospective_v2.lock/1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def safe_relative(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"Expected a non-empty relative path: {relative!r}")
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if resolved_root not in candidate.parents:
        raise ValueError(f"Path escapes lock root: {relative}")
    return candidate


def verify_hashed_artifact(root: Path, record: dict[str, Any]) -> Path:
    if not {"path", "sha256"}.issubset(record):
        raise ValueError(f"Malformed artifact record: {record}")
    path = safe_relative(root, str(record["path"]))
    expected = str(record["sha256"])
    if len(expected) != 64:
        raise ValueError(f"Malformed SHA-256 for {record['path']}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"Artifact hash mismatch for {record['path']}: {actual} != {expected}"
        )
    return path


def load_lock(lock_root: Path, verify_all: bool = True) -> tuple[dict[str, Any], str]:
    lock_root = lock_root.resolve()
    lock_path = lock_root / LOCK_FILENAME
    digest_path = lock_root / LOCK_DIGEST_FILENAME
    raw = lock_path.read_bytes()
    actual_digest = hashlib.sha256(raw).hexdigest()
    expected_digest = digest_path.read_text(encoding="ascii").strip().split()[0]
    if actual_digest != expected_digest:
        raise ValueError(
            f"Lock digest mismatch: {actual_digest} != {expected_digest}"
        )
    lock = json.loads(raw)
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError(f"Unsupported lock schema: {lock.get('schema')!r}")
    if lock.get("status") != "prospective_v2_unrun_not_historical_recovery":
        raise ValueError(f"Unsafe lock status: {lock.get('status')!r}")
    test_policy = lock.get("test_policy", {})
    if test_policy != {
        "decode_allowed": False,
        "generated_samples_allowed": 0,
        "evaluation_entrypoint": None,
    }:
        raise ValueError("The v2 lock must disable every test decode/evaluation path")
    if verify_all:
        for key in ("source_inventory", "source_split", "environment"):
            verify_hashed_artifact(lock_root, lock["artifacts"][key])
        for record in lock["artifacts"]["source_files"]:
            verify_hashed_artifact(lock_root, record)
        verify_hashed_artifact(lock_root, lock["positive_control"])
    return lock, actual_digest


def read_csv_artifact(lock_root: Path, record: dict[str, Any]) -> list[dict[str, str]]:
    path = verify_hashed_artifact(lock_root.resolve(), record)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty locked CSV: {path}")
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = canonical_json_bytes(value)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        import os

        os.fsync(stream.fileno())
    temporary.replace(path)
