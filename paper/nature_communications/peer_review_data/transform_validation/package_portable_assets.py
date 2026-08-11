"""Create a portable transform benchmark manifest without changing event data."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = (
    HERE.parents[3]
    / "npj_robotics"
    / "transform_validation"
    / "synthetic_benchmark_a3_within_scene_n300"
)
SOURCE_MANIFEST = SOURCE_ROOT / "manifest.json"
PATH_FIELDS = (
    "target_path",
    "source_path",
    "source_original_path",
    "target_observed_path",
    "source_observed_path",
    "source_observed_original_path",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def destination_for(source: Path) -> Path:
    parts = [part.lower() for part in source.parts]
    if "assets" in parts:
        return HERE / "assets" / "generated" / source.name
    if "robot_0" in parts:
        return HERE / "assets" / "original" / "robot_0" / source.name
    if "robot_1" in parts:
        return HERE / "assets" / "original" / "robot_1" / source.name
    return HERE / "assets" / "other" / f"{sha256(source)[:12]}_{source.name}"


def main() -> None:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    copied: dict[Path, Path] = {}

    for event in manifest["events"]:
        for field in PATH_FIELDS:
            value = event.get(field)
            if not value:
                continue
            source = Path(value)
            if not source.is_file():
                raise FileNotFoundError(f"Missing source asset: {source}")
            destination = copied.setdefault(source, destination_for(source))
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or sha256(destination) != sha256(source):
                shutil.copy2(source, destination)
            event[field] = destination.relative_to(HERE).as_posix()

    manifest["dataset_root"] = "assets/original"
    manifest["portable_asset_bundle"] = True
    manifest["portable_path_base"] = "."
    manifest["portable_asset_count"] = len(copied)
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote portable manifest with {len(manifest['events'])} events")
    print(f"Copied {len(copied)} unique assets")


if __name__ == "__main__":
    main()
