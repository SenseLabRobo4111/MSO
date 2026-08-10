"""Resolve and hash the repository-local Python dependency closure."""

from __future__ import annotations

import ast
from collections import deque
from pathlib import Path
from typing import Iterable

from .hashing import sha256_file


PROSPECTIVE_ENTRYPOINTS = (
    "training_recovery/build_v2_lock.py",
    "training_recovery/run_v2_locked_campaign.py",
    "training_recovery/train_v2_locked.py",
    "training_recovery/generate_v2_locked_dataset.py",
    "training_recovery/validate_v2_dataset.py",
    "training_recovery/evaluate_v2_positive_control.py",
    "training_recovery/audit_v2_gates.py",
    "training_recovery/summarize_v2_multiseed.py",
    "integrated_offline/reference_registrar.py",
    "integrated_offline/run_paired_ablation.py",
    "sensemap/explore_model/SenseMapNet.py",
)


def _module_candidates(
    repository_root: Path,
    source: Path,
    module: str,
    level: int,
) -> list[Path]:
    parts = [part for part in module.split(".") if part]
    candidates: list[Path] = []
    if level:
        base = source.parent
        for _ in range(level - 1):
            base = base.parent
        candidates.extend((base.joinpath(*parts).with_suffix(".py"),))
        candidates.extend((base.joinpath(*parts, "__init__.py"),))
    else:
        roots = [repository_root, source.parent]
        if source.relative_to(repository_root).parts:
            roots.append(repository_root / source.relative_to(repository_root).parts[0])
        for root in roots:
            candidates.extend((root.joinpath(*parts).with_suffix(".py"),))
            candidates.extend((root.joinpath(*parts, "__init__.py"),))
    return candidates


def _imports(path: Path, repository_root: Path) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    resolved: set[Path] = set()
    for node in ast.walk(tree):
        requests: list[tuple[str, int]] = []
        if isinstance(node, ast.Import):
            requests.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            requests.append((node.module or "", int(node.level)))
            for alias in node.names:
                if alias.name != "*":
                    suffix = ".".join(
                        value for value in (node.module or "", alias.name) if value
                    )
                    requests.append((suffix, int(node.level)))
        for module, level in requests:
            for candidate in _module_candidates(
                repository_root, path, module, level
            ):
                try:
                    candidate = candidate.resolve()
                    candidate.relative_to(repository_root)
                except (FileNotFoundError, ValueError):
                    continue
                if candidate.is_file():
                    resolved.add(candidate)
                    break
    return resolved


def local_import_closure(
    repository_root: Path, entrypoints: Iterable[str]
) -> list[Path]:
    """Return all existing entrypoints and recursively imported local modules."""

    root = repository_root.resolve()
    queue: deque[Path] = deque()
    for relative in entrypoints:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("software entrypoint escapes the repository") from error
        if not path.is_file():
            raise FileNotFoundError(f"missing software entrypoint: {relative}")
        queue.append(path)
    visited: set[Path] = set()
    while queue:
        path = queue.popleft()
        if path in visited:
            continue
        visited.add(path)
        queue.extend(sorted(_imports(path, root).difference(visited)))
    return sorted(visited)


def hash_local_import_closure(
    repository_root: Path, entrypoints: Iterable[str]
) -> dict[str, str]:
    root = repository_root.resolve()
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in local_import_closure(root, entrypoints)
    }
