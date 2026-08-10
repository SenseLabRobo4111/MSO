#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path("/mnt/data1/MSO_12method_benchmark_20260808")
SOURCE_ROOT = ROOT / "sources"
OUTPUT_ROOT = ROOT / "compatibility"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def first_matches(root: Path, patterns: tuple[str, ...], limit: int = 80) -> list[str]:
    matches: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if any(path.match(pattern) or relative.lower().endswith(pattern.lower()) for pattern in patterns):
            matches.append(relative)
        if len(matches) >= limit:
            break
    return matches


def inspect_repository(root: Path) -> dict[str, object]:
    python_files = [
        path for path in root.rglob("*.py") if path.is_file() and ".git" not in path.parts
    ]
    framework_hits: dict[str, int] = {"torch": 0, "tensorflow": 0, "jax": 0}
    input_terms = {"mask": 0, "image": 0, "resolution": 0}
    syntax_errors: list[dict[str, str]] = []
    for path in python_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for key in framework_hits:
            framework_hits[key] += len(re.findall(rf"\b(?:import|from)\s+{key}\b", text))
        for key in input_terms:
            input_terms[key] += len(re.findall(rf"\b{key}\b", text, flags=re.IGNORECASE))
        try:
            compile(text, str(path), "exec")
        except SyntaxError as exc:
            syntax_errors.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "error": f"{exc.msg} at line {exc.lineno}",
                }
            )
    metadata = first_matches(
        root,
        (
            "requirements.txt",
            "requirements-dev.txt",
            "environment.yml",
            "environment.yaml",
            "setup.py",
            "pyproject.toml",
            "Dockerfile",
        ),
    )
    entrypoints = [
        path.relative_to(root).as_posix()
        for path in python_files
        if re.search(r"(^|_)(train|test|eval|inference|demo)(_|\.|$)", path.name, re.IGNORECASE)
    ][:100]
    license_files = [
        path.relative_to(root).as_posix()
        for path in sorted(root.iterdir())
        if path.is_file() and path.name.lower().startswith(("license", "copying"))
    ]
    return {
        "repository": root.name,
        "python_file_count": len(python_files),
        "framework_import_hits": framework_hits,
        "input_term_hits": input_terms,
        "metadata_files": metadata,
        "entrypoint_candidates": entrypoints,
        "license_files": license_files,
        "syntax_error_count": len(syntax_errors),
        "syntax_errors": syntax_errors[:40],
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = [
        inspect_repository(path)
        for path in sorted(SOURCE_ROOT.iterdir())
        if path.is_dir() and (path / ".git").is_dir()
    ]
    report = {
        "status": "source_compatibility_only",
        "result_claim_allowed": False,
        "repositories": records,
    }
    report_path = OUTPUT_ROOT / "source_compatibility.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT_ROOT / "SHA256SUMS").write_text(
        f"{digest(report_path)}  {report_path.name}\n", encoding="utf-8"
    )
    print(json.dumps({"repositories": len(records), "report_sha256": digest(report_path)}))


if __name__ == "__main__":
    main()
