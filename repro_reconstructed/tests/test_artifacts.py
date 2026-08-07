from __future__ import annotations

import importlib.util
from pathlib import Path


def test_committed_candidate_hashes_and_strict_loading():
    package_root = Path(__file__).resolve().parents[1]
    tool_path = package_root / "tools" / "verify_artifacts.py"
    spec = importlib.util.spec_from_file_location("verify_artifacts", tool_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    result = module.verify(package_root)
    assert result["status"] == "verified"
    assert len(result["candidate_loads"]) == 2
    assert all(item["strict_load"] for item in result["candidate_loads"])
    assert all(
        item["trainable_parameter_count"] == 342_771
        for item in result["candidate_loads"]
    )
