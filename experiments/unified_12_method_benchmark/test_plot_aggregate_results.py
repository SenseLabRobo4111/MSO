"""TEST ONLY smoke coverage for final aggregate rendering."""

from __future__ import annotations

import csv
from pathlib import Path

from plot_aggregate_results import METHOD_ORDER, render


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_render_preserves_incomplete_method(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate"
    output = tmp_path / "figures"
    aggregate.mkdir()
    status_rows = []
    summary_rows = []
    metrics = (
        "test_unknown_occupied_f1",
        "test_unknown_occupied_iou",
        "test_unknown_occupied_brier",
        "test_boundary_f1_tolerance_2px",
        "parameter_count",
        "latency_batch1_median_ms",
    )
    for index, method in enumerate(METHOD_ORDER):
        complete = method != "RePaint"
        status_rows.append(
            {
                "method": method,
                "registered_seeds": 5,
                "completed_seeds": 5 if complete else 0,
                "status": "complete" if complete else "incomplete",
                "exclusion_from_complete_case_contrasts": 0 if complete else 1,
            }
        )
        row: dict[str, object] = {"method": method, "n_seeds": 5 if complete else 0}
        for metric in metrics:
            if not complete:
                mean = low = high = ""
            elif metric == "parameter_count":
                mean, low, high = 1_000_000 * (index + 1), 900_000 * (index + 1), 1_100_000 * (index + 1)
            elif metric == "latency_batch1_median_ms":
                mean, low, high = index + 1, index + 0.8, index + 1.2
            else:
                mean, low, high = 0.30 + index * 0.01, 0.29 + index * 0.01, 0.31 + index * 0.01
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = ""
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        summary_rows.append(row)
    write_tsv(aggregate / "method_status.tsv", status_rows)
    write_tsv(aggregate / "method_summary.tsv", summary_rows)
    result = render(aggregate, output)
    assert result["methods_registered"] == 12
    assert result["methods_complete"] == 11
    for suffix in ("svg", "pdf", "png", "tiff"):
        assert (output / f"uniform_benchmark_summary.{suffix}").stat().st_size > 0
