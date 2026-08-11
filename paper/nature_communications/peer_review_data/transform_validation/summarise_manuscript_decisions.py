"""Summarise positive-transform and low-support decisions for the manuscript."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT = HERE / "results_manuscript_reference" / "evaluated_events.csv"
OUTPUT = HERE / "table3_decision_summary.csv"
ORDER = ("observed", "predicted", "predicted_raw")


def truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> None:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            group = counts[row["input_type"]]
            positive = truth(row["gt_positive"])
            accepted = truth(row["accepted"])
            correct = truth(row["correct_within_thresholds"])
            group["events_n"] += 1
            if positive:
                group["positive_n"] += 1
                if accepted and correct:
                    group["positive_correct_accept_n"] += 1
                elif accepted:
                    group["positive_out_of_limit_accept_n"] += 1
                else:
                    group["positive_reject_n"] += 1
            else:
                group["low_support_n"] += 1
                key = "low_support_accept_n" if accepted else "low_support_reject_n"
                group[key] += 1
            group["total_accept_n" if accepted else "total_reject_n"] += 1

    fields = (
        "input_type",
        "events_n",
        "positive_n",
        "positive_correct_accept_n",
        "positive_out_of_limit_accept_n",
        "positive_reject_n",
        "low_support_n",
        "low_support_accept_n",
        "low_support_reject_n",
        "total_accept_n",
        "total_reject_n",
    )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for input_type in ORDER:
            row = {field: counts[input_type].get(field, 0) for field in fields}
            row["input_type"] = input_type
            writer.writerow(row)
    print(f"Wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
