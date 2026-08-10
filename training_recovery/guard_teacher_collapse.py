#!/usr/bin/env python3
"""Pause a campaign parent before students if epoch-30 validation is collapsed."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import signal
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-pid", type=int, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--trigger-epoch", type=int, default=30)
    parser.add_argument("--minimum-f1", type=float, default=0.01)
    parser.add_argument("--minimum-iou", type=float, default=0.005)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args()


def write_state(path: Path, payload: dict[str, object]) -> None:
    payload = {
        **payload,
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> None:
    args = parse_args()
    if args.state.exists():
        raise SystemExit(f"Refusing to overwrite guardian state: {args.state}")
    write_state(args.state, {"status": "watching", "trigger_epoch": args.trigger_epoch})
    while process_exists(args.campaign_pid):
        if args.metrics.exists():
            with args.metrics.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            triggered_rows = [
                row for row in rows if int(row["epoch"]) >= args.trigger_epoch
            ]
            if triggered_rows:
                row = min(triggered_rows, key=lambda item: int(item["epoch"]))
                f1 = float(row["validation_unknown_f1"])
                iou = float(row["validation_unknown_iou"])
                collapsed = f1 < args.minimum_f1 and iou < args.minimum_iou
                if collapsed:
                    os.kill(args.campaign_pid, signal.SIGSTOP)
                    write_state(
                        args.state,
                        {
                            "status": "campaign_parent_paused",
                            "reason": "epoch_30_validation_f1_and_iou_below_guard_floor",
                            "campaign_pid": args.campaign_pid,
                            "observed_epoch": int(row["epoch"]),
                            "validation_unknown_f1": f1,
                            "validation_unknown_iou": iou,
                            "minimum_f1_guard_floor": args.minimum_f1,
                            "minimum_iou_guard_floor": args.minimum_iou,
                            "validation_unknown_bce": float(
                                row["validation_unknown_bce"]
                            ),
                            "boundary": (
                                "teacher child continues under the original protocol; "
                                "the paused parent cannot launch students"
                            ),
                        },
                    )
                else:
                    write_state(
                        args.state,
                        {
                            "status": "guard_cleared",
                            "campaign_pid": args.campaign_pid,
                            "observed_epoch": int(row["epoch"]),
                            "validation_unknown_f1": f1,
                            "validation_unknown_iou": iou,
                        },
                    )
                return
        time.sleep(args.poll_seconds)
    write_state(
        args.state,
        {"status": "campaign_ended_before_guard_decision", "campaign_pid": args.campaign_pid},
    )


if __name__ == "__main__":
    main()
