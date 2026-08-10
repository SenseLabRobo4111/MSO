"""Measure per-topic mean payload size and publishing rate from the recorded
rosbag2 sqlite stores under ``Exp/experiment_video/rosbag/``.

This is used to populate Tab.~\\ref{tab:ros-topics} in the appendix of the
CoRL 2026 submission with empirical bandwidth numbers rather than design
estimates. We compute, per (trial, topic):

* ``rate_hz`` --- ``message_count / duration_s``
* ``mean_bytes`` --- mean over all serialised payloads (one per row in
  ``messages``); we use ``LENGTH(data)`` directly so the figure includes the
  CDR header just like a real wire payload would.

Cross-trial aggregates are then computed across the 10 trials.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from statistics import mean, stdev

# Force UTF-8 output so we can print "±" on a Windows GBK console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
ROSBAG_ROOT = ROOT.parent / "Exp" / "experiment_video" / "rosbag"

# Topics we care about for the inter-robot bandwidth table. Multiple actual
# topic names can map to one conceptual entry in the paper.
TOPIC_GROUPS = {
    "predicted_map (local)": [
        "/robot_0/predicted_map",
        "/robot_1/predicted_map",
    ],
    "predicted_map (global)": [
        "/robot_0/predicted_map_global",
        "/robot_1/predicted_map_global",
    ],
    "merged_map": [
        "/merged_map",
    ],
    "occupancy map (raw)": [
        "/robot_0/map",
        "/robot_1/map",
    ],
    "tf": [
        "/tf",
    ],
}


def discover_db_files(trial_dir: Path) -> list[Path]:
    return sorted(trial_dir.glob("*.db3"))


def measure_one_db(db_path: Path) -> dict[str, dict[str, float]]:
    """Return {topic_name: {count, mean_bytes, total_bytes, duration_s}} for
    one .db3 file. Duration here is local to the file (max-min timestamp on
    any row of ``messages``)."""
    out: dict[str, dict[str, float]] = {}
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        cur = conn.cursor()
        # Topic id -> name
        topic_map = dict(cur.execute("SELECT id, name FROM topics").fetchall())

        # Bag duration via global min/max timestamp.
        cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM messages")
        t_min, t_max = cur.fetchone()
        duration_s = max(1e-9, (t_max - t_min) / 1e9) if t_min is not None else 0.0

        # Per-topic payload accumulator. We avoid loading all data into Python
        # by using SQL aggregates over ``LENGTH(data)``.
        cur.execute(
            "SELECT topic_id, COUNT(*), AVG(LENGTH(data)), SUM(LENGTH(data)) "
            "FROM messages GROUP BY topic_id"
        )
        for topic_id, count, avg_bytes, total_bytes in cur.fetchall():
            name = topic_map.get(topic_id)
            if name is None:
                continue
            out[name] = {
                "count": int(count),
                "mean_bytes": float(avg_bytes or 0.0),
                "total_bytes": int(total_bytes or 0),
                "duration_s": duration_s,
            }
    return out


def aggregate_trial(trial_dir: Path) -> dict[str, dict[str, float]]:
    """Merge all .db3 files under one trial directory into a single per-topic
    summary."""
    merged: dict[str, dict[str, float]] = {}
    duration_s = 0.0
    for db in discover_db_files(trial_dir):
        per_db = measure_one_db(db)
        for name, stats in per_db.items():
            slot = merged.setdefault(
                name,
                {"count": 0, "total_bytes": 0, "duration_s": 0.0},
            )
            slot["count"] += stats["count"]
            slot["total_bytes"] += stats["total_bytes"]
            slot["duration_s"] = max(slot["duration_s"], stats["duration_s"])
        duration_s = max(duration_s, max((s["duration_s"] for s in per_db.values()), default=0.0))
    # Derive mean_bytes and rate now that totals are known.
    for slot in merged.values():
        if slot["count"] > 0 and slot["duration_s"] > 0:
            slot["mean_bytes"] = slot["total_bytes"] / slot["count"]
            slot["rate_hz"] = slot["count"] / slot["duration_s"]
            slot["bytes_per_s"] = slot["total_bytes"] / slot["duration_s"]
        else:
            slot["mean_bytes"] = 0.0
            slot["rate_hz"] = 0.0
            slot["bytes_per_s"] = 0.0
    merged["_trial_duration_s"] = {"duration_s": duration_s}  # type: ignore
    return merged


def fmt_bytes(b: float) -> str:
    if b >= 1024 * 1024:
        return f"{b / 1024 / 1024:.2f} MB"
    if b >= 1024:
        return f"{b / 1024:.2f} KB"
    return f"{b:.0f} B"


def main() -> None:
    if not ROSBAG_ROOT.exists():
        raise SystemExit(f"Rosbag root not found: {ROSBAG_ROOT}")

    trial_dirs = sorted(p for p in ROSBAG_ROOT.iterdir() if p.is_dir())
    print(f"Discovered {len(trial_dirs)} trials under {ROSBAG_ROOT}\n")

    # trial_name -> {topic_name: stats}
    per_trial: dict[str, dict[str, dict[str, float]]] = {}
    for tdir in trial_dirs:
        per_trial[tdir.name] = aggregate_trial(tdir)

    # Per-trial table -------------------------------------------------------
    print("=" * 110)
    print("PER-TRIAL TOPIC STATISTICS")
    print("=" * 110)
    for trial, stats in per_trial.items():
        dur = stats.get("_trial_duration_s", {}).get("duration_s", 0.0)
        print(f"\n[{trial}]  duration {dur:.1f}s")
        rows = []
        for topic, s in stats.items():
            if topic.startswith("_"):
                continue
            if s["count"] == 0:
                continue
            rows.append(
                (
                    topic,
                    s["count"],
                    s["rate_hz"],
                    s["mean_bytes"],
                    s["bytes_per_s"],
                )
            )
        rows.sort(key=lambda r: -r[4])
        print(f"  {'topic':50s}  {'count':>6s}  {'rate':>8s}  {'msg':>10s}  {'bytes/s':>11s}")
        for topic, cnt, rate, mb, bps in rows:
            print(
                f"  {topic:50s}  {cnt:>6d}  {rate:>7.2f}Hz  {fmt_bytes(mb):>10s}  {fmt_bytes(bps):>11s}/s"
            )

    # Cross-trial aggregate per group --------------------------------------
    print("\n" + "=" * 110)
    print("CROSS-TRIAL AGGREGATE (mean over trials, only trials where the topic was non-empty)")
    print("=" * 110)
    for label, names in TOPIC_GROUPS.items():
        rates: list[float] = []
        means: list[float] = []
        bps: list[float] = []
        n_trials = 0
        for trial, stats in per_trial.items():
            for n in names:
                if n in stats and stats[n]["count"] > 0:
                    rates.append(stats[n]["rate_hz"])
                    means.append(stats[n]["mean_bytes"])
                    bps.append(stats[n]["bytes_per_s"])
                    n_trials += 1
        if not rates:
            print(f"\n  {label}: no data in any trial")
            continue

        def _fmt(vals: list[float], unit: str) -> str:
            if len(vals) == 1:
                return f"{vals[0]:.2f}{unit}"
            return f"{mean(vals):.2f}{unit}±{stdev(vals):.2f}"

        print(f"\n  {label}  (n={n_trials} samples)")
        print(f"    rate     : {_fmt(rates, 'Hz')}")
        print(f"    mean msg : {fmt_bytes(mean(means))}  ({mean(means):.0f}B raw mean)")
        print(f"    bytes/s  : {fmt_bytes(mean(bps))}/s")


if __name__ == "__main__":
    main()
