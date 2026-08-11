"""Driver to render real_world progression figures + GIFs for all scenes.

For each scene S:
  1. Calibrate sense{S}-1 and sense{S}-2 (UAV joint fit).
  2. Render sense{S}-1: MSO style (predicted_map_global), full bag.
  3. Render sense{S}-2: Frontier-style (observation-only), sample only the
     first half of bag time but display labels matching trial 1 timestamps.

Skip a scene by passing its number on the command line, e.g.
  py -3 run_real_world_all.py 2 3 4 5
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rosbags.rosbag2 import Reader

ROOT = Path(__file__).resolve().parent
EXP_ROOT = ROOT.parent / "Exp" / "experiment_video"
PY = sys.executable


def bag_duration_s(scene: int, trial: int) -> float:
    bag = EXP_ROOT / "rosbag" / f"sense{scene}-{trial}"
    if not bag.exists():
        raise FileNotFoundError(bag)
    with Reader(bag) as r:
        return (r.duration or 0) / 1e9


def run(env_extra: dict, script: str, label: str):
    env = {**os.environ, **{k: str(v) for k, v in env_extra.items()}, "PYTHONUNBUFFERED": "1"}
    print(f"\n--- {label} ---")
    res = subprocess.run([PY, "-u", script], cwd=str(ROOT), env=env)
    if res.returncode != 0:
        raise RuntimeError(f"{label} exited {res.returncode}")


def process_scene(scene: int):
    print(f"\n===================== sense{scene} =====================")

    # 0. Calibrate both trials (calib script reads scene/trial from sys.argv)
    for tr in (1, 2):
        print(f"\n--- calib sense{scene}-{tr} ---")
        subprocess.run([PY, "-u", "plot_real_world_calib.py", str(scene), str(tr)],
                        cwd=str(ROOT), check=True)

    dur1 = bag_duration_s(scene, 1)
    dur2 = bag_duration_s(scene, 2)
    half2 = dur2 * 0.5
    fracs1 = (0.20, 0.40, 0.60, 0.80, 1.00)
    t_display = ",".join(f"{dur1 * f:.2f}" for f in fracs1)
    print(f"sense{scene}: bag durations T1={dur1:.1f}s, T2={dur2:.1f}s "
          f"(half={half2:.1f}s)")
    print(f"  trial2 will display at t = {t_display}")

    # Invalidate panel caches so per-trial settings take effect
    out_root = ROOT / "figs" / "real_world"
    for tr in (1, 2):
        cache = out_root / f"sense{scene}-{tr}" / "_panels.pkl"
        if cache.exists():
            cache.unlink()

    # 1. Trial 1 (MSO) - progression + gif
    base1 = {"MSO_SCENE": scene, "MSO_TRIAL": 1, "MSO_LABEL": "MSO"}
    run(base1, "plot_real_world_progression.py", f"sense{scene}-1 MSO progression")
    run(base1, "plot_real_world_gif.py",        f"sense{scene}-1 MSO gif")

    # 2. Trial 2 (Frontier-style) - progression + gif
    base2 = {
        "MSO_SCENE": scene, "MSO_TRIAL": 2, "MSO_LABEL": "Frontier",
        "MSO_OBSERVATION_ONLY": 1,
    }
    base2_prog = {**base2,
                  "MSO_T_FRACTIONS": "0.10,0.20,0.30,0.40,0.50",
                  "MSO_T_DISPLAY": t_display}
    run(base2_prog, "plot_real_world_progression.py",
        f"sense{scene}-2 Frontier progression")
    base2_gif = {**base2,
                 "MSO_GIF_BAG_DURATION_S": f"{half2:.4f}",
                 "MSO_GIF_DISP_DURATION_S": f"{dur1:.4f}"}
    run(base2_gif, "plot_real_world_gif.py",
        f"sense{scene}-2 Frontier gif")


def main():
    scenes = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else [2, 3, 4, 5]
    for s in scenes:
        process_scene(s)


if __name__ == "__main__":
    main()
