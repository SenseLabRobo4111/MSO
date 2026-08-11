"""Single-robot real-world coverage comparison for MapEx / IG-Hector / UPEN / MSO.

Reads the `/map` OccupancyGrid stream from each deployment rosbag under
``Exp/arena1/<method>_trial<i>/``, computes explored area over time, and
produces:

1. Per-deployment CSV  ``Exp/arena1/<method>_trial<i>_coverage.csv``
   columns: ``t_s, known_cells, free_cells, occ_cells, known_area_m2,
   coverage_frac``
2. A comparison figure ``figs/appendix/realworld_single_coverage.png/pdf``
   - left: coverage-over-time, mean of the two trials per method (shaded band
     = trial range)
   - right: final-coverage and exploration-time bars per method
3. A printed summary table.

Coverage fraction is the observed (known = free + occupied) area normalised by
the largest observed area reached by any deployment, which approximates the
arena's explorable extent. This avoids assuming a hand-measured arena area and
keeps the four methods on one scale.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
ARENA_DIR = ROOT.parent / "Exp" / "arena1"
FIG_DIR = ROOT / "figs" / "appendix"
OUT_PNG = FIG_DIR / "realworld_single_coverage.png"
OUT_PDF = FIG_DIR / "realworld_single_coverage.pdf"

TS = get_typestore(Stores.ROS2_HUMBLE)
OCC_THRESH = 50  # >= counts as occupied

# Match the main real-world figure (Fig. 5) colour scheme.
COLOR_FREE = (255, 255, 255)
COLOR_UNKNOWN = (213, 213, 213)
COLOR_OBS = (115, 74, 173)   # purple, as used for the "Predicted Map" obstacles

# Display order, labels, colours. Colour-blind-safe and chosen to contrast with
# the purple obstacle pixels in the map panels (red / blue / green / amber).
METHODS = {
    "mso":      {"label": "MSO (ours)",   "color": "#D32F2F", "lw": 2.9, "zorder": 9},
    "mapex":    {"label": "MapEx",        "color": "#0072B2", "lw": 2.2, "zorder": 7},
    "upen":     {"label": "UPEN",         "color": "#009E73", "lw": 2.2, "zorder": 6},
    "ighector": {"label": "IG-Hector",    "color": "#E69F00", "lw": 2.2, "zorder": 5},
}
# Trials to include per method. ighector_trial1 terminated after ~13 s (a
# failed run), so IG-Hector uses trial 2 only; the rest use both trials.
TRIALS_PER_METHOD = {
    "mso":      (1, 2),
    "mapex":    (1, 2),
    "upen":     (1, 2),
    "ighector": (2,),
}


def read_video_frame(bag_dir: Path, frac: float = 0.8):
    """Grab a representative RGB frame from the deployment's .mp4 (default at
    `frac` of the way through the run, when exploration is well underway)."""
    import cv2
    vids = list(bag_dir.glob("*.mp4"))
    if not vids:
        return None
    cap = cv2.VideoCapture(str(vids[0]))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    idx = int(np.clip(frac, 0.0, 0.99) * n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _mat_from_tf(tr) -> np.ndarray:
    """3x3 homogeneous 2D transform from a TransformStamped (ignoring z)."""
    q = tr.transform.rotation
    # yaw from quaternion
    yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    c, s = np.cos(yaw), np.sin(yaw)
    tx, ty = tr.transform.translation.x, tr.transform.translation.y
    return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1]], dtype=float)


def read_final_map_and_traj(bag_dir: Path) -> dict:
    """Return the final /map grid as an image plus the map-frame trajectory.

    Image convention: 0.0 = occupied (black), 1.0 = free (white),
    0.78 = unknown (light gray). Returns extent (m) for imshow and a
    trajectory array of (x, y) in the map frame.
    """
    last_map = None
    map_to_odom = np.eye(3)
    odom_to_base = None
    traj: list[tuple[float, float]] = []
    with AnyReader([bag_dir], default_typestore=TS) as reader:
        map_conns = [c for c in reader.connections if c.topic == "/map"]
        tf_conns = [c for c in reader.connections if c.topic in ("/tf", "/tf_static")]
        for con, t_ns, raw in reader.messages(connections=map_conns + tf_conns):
            if con.topic == "/map":
                last_map = reader.deserialize(raw, con.msgtype)
            else:
                m = reader.deserialize(raw, con.msgtype)
                for tr in m.transforms:
                    pair = (tr.header.frame_id, tr.child_frame_id)
                    if pair == ("map", "odom"):
                        map_to_odom = _mat_from_tf(tr)
                    elif pair == ("odom", "base_link"):
                        odom_to_base = _mat_from_tf(tr)
                        mb = map_to_odom @ odom_to_base
                        traj.append((mb[0, 2], mb[1, 2]))
    if last_map is None:
        return {}
    info = last_map.info
    res = float(info.resolution)
    w, h = int(info.width), int(info.height)
    ox, oy = float(info.origin.position.x), float(info.origin.position.y)
    data = np.asarray(last_map.data, dtype=np.int8).reshape(h, w)
    # RGB image matching Fig. 5: white free, light-gray unknown, purple obstacle.
    img = np.full((h, w, 3), COLOR_UNKNOWN, dtype=np.uint8)
    img[data == 0] = COLOR_FREE
    img[data >= OCC_THRESH] = COLOR_OBS
    extent = (ox, ox + w * res, oy, oy + h * res)
    return {"img": img, "extent": extent,
            "traj": np.asarray(traj, dtype=float) if traj else np.empty((0, 2))}


def read_map_series(bag_dir: Path) -> dict[str, np.ndarray]:
    """Return per-message arrays {t_s, known, free, occ, res} for /map."""
    t_list, known_list, free_list, occ_list = [], [], [], []
    res_val = None
    t0 = None
    with AnyReader([bag_dir], default_typestore=TS) as reader:
        conns = [c for c in reader.connections if c.topic == "/map"]
        if not conns:
            return {}
        for con, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, con.msgtype)
            res_val = float(msg.info.resolution)
            data = np.asarray(msg.data, dtype=np.int8)
            if t0 is None:
                t0 = t_ns
            t_list.append((t_ns - t0) / 1e9)
            known_list.append(int((data >= 0).sum()))
            free_list.append(int((data == 0).sum()))
            occ_list.append(int((data >= OCC_THRESH).sum()))
    if not t_list:
        return {}
    return {
        "t": np.asarray(t_list, dtype=float),
        "known": np.asarray(known_list, dtype=float),
        "free": np.asarray(free_list, dtype=float),
        "occ": np.asarray(occ_list, dtype=float),
        "res": res_val,
    }


def _align_map(mv, margin_px: int = 8):
    """Rotate a map so its dominant rectangular structure is axis-aligned with
    the image frame, crop to the observed content, and return
    ``(rgb_top, traj_px)`` where ``rgb_top`` has row 0 at the top and
    ``traj_px`` is the trajectory in that pixel frame (cols=x, rows=y)."""
    import cv2

    img_bottomup = mv["img"]                      # row 0 = bottom (origin lower)
    img = np.ascontiguousarray(img_bottomup[::-1])  # row 0 = top
    h, w = img.shape[:2]
    ox, ox1, oy, oy1 = mv["extent"]
    res = (ox1 - ox) / w

    # Trajectory world -> top-origin pixel coords.
    traj = mv["traj"]
    if traj.shape[0] >= 1:
        cols = (traj[:, 0] - ox) / res
        rows = (h - 1) - (traj[:, 1] - oy) / res
        traj_px = np.column_stack([cols, rows]).astype(np.float32)
    else:
        traj_px = np.empty((0, 2), dtype=np.float32)

    # Dominant orientation from observed (non-unknown) pixels.
    known = (img != COLOR_UNKNOWN[0]).any(axis=2) | \
            (img[:, :, 1] != COLOR_UNKNOWN[1]) | (img[:, :, 2] != COLOR_UNKNOWN[2])
    ys, xs = np.where(known)
    if xs.size < 20:
        return img, traj_px
    pts = np.column_stack([xs, ys]).astype(np.float32)
    angle = cv2.minAreaRect(pts)[-1]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90

    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST,
                         borderValue=COLOR_UNKNOWN)
    if traj_px.shape[0] >= 1:
        ones = np.ones((traj_px.shape[0], 1), dtype=np.float32)
        traj_px = (np.hstack([traj_px, ones]) @ M.T).astype(np.float32)

    # Crop to observed content of the rotated image.
    kr = (rot != COLOR_UNKNOWN[0]).any(axis=2) | \
         (rot[:, :, 1] != COLOR_UNKNOWN[1]) | (rot[:, :, 2] != COLOR_UNKNOWN[2])
    ys2, xs2 = np.where(kr)
    if xs2.size == 0:
        return rot, traj_px
    x0 = max(0, xs2.min() - margin_px); x1 = min(w, xs2.max() + margin_px + 1)
    y0 = max(0, ys2.min() - margin_px); y1 = min(h, ys2.max() + margin_px + 1)
    crop = rot[y0:y1, x0:x1]
    traj_px = traj_px - np.array([x0, y0], dtype=np.float32)
    return crop, traj_px


def _draw_combined_bars(fig, spec, order, summary, label="(d)"):
    """Single grouped-bar panel with a twin y-axis: per method, a solid bar
    for final explored free area (left axis, m^2) and a hatched bar for
    exploration time (right axis, s). Per-bar labels sit on their own bar so
    nothing overlaps."""
    from matplotlib.patches import Patch

    x = np.arange(len(order))
    labels = [METHODS[m]["label"] for m in order]
    colors = [METHODS[m]["color"] for m in order]
    farea = [dict((m, fa) for m, fa, _, _ in summary)[m] for m in order]
    durs = [dict((m, d) for m, _, d, _ in summary)[m] for m in order]
    w = 0.40

    ax = fig.add_subplot(spec)
    ax2 = ax.twinx()
    ba = ax.bar(x - w / 2, farea, w, color=colors, edgecolor="#1F2933",
                linewidth=0.5, zorder=3)
    bb = ax2.bar(x + w / 2, durs, w, color=colors, edgecolor="#1F2933",
                 linewidth=0.5, hatch="////", alpha=0.55, zorder=3)

    ax.set_ylim(0, max(farea) * 1.22)
    ax2.set_ylim(0, max(durs) * 1.22)
    ax.set_ylabel("Final explored free area (m$^2$)")
    ax2.set_ylabel("Exploration time (s)")
    ax.set_xticks(x, labels, rotation=22, ha="right", fontsize=9)
    ax.bar_label(ba, fmt="%.1f", padding=2, fontsize=8.5)
    ax2.bar_label(bb, fmt="%.0f s", padding=2, fontsize=8.5)
    ax.grid(True, axis="y", color="#D7DCE2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.legend(handles=[Patch(facecolor="#777", edgecolor="#1F2933", label="Area (left)"),
                       Patch(facecolor="#777", edgecolor="#1F2933", hatch="////",
                             alpha=0.55, label="Time (right)")],
              loc="upper right", fontsize=8.5, frameon=True, framealpha=0.85,
              edgecolor="#888", borderpad=0.35, handlelength=1.4)
    if label:
        ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=14,
                fontweight="bold", va="top")
    return ax, ax2


def export_panels(panel_dir: Path, order, map_vis, method_curves, grid,
                  t_max, summary) -> None:
    """Save each sub-panel as its own standalone PNG+PDF for manual layout.

    Photos and maps are exported clean (no titles/labels) so they can be
    re-arranged freely; the curve and bar charts are exported as fully
    labelled standalone plots.
    """
    import matplotlib.image as mpimg  # noqa: F401  (kept for parity)

    # Clean photos.
    for method in order:
        photo = map_vis.get(method, {}).get("photo")
        if photo is None:
            continue
        f, a = plt.subplots(figsize=(3.0, 3.0 * photo.shape[0] / photo.shape[1]))
        a.imshow(photo)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor("#3a3a3a"); sp.set_linewidth(0.6)
        f.savefig(panel_dir / f"photo_{method}.png", dpi=300, bbox_inches="tight",
                  pad_inches=0.01)
        f.savefig(panel_dir / f"photo_{method}.pdf", bbox_inches="tight",
                  pad_inches=0.01)
        plt.close(f)

    # Clean maps + trajectory, axis-aligned and padded to one common size.
    # Pass 1: rotate + crop each map; record the max content size.
    aligned: dict[str, tuple] = {}
    maxh = maxw = 0
    for method in order:
        mv = map_vis.get(method)
        if mv is None:
            continue
        crop, traj_px = _align_map(mv)
        aligned[method] = (crop, traj_px)
        maxh = max(maxh, crop.shape[0]); maxw = max(maxw, crop.shape[1])

    # Pass 2: pad every cropped map to (maxh, maxw) centred, then render.
    for method, (crop, traj_px) in aligned.items():
        ch, cw = crop.shape[:2]
        pad_t = (maxh - ch) // 2
        pad_l = (maxw - cw) // 2
        canvas = np.full((maxh, maxw, 3), COLOR_UNKNOWN, dtype=np.uint8)
        canvas[pad_t:pad_t + ch, pad_l:pad_l + cw] = crop
        tp = traj_px + np.array([pad_l, pad_t], dtype=np.float32) \
            if traj_px.shape[0] else traj_px

        f, a = plt.subplots(figsize=(3.0, 3.0 * maxh / maxw))
        a.imshow(canvas, interpolation="nearest")
        if tp.shape[0] >= 2:
            tcol = METHODS[method]["color"]
            a.plot(tp[:, 0], tp[:, 1], color=tcol, linewidth=2.0,
                   solid_capstyle="round", solid_joinstyle="round")
            a.plot(tp[0, 0], tp[0, 1], marker="*", markersize=13,
                   markerfacecolor=tcol, markeredgecolor="white",
                   markeredgewidth=1.0, linestyle="none")
            a.plot(tp[-1, 0], tp[-1, 1], marker="o", markersize=8,
                   markerfacecolor=tcol, markeredgecolor="white",
                   markeredgewidth=1.2, linestyle="none")
        a.set_xlim(0, maxw); a.set_ylim(maxh, 0)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor("#3a3a3a"); sp.set_linewidth(0.6)
        f.savefig(panel_dir / f"map_{method}.png", dpi=300, bbox_inches="tight",
                  pad_inches=0.01)
        f.savefig(panel_dir / f"map_{method}.pdf", bbox_inches="tight",
                  pad_inches=0.01)
        plt.close(f)
    if aligned:
        print(f"  aligned maps padded to {maxw}x{maxh}px (common size)")

    # Coverage-over-time curve (standalone).
    f, ax = plt.subplots(figsize=(4.6, 3.2))
    y_max = max(c["hi"].max() for c in method_curves.values())
    for method, spec in METHODS.items():
        if method not in method_curves:
            continue
        c = method_curves[method]
        if not np.allclose(c["lo"], c["hi"]):
            ax.fill_between(grid, c["lo"], c["hi"], color=spec["color"],
                            alpha=0.15, linewidth=0, zorder=spec["zorder"] - 1)
        ax.plot(grid, c["mean"], label=spec["label"], color=spec["color"],
                linewidth=spec["lw"], alpha=0.85, zorder=spec["zorder"],
                solid_capstyle="round")
    ax.set_xlim(0, t_max); ax.set_ylim(0, y_max * 1.08)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Explored free area (m$^2$)")
    ax.grid(True, color="#D7DCE2", linewidth=0.8); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.8, edgecolor="#888",
              borderpad=0.4, labelspacing=0.3, fontsize=9)
    f.tight_layout()
    f.savefig(panel_dir / "curve_area_over_time.png", dpi=300, bbox_inches="tight")
    f.savefig(panel_dir / "curve_area_over_time.pdf", bbox_inches="tight")
    plt.close(f)

    # Final-area + time grouped bars (standalone single panel, twin axis).
    f = plt.figure(figsize=(4.6, 3.4))
    gsb = f.add_gridspec(1, 1)
    _draw_combined_bars(f, gsb[0, 0], order, summary, label=None)
    f.tight_layout()
    f.savefig(panel_dir / "bars_final_area_time.png", dpi=300, bbox_inches="tight")
    f.savefig(panel_dir / "bars_final_area_time.pdf", bbox_inches="tight")
    plt.close(f)
    print(f"\n  Wrote individual panels to {panel_dir}")


def main() -> None:
    # 1. Read every deployment ------------------------------------------------
    raw: dict[tuple[str, int], dict] = {}
    for method, trials in TRIALS_PER_METHOD.items():
        for trial in trials:
            bag = ARENA_DIR / f"{method}_trial{trial}"
            if not bag.exists():
                print(f"  WARN missing {bag}")
                continue
            s = read_map_series(bag)
            if not s:
                print(f"  WARN no /map in {bag}")
                continue
            # Explored free (navigable) area in m^2 — absolute, no normalisation.
            s["free_area"] = s["free"] * s["res"] ** 2
            s["known_area"] = s["known"] * s["res"] ** 2
            raw[(method, trial)] = s
            print(
                f"  {method}_trial{trial}: {len(s['t'])} maps, "
                f"dur={s['t'][-1]:.1f}s, final free_area={s['free_area'][-1]:.2f} m2"
            )

    if not raw:
        raise SystemExit("No deployments read; check Exp/arena1 layout.")

    # 2. Per-deployment CSV ---------------------------------------------------
    for (method, trial), s in raw.items():
        out_csv = ARENA_DIR / f"{method}_trial{trial}_coverage.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "known_cells", "free_cells", "occ_cells",
                        "free_area_m2", "known_area_m2"])
            for i in range(len(s["t"])):
                w.writerow([f"{s['t'][i]:.3f}", int(s["known"][i]),
                            int(s["free"][i]), int(s["occ"][i]),
                            f"{s['free_area'][i]:.4f}", f"{s['known_area'][i]:.4f}"])

    # 3. Per-method mean curve over its included trials -----------------------
    t_max = max(s["t"][-1] for s in raw.values())
    grid = np.linspace(0, t_max, 300)

    def interp_area(s: dict) -> np.ndarray:
        return np.interp(grid, s["t"], s["free_area"],
                         left=s["free_area"][0], right=s["free_area"][-1])

    summary = []  # (method, final_area_mean, time_mean, n_trials)
    method_curves: dict[str, dict] = {}
    for method, trials in TRIALS_PER_METHOD.items():
        series = [raw[(method, t)] for t in trials if (method, t) in raw]
        if not series:
            continue
        stack = np.stack([interp_area(s) for s in series], axis=0)
        mean = stack.mean(axis=0)
        lo, hi = stack.min(axis=0), stack.max(axis=0)
        method_curves[method] = {"mean": mean, "lo": lo, "hi": hi}
        final_area = np.mean([s["free_area"][-1] for s in series])
        dur = np.mean([s["t"][-1] for s in series])
        summary.append((method, final_area, dur, len(series)))

    # 4b. Final-map visualisation per method (representative trial) -----------
    REPR_TRIAL = {"mso": 1, "mapex": 1, "upen": 1, "ighector": 2}
    map_vis: dict[str, dict] = {}
    for method in METHODS:
        trial = REPR_TRIAL[method]
        if (method, trial) not in raw:
            # fall back to any available trial
            avail = [t for (m, t) in raw if m == method]
            if not avail:
                continue
            trial = avail[0]
        bag = ARENA_DIR / f"{method}_trial{trial}"
        mv = read_final_map_and_traj(bag)
        if mv:
            mv["photo"] = read_video_frame(bag, frac=0.8)
            map_vis[method] = mv
            print(f"  {method}: rendered final map (trial{trial}), "
                  f"{mv['traj'].shape[0]} traj pts, "
                  f"photo={'yes' if mv.get('photo') is not None else 'no'}")

    # 5. Figure ---------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 13, "axes.titlesize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    order = [m for m in METHODS if m in method_curves]
    fig = plt.figure(figsize=(9.2, 7.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.92, 1.0, 1.0],
                          hspace=0.34, wspace=0.28)
    gs_photo = gs[0, :].subgridspec(1, 4, wspace=0.08)
    gs_map = gs[1, :].subgridspec(1, 4, wspace=0.12)

    # Row (a): real overhead/handheld footage per method.
    for j, method in enumerate(order):
        axp = fig.add_subplot(gs_photo[0, j])
        photo = map_vis.get(method, {}).get("photo")
        if photo is not None:
            axp.imshow(photo)
        axp.set_title(METHODS[method]["label"], fontsize=10.5,
                      color=METHODS[method]["color"], fontweight="bold", pad=3)
        axp.set_xticks([]); axp.set_yticks([])
        for sp in axp.spines.values():
            sp.set_edgecolor("#3a3a3a"); sp.set_linewidth(0.5)
        if j == 0:
            axp.set_ylabel("Real footage", fontsize=10)
            axp.text(0.04, 0.95, "(a)", transform=axp.transAxes, fontsize=14,
                     fontweight="bold", va="top", ha="left", color="white")

    # Row (b): final registered map + trajectory per method, axis-aligned and
    # padded to one common size (same as the standalone panels).
    aligned_b: dict[str, tuple] = {}
    mbh = mbw = 0
    for method in order:
        mv = map_vis.get(method)
        if mv is None:
            continue
        crop, traj_px = _align_map(mv)
        aligned_b[method] = (crop, traj_px)
        mbh = max(mbh, crop.shape[0]); mbw = max(mbw, crop.shape[1])
    for j, method in enumerate(order):
        axm = fig.add_subplot(gs_map[0, j])
        if method in aligned_b:
            crop, traj_px = aligned_b[method]
            ch, cw = crop.shape[:2]
            pt, pl = (mbh - ch) // 2, (mbw - cw) // 2
            canvas = np.full((mbh, mbw, 3), COLOR_UNKNOWN, dtype=np.uint8)
            canvas[pt:pt + ch, pl:pl + cw] = crop
            axm.imshow(canvas, interpolation="nearest")
            tp = traj_px + np.array([pl, pt], dtype=np.float32) \
                if traj_px.shape[0] else traj_px
            if tp.shape[0] >= 2:
                tcol = METHODS[method]["color"]
                axm.plot(tp[:, 0], tp[:, 1], color=tcol, linewidth=1.7,
                         solid_capstyle="round", solid_joinstyle="round")
                axm.plot(tp[0, 0], tp[0, 1], marker="*", markersize=11,
                         markerfacecolor=tcol, markeredgecolor="white",
                         markeredgewidth=0.9, linestyle="none")
                axm.plot(tp[-1, 0], tp[-1, 1], marker="o", markersize=7,
                         markerfacecolor=tcol, markeredgecolor="white",
                         markeredgewidth=1.1, linestyle="none")
            axm.set_xlim(0, mbw); axm.set_ylim(mbh, 0)
        axm.set_xticks([]); axm.set_yticks([])
        for sp in axm.spines.values():
            sp.set_edgecolor("#3a3a3a"); sp.set_linewidth(0.5)
        if j == 0:
            axm.set_ylabel("Final map", fontsize=10)
            axm.text(0.04, 0.95, "(b)", transform=axm.transAxes, fontsize=14,
                     fontweight="bold", va="top", ha="left", color="#111")

    gs_data = gs[2, :].subgridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.5)
    ax = fig.add_subplot(gs_data[0, 0])
    y_max = max(c["hi"].max() for c in method_curves.values())
    for method, spec in METHODS.items():
        if method not in method_curves:
            continue
        c = method_curves[method]
        if c["lo"] is not c["hi"]:
            ax.fill_between(grid, c["lo"], c["hi"], color=spec["color"],
                            alpha=0.15, linewidth=0, zorder=spec["zorder"] - 1)
        ax.plot(grid, c["mean"], label=spec["label"], color=spec["color"],
                linewidth=spec["lw"], alpha=0.85, zorder=spec["zorder"],
                solid_capstyle="round")
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, y_max * 1.08)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Explored free area (m$^2$)")
    ax.set_title("Single-robot explored area over time")
    ax.grid(True, color="#D7DCE2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(0.012, 0.965, "(c)", transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top")
    ax.legend(loc="lower right", frameon=True, framealpha=0.8,
              edgecolor="#888", borderpad=0.4, labelspacing=0.3)

    # Panel (d): combined area + time grouped bars.
    _draw_combined_bars(fig, gs_data[0, 1], order, summary, label="(d)")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=350, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    # 5b. Individual panels for manual layout --------------------------------
    panel_dir = FIG_DIR / "single_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    export_panels(panel_dir, order, map_vis, method_curves, grid, t_max,
                  summary)

    # 6. Summary table --------------------------------------------------------
    print("\n" + "=" * 64)
    print("SINGLE-ROBOT REAL-WORLD SUMMARY (arena1)")
    print("=" * 64)
    print(f"  {'method':12s}  {'free area':>12s}  {'expl time':>10s}  {'trials':>6s}")
    for method, fa, d, n in summary:
        print(f"  {METHODS[method]['label']:12s}  {fa:>10.2f} m2  {d:>9.1f}s  {n:>6d}")
    print(f"\n  Wrote {OUT_PNG}")
    print(f"  Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
