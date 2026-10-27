"""Per-environment, per-stage movement summary of the simulated trajectories.

For each trajectory (seeds 1-26): path length, mean speed, mean |turning rate|,
fraction of sharp turns and the number of visited 25x25 position bins (the
analysis bins; obstacle bins are excluded from the count of free bins).
A sharp turn is a heading change above SHARP_TURN_RAD per saved sample, i.e.
above 90 deg/s in rat time (0.1 s samples); robot-scaled samples keep the
per-sample heading change, so the threshold is the same per sample.
Values are averaged over seeds; "union" counts bins visited by any seed.
"""

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from utils.metrics import bin_data  # noqa: E402

STAGES = ["crawl", "walk", "run", "adult"]
N_BINS = 25
SHARP_TURN_RAD = np.pi / 2 * 0.1


def obstacle_mask(config, env_dim):
    mask = np.zeros((N_BINS, N_BINS), bool)
    bin_size = env_dim / N_BINS
    for obstacle in config.get("OBSTACLES") or []:
        (cx, cy), (w, d) = obstacle["CENTER"], np.broadcast_to(obstacle["SIZE"], 2)
        x0, x1, y0, y1 = np.round(
            np.array([cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2]) / bin_size
        ).astype(int)
        mask[x0:x1, y0:y1] = True
    return mask


def summarize(data_root, label, env_dim):
    rows = []
    for stage in STAGES:
        dirs = sorted(glob.glob(str(REPO_ROOT / data_root / "box" / stage / "exp_*_seed*")))[:26]
        config = yaml.safe_load(Path(dirs[0], "config.yaml").read_text())
        interval = config.get("ROBOT_SCALING", {}).get("FRAME_INTERVAL_S", 1 / config["FPS"])
        mask = obstacle_mask(config, env_dim)
        per, union = [], np.zeros((N_BINS, N_BINS))
        for d in dirs:
            sim = Path(d, "riab_simulation")
            pos = np.load(sim / "positions.npy")
            speed = np.linalg.norm(np.load(sim / "velocities.npy"), axis=1)
            dtheta = np.abs(np.load(sim / "rot_velocities.npy"))
            occ = bin_data(pos, n_bins=N_BINS, limits=[(0, env_dim), (0, env_dim)])
            union += occ
            per.append([
                np.linalg.norm(np.diff(pos, axis=0), axis=1).sum(),
                speed.mean(),
                dtheta.mean() / interval,
                (dtheta > SHARP_TURN_RAD).mean(),
                ((occ > 0) & ~mask).sum(),
            ])
        p = np.mean(per, axis=0)
        rows.append({
            "environment": label, "stage": stage, "n_seeds": len(dirs),
            "box_side_m": env_dim, "duration_s": round(len(pos) * interval, 1),
            "path_length_m": round(p[0], 2), "path_length_per_side": round(p[0] / env_dim, 1),
            "mean_speed_m_s": round(p[1], 4), "mean_abs_turn_rad_s": round(p[2], 3),
            "sharp_turn_frac": round(p[3], 4),
            "visited_bins_per_traj": round(p[4], 1),
            "visited_bins_union": int(((union > 0) & ~mask).sum()),
            "free_bins": int((~mask).sum()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports/movement_summary/movement_summary.csv")
    args = parser.parse_args()
    envs = [
        ("data", "paper box", 0.635),
        ("data_robot_room", "robot-scaled room", 5.4415),
        ("data_obst4grid", "4 pillars", 0.635),
        ("data_center", "centre block", 0.635),
        ("data_centerU", "centre + U + thin wall", 0.635),
    ]
    rows = [r for root, label, dim in envs for r in summarize(root, label, dim)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[+] Saved {args.out}")


if __name__ == "__main__":
    main()
