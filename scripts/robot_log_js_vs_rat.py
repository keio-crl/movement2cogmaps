"""Jensen-Shannon distance between real TurtleBot rat_walker logs and the rat data.

Speed and rotational-speed histograms are built as for the rat data
(real_data_exploration/utils/process_data.py): heading from positions smoothed
over SMOOTH_THETA seconds of rat time (calculate_thetas_smooth), rotational
speed from its per-sample change, zeros dropped, the bins and ranges of
real_data_exploration/cluster_locomotion/config.yaml, and scipy's
jensenshannon against data_agg_export_9.pkl (row 0 crawl ... 3 adult).

Each log is compared in two unit systems that share the same samples:
  raw : robot units (m/s, rad/s over the 0.1 k s save interval)
  rat : converted to rat units (speed x k / s, time / k)

Two motion sources are compared:
  odom    : what the robot did (odometry x, y)
  command : what the scaled motion model asked for (v_net_x, v_net_y)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import jensenshannon

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from real_data_exploration.utils.process_data import calculate_hist_occ  # noqa: E402
from simulation.riab_simulation.utils import (  # noqa: E402
    calculate_rot_velocity,
    calculate_thetas_smooth,
)

STAGES = ["crawl", "walk", "run", "adult"]
RAT_ROW = {"crawl": 0, "walk": 1, "run": 2, "adult": 3}
RAT_CONFIG = REPO_ROOT / "real_data_exploration" / "cluster_locomotion" / "config.yaml"
RAT_PICKLE = REPO_ROOT / "data_agg_export_9.pkl"
WALKER_CONFIG = (REPO_ROOT / "turtlebot_ros" / "src" / "turtlebot_obstacle_fusion"
                 / "config" / "rat_walker.yaml")
DEFAULT_LOGS = [REPO_ROOT / "rat_walker_logs" / "crawl_repel0.200_20261121_142448.csv"]


def log_motion(log: pd.DataFrame, source: str, interval_rat: float, k: float, smooth_s: float):
    """Speed (robot metres per rat second) and heading change per sample for one log."""
    pos = log[["x", "y"]].to_numpy()
    if source == "odom":
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        speed = np.append(speed, speed[-1]) / interval_rat
    else:
        # v_net is in robot m/s; integrate it so the heading follows the same rule as odometry.
        speed = np.hypot(log["v_net_x"], log["v_net_y"]).to_numpy() * k
        step = log[["v_net_x", "v_net_y"]].to_numpy() * interval_rat * k
        pos = np.vstack([[0.0, 0.0], np.cumsum(step, axis=0)[:-1]])
    window = int(smooth_s / interval_rat)
    window = window if window % 2 else window + 1
    dtheta = calculate_rot_velocity(calculate_thetas_smooth(pos, window))
    return speed, dtheta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--stage", default="crawl", choices=STAGES)
    args = parser.parse_args()

    cfg = yaml.safe_load(RAT_CONFIG.read_text())
    walker = yaml.safe_load(WALKER_CONFIG.read_text())["rat_walker"]["ros__parameters"]
    s, k = walker["length_scale"], walker["time_scale"]
    interval_rat = walker["save_period_k"]  # 0.1 k robot seconds = 0.1 s of rat time
    interval_robot = interval_rat * k
    rat = pd.read_pickle(RAT_PICKLE)

    logs = [pd.read_csv(path) for path in args.logs]
    for path, log in zip(args.logs, logs):
        print(f"{path.name}: {len(log)} rows ({len(log) * interval_robot:.0f} s robot, "
              f"{len(log) * interval_rat:.0f} s rat time)")

    rows = []
    for source in ["odom", "command"]:
        motion = [log_motion(log, source, interval_rat, k, cfg["SMOOTH_THETA"]) for log in logs]
        for units in ["raw", "rat"]:
            # Robot length per rat second -> robot m/s (/ k) or rat m/s (/ s).
            speed_scale = 1 / k if units == "raw" else 1 / s
            interval = interval_robot if units == "raw" else interval_rat
            speeds = [m[0] * speed_scale for m in motion]
            rots = [m[1] / interval for m in motion]
            occ_s = calculate_hist_occ(speeds, cfg["SPEED_BINS"], cfg["SPEED_RANGE"])
            occ_rs = calculate_hist_occ(rots, cfg["RS_BINS"], [-np.pi, np.pi])
            all_s, all_rs = np.concatenate(speeds), np.concatenate(rots)
            for rat_stage, row in RAT_ROW.items():
                js_s = float(jensenshannon(occ_s, rat.loc[row, "speed_hist"]))
                js_rs = float(jensenshannon(occ_rs, rat.loc[row, "rot_speed_hist"]))
                rows.append({"source": source, "units": units, "rat_stage": rat_stage,
                             "speed_js": js_s, "rs_js": js_rs, "sum_js": js_s + js_rs,
                             "speed_mean": all_s.mean(), "rs_sd": all_rs.std(),
                             "speed_out": (all_s > cfg["SPEED_RANGE"][1]).mean(),
                             "rs_out": (np.abs(all_rs) > np.pi).mean()})

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\nAgainst rat {args.stage}:")
    print(df[df.rat_stage == args.stage].drop(columns="rat_stage").round(3).to_string(index=False))
    print("\nsum_js against every rat stage:")
    print(df.pivot_table(index=["source", "units"], columns="rat_stage", values="sum_js",
                         sort=False)[STAGES].round(3).to_string())


if __name__ == "__main__":
    main()
