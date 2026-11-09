"""Jensen-Shannon distance between simulated training trajectories and the rat data.

The rat side is data/cluster_locomotion/by_day/data_agg_export_9.pkl (real rats,
one row per developmental stage: 0 crawl, 1 walk, 2 run, 3 adult; columns
speed_hist, rot_speed_hist, folded transition_matrix), made with
data_agg_export_9_config.yaml next to it. The simulated side is built the same
way as in real_data_exploration/cluster_locomotion/03_riab_grid_search.py:
speed and rotational-speed histograms (calculate_hist_occ, zeros dropped) and the
folded transition matrix (generate_tm_loop, window THRESHOLD_SEC, discount DF with
the per-frame damping converted to the trajectory's frame interval), compared with
compare_parameters' metrics (scipy jensenshannon; mean over start bins for the
transition matrix).

Robot-room data are compared twice:
  raw : robot units as recorded (m/s, rad/s, the 5.44 m room, 0.725 s per sample)
  rat : converted back to rat units (speed x k/s, time / k, positions into the box)

Writes <out-dir>/movement_js.csv (every dataset x stage x rat stage) and
movement_js_matched.csv (each stage against the same rat stage) and prints them.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import jensenshannon

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from real_data_exploration.utils.process_data import calculate_hist_occ  # noqa: E402
from real_data_exploration.utils.transition_matrix_helper import (  # noqa: E402
    fold_transition_matrix,
    generate_tm_loop,
    reord_folded_transition_matrix,
    reorder_transition_matrix,
)
from utils.metrics import get_jensen_shannon  # noqa: E402

STAGES = ["crawl", "walk", "run", "adult"]
# Row of data_agg_export_9.pkl holding each rat stage.
RAT_ROW = {"crawl": 0, "walk": 1, "run": 2, "adult": 3}
DATASETS = [
    "paper:data:raw",
    "paper_o1000:data_box_o1000:raw",
    "paper_o2000:data_box_o2000:raw",
    "paper_o3000:data_box_o3000:raw",
    "obstacles:data_obst4grid:raw",
    "robot_raw:data_robot_room:raw",
    "robot_rat:data_robot_room:rat",
    "robot_o1000_raw:data_robot_room_o1000:raw",
    "robot_o1000_rat:data_robot_room_o1000:rat",
    "robot_o2000_raw:data_robot_room_o2000:raw",
    "robot_o2000_rat:data_robot_room_o2000:rat",
]
RAT_DIR = REPO_ROOT / "data" / "cluster_locomotion" / "by_day"
RAT_BOX, RAT_EPS = 0.635, 0.005


def load_trajectories(root: Path, stage: str, seeds: list[int], units: str):
    """Speeds (m/s), rotational speeds (rad/s), positions and box size, frame interval (s)."""
    speeds, rot_speeds, xs, ys = [], [], [], []
    for seed in seeds:
        exp_dir = next((root / "box" / stage).glob(f"exp_dim*_seed{seed:02d}"))
        sim = exp_dir / "riab_simulation"
        config = yaml.safe_load((exp_dir / "config.yaml").read_text())
        scaling = config.get("ROBOT_SCALING")
        pos = np.load(sim / "positions.npy")
        speed = np.linalg.norm(np.load(sim / "velocities.npy"), axis=1)
        # rot_velocities.npy holds the heading change per sample (rad / frame).
        rot = np.load(sim / "rot_velocities.npy").ravel()
        if scaling is None:
            interval, box = 1 / config["FPS"], RAT_BOX
        elif units == "raw":
            interval, box = scaling["FRAME_INTERVAL_S"], config["ENV_DIM"]
        else:
            s, k = scaling["SCALE_S"], scaling["SCALE_K"]
            interval, box = scaling["FRAME_INTERVAL_S"] / k, RAT_BOX
            pos = (pos - config["ENV_EPS"]) / s + RAT_EPS
            speed = speed * k / s
        speeds.append(speed)
        rot_speeds.append(rot / interval)
        xs.append(pos[:, 0])
        ys.append(pos[:, 1])
    return speeds, rot_speeds, xs, ys, box, interval


def simulated_metrics(root, stage, seeds, units, cfg):
    speeds, rots, xs, ys, box, interval = load_trajectories(root, stage, seeds, units)
    occ_s = calculate_hist_occ(speeds, cfg["SPEED_BINS"], cfg["SPEED_RANGE"])
    occ_rs = calculate_hist_occ(rots, cfg["RS_BINS"], [-np.pi, np.pi])
    fps = 1 / interval
    bins = cfg["B"]
    tm, _ = generate_tm_loop(
        xs, ys, box, bins,
        cfg["DF"], 1 / (2 * fps), int(round(fps * cfg["THRESHOLD_SEC"])), cfg["ATOL"],
        cfg["SIGMA_SMOOTHING"],
    )
    tm_reord = reorder_transition_matrix(tm, bins)
    tm_quarter, _, _, _ = fold_transition_matrix(tm, tm_reord, bins, bins**2 // 2)
    tm_fold = reord_folded_transition_matrix(tm_quarter, bins)
    stats = {
        "speed_mean": float(np.concatenate(speeds).mean()),
        "rs_sd": float(np.concatenate(rots).std()),
        "speed_out_of_range": float((np.concatenate(speeds) > cfg["SPEED_RANGE"][1]).mean()),
        "rs_out_of_range": float((np.abs(np.concatenate(rots)) > np.pi).mean()),
    }
    return occ_s, occ_rs, tm_fold, stats


def js_all(sim, rat_row):
    occ_s, occ_rs, tm_fold, _ = sim
    return (
        float(jensenshannon(occ_s, rat_row["speed_hist"])),
        float(jensenshannon(occ_rs, rat_row["rot_speed_hist"])),
        float(get_jensen_shannon(tm_fold, rat_row["transition_matrix"], axis=(-1, -2)).mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        help="label:data-root:units (units raw or rat; rat only matters for robot rooms)")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 18)),
                        help="training seeds (default 1-17)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "movement_js")
    args = parser.parse_args()

    cfg = yaml.safe_load((RAT_DIR / "data_agg_export_9_config.yaml").read_text())
    rat = pd.read_pickle(RAT_DIR / "data_agg_export_9.pkl")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in args.datasets:
        label, root, units = spec.split(":")
        for stage in STAGES:
            sim = simulated_metrics(REPO_ROOT / root, stage, args.seeds, units, cfg)
            for rat_stage, row in RAT_ROW.items():
                js_s, js_rs, js_tm = js_all(sim, rat.loc[row])
                rows.append({"dataset": label, "root": root, "units": units, "stage": stage,
                             "rat_stage": rat_stage, "speed_js": js_s, "rs_js": js_rs,
                             "tm_js": js_tm, "sum_js": js_s + js_rs + js_tm, **sim[3]})
            print(f"{label} {stage}: speed mean {sim[3]['speed_mean']:.3f} m/s, "
                  f"rs sd {sim[3]['rs_sd']:.2f} rad/s", flush=True)

    with open(args.out_dir / "movement_js.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    matched = df[df["stage"] == df["rat_stage"]]
    matched.to_csv(args.out_dir / "movement_js_matched.csv", index=False)
    for metric in ["speed_js", "rs_js", "tm_js", "sum_js"]:
        print(f"\n{metric}, each stage against the same rat stage")
        table = matched.pivot_table(index="dataset", columns="stage", values=metric, sort=False)
        table["total"] = table.sum(axis=1)
        print(table[STAGES + ["total"]].round(3).to_string())
    print(f"\n[+] Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
