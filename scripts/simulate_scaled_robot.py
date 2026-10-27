"""Simulate rat trajectories scaled to a robot in a square room (no obstacles).

Lengths are scaled by s = (L - 2r) / 0.625 and times by
k = max(3.03 s sigma_v,max / v_max, 3 sigma_w,max / w_max), so that the scaled
speeds (Rayleigh, mode sigma_v s/k) and turning rates (normal, std sigma_w/k)
stay within the robot limits. Per behaviour of simulation/config.yaml:
speed_mean * s/k, speed_coherence_time * k, rotational_velocity_std / k,
rotational_velocity_coherence_time * k, wall_repel_distance * s, thigmotaxis
unchanged. Samples are saved every 0.1 k s (7200 per trajectory, 720 k s), which
keeps the frame-based training pipeline unchanged; the internal step divides
each saved interval into --substeps steps. Thetas and rotational velocities are
computed as in simulation/riab_simulation/utils.py:run_simulation.

Writes <data-root>/box/<behaviour>/exp_dim<L>_fps10_s720_seedXX/
{config.yaml, riab_simulation/*.npy, riab_simulation/motion_stats.json}.
Positions are room coordinates in [r, L - r] (ENV_DIM = L, ENV_EPS = r).
"""

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from ratinabox.Agent import Agent
from ratinabox.Environment import Environment

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from simulation.riab_simulation.utils import (  # noqa: E402
    calculate_rot_velocity,
    calculate_thetas_smooth,
)

RAT_BOX = 0.625  # movable side of the rat box (ENV_DIM - 2 ENV_EPS)
RAT_WALL_REPEL = 0.1
N_SAMPLES = 7200  # saved samples per trajectory (FPS 10 x SECONDS 720 in rat time)
RAT_FRAME_INTERVAL = 0.1
RAT_SMOOTH_THETA = 1.0  # seconds, as in riabox_simulation.py


def scaling(args, behaviours):
    room = math.sqrt(args.room_x * args.room_y)
    s = (room - 2 * args.robot_radius) / RAT_BOX
    sigma_v = max(b["SPEED_MEAN"] for b in behaviours.values())
    sigma_w = max(b["ROT_SPEED_STD"] * np.pi for b in behaviours.values())
    k_v = 3.03 * s * sigma_v / args.v_max
    k_w = 3 * sigma_w / args.w_max
    return room, s, max(k_v, k_w), k_v, k_w


def scaled_behaviour(b, s, k):
    return {
        "SPEED_MEAN": b["SPEED_MEAN"] * s / k,
        "SPEED_CT": b["SPEED_CT"] * k,
        "ROT_SPEED_STD": b["ROT_SPEED_STD"] / k,  # in units of pi rad/s, as in config.yaml
        "ROT_SPEED_CT": b["ROT_SPEED_CT"] * k,
        "THIGMOTAXIS": b["THIGMOTAXIS"],
    }


def motion_stats(vel, dt):
    """Percentiles of speed, turning rate and their rates of change at step dt."""
    speed = np.linalg.norm(vel, axis=1)
    heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    omega = np.diff(heading) / dt
    stats = {}
    for name, x in (
        ("speed_m_s", speed),
        ("abs_omega_rad_s", np.abs(omega)),
        ("abs_lin_acc_m_s2", np.abs(np.diff(speed)) / dt),
        ("abs_ang_acc_rad_s2", np.abs(np.diff(omega)) / dt),
    ):
        stats[name] = {q: float(np.percentile(x, float(q))) for q in ("50", "99", "99.7", "99.9")}
    return stats, speed, np.abs(omega)


def simulate(task):
    behaviour, seed, params, geom, out_dir = task
    room, r, s, k, substeps, v_max, w_max = geom
    frame_interval = RAT_FRAME_INTERVAL * k
    dt = frame_interval / substeps
    np.random.seed(seed)
    env = Environment(params={"scale": room - 2 * r, "aspect": 1})
    agent = Agent(env, params={
        "dt": dt,
        "speed_mean": params["SPEED_MEAN"],
        "speed_coherence_time": params["SPEED_CT"],
        "rotational_velocity_std": params["ROT_SPEED_STD"] * np.pi,
        "rotational_velocity_coherence_time": params["ROT_SPEED_CT"],
        "thigmotaxis": params["THIGMOTAXIS"],
        "wall_repel_distance": RAT_WALL_REPEL * s,
    })
    agent.velocity /= 1e3  # slow start, as in run_simulation
    for _ in range(N_SAMPLES * substeps):
        agent.update(dt=dt)

    t = np.array(agent.history["t"])
    pos = np.array(agent.history["pos"])
    vel = np.array(agent.history["vel"])
    if np.isnan(pos).any():
        raise ValueError(f"NaN in positions: {behaviour} seed {seed}")

    window = int(RAT_SMOOTH_THETA * k / dt)
    thetas = np.array(calculate_thetas_smooth(pos, window if window % 2 else window + 1))
    rot = calculate_rot_velocity(thetas)
    stats, speed, abs_omega = motion_stats(vel, dt)
    stats["frac_speed_above_v_max"] = float((speed > v_max).mean())
    stats["frac_omega_above_w_max"] = float((abs_omega > w_max).mean())
    stats["dt_s"] = dt

    # Back to one sample per saved interval, exactly as run_simulation.
    keep = slice(substeps - 1, None, substeps)
    vel_s = np.add.reduceat(vel, np.arange(0, len(vel), substeps)) / substeps
    rot_s = np.add.reduceat(rot, np.arange(0, len(rot), substeps))
    rot_s = np.hstack([rot_s[1:], rot_s[-1]])
    rot_s[rot_s > np.pi] -= 2 * np.pi
    rot_s[rot_s < -np.pi] += 2 * np.pi
    data = {
        "timestamps": t[keep],
        "positions": pos[keep] + r,
        "thetas": thetas[keep],
        "velocities": np.vstack([vel_s[1:], vel_s[-1]]),
        "rot_velocities": rot_s,
    }
    sim_dir = out_dir / "riab_simulation"
    sim_dir.mkdir(parents=True, exist_ok=True)
    for name, values in data.items():
        np.save(sim_dir / f"{name}.npy", values)
    (sim_dir / "motion_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return f"{behaviour} seed {seed:02d}: {len(data['positions'])} samples"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-x", type=float, default=4.7, help="m")
    parser.add_argument("--room-y", type=float, default=6.3, help="m")
    parser.add_argument("--robot-radius", type=float, default=0.341 / 2, help="m")
    parser.add_argument("--v-max", type=float, default=0.31, help="m/s")
    parser.add_argument("--w-max", type=float, default=1.9, help="rad/s")
    parser.add_argument("--substeps", type=int, default=36, help="internal steps per saved sample")
    parser.add_argument("--behaviours", nargs="+", default=["crawl", "walk", "run", "adult"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 27)))
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_robot_room")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base = yaml.safe_load((REPO_ROOT / "simulation" / "config.yaml").read_text())
    room, s, k, k_v, k_w = scaling(args, base["BEHAVIOURS"])
    room = round(room, 4)
    behaviours = {name: scaled_behaviour(b, s, k) for name, b in base["BEHAVIOURS"].items()}
    print(f"L = {room} m, s = {s:.4f}, k = {k:.4f} (speed {k_v:.4f}, turning {k_w:.4f})")

    config = {
        **{key: base[key] for key in ("FPS", "SECONDS", "SPATIAL_RES", "FRAME_DIM")},
        "ENV_DIM": room,
        "ENV_EPS": args.robot_radius,
        "BEHAVIOURS": behaviours,
        "WALL_REPEL_DISTANCE": RAT_WALL_REPEL * s,
        "ROBOT_SCALING": {
            "note": "FPS/SECONDS count samples (rat time); real time = FPS/SECONDS scaled by k",
            "ROOM_XY_M": [args.room_x, args.room_y],
            "ROBOT_RADIUS_M": args.robot_radius,
            "V_MAX_M_S": args.v_max,
            "W_MAX_RAD_S": args.w_max,
            "SCALE_S": s,
            "SCALE_K": k,
            "FRAME_INTERVAL_S": RAT_FRAME_INTERVAL * k,
            "DURATION_S": RAT_FRAME_INTERVAL * k * N_SAMPLES,
            "SUBSTEPS": args.substeps,
        },
    }
    tasks = []
    for behaviour in args.behaviours:
        for seed in args.seeds:
            out_dir = (args.data_root / "box" / behaviour
                       / f"exp_dim{room}_fps{base['FPS']}_s{base['SECONDS']}_seed{seed:02d}")
            if (out_dir / "riab_simulation" / "positions.npy").exists() and not args.overwrite:
                raise FileExistsError(f"{out_dir} exists (use --overwrite)")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            geom = (room, args.robot_radius, s, k, args.substeps, args.v_max, args.w_max)
            tasks.append((behaviour, seed, behaviours[behaviour], geom, out_dir))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for message in pool.map(simulate, tasks):
            print(message, flush=True)


if __name__ == "__main__":
    main()
