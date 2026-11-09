"""Map robot-room trajectories back into the rat box so they render like the paper data.

Positions p in [r, L - r] become (p - r) / s + RAT_EPS in [RAT_EPS, RAT_DIM - RAT_EPS];
velocities are multiplied by k / s, rotational velocities by k and timestamps divided
by k (rat units). Thetas are unchanged. The config drops ROBOT_SCALING so Blender uses
the positions as they are, and ENV_DIM / ENV_EPS are those of the rat box.

Writes <out-root>/box/<behaviour>/exp_dim0.635_fps10_s720_seedXX/{config.yaml, riab_simulation/*.npy}.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

RAT_DIM = 0.635
RAT_EPS = 0.005


def convert(src: Path, dst: Path) -> None:
    with open(src / "config.yaml") as f:
        config = yaml.safe_load(f)
    scaling = config.pop("ROBOT_SCALING")
    s, k, r = scaling["SCALE_S"], scaling["SCALE_K"], config["ENV_EPS"]
    sim_src, sim_dst = src / "riab_simulation", dst / "riab_simulation"
    sim_dst.mkdir(parents=True, exist_ok=True)
    np.save(sim_dst / "positions.npy", (np.load(sim_src / "positions.npy") - r) / s + RAT_EPS)
    np.save(sim_dst / "velocities.npy", np.load(sim_src / "velocities.npy") * k / s)
    np.save(sim_dst / "rot_velocities.npy", np.load(sim_src / "rot_velocities.npy") * k)
    np.save(sim_dst / "timestamps.npy", np.load(sim_src / "timestamps.npy") / k)
    np.save(sim_dst / "thetas.npy", np.load(sim_src / "thetas.npy"))
    config.update(ENV_DIM=RAT_DIM, ENV_EPS=RAT_EPS)
    config["CONVERTED_FROM_ROBOT_ROOM"] = {"SOURCE": str(src), **scaling}
    with open(dst / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, default=Path("data_robot_room"))
    parser.add_argument("--out-root", type=Path, default=Path("data_robot_room_ratrange"))
    parser.add_argument("--behaviours", nargs="+", default=["crawl", "walk", "run", "adult"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 27)))
    args = parser.parse_args()
    for b in args.behaviours:
        for seed in args.seeds:
            name = f"seed{seed:02d}"
            (src,) = (args.src_root / "box" / b).glob(f"exp_dim*_fps10_s720_{name}")
            dst = args.out_root / "box" / b / f"exp_dim{RAT_DIM}_fps10_s720_{name}"
            convert(src, dst)
        print(f"{b}: {len(args.seeds)} trajectories -> {args.out_root / 'box' / b}")


if __name__ == "__main__":
    main()
