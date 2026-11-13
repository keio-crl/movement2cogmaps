"""Linear and angular speed over time of the RatInABox trajectories (paper data).

Reads data/box/<stage>/exp_dim0.635_fps10_s720_seed<NN>/riab_simulation/ and
writes one PNG per stage (crawl, walk, run, adult) with two panels sharing the
time axis, in the layout of ou_cmd_<stage>.png:
  - linear speed (|velocity|, m/s) with the mean of the Rayleigh speed
    distribution, speed_mean * sqrt(pi / 2), as a dashed line;
  - angular speed (rot_velocities.npy * FPS, rad/s; the heading change derived
    from the positions, i.e. the signal fed to the models) with +-rot_std, the
    std of the rotational-velocity OU process, as dotted lines.
The behaviour parameters are those of simulation/config.yaml. Axis limits are
shared across stages.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
STAGES = ["crawl", "walk", "run", "adult"]


def load(stage, seed, seconds, fps):
    d = REPO / "data" / "box" / stage / f"exp_dim0.635_fps10_s720_seed{seed:02d}" / "riab_simulation"
    n = int(seconds * fps)
    t = np.load(d / "timestamps.npy")[:n]
    speed = np.linalg.norm(np.load(d / "velocities.npy")[:n], axis=1)
    rot = np.load(d / "rot_velocities.npy")[:n] * fps
    return t, speed, rot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--out-dir", type=Path, default=REPO / "reports" / "riab_speed")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((REPO / "simulation" / "config.yaml").read_text())
    fps = config["FPS"]
    data = {s: load(s, args.seed, args.seconds, fps) for s in STAGES}
    params = {s: config["BEHAVIOURS"][s] for s in STAGES}
    speed_top = 1.05 * max(max(d[1].max(), p["SPEED_MEAN"] * np.sqrt(np.pi / 2))
                           for d, p in zip(data.values(), params.values()))
    rot_top = 1.05 * max(max(np.abs(d[2]).max(), p["ROT_SPEED_STD"] * np.pi)
                         for d, p in zip(data.values(), params.values()))

    for stage, (t, speed, rot) in data.items():
        p = params[stage]
        rayleigh = p["SPEED_MEAN"] * np.sqrt(np.pi / 2)
        rot_std = p["ROT_SPEED_STD"] * np.pi
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        a1.plot(t, speed, color="tab:blue", lw=1.2, label="linear speed")
        a1.axhline(rayleigh, color="tab:green", ls="--", lw=1, label=f"Rayleigh mean {rayleigh:.3f}")
        a1.set_ylim(0, speed_top)
        a1.set_ylabel("linear speed [m/s]")
        a1.legend(loc="upper right", fontsize=9, ncol=2)

        a2.plot(t, rot, color="tab:orange", lw=1.2, label="angular speed")
        a2.axhline(0, color="black", lw=0.6)
        for sgn in (1, -1):
            a2.axhline(sgn * rot_std, color="tab:red", ls=":", lw=1,
                       label=f"rot_std ±{rot_std:.2f}" if sgn > 0 else None)
        a2.set_ylim(-rot_top, rot_top)
        a2.set_ylabel("angular speed [rad/s]")
        a2.set_xlabel("time [s]")
        a2.legend(loc="upper right", fontsize=9, ncol=2)

        fig.suptitle(f"{stage}: speed_mean={p['SPEED_MEAN']:.3f} m/s, speed_ct={p['SPEED_CT']:g} s, "
                     f"rot_std={rot_std:.3f} rad/s, rot_ct={p['ROT_SPEED_CT']:g} s, dt={1 / fps:g} s "
                     f"(seed {args.seed:02d})", fontsize=10)
        fig.tight_layout()
        out = args.out_dir / f"{stage}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[+] {out}  mean speed {speed.mean():.3f} m/s, angular sd {rot.std():.2f} rad/s")


if __name__ == "__main__":
    main()
