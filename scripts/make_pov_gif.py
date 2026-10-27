"""Make a real-time GIF of the rendered rat's-eye frames (box_messy/frame*.png).

The first --seconds x FPS frames of an experiment are written unchanged (128x64,
grayscale) at FPS, looping, e.g.
reports/riab_gifs/data_center_adult_seed01_blender_first60s_realtime.gif.
For robot-scaled data (ROBOT_SCALING in config.yaml) the same frames span
FRAME_INTERVAL_S each, so the GIF is a sped-up view and named accordingly;
with --realtime each frame is instead shown for FRAME_INTERVAL_S.
--stride N keeps every N-th frame (offset 0), i.e. the frames one RNN input
sequence sees with stride N, each shown for the time N frames span, so the
playback speed is unchanged and the name gets a _stN suffix.
"""

import argparse
from pathlib import Path

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", type=Path, help="<data-root>/box/<stage>/exp_dim..._seedXX")
    parser.add_argument("--env", default="box_messy")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="for robot-scaled data, show each frame for FRAME_INTERVAL_S",
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="keep every N-th frame, as the RNN input does"
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "riab_gifs")
    args = parser.parse_args()

    config = yaml.safe_load((args.exp_dir / "config.yaml").read_text())
    fps = config["FPS"]
    n_frames = args.seconds * fps
    paths = sorted((args.exp_dir / args.env).glob("frame*.png"))[:n_frames]
    if len(paths) < n_frames:
        raise ValueError(f"Only {len(paths)} frames in {args.exp_dir / args.env}")
    frames = [Image.open(p).convert("L") for p in paths[:: args.stride]]

    data_root = args.exp_dir.parents[2].name
    stage = args.exp_dir.parent.name
    seed = args.exp_dir.name.rsplit("_", 1)[-1]
    frame_time = 1 / fps
    if "ROBOT_SCALING" in config:
        interval = config["ROBOT_SCALING"]["FRAME_INTERVAL_S"]
        if args.realtime:
            frame_time = interval
            name = f"first{n_frames}frames_realtime"
        else:
            # One frame per FRAME_INTERVAL_S robot seconds, played at 1/FPS s per frame.
            name = f"first{n_frames}frames_x{interval * fps:.2f}speed"
    else:
        name = f"first{args.seconds}s_realtime"
    if args.stride > 1:
        name += f"_st{args.stride}"
    # GIF delays are in 10 ms units; round the cumulative time so the total
    # length stays exact.
    step = frame_time * args.stride
    ends = [round(i * step * 100) * 10 for i in range(len(frames) + 1)]
    duration = [b - a for a, b in zip(ends, ends[1:])]
    out = args.out_dir / f"{data_root}_{stage}_{seed}_blender_{name}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out, save_all=True, append_images=frames[1:], duration=duration, loop=0
    )
    print(f"[+] Saved {out}")


if __name__ == "__main__":
    main()
