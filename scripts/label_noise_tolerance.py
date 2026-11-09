"""Tolerance of the cognitive-map metrics to noise in the analysis labels.

The saved activity of each model (no retraining, no forward pass) is analysed
again after adding noise only to the positions and head directions that the
metrics are computed against:

  position noise -> SIr, place-cell candidates (SIr > 0.3), sRSA
  HD noise       -> SId, RVL, HD-cell candidates (SId > 0.2 or RVL > 0.3)

Cell counts follow selected_units_analysis: place = place candidates that are
not HD candidates, HD = the reverse, conjunctive = both; the label without
noise is used for the other candidate set.

Two noise types are used, each repeated with several random draws:
  white : independent Gaussian noise in every frame
  drift : Ornstein-Uhlenbeck noise (same stationary SD, time constant
          DRIFT_TAU_FRAMES), generated on the original frame timeline of each
          analysis trajectory so the 10 stride offsets share it
Positions are reflected at the reachable limits [ENV_EPS, ENV_DIM - ENV_EPS];
head directions are wrapped to [-pi, pi).

Statistics follow R_stats/JonckTest_WilcoxTest.r on the 500 units per stage:
Jonckheere-Terpstra (walk < run < adult, 10,000 permutations) and pairwise
one-sided Wilcoxon rank-sum tests with BH correction.

Position and HD decoding errors (linear regression as in RNNActiviter.pos_hd_decoding,
fitted on noisy labels) are scored against the noisy labels and the true ones.

Writes <out-dir>/<model>.json (one record per condition) as each model finishes.
"""

import argparse
import json
import os
import sys

# Keep each worker to a couple of BLAS threads on the shared machine.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, rankdata

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.spatial_units import PolarMaps, RateMaps  # noqa: E402

STAGES = ["walk", "run", "adult"]
HIST = {"walk": "crawl", "run": "crawl_walk", "adult": "crawl_walk_run"}
TAIL = "_f1_w9_st10_fss4_do[0,0,0]_lat500_nlsigmoid_hreg0.0_wreg0.0_s"
STRIDE = 10
DRIFT_TAU_FRAMES = 100
HD_LEVELS_DEG = [0, 3, 6, 10, 20]
ENVS = {
    "paper": {"env_dim": 0.635, "eps": 0.005, "pos_levels_cm": [0, 2, 5, 10, 20]},
    # Real robot units; 16-163 cm are 2-20 cm of the rat box (x SCALE_S = 8.1608).
    "robot": {"env_dim": 5.4415, "eps": 0.1705,
              "pos_levels_cm": [0, 2, 5, 10, 20, 16.3, 40.8, 81.6, 163.2]},
}


def curriculum_dir(root, prefix, seed, stage):
    return (REPO / root / "box" / stage / "predictions" / "box_messy" / HIST[stage]
            / f"{stage}_RNN_{prefix}_ft{TAIL}{seed:02d}" / f"act_{stage}_epoch1500_exclunvisited")


MODELS = {
    "paper_s1": ("paper", lambda s: REPO / "data/box" / s / "predictions/box_messy/vanilla"
                 / f"RNN_0716_{s}_ft{TAIL}01" / f"act_{s}_epoch1500"),
    "paper_s2": ("paper", lambda s: curriculum_dir("data", "seed_1029", 2, s)),
    "paper_s3": ("paper", lambda s: curriculum_dir("data", "seed_1029", 3, s)),
    "paper_s4": ("paper", lambda s: curriculum_dir("data", "seed_1029", 4, s)),
    "robot_s1": ("robot", lambda s: curriculum_dir("data_robot_room", "robotroom_1023", 1, s)),
    "robot_s2": ("robot", lambda s: curriculum_dir("data_robot_room", "robotseed_1102", 2, s)),
    "robot_s3": ("robot", lambda s: curriculum_dir("data_robot_room", "robotseed_1102", 3, s)),
    # Trained and analysed on the trajectory-seed o2000 set; its SIr trend is significant.
    "robot_o2000": ("robot", lambda s: curriculum_dir("data_robot_room_o2000", "robotO2000_1028", 1, s)),
}


def reflect(x, lo, hi):
    period = 2 * (hi - lo)
    y = np.mod(x - lo, period)
    return lo + np.where(y > hi - lo, period - y, y)


def frame_noise(rng, kind, sd, n_seq, n_steps, n_dim):
    """Noise of shape (n_seq, n_steps, n_dim); sequence k is trajectory k // STRIDE
    sampled every STRIDE frames from offset k % STRIDE."""
    n_traj, n_frames = n_seq // STRIDE, STRIDE * (n_steps + 1)
    if kind == "white":
        noise = rng.normal(0, sd, (n_traj, n_frames, n_dim))
    else:
        a = np.exp(-1 / DRIFT_TAU_FRAMES)
        eps = rng.normal(0, sd * np.sqrt(1 - a**2), (n_traj, n_frames, n_dim))
        noise = np.empty_like(eps)
        noise[:, 0] = rng.normal(0, sd, (n_traj, n_dim))
        for t in range(1, n_frames):
            noise[:, t] = a * noise[:, t - 1] + eps[:, t]
    frames = np.arange(n_seq)[:, None] % STRIDE + STRIDE * np.arange(n_steps)[None]
    return noise[np.arange(n_seq)[:, None] // STRIDE, frames]


def srsa(lact, pos):
    """Vectorised RNNActiviter.calculate_sRSA (upper triangle incl. diagonal)."""
    n_steps = lact.shape[1]
    iu = np.triu_indices(n_steps)
    vals = []
    for a, p in zip(lact.astype(np.float64), pos.astype(np.float64)):
        an = a / np.linalg.norm(a, axis=1, keepdims=True)
        d_c = 1 - an @ an.T
        d_e = np.linalg.norm(p[:, None] - p[None], axis=-1)
        r1, r2 = rankdata(d_e[iu]), rankdata(d_c[iu])
        vals.append(np.corrcoef(r1, r2)[0, 1])
    return np.array(vals)


def jt_p(groups, n_perm=10_000, seed=0):
    """One-sided (increasing) Jonckheere-Terpstra permutation p-value."""
    x = np.concatenate(groups)
    lab = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    k = len(groups)

    def stat(lab):
        s = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                a, b = x[lab == i], x[lab == j]
                r = rankdata(np.concatenate([a, b]))
                s += r[len(a):].sum() - len(b) * (len(b) + 1) / 2
        return s

    obs = stat(lab)
    rng = np.random.default_rng(seed)
    ge = sum(stat(rng.permutation(lab)) >= obs for _ in range(n_perm))
    return (ge + 1) / (n_perm + 1)


def pairwise_wilcox(groups):
    """R pairwise.wilcox.test(alternative='greater', BH): later stage > earlier."""
    pairs = [(0, 1), (1, 2), (0, 2)]
    p = np.array([mannwhitneyu(groups[j], groups[i], alternative="greater",
                               method="asymptotic", use_continuity=True).pvalue for i, j in pairs])
    order = np.argsort(p)[::-1]
    adj = np.minimum.accumulate(p[order] * len(p) / (len(p) - np.arange(len(p))))
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1)
    return {f"{STAGES[i]}<{STAGES[j]}": float(out[n]) for n, (i, j) in enumerate(pairs)}


def place_metrics(act, pos, env_dim):
    rm = RateMaps(pos, env_dim, exclude_unvisited=True)
    rate_maps, occ = rm.calculate_rate_maps(act)
    return rm.calculate_metrics(rate_maps, occ, norm="minmax")


def hd_metrics(act, thetas):
    pm = PolarMaps(thetas)
    polar_maps, occ = pm.calculate_polar_maps(act)
    si, rvl, _ = pm.calculate_metrics(polar_maps, occ)
    return si, rvl


def decoding(lact, pos_noisy, th_noisy, pos_true, th_true):
    """RNNActiviter.pos_hd_decoding on (possibly noisy) labels: a linear regression
    fitted on the first half of the sequences, scored on the second half against both
    the noisy labels and the true ones. Returns mean errors (cm, degrees)."""
    from sklearn.linear_model import LinearRegression

    h = lact.shape[0] // 2
    flat = lambda a: a.reshape(-1, a.shape[-1])
    labels = np.concatenate([pos_noisy, th_noisy[..., None]], -1)
    pred = LinearRegression().fit(flat(lact[:h]), flat(labels[:h])).predict(flat(lact[h:]))
    out = {}
    for name, p, t in [("noisy", pos_noisy, th_noisy), ("true", pos_true, th_true)]:
        out[f"pos_{name}"] = float(np.mean(np.linalg.norm(flat(p[h:]) * 100 - pred[:, :2] * 100, axis=1)))
        out[f"hd_{name}"] = float(np.mean(np.abs(np.degrees(t[h:].ravel()) - np.degrees(pred[:, 2]))))
    return out


def load(path):
    rec = path / "recurrent_activity.npy"
    # Older runs saved only latent_activity, which for a plain RNN is the hidden state.
    act = np.load(rec if rec.exists() else path / "latent_activity.npy")
    lact = np.load(path / "latent_activity.npy")
    return act, lact, np.load(path / "positions.npy"), np.load(path / "thetas.npy")[..., 0]


def summarise(per_stage, key):
    groups = [per_stage[s][key] for s in STAGES]
    return {"median": [float(np.nanmedian(g)) for g in groups],
            "jt_p": jt_p([g[~np.isnan(g)] for g in groups]),
            "wilcox_bh": pairwise_wilcox([g[~np.isnan(g)] for g in groups])}


def run_model(name, n_rep, kinds=("white", "drift"), pos_levels=None, hd_levels=None):
    env_name, path = MODELS[name]
    env = ENVS[env_name]
    lo, hi = env["eps"], env["env_dim"] - env["eps"]
    data = {s: load(path(s)) for s in STAGES}

    conditions = [("none", "none", 0.0, 0)]
    pos_levels = env["pos_levels_cm"] if pos_levels is None else pos_levels
    hd_levels = HD_LEVELS_DEG if hd_levels is None else hd_levels
    for kind in kinds:
        conditions += [("pos", kind, lv, r) for lv in pos_levels if lv > 0 for r in range(n_rep)]
        conditions += [("hd", kind, lv, r) for lv in hd_levels if lv > 0 for r in range(n_rep)]

    records = []
    base = {}
    for target, kind, level, rep in conditions:
        per_stage = {}
        for si_, s in enumerate(STAGES):
            act, lact, pos, th = data[s]
            n_seq, n_steps = pos.shape[:2]
            rng = np.random.default_rng([rep, si_, int(level * 10), kind == "drift", target == "hd"])
            res = {}
            p, t = pos, th
            if target in ("none", "pos"):
                if target == "pos":
                    p = reflect(pos + frame_noise(rng, kind, level / 100, n_seq, n_steps, 2), lo, hi)
                res["SIr"] = place_metrics(act.reshape(-1, act.shape[-1]), p.reshape(-1, 2), env["env_dim"])
                res["sRSA"] = srsa(lact, p)
            if target in ("none", "hd"):
                if target == "hd":
                    t = np.mod(th + frame_noise(rng, kind, np.radians(level), n_seq, n_steps, 1)[..., 0]
                               + np.pi, 2 * np.pi) - np.pi
                res["SId"], res["RVL"] = hd_metrics(act.reshape(-1, act.shape[-1]), t.reshape(-1))
            res["dec"] = decoding(lact, p, t, pos, th)
            if target == "none":
                base[s] = res
                res["check_pos_dec"] = res["dec"]["pos_true"] - float(np.mean(np.load(path(s) / "pos_dec_err.npy")))
                res["check_hd_dec"] = res["dec"]["hd_true"] - float(np.mean(np.load(path(s) / "thet_dec_err.npy")))
                saved = path(s) / "place" / "si.npy"
                if "exclunvisited" in path(s).name:
                    res["check_SIr"] = float(np.nanmax(np.abs(res["SIr"] - np.load(saved))))
                    res["check_SId"] = float(np.nanmax(np.abs(res["SId"] - np.load(path(s) / "hd" / "si.npy"))))
                res["check_sRSA"] = float(np.max(np.abs(res["sRSA"] - np.load(path(s) / "sRSA_values.npy"))))
            full = {**base[s], **res}
            pc = full["SIr"] > RateMaps.PLACE_SI_TH
            hdc = (full["SId"] > PolarMaps.HD_SI_TH) | (full["RVL"] > PolarMaps.HD_RVL_TH)
            full["counts"] = {"place": int((pc & ~hdc).sum()), "hd": int((hdc & ~pc).sum()),
                              "conjunctive": int((pc & hdc).sum())}
            per_stage[s] = full

        rec = {"model": name, "env": env_name, "target": target, "kind": kind, "level": level, "rep": rep,
               "counts": {s: per_stage[s]["counts"] for s in STAGES},
               "sRSA": {s: float(np.mean(per_stage[s]["sRSA"])) for s in STAGES},
               "decoding": {s: per_stage[s]["dec"] for s in STAGES}}
        keys = {"none": ["SIr", "SId", "RVL"], "pos": ["SIr"], "hd": ["SId", "RVL"]}[target]
        for k in keys:
            rec[k] = summarise(per_stage, k)
        if target == "none":
            rec["check"] = {s: {k: v for k, v in base[s].items() if k.startswith("check")} for s in STAGES}
        records.append(rec)
        print(f"{name} {target} {kind} {level} r{rep}: "
              + " ".join(f"{k} jt {rec[k]['jt_p']:.4f}" for k in keys)
              + f" sRSA {[round(v, 3) for v in rec['sRSA'].values()]}"
              + f" dec pos/hd true {[(round(d['pos_true'], 1), round(d['hd_true'], 1)) for d in rec['decoding'].values()]}",
              flush=True)
    return name, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--n-rep", type=int, default=5)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=REPO / "reports" / "label_noise")
    parser.add_argument("--kinds", nargs="+", default=["white", "drift"], choices=["white", "drift"])
    parser.add_argument("--pos-levels", nargs="+", type=float, default=None,
                        help="position noise SDs in cm (default: the environment's list)")
    parser.add_argument("--hd-levels", nargs="+", type=float, default=None,
                        help="HD noise SDs in degrees (default: HD_LEVELS_DEG)")
    parser.add_argument("--tag", default="", help="suffix of the output file name")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with Pool(args.workers) as pool:
        jobs = [pool.apply_async(run_model, (m, args.n_rep, args.kinds, args.pos_levels, args.hd_levels)) for m in args.models]
        for job in jobs:
            name, records = job.get()
            (args.out_dir / f"{name}{args.tag}.json").write_text(json.dumps(records, indent=1))
            print(f"[+] saved {name}", flush=True)


if __name__ == "__main__":
    main()
