"""Summarise reports/label_noise/*.json from label_noise_tolerance.py.

For every environment, noise type and level, a metric is unchanged when, for
every model (averaged over the random draws):
  - stage-trend metrics (SIr, SId, RVL): the Jonckheere verdict (p < 0.05,
    median over draws) equals the verdict without noise, and
  - every metric: the change of each stage's value (unit median, cell count or
    mean sRSA) stays within the SD of that value over the training seeds
    without noise.
The tolerance is the largest level up to which every level passes.

Writes <dir>/summary.csv (per model, condition and metric) and
<dir>/tolerance.csv, and prints both.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
STAGES = ["walk", "run", "adult"]
TREND = ["SIr", "SId", "RVL"]
METRICS = {"pos": ["SIr", "place", "conjunctive", "sRSA"], "hd": ["SId", "RVL", "hd", "conjunctive"]}
ALPHA = 0.05
# Stage pattern of the paper: every step significant except SIr run -> adult.
REQUIRED = {"SIr": ["jt", "walk<run", "walk<adult"],
            "SId": ["jt", "walk<run", "run<adult", "walk<adult"],
            "RVL": ["jt", "walk<run", "run<adult", "walk<adult"]}


def values(rec, metric):
    if metric in TREND:
        return np.array(rec[metric]["median"])
    if metric == "sRSA":
        return np.array([rec["sRSA"][s] for s in STAGES])
    return np.array([rec["counts"][s][metric] for s in STAGES], dtype=float)


def pattern_holds(rec, metric):
    p = {"jt": rec[metric]["jt_p"], **rec[metric]["wilcox_bh"]}
    return all(p[k] < ALPHA for k in REQUIRED[metric])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=REPO / "reports" / "label_noise")
    args = parser.parse_args()

    recs = [r for f in sorted(args.dir.glob("*_s*.json")) for r in json.loads(f.read_text())]
    base = {r["model"]: r for r in recs if r["target"] == "none"}

    seed_sd = {}
    for env in {r["env"] for r in base.values()}:
        models = [m for m, r in base.items() if r["env"] == env]
        for metric in {m for ms in METRICS.values() for m in ms}:
            seed_sd[env, metric] = np.std([values(base[m], metric) for m in models], axis=0, ddof=1)

    rows = []
    groups = {}
    for r in recs:
        if r["target"] != "none":
            groups.setdefault((r["model"], r["target"], r["kind"], r["level"]), []).append(r)
    for (model, target, kind, level), rs in sorted(groups.items()):
        env = rs[0]["env"]
        for metric in METRICS[target]:
            delta = np.mean([values(r, metric) for r in rs], axis=0) - values(base[model], metric)
            sd = seed_sd[env, metric]
            row = {"env": env, "model": model, "target": target, "kind": kind, "level": level,
                   "metric": metric, **{f"delta_{s}": d for s, d in zip(STAGES, delta)},
                   **{f"seed_sd_{s}": v for s, v in zip(STAGES, sd)},
                   "within_sd": bool(np.all(np.abs(delta) <= sd))}
            if metric in TREND:
                p0 = base[model][metric]["jt_p"]
                p = float(np.median([r[metric]["jt_p"] for r in rs]))
                row.update(jt_p_base=p0, jt_p=p, verdict_same=(p < ALPHA) == (p0 < ALPHA),
                           run_adult_p=float(np.median([r[metric]["wilcox_bh"]["run<adult"] for r in rs])),
                           run_adult_p_base=base[model][metric]["wilcox_bh"]["run<adult"],
                           pattern_base=pattern_holds(base[model], metric),
                           pattern_draws=sum(pattern_holds(r, metric) for r in rs), n_draws=len(rs))
            row["ok"] = row["within_sd"] and row.get("verdict_same", True)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.dir / "summary.csv", index=False)

    tol = []
    for (env, target, kind, metric), g in df.groupby(["env", "target", "kind", "metric"]):
        ok = g.groupby("level")["ok"].all().sort_index()
        passed = [lv for lv in ok.index if ok[:lv].all()]
        first_fail = next((lv for lv in ok.index if not ok[lv]), None)
        tol.append({"env": env, "target": target, "kind": kind, "metric": metric,
                    "tolerance": max(passed) if passed else 0, "first_fail": first_fail})
    tol = pd.DataFrame(tol)
    tol.to_csv(args.dir / "tolerance.csv", index=False)

    print("\nDraws keeping the paper's stage pattern (models whose noise-free result has it)")
    pat = df[df.metric.isin(TREND) & df.pattern_base.astype(bool)]
    print(pat.pivot_table(index=["env", "model", "metric"], columns=["kind", "level"],
                          values="pattern_draws").to_string())

    pd.set_option("display.width", 200)
    for target, unit in [("pos", "cm"), ("hd", "deg")]:
        print(f"\nTolerance ({unit}), largest level with every model unchanged")
        print(tol[tol.target == target].pivot_table(index=["env", "kind"], columns="metric",
                                                    values="tolerance").to_string())
    for metric in TREND:
        print(f"\n{metric}: Jonckheere p (median over draws)")
        g = df[df.metric == metric]
        print(g.pivot_table(index=["env", "model"], columns=["kind", "level"], values="jt_p").round(4).to_string())


if __name__ == "__main__":
    main()
