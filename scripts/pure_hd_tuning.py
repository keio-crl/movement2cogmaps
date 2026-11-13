"""Percentage, SId and RVL of pure HD cells across stages, for each model.

Pure HD cells are units with SId > 0.2 or RVL > 0.3 and SIr <= 0.3 (the models
and their data are defined in cell_percentage_stats.py). Only the models are
drawn; the rat cells are classified by other criteria.

- Percentage: pure HD units / all units, mean +- SEM over training seeds with a
  thin line per seed (no SEM with one seed). Not tested.
- SId and RVL: the pure HD units of all seeds are pooled per stage; mean +-
  SEM over those units, thin lines for each seed's mean. Walk vs run and run vs
  adult are compared with one-sided (increase) Wilcoxon rank-sum tests on the
  pooled units, BH-corrected over the two comparisons, and drawn as solid
  brackets above the data. Crawl is drawn but not tested.

Writes reports/cell_percentage/<model>/pure_hd.{png,svg}, pure_hd_values.csv
(per seed and stage) and pure_hd_tests.csv.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell_percentage_stats import (COLOR, GRID, HD_RVL_TH, HD_SI_TH, INK, MODELS, MUTED,  # noqa: E402
                                   PLACE_SI_TH, REPO, STAGES, stars, unit_metrics)

PANELS = {"perc": "Pure HD cells (%)", "sid": "SId of pure HD cells", "rvl": "RVL of pure HD cells"}
PAIRS = [("walk", "run"), ("run", "adult")]


def pure_hd(export, crawl_dir):
    """Per stage: (percentage, SId array, RVL array) of the pure HD units, or None."""
    out = []
    for metrics in unit_metrics(export, crawl_dir):
        if metrics is None:
            out.append(None)
            continue
        sir, sid, rvl = metrics
        hd = (sid > HD_SI_TH) | (rvl > HD_RVL_TH)
        pure = hd & ~(np.nan_to_num(sir, nan=-np.inf) > PLACE_SI_TH)
        out.append((100 * pure.mean(), sid[pure], rvl[pure]))
    return out


def wilcox_greater(a, b):
    """R wilcox.test(b, a, alternative='greater'): exact for n < 50 without ties."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    exact = len(a) < 50 and len(b) < 50 and len(np.unique(np.r_[a, b])) == len(a) + len(b)
    return mannwhitneyu(b, a, alternative="greater", method="exact" if exact else "asymptotic",
                        use_continuity=True).pvalue


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    order = np.argsort(q)[::-1]
    adj = np.minimum.accumulate(q[order] * len(q) / (len(q) - np.arange(len(q))))
    res = np.empty_like(q)
    res[order] = np.minimum(adj, 1)
    out[ok] = res
    return out


def sem(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=12, color=INK, loc="left")
    ax.set_xticks(np.arange(len(STAGES)), [s.capitalize() for s in STAGES])
    ax.set_xlim(-0.35, len(STAGES) - 0.65)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def brackets(ax, pvals, y, step):
    """Solid square brackets for walk-run and run-adult with the significance mark above."""
    tick = 0.25 * step
    for (a, b), p in zip(PAIRS, pvals):
        i, j = STAGES.index(a), STAGES.index(b)
        xi, xj = i + (0.04 if a == "run" else 0), j - (0.04 if b == "run" else 0)
        ax.plot([xi, xi, xj, xj], [y - tick, y, y, y - tick], color=INK, lw=1.2, clip_on=False)
        mark = "n/a" if np.isnan(p) else stars(p)
        ax.text((xi + xj) / 2, y + 0.06 * step, mark, ha="center", va="bottom",
                fontsize=11 if mark not in ("ns", "n/a") else 9,
                color=INK if mark not in ("ns", "n/a") else MUTED, clip_on=False)
    return y + step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO / "reports" / "cell_percentage")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()

    x = np.arange(len(STAGES))
    for name in args.models:
        label, sources = MODELS[name]
        seeds = [pure_hd(*src) for src in sources]
        n_seeds = len(seeds)
        out = args.out_dir / name
        out.mkdir(parents=True, exist_ok=True)

        pd.DataFrame([{"seed_index": k + 1, "stage": st,
                       "pure_hd_percent": np.nan if v is None else v[0],
                       "n_pure_hd": 0 if v is None else len(v[1]),
                       "mean_sid": np.nan if v is None or not len(v[1]) else v[1].mean(),
                       "mean_rvl": np.nan if v is None or not len(v[2]) else v[2].mean()}
                      for k, seed in enumerate(seeds) for st, v in zip(STAGES, seed)]
                     ).to_csv(out / "pure_hd_values.csv", index=False)

        pooled = {m: [np.concatenate([s[i][k] for s in seeds if s[i] is not None])
                      if any(s[i] is not None for s in seeds) else None
                      for i in range(len(STAGES))] for k, m in ((1, "sid"), (2, "rvl"))}
        test_rows = []
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
        for ax, (m, title) in zip(axes, PANELS.items()):
            if m == "perc":
                table = np.array([[np.nan if v is None else v[0] for v in s] for s in seeds])
                mean = np.nanmean(table, 0) if n_seeds > 1 else table[0]
                err = np.array([sem(c) for c in table.T]) if n_seeds > 1 else np.zeros(len(STAGES))
                seed_lines = table if n_seeds > 1 else []
            else:
                k = 1 if m == "sid" else 2
                vals = pooled[m]
                mean = np.array([np.nan if v is None or not len(v) else v.mean() for v in vals])
                err = np.array([0.0 if v is None else sem(v) for v in vals])
                seed_lines = ([[np.nan if v is None or not len(v[k]) else v[k].mean() for v in s] for s in seeds]
                              if n_seeds > 1 else [])
            for v in seed_lines:
                ax.plot(x, v, color=COLOR["model"], lw=0.8, alpha=0.3, zorder=2)
            ok = ~np.isnan(mean)
            ax.errorbar(x[ok], mean[ok], yerr=err[ok], color=COLOR["model"], lw=2, marker="o", ms=8,
                        mec="white", mew=1.5, capsize=3, zorder=3)
            vals_top = np.r_[mean + err, np.ravel(seed_lines)] if len(seed_lines) else mean + err
            top = np.nanmax(vals_top) if np.isfinite(vals_top).any() else 0.0
            if top <= 0:
                # No pure HD unit at any stage: keep a readable axis.
                top = {"perc": 1.0, "sid": HD_SI_TH, "rvl": HD_RVL_TH}[m]
            step = 0.09 * top
            hi = top * 1.12
            if m != "perc":
                vals = pooled[m]
                raw = [wilcox_greater(vals[STAGES.index(a)], vals[STAGES.index(b)])
                       if vals[STAGES.index(a)] is not None and vals[STAGES.index(b)] is not None else np.nan
                       for a, b in PAIRS]
                adj = bh(raw)
                hi = brackets(ax, adj, top + 0.8 * step, step)
                for (a, b), r, p in zip(PAIRS, raw, adj):
                    test_rows.append({"model": name, "metric": m, "pair": f"{a}<{b}",
                                      "n_a": len(vals[STAGES.index(a)]), "n_b": len(vals[STAGES.index(b)]),
                                      "p_raw": r, "p_bh": p, "mark": stars(p) if not np.isnan(p) else "n/a"})
                counts = ["no data" if v is None else f"n={len(v)}" for v in vals]
                for xi, n in zip(x, counts):
                    ax.text(xi, 0.02, n, transform=ax.get_xaxis_transform(), ha="center",
                            va="bottom", fontsize=8.5, color=MUTED)
            ax.set_ylim(0, hi + 0.2 * step)
            style(ax, title, {"perc": "Percentage of units (%)", "sid": "SId", "rvl": "RVL"}[m])
        pd.DataFrame(test_rows).to_csv(out / "pure_hd_tests.csv", index=False)

        seed_txt = f"{n_seeds} seed{'s' if n_seeds > 1 else ''}"
        fig.suptitle(f"{label}: pure HD cells by stage ({seed_txt})", fontsize=13, color=INK, x=0.01,
                     ha="left")
        note = ("Pure HD = (SId > 0.2 or RVL > 0.3) and SIr ≤ 0.3. Percentage: "
                + ("mean ± SEM over seeds. " if n_seeds > 1 else "one seed. ")
                + "SId / RVL: mean ± SEM over pure HD units"
                + (" pooled across seeds (n = units); thin lines: each seed's mean." if n_seeds > 1
                   else " (n = units).")
                + "\nBrackets: one-sided Wilcoxon rank-sum (increase) on units, BH over walk–run and run–adult; "
                  "* p≤0.05, ** p≤0.01, *** p≤0.001, ns p>0.05. Crawl is not tested.")
        fig.text(0.01, -0.08, note, fontsize=8, color=MUTED)
        for ext in ("png", "svg"):
            fig.savefig(out / f"pure_hd.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[+] {out}/pure_hd.png")


if __name__ == "__main__":
    main()
