"""Developmental tests and line plots of the percentage of spatial cells.

For every model, the percentage of place, HD and conjunctive cells at crawl,
walk, run and adult is computed from the per-unit SIr, SId and RVL, with the
definitions of the rat data: place = SIr > 0.3, HD = SId > 0.2 or RVL > 0.3,
conjunctive = both, so place and HD include the conjunctive cells. Walk, run
and adult come from the R_stats exports; crawl, which the exports and the rat
data lack, comes from the crawl model's activity folder.

Models are plotted only (mean +- SEM over training seeds, thin lines per
seed). The rat sessions (R_stats/data/*_perc_real.npy) are tested with the R
pipeline's tests: an increasing Jonckheere-Terpstra test (10,000 permutations,
CSV only) and one-sided pairwise Wilcoxon rank-sum tests with BH correction
over the three pairs; the figure shows walk-run and run-adult.

Writes reports/cell_percentage/<model>/{cell_percentage.png,.svg,values.csv,
tests.csv} and reports/cell_percentage/summary_tests.csv.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "R_stats" / "results"
STAGES = ["crawl", "walk", "run", "adult"]
TESTED = STAGES[1:]
PAIRS = [(0, 1), (1, 2), (0, 2)]
SHOWN = [(0, 1), (1, 2)]
CELLS = {"place": "Place cells", "hd": "HD cells", "conjunctive": "Conjunctive cells"}
REAL = {"place": "pc_perc_real", "hd": "hdc_perc_real", "conjunctive": "phdc_perc_real"}
PLACE_SI_TH, HD_SI_TH, HD_RVL_TH = 0.3, 0.2, 0.3

RNN_TAIL = "_f1_w9_st10_fss4_do[0,0,0]_lat500_nlsigmoid_hreg0.0_wreg0.0_s0{}"


def crawl(root, name, act):
    return REPO / root / "box/crawl/predictions/box_messy/vanilla" / name / act



# MTRSSM latent = [lower determ 106, lower stoch 6x32, upper determ 61, upper stoch 3x32]
# (MTRSSM._packed_state); the walk/run folders are named act_crawl but hold the
# walk/run behaviour (positions match the RNN500 walk/run analyses).
MTRSSM_NAME = ("MTRSSM_mtrssm_cat_rnn500match{ft}_f1_w9_st10_fss4_do{do}_lat106_stoch6_kl1.0_fn3.0"
               "_hreg0.0_wreg0.0_s01_hlat61_hstoch3_ta5_tau2.0-8.0_cat32")
MTRSSM_DIRS = {
    "crawl": REPO / "data/box/crawl/predictions/box_messy/vanilla"
    / MTRSSM_NAME.format(ft="", do="[0,0,0]") / "act_crawl_epoch1500",
    "walk": REPO / "data/box/walk/predictions/box_messy/crawl"
    / MTRSSM_NAME.format(ft="_ft", do="[0,0,0]") / "act_crawl_epoch1500",
    "run": REPO / "data/box/run/predictions/box_messy/crawl_walk"
    / MTRSSM_NAME.format(ft="_ft", do="[0,0,0]") / "act_crawl_epoch1500",
    "adult": REPO / "data/box/adult/predictions/box_messy/crawl_walk_run"
    / MTRSSM_NAME.format(ft="_ft", do="[0.0,0.0,0.0]") / "act_adult_epoch1500",
}
PART_LABEL = {"det": "deterministic", "stoch": "stochastic", "both": "deterministic + stochastic"}
MTRSSM_UNITS = {
    "lower": {"det": slice(0, 106), "stoch": slice(106, 298), "both": slice(0, 298)},
    "upper": {"det": slice(298, 359), "stoch": slice(359, 455), "both": slice(298, 455)},
}

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
COLOR = {"model": "#2a78d6", "real": "#eb6834"}

MODELS = {
    "RNN500": ("RNN500 (paper box)", [
        (RES / "compare_rnn500/rnn0716/RNN_0716_hidden",
         crawl("data", "RNN_crawl_0714" + RNN_TAIL.format(1), "act_crawl_epoch1500")),
        *[(RES / f"seeds/rnn500_s{s}/RNN500_seed{s}_hidden",
           crawl("data", "crawl_RNN_seed_1029" + RNN_TAIL.format(s), "act_crawl_epoch1500_exclunvisited"))
          for s in (2, 3, 4)]]),
    "RNN500_robotroom": ("RNN500, enlarged room (robotroom)", [
        (RES / "robotroom/rnn500/RNN500_robotroom_hidden",
         crawl("data_robot_room", "crawl_RNN_robotroom_1023" + RNN_TAIL.format(1),
               "act_crawl_epoch1500_exclunvisited")),
        *[(RES / f"robotroom/train_seeds/s{s}/RNN500_robotseed{s}_hidden",
           crawl("data_robot_room", "crawl_RNN_robotseed_1102" + RNN_TAIL.format(s),
                 "act_crawl_epoch1500_exclunvisited")) for s in (2, 3)]]),
    "RNN500_robot_o2000": ("RNN500, enlarged room (trajectory o2000)", [
        (RES / "robotroom/change_seed/rnn500_o2000/RNN500_robotO2000_hidden",
         crawl("data_robot_room_o2000", "crawl_RNN_robotO2000_1028" + RNN_TAIL.format(1),
               "act_crawl_epoch1500_exclunvisited"))]),
    "GRU267": ("GRU267 (sigmoid candidate)", [
        (RES / "gru_sigmoid/GRU267_sigmoid_hidden",
         crawl("data", "crawl_GRU_1108_267dim_sig_nonneg", "act_crawl_epoch1500_exclunvisited"))]),
    "LSTM221": ("LSTM221 (non-negative)", [(RES / "nonneg/lstm221/LSTM221_nonneg_hidden", None)]),
    "RSSM_GRU": ("RSSM with GRU", [(RES / "compare_rnn500/rssm_gru/RSSM_combined", None)]),
    "RSSM_RNN": ("RSSM with RNN", [
        (RES / "compare_rnn500/rssm_d512/RSSM_combined",
         crawl("data", "crawl_RSSM_lat512_rssm_lat512_stoch8rssm_nlsigmoid_cat8",
               "act_crawl_epoch1500_cells-combined"))]),
    **{f"CTRSSM_tau{t}": (f"CTRSSM (tau {t}, 1017)", [
        (RES / f"compare_rnn500/ctrssm/ctrssm_tau{t}/RSSM_combined", None)]) for t in (2,)},
    "CTRSSM_tau3": ("CTRSSM (tau 3, 1007)", [
        (RES / "ctrssm/ctrssm_1007_tau3_results/RSSM_combined",
         crawl("data", "crawl_CTRSSM_1007_lat512_stoch8rssm_nlsigmoid_tau3_cat8",
               "act_crawl_epoch1500_cells-combined"))]),
    **{f"CTRSSM_tau{t}": (f"CTRSSM (tau {t}, 1017)", [
        (RES / f"compare_rnn500/ctrssm/ctrssm_tau{t}/RSSM_combined", None)]) for t in (4, 8, 16)},
    **{f"MTRSSM_{layer}_{part}": (f"MTRSSM, {layer} layer, {PART_LABEL[part]}",
                                   [({"dirs": MTRSSM_DIRS, "units": units}, None)])
       for layer, parts in MTRSSM_UNITS.items() for part, units in parts.items()},
}


def unit_metrics(export, crawl_dir):
    """(SIr, SId, RVL) per stage; None for crawl when it has no activity folder.

    ``export`` is an R_stats export folder, or {"dirs": {stage: activity folder},
    "units": slice} to read every stage from activity folders and keep a subset
    of units.
    """
    if isinstance(export, dict):
        return [tuple(np.load(export["dirs"][st] / f).astype(float).ravel()[export["units"]]
                      for f in ("place/si.npy", "hd/si.npy", "hd/rvl.npy")) for st in STAGES]
    d = export / "data"
    g = np.load(d / "g_model.npy", allow_pickle=True).astype(int)
    m = [np.load(d / f"{k}_model.npy", allow_pickle=True).astype(float) for k in ("sir", "sid", "rvl")]
    first = None
    if crawl_dir is not None:
        first = tuple(np.load(crawl_dir / f).astype(float).ravel()
                      for f in ("place/si.npy", "hd/si.npy", "hd/rvl.npy"))
    return [first] + [tuple(a[g == i + 1] for a in m) for i in range(3)]


def percentages(export, crawl_dir):
    """{cell: [crawl, walk, run, adult]} in percent of all units (NaN when missing)."""
    out = {c: [] for c in CELLS}
    for metrics in unit_metrics(export, crawl_dir):
        if metrics is None:
            for c in CELLS:
                out[c].append(np.nan)
            continue
        sir, sid, rvl = metrics
        place = np.nan_to_num(sir, nan=-np.inf) > PLACE_SI_TH
        hd = (sid > HD_SI_TH) | (rvl > HD_RVL_TH)
        for c, mask in (("place", place), ("hd", hd), ("conjunctive", place & hd)):
            out[c].append(100 * mask.mean())
    return out


def jt_p(groups, n_perm=10_000, seed=0):
    x = np.concatenate(groups)
    lab = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])

    def stat(lab):
        s = 0.0
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = x[lab == i], x[lab == j]
                r = rankdata(np.concatenate([a, b]))
                s += r[len(a):].sum() - len(b) * (len(b) + 1) / 2
        return s

    obs = stat(lab)
    rng = np.random.default_rng(seed)
    ge = sum(stat(rng.permutation(lab)) >= obs for _ in range(n_perm))
    return (ge + 1) / (n_perm + 1)


def pairwise_wilcox(groups):
    """R pairwise.wilcox.test(alternative='greater', p.adjust.method='BH')."""
    p = []
    for i, j in PAIRS:
        a, b = np.asarray(groups[j]), np.asarray(groups[i])
        exact = len(a) < 50 and len(b) < 50 and len(np.unique(np.r_[a, b])) == len(a) + len(b)
        p.append(mannwhitneyu(a, b, alternative="greater", method="exact" if exact else "asymptotic",
                              use_continuity=True).pvalue)
    p = np.array(p)
    order = np.argsort(p)[::-1]
    adj = np.minimum.accumulate(p[order] * len(p) / (len(p) - np.arange(len(p))))
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1)
    return out


def pair_name(i, j):
    return f"{TESTED[i]}<{TESTED[j]}"


def tests(groups):
    return {"jt": jt_p(groups), **{pair_name(i, j): p for (i, j), p in zip(PAIRS, pairwise_wilcox(groups))}}


def stars(p):
    p = round(p, 12)
    return "***" if p <= 0.001 else "**" if p <= 0.01 else "*" if p <= 0.05 else "ns"


def mean_sem(groups):
    m = np.array([np.mean(g) for g in groups])
    s = np.array([np.std(g, ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0 for g in groups])
    return m, s


def draw_brackets(ax, t, y, step):
    """Dashed square brackets above the data for walk-run and run-adult."""
    tick = 0.25 * step
    for i, j in SHOWN:
        xi = i + 1 + (0.04 if i == 1 else 0)
        xj = j + 1 - (0.04 if j == 1 else 0)
        ax.plot([xi, xi, xj, xj], [y - tick, y, y, y - tick], color=INK, lw=1.2,
                ls=(0, (4, 2.5)), clip_on=False)
        mark = stars(t[pair_name(i, j)])
        ax.text((xi + xj) / 2, y + 0.06 * step, mark, ha="center", va="bottom",
                fontsize=11 if mark != "ns" else 9, color=INK if mark != "ns" else MUTED, clip_on=False)
    return y + step


def plot(label, model, real, rtests, n_seeds, n_real, out):
    fig, axes = plt.subplots(1, len(CELLS), figsize=(13, 4.8), constrained_layout=True)
    x = np.arange(len(STAGES))
    xr = x[1:]
    for ax, c in zip(axes, CELLS):
        mm, ms = mean_sem(model[c])
        rm, rs = mean_sem(real[c])
        ok = ~np.isnan(mm)
        top = max(np.nanmax(mm + ms), (rm + rs).max())
        low = min(np.nanmin(mm - ms), (rm - rs).min())
        step = 0.09 * max(top, 10)
        ax.errorbar(xr, rm, yerr=rs, color=COLOR["real"], ls=(0, (4, 2.5)), lw=2, marker="s", ms=7,
                    mec="white", mew=1.5, capsize=3, label=f"Rat (sessions {'/'.join(map(str, n_real))})")
        ax.errorbar(x[ok], mm[ok], yerr=ms[ok] if n_seeds > 1 else None, color=COLOR["model"], lw=2,
                    marker="o", ms=8, mec="white", mew=1.5, capsize=3,
                    label=f"Model ({n_seeds} seed{'s' if n_seeds > 1 else ''})", zorder=3)
        if n_seeds > 1:
            for v in np.array(model[c]).T:
                ax.plot(x, v, color=COLOR["model"], lw=0.8, alpha=0.3, zorder=2)

        hi = draw_brackets(ax, rtests[c], top + 0.8 * step, step)
        ax.set_ylim(min(0, low - 0.6 * step), hi + 0.2 * step)
        ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= ax.get_ylim()[1]])

        ax.set_title(CELLS[c], fontsize=12, color=INK, loc="left")
        ax.set_xticks(x, [s.capitalize() for s in STAGES])
        ax.set_xlim(-0.35, len(STAGES) - 0.65)
        ax.set_ylabel("Percentage of units / cells (%)" if c == "place" else "", color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper right", ncol=2, frameon=False, fontsize=9.5,
               bbox_to_anchor=(1.0, 1.06))
    fig.suptitle(f"{label}: percentage of spatial cells by stage", fontsize=13, color=INK, x=0.01,
                 ha="left")
    note = ("Mean ± SEM. Thin lines: individual training seeds. Brackets: rat data, one-sided pairwise "
            "Wilcoxon (BH over walk-run, run-adult and walk-adult);\n"
            "* p≤0.05, ** p≤0.01, *** p≤0.001, ns p>0.05. Model is not tested. "
            "Place and HD include conjunctive cells.")
    fig.text(0.01, -0.06, note, fontsize=8, color=MUTED)
    for ext in ("png", "svg"):
        fig.savefig(out / f"cell_percentage.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO / "reports" / "cell_percentage")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()

    g = np.load(REPO / "R_stats/data/g_real_perc.npy").astype(int)
    real = {c: [np.load(REPO / f"R_stats/data/{f}.npy").astype(float)[g == i + 1] for i in range(3)]
            for c, f in REAL.items()}
    n_real = [int((g == i + 1).sum()) for i in range(3)]
    rtests = {c: tests(real[c]) for c in CELLS}

    summary = []
    for name in args.models:
        label, sources = MODELS[name]
        for export, crawl_dir in sources:
            dirs = list(export["dirs"].values()) if isinstance(export, dict) else [export]
            for d in dirs + ([crawl_dir] if crawl_dir is not None else []):
                if not d.exists():
                    raise FileNotFoundError(d)
        per_seed = [percentages(*src) for src in sources]
        model = {c: [[p[c][i] for p in per_seed] for i in range(len(STAGES))] for c in CELLS}
        out = args.out_dir / name
        out.mkdir(parents=True, exist_ok=True)

        pd.DataFrame([{"seed_index": k + 1, "export": str(src[0]), "crawl_dir": str(src[1]), "stage": st,
                       **{c: per_seed[k][c][i] for c in CELLS}}
                      for k, src in enumerate(sources) for i, st in enumerate(STAGES)]
                     ).to_csv(out / "values.csv", index=False)
        rows = []
        for c in CELLS:
            means = mean_sem(model[c])[0]
            row = {"model": name, "cell": c, "source": "model", "n": len(sources),
                   **{f"mean_{s}": v for s, v in zip(STAGES, means)}}
            rows.append(row)
            rows.append({"model": name, "cell": c, "source": "rat", "n": "/".join(map(str, n_real)),
                         **{f"mean_{s}": v for s, v in zip(TESTED, mean_sem(real[c])[0])},
                         "jt_p": rtests[c]["jt"], **{k: v for k, v in rtests[c].items() if k != "jt"}})
        pd.DataFrame(rows).to_csv(out / "tests.csv", index=False)
        summary += rows
        plot(label, model, real, rtests, len(sources), n_real, out)
        print(f"[+] {out}")

    pd.DataFrame(summary).to_csv(args.out_dir / "summary_tests.csv", index=False)


if __name__ == "__main__":
    main()
