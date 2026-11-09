"""HTML table of the per-draw label-noise results of selected models.

Reads reports/label_noise/<model>.json written by label_noise_tolerance.py and
writes one row per condition and random draw (no averaging) to
reports/label_noise/raw_<models>.html. Metrics that a condition does not
recompute (SId/RVL under position noise, SIr/sRSA under HD noise) are shown as
a dash; cell counts are recomputed in every condition.
"""

import argparse
import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGES = ["walk", "run", "adult"]
PAIRS = ["walk<run", "run<adult", "walk<adult"]
SCALE_S = 8.1608
LABELS = {"paper_s1": "論文の箱・論文の seed（RNN_0716 s1）",
          "robot_s1": "拡大環境（robotroom_1023 s1）",
          "robot_o2000": "拡大環境・軌跡 seed o2000（robotO2000_1028 s1）"}
TARGET = {"none": "なし", "pos": "位置", "hd": "頭方向"}
KIND = {"none": "—", "white": "独立", "drift": "ずれ"}

CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#d9d9de; --head:#f3f3f6;
        --base:#fff7d6; --sig:#c4122f; --dash:#b0b0b8; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161618; --fg:#ececf0; --muted:#9a9aa2; --line:#3a3a40; --head:#232327;
          --base:#3a3218; --sig:#ff6b7d; --dash:#5a5a62; } }
body { margin:0; padding:16px; background:var(--bg); color:var(--fg);
       font:13px/1.45 system-ui, "Hiragino Sans", "Noto Sans JP", sans-serif; }
h1 { font-size:18px; margin:0 0 4px; } h2 { font-size:15px; margin:24px 0 6px; }
p.note { color:var(--muted); margin:2px 0; max-width:70em; }
.controls { margin:12px 0; display:flex; gap:12px; flex-wrap:wrap; }
.wrap { overflow:auto; max-height:80vh; border:1px solid var(--line); }
table { border-collapse:collapse; font-variant-numeric:tabular-nums; white-space:nowrap; }
th, td { border:1px solid var(--line); padding:3px 6px; text-align:right; }
th { background:var(--head); position:sticky; top:0; z-index:2; }
thead tr:nth-child(2) th { top:25px; }
td.l, th.l { text-align:left; }
td.sticky, th.sticky { position:sticky; left:0; background:var(--bg); z-index:1; }
th.sticky { background:var(--head); z-index:3; }
tr.base td { background:var(--base); }
.sig { color:var(--sig); font-weight:600; } .dash { color:var(--dash); }
"""

JS = """
function apply() {
  const t = document.getElementById('ft').value, k = document.getElementById('fk').value;
  document.querySelectorAll('tbody tr').forEach(r => {
    const ok = (t === 'all' || r.dataset.t === t || r.dataset.t === 'none') &&
               (k === 'all' || r.dataset.k === k || r.dataset.k === 'none');
    r.style.display = ok ? '' : 'none';
  });
}
"""


def fmt(v, digits=4, p=False):
    if v is None:
        return '<td class="dash">—</td>'
    cls = ' class="sig"' if p and v < 0.05 else ""
    return f"<td{cls}>{v:.{digits}g}</td>" if p else f"<td>{v:.{digits}f}</td>"


def table(model, recs):
    robot = model.startswith("robot")
    lvl_head = "大きさ（実寸）</th><th rowspan=2>ラット換算" if robot else "大きさ"
    groups = [("SIr 中央値", 3), ("SIr 検定 p", 4), ("SId 中央値", 3), ("SId 検定 p", 4),
              ("RVL 中央値", 3), ("RVL 検定 p", 4), ("sRSA", 3),
              ("place 細胞", 3), ("HD 細胞", 3), ("conjunctive 細胞", 3)]
    h1 = (f'<th rowspan=2 class="l sticky">ノイズ</th><th rowspan=2 class="l">種類</th>'
          f'<th rowspan=2>{lvl_head}</th><th rowspan=2>回</th>'
          + "".join(f"<th colspan={n}>{g}</th>" for g, n in groups))
    h2 = ""
    for g, n in groups:
        h2 += "".join(f"<th>{c}</th>" for c in (["JT"] + PAIRS if n == 4 else STAGES))
    rows = []
    for r in recs:
        t, k, lv = r["target"], r["kind"], r["level"]
        unit = "cm" if t == "pos" else "°" if t == "hd" else ""
        cells = (f'<td class="l sticky">{TARGET[t]}</td><td class="l">{KIND[k]}</td>'
                 f"<td>{lv:g}{unit}</td>")
        if robot:
            cells += f"<td>{lv / SCALE_S:.2f}cm</td>" if t == "pos" else '<td class="dash">—</td>'
        cells += f"<td>{r['rep'] + 1}</td>"
        for m in ["SIr", "SId", "RVL"]:
            s = r.get(m)
            cells += "".join(fmt(s["median"][i] if s else None) for i in range(3))
            cells += fmt(s["jt_p"] if s else None, 3, p=True)
            cells += "".join(fmt(s["wilcox_bh"][q] if s else None, 3, p=True) for q in PAIRS)
        srsa = t != "hd"
        cells += "".join(fmt(r["sRSA"][s] if srsa else None) for s in STAGES)
        for c in ["place", "hd", "conjunctive"]:
            cells += "".join(f"<td>{r['counts'][s][c]}</td>" for s in STAGES)
        cls = ' class="base"' if t == "none" else ""
        rows.append(f'<tr{cls} data-t="{t}" data-k="{k}">{cells}</tr>')
    return (f"<h2>{html.escape(LABELS.get(model, model))}</h2>"
            f'<div class="wrap"><table><thead><tr>{h1}</tr><tr>{h2}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["paper_s1", "robot_s1"])
    parser.add_argument("--dir", type=Path, default=REPO / "reports" / "label_noise")
    args = parser.parse_args()

    body = "".join(table(m, json.loads((args.dir / f"{m}.json").read_text())) for m in args.models)
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label noise results</title><style>{CSS}</style></head><body>
<h1>位置・頭方向ラベルのノイズ：集計前の結果</h1>
<p class="note">保存済みの活動（walk・run・adult、各 500 ユニット）に対し、解析用の位置と頭方向にだけノイズを足して計算し直した値。1 行 = 1 条件の 1 回分（乱数を変えて 5 回）。黄色の行はノイズなし。</p>
<p class="note">中央値は 500 ユニットの中央値、sRSA は 90 系列の平均。検定 p：JT は walk&lt;run&lt;adult の Jonckheere 検定（10,000 回の並べ替え）、ほかは片側 Wilcoxon 順位和検定（BH 補正）。赤字は p &lt; 0.05。「—」はその条件では計算し直していない値（ノイズなしと同じ）。細胞数は place = place のみ、HD = HD のみ、conjunctive = 両方。</p>
<p class="note">種類：独立 = 毎フレーム独立のガウスノイズ、ずれ = 時定数 100 フレームのなめらかなずれ。位置は壁で折り返し。拡大環境のラット換算は 実寸 ÷ 8.16。</p>
<div class="controls"><label>ノイズ <select id="ft" onchange="apply()"><option value="all">すべて</option>
<option value="pos">位置</option><option value="hd">頭方向</option></select></label>
<label>種類 <select id="fk" onchange="apply()"><option value="all">すべて</option>
<option value="white">独立</option><option value="drift">ずれ</option></select></label></div>
{body}<script>{JS}</script></body></html>"""
    out = args.dir / f"raw_{'_'.join(args.models)}.html"
    out.write_text(page)
    print(f"[+] Saved {out}")


if __name__ == "__main__":
    main()
