#!/usr/bin/env python3
"""
experiments/report_phase3.py
============================
Generate an HTML report for Phase 3 (PCA Sweep) results.

Usage:
    python experiments/report_phase3.py [--out results/phase_3_report.html]
"""

from __future__ import annotations

import argparse
import math
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GROUPS = {
    "09_pca_none": "No PCA",
    "10_pca3":     "PCA=3",
    "11_pca4":     "PCA=4",
}

STRATEGIES = [
    "adx_dm", "channel_breakout", "dema", "donchian", "hmm_mr",
    "ichimoku", "macd", "parabolic_sar", "rsi", "sma",
    "tsmom", "turtle", "vol_adj",
]

TIMEFRAMES = ["is2_oos1", "is3_oos1"]

# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt(val, decimals=3):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "n/a"
    return f"{val:.{decimals}f}"


def _color(val, higher_better=True):
    if val is None or (isinstance(val, float) and math.isnan(val)) or higher_better is None:
        return ""
    if higher_better:
        return "color:#1a7a1a;font-weight:600" if val > 0 else ("color:#b22222;font-weight:600" if val < 0 else "")
    else:
        return "color:#1a7a1a;font-weight:600" if val < 0 else ("color:#b22222;font-weight:600" if val > 0 else "")


def load_master(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["phase"] == 3].copy()
    return df


# ── section builders ──────────────────────────────────────────────────────────

def pca_summary_table(df: pd.DataFrame) -> str:
    best_sharpe = -1e9
    best_gid = None
    summary_rows = []

    for gid, label in GROUPS.items():
        sub = df[df["group_id"] == gid]
        if sub.empty:
            continue
        row = dict(
            gid=gid, label=label, n=len(sub),
            win_rate=sub["win_rate"].mean(),
            bl_sharpe=sub["bl_oos_sharpe_mean"].mean(),
            hmm_sharpe=sub["hmm_oos_sharpe_mean"].mean(),
            delta_sharpe=sub["delta_sharpe"].mean(),
            bl_cagr=sub["bl_oos_annual_mean"].mean(),
            hmm_cagr=sub["hmm_oos_annual_mean"].mean(),
            delta_cagr=sub["delta_annual"].mean(),
            bl_dd=sub["bl_oos_dd_mean"].mean(),
            hmm_dd=sub["hmm_oos_dd_mean"].mean(),
            delta_dd=sub["delta_dd"].mean(),
            bl_calmar=sub["bl_oos_calmar_mean"].mean(),
            hmm_calmar=sub["hmm_oos_calmar_mean"].mean(),
            delta_calmar=sub["delta_calmar"].mean(),
            bl_trades=sub["bl_oos_trades_mean"].mean(),
            hmm_trades=sub["hmm_oos_trades_mean"].mean(),
            delta_trades=sub["delta_trades"].mean(),
        )
        summary_rows.append(row)
        if row["hmm_sharpe"] > best_sharpe:
            best_sharpe = row["hmm_sharpe"]
            best_gid = gid

    rows_html = ""
    for r in summary_rows:
        hi = ' style="background:#fffde7"' if r["gid"] == best_gid else ""
        star = "&nbsp;⭐" if r["gid"] == best_gid else ""
        rows_html += f"""
        <tr{hi}>
          <td><strong>{r['label']}</strong>{star}</td>
          <td>{r['n']}</td>
          <td>{_fmt(r['win_rate'], 1)}%</td>
          <td>{_fmt(r['bl_sharpe'])}</td>
          <td style="{_color(r['delta_sharpe'])}">{_fmt(r['hmm_sharpe'])}</td>
          <td style="{_color(r['delta_sharpe'])}">{_fmt(r['delta_sharpe'])}</td>
          <td>{_fmt(r['bl_cagr'], 2)}%</td>
          <td style="{_color(r['delta_cagr'])}">{_fmt(r['hmm_cagr'], 2)}%</td>
          <td style="{_color(r['delta_cagr'])}">{_fmt(r['delta_cagr'], 2)}%</td>
          <td>{_fmt(r['bl_dd'], 2)}%</td>
          <td style="{_color(r['delta_dd'], higher_better=False)}">{_fmt(r['hmm_dd'], 2)}%</td>
          <td style="{_color(r['delta_dd'], higher_better=False)}">{_fmt(r['delta_dd'], 2)}%</td>
          <td>{_fmt(r['bl_calmar'])}</td>
          <td style="{_color(r['delta_calmar'])}">{_fmt(r['hmm_calmar'])}</td>
          <td style="{_color(r['delta_calmar'])}">{_fmt(r['delta_calmar'])}</td>
          <td>{_fmt(r['bl_trades'], 1)}</td>
          <td>{_fmt(r['hmm_trades'], 1)}</td>
          <td>{_fmt(r['delta_trades'], 1)}</td>
        </tr>"""

    return f"""
    <h2>1. PCA Summary (mean across all strategies &amp; timeframes)</h2>
    <div class="scroll">
    <table>
      <thead>
        <tr>
          <th>PCA</th><th>Rows</th><th>Win%</th>
          <th>BL Sharpe</th><th>HMM Sharpe</th><th>Δ Sharpe</th>
          <th>BL CAGR%</th><th>HMM CAGR%</th><th>Δ CAGR%</th>
          <th>BL MaxDD%</th><th>HMM MaxDD%</th><th>Δ MaxDD%</th>
          <th>BL Calmar</th><th>HMM Calmar</th><th>Δ Calmar</th>
          <th>BL Trades</th><th>HMM Trades</th><th>Δ Trades</th>
        </tr>
      </thead>
      <tbody>{rows_html}
      </tbody>
    </table>
    </div>
    <p class="note">⭐ = best mean HMM Sharpe &nbsp;|&nbsp; Δ = HMM − Baseline &nbsp;|&nbsp;
    Win% = % of OOS time windows where HMM beat baseline Sharpe</p>
    """


def strategy_heatmap(df: pd.DataFrame, metric: str, metric_label: str,
                     higher_better, decimals: int = 3, section_num: int = 2) -> str:
    col_order = [(gid, tf) for gid in GROUPS for tf in TIMEFRAMES]
    col_labels = [f"{GROUPS[g]}<br><small>{tf}</small>" for g, tf in col_order]

    def cell_val(gid, tf, strat):
        sub = df[(df["group_id"] == gid) & (df["timeframe"] == tf) & (df["strategy"] == strat)]
        if sub.empty:
            return None
        v = sub[metric].iloc[0]
        return None if (isinstance(v, float) and math.isnan(v)) else v

    all_vals = [cell_val(g, tf, s) for g, tf in col_order for s in STRATEGIES]
    all_vals = [v for v in all_vals if v is not None]
    vmax = max(abs(v) for v in all_vals) if all_vals else 1.0

    def bg_color(val):
        if val is None:
            return "#f5f5f5"
        intensity = min(abs(val) / (vmax + 1e-9), 1.0)
        if higher_better is None:
            b = int(180 + 75 * intensity)
            return f"rgb(200,200,{b})"
        if (higher_better and val > 0) or (not higher_better and val < 0):
            g = int(180 + 75 * intensity)
            return f"rgb(200,{g},200)"
        elif val != 0:
            r = int(180 + 75 * intensity)
            return f"rgb({r},200,200)"
        return "#ffffff"

    header = "<tr><th>Strategy</th>" + "".join(f"<th>{lbl}</th>" for lbl in col_labels) + "</tr>"
    body = ""
    for strat in STRATEGIES:
        body += f"<tr><td><strong>{strat}</strong></td>"
        for gid, tf in col_order:
            val = cell_val(gid, tf, strat)
            body += f'<td style="background:{bg_color(val)};text-align:center">{_fmt(val, decimals) if val is not None else "—"}</td>'
        body += "</tr>"

    note = "Green = favourable &nbsp;|&nbsp; Red = unfavourable" if higher_better is not None \
           else "Blue intensity = magnitude of trade-count change"

    return f"""
    <h2>{section_num}. {metric_label} by Strategy &amp; PCA Setting</h2>
    <p class="note">{note}</p>
    <div class="scroll">
    <table>
      <thead>{header}</thead>
      <tbody>{body}</tbody>
    </table>
    </div>
    """


def win_rate_table(df: pd.DataFrame) -> str:
    col_order = [(gid, tf) for gid in GROUPS for tf in TIMEFRAMES]
    col_labels = [f"{GROUPS[g]}<br><small>{tf}</small>" for g, tf in col_order]

    header = "<tr><th>Strategy</th>" + "".join(f"<th>{lbl}</th>" for lbl in col_labels) + "</tr>"
    body = ""
    for strat in STRATEGIES:
        body += f"<tr><td><strong>{strat}</strong></td>"
        for gid, tf in col_order:
            sub = df[(df["group_id"] == gid) & (df["timeframe"] == tf) & (df["strategy"] == strat)]
            if sub.empty:
                body += "<td style='text-align:center'>—</td>"
                continue
            wr = sub["win_rate"].iloc[0]
            if math.isnan(wr):
                body += "<td style='text-align:center'>—</td>"
                continue
            bg = f"rgb({int(255 - wr*1.5)},{int(180 + wr*0.75)},{int(255 - wr*1.5)})"
            body += f'<td style="background:{bg};text-align:center">{wr:.0f}%</td>'
        body += "</tr>"

    return f"""
    <h2>8. Window Win Rate (% windows HMM improved OOS Sharpe)</h2>
    <p class="note">Darker green = more windows improved &nbsp;|&nbsp;
    Win rate = % of OOS <em>time windows</em> where HMM beat baseline (not per-trade win rate)</p>
    <div class="scroll">
    <table>
      <thead>{header}</thead>
      <tbody>{body}</tbody>
    </table>
    </div>
    """


def top_configs_table(df: pd.DataFrame, n: int = 15) -> str:
    top = df.nlargest(n, "hmm_oos_sharpe_mean").copy()
    rows = ""
    for _, r in top.iterrows():
        rows += f"""<tr>
          <td>{GROUPS.get(r['group_id'], r['group_id'])}</td>
          <td>{r['strategy']}</td>
          <td>{r['timeframe']}</td>
          <td>{_fmt(r['win_rate'], 1)}%</td>
          <td>{_fmt(r['bl_oos_sharpe_mean'])}</td>
          <td style="{_color(r['delta_sharpe'])}">{_fmt(r['hmm_oos_sharpe_mean'])}</td>
          <td style="{_color(r['delta_sharpe'])}">{_fmt(r['delta_sharpe'])}</td>
          <td>{_fmt(r['bl_oos_annual_mean'], 2)}%</td>
          <td style="{_color(r['delta_annual'])}">{_fmt(r['hmm_oos_annual_mean'], 2)}%</td>
          <td>{_fmt(r['bl_oos_dd_mean'], 2)}%</td>
          <td style="{_color(r['delta_dd'], higher_better=False)}">{_fmt(r['hmm_oos_dd_mean'], 2)}%</td>
          <td>{_fmt(r['bl_oos_calmar_mean'])}</td>
          <td style="{_color(r['delta_calmar'])}">{_fmt(r['hmm_oos_calmar_mean'])}</td>
          <td>{_fmt(r['bl_oos_trades_mean'], 1)}</td>
          <td>{_fmt(r['hmm_oos_trades_mean'], 1)}</td>
        </tr>"""
    return f"""
    <h2>9. Top {n} Configurations by HMM OOS Sharpe</h2>
    <div class="scroll">
    <table>
      <thead>
        <tr>
          <th>PCA</th><th>Strategy</th><th>Timeframe</th><th>Win%</th>
          <th>BL Sharpe</th><th>HMM Sharpe</th><th>Δ Sharpe</th>
          <th>BL CAGR%</th><th>HMM CAGR%</th>
          <th>BL MaxDD%</th><th>HMM MaxDD%</th>
          <th>BL Calmar</th><th>HMM Calmar</th>
          <th>BL Trades</th><th>HMM Trades</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    </div>
    """


def findings_section(df: pd.DataFrame) -> str:
    pca_sharpes = {}
    for gid, label in GROUPS.items():
        sub = df[df["group_id"] == gid]
        if not sub.empty:
            pca_sharpes[label] = sub["hmm_oos_sharpe_mean"].mean()

    best_label = max(pca_sharpes, key=pca_sharpes.get)
    best_gid = [g for g, l in GROUPS.items() if l == best_label][0]
    best_df = df[df["group_id"] == best_gid]
    improved = best_df[best_df["delta_sharpe"] > 0]["strategy"].unique().tolist()
    hurt     = best_df[best_df["delta_sharpe"] < 0]["strategy"].unique().tolist()

    improved_str = ", ".join(f"<code>{s}</code>" for s in sorted(set(improved)))
    hurt_str     = ", ".join(f"<code>{s}</code>" for s in sorted(set(hurt)))

    ranking_rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
        for k, v in sorted(pca_sharpes.items(), key=lambda x: -x[1])
    )

    return f"""
    <h2>10. Key Findings &amp; Recommendation</h2>
    <h3>10.1 PCA Rankings (mean HMM OOS Sharpe)</h3>
    <table style="width:280px">
      <thead><tr><th>PCA Setting</th><th>Mean HMM Sharpe</th></tr></thead>
      <tbody>{ranking_rows}</tbody>
    </table>

    <h3>10.2 Winner: <span style="color:#1a7a1a">{best_label}</span></h3>
    <p>Using raw features without dimensionality reduction preserves the full
    information content of the feature space for the HMM. PCA can hurt when
    the number of components is too low relative to the true signal dimensions,
    or when the rotation discards regime-relevant variance.</p>

    <h3>10.3 Strategies benefiting from {best_label}</h3>
    <p>Positive Δ Sharpe in at least one timeframe: {improved_str}</p>

    <h3>10.4 Strategies not benefiting</h3>
    <p>Consistently negative Δ Sharpe: {hurt_str}</p>

    <h3>10.5 Recommendation for Phase 4</h3>
    <p>Proceed with <strong>mode = score</strong>, <strong>K = 4</strong>,
    <strong>{best_label}</strong>.
    Update <code>groups.yaml</code> Phase 4 entries and sweep feature sets.</p>
    """


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; background: #fafafa; color: #222; }
.header { background: #4a148c; color: #fff; padding: 28px 40px; }
.header h1 { margin: 0; font-size: 1.8em; }
.header p { margin: 6px 0 0; opacity: 0.85; }
.content { padding: 32px 40px; max-width: 1600px; }
h2 { border-bottom: 2px solid #4a148c; padding-bottom: 6px; color: #4a148c; margin-top: 40px; }
h3 { color: #333; }
table { border-collapse: collapse; font-size: 0.85em; width: 100%; margin-bottom: 16px; }
th { background: #4a148c; color: #fff; padding: 7px 10px; text-align: left; white-space: nowrap; }
td { padding: 5px 10px; border: 1px solid #ddd; white-space: nowrap; }
tr:nth-child(even) { background: #f3e5f5; }
tr:hover { background: #e1bee7; }
.note { font-size: 0.82em; color: #666; margin-top: -8px; }
.scroll { overflow-x: auto; }
code { background: #f3e5f5; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }
"""


# ── build ─────────────────────────────────────────────────────────────────────

def build_report(master_path: str, out_path: str):
    df = load_master(master_path)

    pca_sum      = pca_summary_table(df)
    sharpe_heat  = strategy_heatmap(df, "delta_sharpe",  "Δ OOS Sharpe (HMM − Baseline)",       higher_better=True,  decimals=3, section_num=2)
    return_heat  = strategy_heatmap(df, "delta_return",  "Δ OOS Return % (HMM − Baseline)",     higher_better=True,  decimals=2, section_num=3)
    cagr_heat    = strategy_heatmap(df, "delta_annual",  "Δ OOS CAGR % (HMM − Baseline)",       higher_better=True,  decimals=2, section_num=4)
    calmar_heat  = strategy_heatmap(df, "delta_calmar",  "Δ OOS Calmar Ratio (HMM − Baseline)", higher_better=True,  decimals=3, section_num=5)
    dd_heat      = strategy_heatmap(df, "delta_dd",      "Δ OOS MaxDD % (HMM − Baseline)",      higher_better=False, decimals=2, section_num=6)
    trades_heat  = strategy_heatmap(df, "delta_trades",  "Δ OOS Trades (HMM − Baseline)",       higher_better=None,  decimals=1, section_num=7)
    wr_table     = win_rate_table(df)
    top_table    = top_configs_table(df, n=15)
    findings     = findings_section(df)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Phase 3 Report — PCA Sweep</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="header">
    <h1>Phase 3 Report &mdash; PCA (Dimensionality Reduction) Sweep</h1>
    <p>SPY &amp; QQQ &nbsp;|&nbsp; 2010–2026 &nbsp;|&nbsp; 13 strategies &nbsp;|&nbsp;
       IS 2yr&amp;3yr / OOS 1yr walk-forward &nbsp;|&nbsp;
       Mode: Score &nbsp;|&nbsp; K=4 &nbsp;|&nbsp; PCA ∈ {{None, 3, 4}}</p>
    <p style="opacity:0.75;font-size:0.85em">
      <strong>Win rate</strong> = % of OOS <em>time windows</em> where HMM OOS Sharpe &gt; baseline
      (not per-trade win rate).
    </p>
  </div>
  <div class="content">
    {pca_sum}
    {sharpe_heat}
    {return_heat}
    {cagr_heat}
    {calmar_heat}
    {dd_heat}
    {trades_heat}
    {wr_table}
    {top_table}
    {findings}
    <p style="margin-top:60px;font-size:0.75em;color:#aaa">
      Generated from {master_path}
    </p>
  </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master", default=os.path.join(ROOT, "results", "master_results_phase3.csv"),
                        help="Path to phase 3 master CSV")
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "phase_3_report.html"),
                        help="Output HTML path")
    args = parser.parse_args()
    build_report(args.master, args.out)


if __name__ == "__main__":
    main()
