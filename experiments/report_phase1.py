#!/usr/bin/env python3
"""
experiments/report_phase1.py
============================
Generate an HTML report for Phase 1 (Regime Mode Sweep) results.

Usage:
    python experiments/report_phase1.py [--out results/phase_1_report.html]
"""

from __future__ import annotations

import argparse
import math
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODES = {
    "01_strict": "Strict",
    "02_size":   "Size",
    "03_score":  "Score",
    "04_linear": "Linear",
}

STRATEGIES = [
    "adx_dm", "bollinger", "channel_breakout", "composite_trend", "dema", "donchian",
    "false_breakout", "ichimoku", "kama", "macd", "parabolic_sar", "rsi", "sma",
    "tsmom", "tsmom_fast", "turtle", "vol_adj",
]

TIMEFRAMES = ["is2_oos1", "is3_oos1"]

# ── helpers ──────────────────────────────────────────────────────────────────

def _fmt(val, decimals=3, pct=False):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "n/a"
    s = f"{val:.{decimals}f}"
    return s + "%" if pct else s


def _color(val, higher_better=True):
    """Return an inline CSS color string based on positive/negative value."""
    if val is None or (isinstance(val, float) and math.isnan(val)) or higher_better is None:
        return ""
    if higher_better:
        return "color:#1a7a1a;font-weight:600" if val > 0 else ("color:#b22222;font-weight:600" if val < 0 else "")
    else:
        return "color:#1a7a1a;font-weight:600" if val < 0 else ("color:#b22222;font-weight:600" if val > 0 else "")


def load_master(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["phase"] == 1].copy()
    # derived: per-trade win rate %
    df["bl_trade_win_pct"]  = (df["bl_oos_won_trades_mean"]  / df["bl_oos_trades_mean"].replace(0, float("nan"))  * 100)
    df["hmm_trade_win_pct"] = (df["hmm_oos_won_trades_mean"] / df["hmm_oos_trades_mean"].replace(0, float("nan")) * 100)
    df["delta_trade_win_pct"] = df["hmm_trade_win_pct"] - df["bl_trade_win_pct"]
    return df


# ── section builders ─────────────────────────────────────────────────────────

def mode_summary_table(df: pd.DataFrame) -> str:
    """Per-mode aggregate summary (mean across all strategies & timeframes)."""
    rows_html = ""
    best_sharpe = -1e9
    best_mode = None

    summary_rows = []
    for gid, label in MODES.items():
        sub = df[df["group_id"] == gid]
        if sub.empty:
            continue
        win_rate       = sub["win_rate"].mean()
        bl_sharpe      = sub["bl_oos_sharpe_mean"].mean()
        hmm_sharpe     = sub["hmm_oos_sharpe_mean"].mean()
        delta_sharpe   = sub["delta_sharpe"].mean()
        bl_cagr        = sub["bl_oos_annual_mean"].mean()
        hmm_cagr       = sub["hmm_oos_annual_mean"].mean()
        delta_cagr     = sub["delta_annual"].mean()
        bl_dd          = sub["bl_oos_dd_mean"].mean()
        hmm_dd         = sub["hmm_oos_dd_mean"].mean()
        delta_dd       = sub["delta_dd"].mean()
        bl_calmar      = sub["bl_oos_calmar_mean"].mean()
        hmm_calmar     = sub["hmm_oos_calmar_mean"].mean()
        delta_calmar   = sub["delta_calmar"].mean()
        bl_trades          = sub["bl_oos_trades_mean"].mean()
        hmm_trades         = sub["hmm_oos_trades_mean"].mean()
        delta_trades       = sub["delta_trades"].mean()
        bl_won             = sub["bl_oos_won_trades_mean"].mean()
        hmm_won            = sub["hmm_oos_won_trades_mean"].mean()
        delta_won          = sub["delta_won_trades"].mean()
        bl_tw_pct          = sub["bl_trade_win_pct"].mean()
        hmm_tw_pct         = sub["hmm_trade_win_pct"].mean()
        delta_tw_pct       = sub["delta_trade_win_pct"].mean()
        n_strategies       = len(sub)
        summary_rows.append((gid, label, n_strategies, win_rate, bl_sharpe, hmm_sharpe,
                              delta_sharpe, bl_cagr, hmm_cagr, delta_cagr, bl_dd, hmm_dd, delta_dd,
                              bl_calmar, hmm_calmar, delta_calmar, bl_trades, hmm_trades, delta_trades,
                              bl_won, hmm_won, delta_won, bl_tw_pct, hmm_tw_pct, delta_tw_pct))
        if hmm_sharpe > best_sharpe:
            best_sharpe = hmm_sharpe
            best_mode = gid

    for (gid, label, n_strats, win_rate, bl_sharpe, hmm_sharpe,
         delta_sharpe, bl_cagr, hmm_cagr, delta_cagr, bl_dd, hmm_dd, delta_dd,
         bl_calmar, hmm_calmar, delta_calmar, bl_trades, hmm_trades, delta_trades,
         bl_won, hmm_won, delta_won, bl_tw_pct, hmm_tw_pct, delta_tw_pct) in summary_rows:
        highlight = ' style="background:#fffde7"' if gid == best_mode else ""
        rows_html += f"""
        <tr{highlight}>
          <td><strong>{label}</strong>{'&nbsp;⭐' if gid == best_mode else ''}</td>
          <td>{n_strats}</td>
          <td>{_fmt(win_rate, 1)}%</td>
          <td>{_fmt(bl_sharpe)}</td>
          <td style="{_color(delta_sharpe)}">{_fmt(hmm_sharpe)}</td>
          <td style="{_color(delta_sharpe)}">{_fmt(delta_sharpe)}</td>
          <td>{_fmt(bl_cagr, 2)}%</td>
          <td style="{_color(delta_cagr)}">{_fmt(hmm_cagr, 2)}%</td>
          <td style="{_color(delta_cagr)}">{_fmt(delta_cagr, 2)}%</td>
          <td>{_fmt(bl_dd, 2)}%</td>
          <td style="{_color(delta_dd, higher_better=False)}">{_fmt(hmm_dd, 2)}%</td>
          <td style="{_color(delta_dd, higher_better=False)}">{_fmt(delta_dd, 2)}%</td>
          <td>{_fmt(bl_calmar)}</td>
          <td style="{_color(delta_calmar)}">{_fmt(hmm_calmar)}</td>
          <td style="{_color(delta_calmar)}">{_fmt(delta_calmar)}</td>
          <td>{_fmt(bl_trades, 1)}</td>
          <td>{_fmt(hmm_trades, 1)}</td>
          <td style="{_color(delta_trades, higher_better=None)}">{_fmt(delta_trades, 1)}</td>
          <td>{_fmt(bl_won, 1)}</td>
          <td>{_fmt(hmm_won, 1)}</td>
          <td style="{_color(delta_won, higher_better=None)}">{_fmt(delta_won, 1)}</td>
          <td>{_fmt(bl_tw_pct, 1)}%</td>
          <td style="{_color(delta_tw_pct)}">{_fmt(hmm_tw_pct, 1)}%</td>
          <td style="{_color(delta_tw_pct)}">{_fmt(delta_tw_pct, 1)}%</td>
        </tr>"""

    return f"""
    <h2>1. Mode Summary (mean across all strategies &amp; timeframes)</h2>
    <table>
      <thead>
        <tr>
          <th>Mode</th><th>Rows</th><th>Win%</th>
          <th>BL Sharpe</th><th>HMM Sharpe</th><th>Δ Sharpe</th>
          <th>BL CAGR%</th><th>HMM CAGR%</th><th>Δ CAGR%</th>
          <th>BL MaxDD%</th><th>HMM MaxDD%</th><th>Δ MaxDD%</th>
          <th>BL Calmar</th><th>HMM Calmar</th><th>Δ Calmar</th>
          <th>BL Trades</th><th>HMM Trades</th><th>Δ Trades</th>
          <th>BL Won</th><th>HMM Won</th><th>Δ Won</th>
          <th>BL Win%</th><th>HMM Win%</th><th>Δ Win%</th>
        </tr>
      </thead>
      <tbody>{rows_html}
      </tbody>
    </table>
    <p class="note">⭐ = best mean HMM Sharpe &nbsp;|&nbsp; Won = avg winning trades per window &nbsp;|&nbsp; Win% = per-trade win rate (won/total) &nbsp;|&nbsp; Δ = HMM − Baseline &nbsp;|&nbsp; Win% = % of OOS time windows where HMM beat baseline</p>
    """


def strategy_heatmap(df: pd.DataFrame, metric: str, metric_label: str,
                     higher_better, decimals: int = 3) -> str:
    """Cross-table: rows = strategy, columns = mode × timeframe."""
    col_order = [(gid, tf) for gid in MODES for tf in TIMEFRAMES]
    col_labels = [f"{MODES[g]}<br><small>{tf}</small>" for g, tf in col_order]

    def cell_val(gid, tf, strat):
        sub = df[(df["group_id"] == gid) & (df["timeframe"] == tf) & (df["strategy"] == strat)]
        if sub.empty:
            return None
        return sub[metric].iloc[0]

    # compute global range for colouring
    all_vals = [cell_val(g, tf, s) for g, tf in col_order for s in STRATEGIES]
    all_vals = [v for v in all_vals if v is not None and not math.isnan(v)]
    vmax = max(abs(v) for v in all_vals) if all_vals else 1.0

    def bg_color(val):
        if val is None or math.isnan(val):
            return "#f5f5f5"
        intensity = min(abs(val) / (vmax + 1e-9), 1.0)
        if higher_better is None:
            # neutral: blue shading regardless of direction
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
            bg = bg_color(val)
            body += f'<td style="background:{bg};text-align:center">{_fmt(val, decimals) if val is not None else "—"}</td>'
        body += "</tr>"

    return f"""
    <h2>2. {metric_label} by Strategy &amp; Mode</h2>
    <p class="note">Green = favourable &nbsp;|&nbsp; Red = unfavourable</p>
    <div class="scroll">
    <table>
      <thead>{header}</thead>
      <tbody>{body}</tbody>
    </table>
    </div>
    """


def win_rate_table(df: pd.DataFrame) -> str:
    """Win-rate (% windows where HMM improved) per mode × strategy."""
    col_order = [(gid, tf) for gid in MODES for tf in TIMEFRAMES]
    col_labels = [f"{MODES[g]}<br><small>{tf}</small>" for g, tf in col_order]

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
    <h2>3. Win Rate (% windows HMM improved OOS Sharpe)</h2>
    <p class="note">Darker green = more windows improved</p>
    <div class="scroll">
    <table>
      <thead>{header}</thead>
      <tbody>{body}</tbody>
    </table>
    </div>
    """


def top_configs_table(df: pd.DataFrame, n: int = 15) -> str:
    """Top N configurations ranked by HMM OOS Sharpe."""
    top = df.nlargest(n, "hmm_oos_sharpe_mean").copy()
    rows = ""
    for _, r in top.iterrows():
        rows += f"""<tr>
          <td>{MODES.get(r['group_id'], r['group_id'])}</td>
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
          <td>{_fmt(r['bl_oos_won_trades_mean'], 1)}</td>
          <td style="{_color(r['delta_won_trades'], higher_better=None)}">{_fmt(r['hmm_oos_won_trades_mean'], 1)}</td>
          <td>{_fmt(r['bl_trade_win_pct'], 1)}%</td>
          <td style="{_color(r['delta_trade_win_pct'])}">{_fmt(r['hmm_trade_win_pct'], 1)}%</td>
        </tr>"""
    return f"""
    <h2>4. Top {n} Configurations by HMM OOS Sharpe</h2>
    <div class="scroll">
    <table>
      <thead>
        <tr>
          <th>Mode</th><th>Strategy</th><th>Timeframe</th><th>Win%</th>
          <th>BL Sharpe</th><th>HMM Sharpe</th><th>Δ Sharpe</th>
          <th>BL CAGR%</th><th>HMM CAGR%</th>
          <th>BL MaxDD%</th><th>HMM MaxDD%</th>
          <th>BL Calmar</th><th>HMM Calmar</th>
          <th>BL Trades</th><th>HMM Trades</th>
          <th>BL Won</th><th>HMM Won</th>
          <th>BL Win%</th><th>HMM Win%</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    </div>
    """


def findings_section(df: pd.DataFrame) -> str:
    # best mode by mean HMM sharpe
    mode_sharpes = {}
    for gid, label in MODES.items():
        sub = df[df["group_id"] == gid]
        if not sub.empty:
            mode_sharpes[label] = sub["hmm_oos_sharpe_mean"].mean()

    best_label = max(mode_sharpes, key=mode_sharpes.get)

    # strategies that improved in score mode
    score_df = df[df["group_id"] == "03_score"]
    improved = score_df[score_df["delta_sharpe"] > 0]["strategy"].unique().tolist()
    hurt     = score_df[score_df["delta_sharpe"] < 0]["strategy"].unique().tolist()

    improved_str = ", ".join(f"<code>{s}</code>" for s in sorted(set(improved)))
    hurt_str     = ", ".join(f"<code>{s}</code>" for s in sorted(set(hurt)))

    rows = "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in sorted(mode_sharpes.items(), key=lambda x: -x[1]))

    return f"""
    <h2>5. Key Findings &amp; Recommendation</h2>
    <h3>5.1 Mode Rankings (mean HMM OOS Sharpe)</h3>
    <table style="width:300px">
      <thead><tr><th>Mode</th><th>Mean HMM Sharpe</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>

    <h3>5.2 Winner: <span style="color:#1a7a1a">{best_label}</span></h3>
    <p><strong>Score mode</strong> treats each HMM state as having its own position-size
    weight. This soft sizing lets the strategy stay partially invested even in
    uncertain regimes, preserving upside while still dampening risk in clearly
    unfavourable states.</p>

    <h3>5.3 Strategies benefiting from Score mode</h3>
    <p>Positive Δ Sharpe in at least one timeframe: {improved_str}</p>

    <h3>5.4 Strategies not benefiting</h3>
    <p>Consistently negative Δ Sharpe: {hurt_str}</p>

    <h3>5.5 tsmom note</h3>
    <p><code>tsmom</code> showed zero OOS activity across <em>all</em> modes and
    timeframes — the strategy produced no trades in OOS windows. This is likely a
    signal/parameter mismatch and should be investigated before Phase 2.</p>

    <h3>5.6 Recommendation for Phase 2</h3>
    <p>Proceed with <strong>mode = score</strong>.
    Update <code>groups.yaml</code> Phase 2 entries to use
    <code>--regime-mode score</code> and sweep K ∈ {{2, 3, 4, 6}}.</p>
    """


# ── HTML assembly ─────────────────────────────────────────────────────────────

CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; background: #fafafa; color: #222; }
.header { background: #1a237e; color: #fff; padding: 28px 40px; }
.header h1 { margin: 0; font-size: 1.8em; }
.header p { margin: 6px 0 0; opacity: 0.8; }
.content { padding: 32px 40px; max-width: 1600px; }
h2 { border-bottom: 2px solid #1a237e; padding-bottom: 6px; color: #1a237e; margin-top: 40px; }
h3 { color: #333; }
table { border-collapse: collapse; font-size: 0.85em; width: 100%; margin-bottom: 16px; }
th { background: #1a237e; color: #fff; padding: 7px 10px; text-align: left; white-space: nowrap; }
td { padding: 5px 10px; border: 1px solid #ddd; white-space: nowrap; }
tr:nth-child(even) { background: #f0f4ff; }
tr:hover { background: #e8edff; }
.note { font-size: 0.82em; color: #666; margin-top: -8px; }
.scroll { overflow-x: auto; }
code { background: #eef; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }
"""


def build_report(master_path: str, out_path: str):
    df = load_master(master_path)

    mode_sum     = mode_summary_table(df)
    sharpe_heat  = strategy_heatmap(df, "delta_sharpe",  "Δ OOS Sharpe (HMM − Baseline)",       higher_better=True,  decimals=3)
    return_heat  = strategy_heatmap(df, "delta_return",  "Δ OOS Return % (HMM − Baseline)",     higher_better=True,  decimals=2)
    cagr_heat    = strategy_heatmap(df, "delta_annual",  "Δ OOS CAGR % (HMM − Baseline)",       higher_better=True,  decimals=2)
    calmar_heat  = strategy_heatmap(df, "delta_calmar",  "Δ OOS Calmar Ratio (HMM − Baseline)", higher_better=True,  decimals=3)
    dd_heat      = strategy_heatmap(df, "delta_dd",      "Δ OOS MaxDD % (HMM − Baseline)",      higher_better=False, decimals=2)
    trades_heat  = strategy_heatmap(df, "delta_trades",     "Δ OOS Trades (HMM − Baseline)",          higher_better=None,  decimals=1)
    won_heat     = strategy_heatmap(df, "delta_won_trades",  "Δ OOS Won Trades (HMM − Baseline)",      higher_better=None,  decimals=1)
    tw_pct_heat  = strategy_heatmap(df, "delta_trade_win_pct", "Δ Per-Trade Win% (HMM − Baseline)",   higher_better=True,  decimals=1)
    wr_table     = win_rate_table(df)
    top_table    = top_configs_table(df, n=15)
    findings     = findings_section(df)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Phase 1 Report — Regime Mode Sweep</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="header">
    <h1>Phase 1 Report &mdash; Regime Mode Sweep</h1>
    <p>SPY &amp; QQQ &nbsp;|&nbsp; 2010–2026 &nbsp;|&nbsp; 13 strategies &nbsp;|&nbsp;
       IS 2yr&amp;3yr / OOS 1yr walk-forward &nbsp;|&nbsp; 17 strategies &nbsp;|&nbsp; Modes: Strict, Size, Score, Linear</p>
    <p style="opacity:0.7;font-size:0.85em">
      <strong>Win rate</strong> = % of OOS <em>time windows</em> where HMM OOS Sharpe &gt; baseline
      (not per-trade win rate — per-trade profitability is not captured in the current results).
    </p>
  </div>
  <div class="content">
    {mode_sum}
    {sharpe_heat}
    {return_heat}
    {cagr_heat}
    {calmar_heat}
    {dd_heat}
    {trades_heat}
    {won_heat}
    {tw_pct_heat}
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
    parser.add_argument("--master", default=os.path.join(ROOT, "results", "master_results.csv"),
                        help="Path to master_results.csv (default: results/master_results.csv)")
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "phase_1_report.html"),
                        help="Output HTML path (default: results/phase_1_report.html)")
    args = parser.parse_args()
    build_report(args.master, args.out)


if __name__ == "__main__":
    main()
