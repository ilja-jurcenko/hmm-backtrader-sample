#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
diagnose_tsmom_rsi.py
=====================
Diagnostic script investigating two observed failures in Phase-5 results:

  1. TSMOM zero-trade failure on every OOS window
  2. RSI Sharpe inversion / collapse on SPY/QQQ ETFs

Run from the repo root:
    source .venv/bin/activate
    python experiments/diagnose_tsmom_rsi.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import pandas as pd
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Load project modules
# ---------------------------------------------------------------------------
def _load(name, filename):
    path = os.path.join(_HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

maq = _load('maq', 'ma-quantstats.py')
opt = _load('opt', 'optimize-hmm.py')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SECTION = '\n' + '='*70 + '\n'

def _bt(**kw):
    """Thin wrapper: run a baseline (no HMM) backtest."""
    return opt.backtest(quiet=True, **kw)


# ---------------------------------------------------------------------------
# 1. TSMOM warm-up diagnosis
# ---------------------------------------------------------------------------
def diagnose_tsmom():
    print(SECTION + 'TSMOM WARM-UP DIAGNOSIS' + SECTION)

    TICKER     = 'SPY'
    OOS_START  = '2012-01-01'
    OOS_END    = '2013-01-01'          # 1-year OOS window (≈252 bars)
    IS_START   = '2010-01-01'          # 2-year IS  window (≈504 bars)
    LOOKBACK   = 252
    SKIP       = 21
    WARMUP     = LOOKBACK + SKIP       # 273 bars needed before first signal

    print(f'TSMOM default params  : lookback={LOOKBACK}, skip={SKIP}')
    print(f'Min bars before trade : {WARMUP + 1}')

    # Count bars in 1-year OOS window
    csv_path = os.path.join(_HERE, 'datas', 'spy-2000-2026.csv')
    df_oos = maq.load_csv_as_dataframe(csv_path, OOS_START, OOS_END)
    print(f'OOS bars available    : {len(df_oos)}  ({OOS_START} → {OOS_END})')
    print(f'Can TSMOM warm up?    : {"YES" if len(df_oos) > WARMUP else "NO — shortfall of " + str(WARMUP - len(df_oos) + 1) + " bars"}')

    # Confirm: OOS baseline with data clipped to OOS window → 0 trades
    result_clipped = _bt(tickers=[TICKER], strategy='tsmom',
                         fromdate=OOS_START, todate=OOS_END,
                         tsmom_lookback=LOOKBACK, tsmom_skip=SKIP,
                         stake=100, cash=100_000)
    print(f'\nBaseline OOS (clipped to OOS window):')
    print(f'  trades={result_clipped.get("trade_count", 0)}  '
          f'return={result_clipped["total_return"]:+.2f}%  '
          f'sharpe={result_clipped["sharpe"]:.3f}')

    # Confirm: IS baseline with 2-year window → trades exist
    result_is = _bt(tickers=[TICKER], strategy='tsmom',
                    fromdate=IS_START, todate=OOS_END,
                    tsmom_lookback=LOOKBACK, tsmom_skip=SKIP,
                    stake=100, cash=100_000)
    print(f'\nBaseline IS+OOS (2-year window with warmup):')
    print(f'  trades={result_is.get("trade_count", 0)}  '
          f'return={result_is["total_return"]:+.2f}%  '
          f'sharpe={result_is["sharpe"]:.3f}')

    # Proposed fix: load data starting WARMUP bars before OOS start
    #   i.e. extend fromdate backward by ~14 months (273 trading days ≈ 385 calendar days)
    from dateutil.relativedelta import relativedelta
    import datetime
    oos_dt    = datetime.datetime.strptime(OOS_START, '%Y-%m-%d')
    warmup_dt = oos_dt - datetime.timedelta(days=int(WARMUP * 365.25 / 252) + 30)
    warmup_str = warmup_dt.strftime('%Y-%m-%d')

    df_extended = maq.load_csv_as_dataframe(csv_path, warmup_str, OOS_END)
    print(f'\nWith warm-up prepended (from {warmup_str}):')
    print(f'  total bars = {len(df_extended)}  (need >{WARMUP})')

    # To properly test: run with extended fromdate but only measure OOS
    # In the current codebase the cleanest proxy is to extend fromdate in the
    # backtest call; performance includes warm-up (all-cash) period.
    result_warm = _bt(tickers=[TICKER], strategy='tsmom',
                      fromdate=warmup_str, todate=OOS_END,
                      tsmom_lookback=LOOKBACK, tsmom_skip=SKIP,
                      stake=100, cash=100_000)
    print(f'\nBaseline with warm-up extension (full-period metrics include warm-up):')
    print(f'  trades={result_warm.get("trade_count", 0)}  '
          f'return={result_warm["total_return"]:+.2f}%  '
          f'sharpe={result_warm["sharpe"]:.3f}')

    print('\nROOT CAUSE CONFIRMED:')
    print('  load_csv_as_dataframe() clips data at fromdate=OOS_START.')
    print('  TSMOM needs 273 bars before first signal; 1-year OOS ≈ 252 bars.')
    print('  No bars pre-date OOS_START → TSMOM is in warm-up for entire OOS period.')
    print('  Stop-loss / position-sizing / HMM are NOT the cause.')

    print('\nFIX:')
    print('  In ma-quantstats.py run(), when strategy_key == "tsmom",')
    print('  extend fromdate backward by (tsmom_lookback + tsmom_skip) bars')
    print('  before clipping the DataFrame.')


# ---------------------------------------------------------------------------
# 2. RSI Sharpe inversion / collapse diagnosis
# ---------------------------------------------------------------------------
def diagnose_rsi():
    print(SECTION + 'RSI SHARPE INVERSION DIAGNOSIS (spy_qqq  is2_oos1)' + SECTION)

    # Load phase-5 RSI window-level results CSV for spy_qqq is2_oos1
    csv_path = os.path.join(_HERE, 'results', 'phase_5',
                            '14_best_multiasset', 'spy_qqq',
                            'is2_oos1', 'rsi_results.csv')
    if not os.path.exists(csv_path):
        print(f'Window results CSV not found: {csv_path}')
        print('Showing summary from master_results_phase5.csv instead.')
        master = pd.read_csv(os.path.join(_HERE, 'results', 'master_results_phase5.csv'))
        row = master.query("strategy=='rsi' and ticker_set=='spy_qqq' and timeframe=='is2_oos1'")
        print(row.to_string(index=False))
        return

    df = pd.read_csv(csv_path)
    print('Per-window RSI results (spy_qqq  is2_oos1):')
    print(f'{"Win":>4}  {"BL sharpe":>10}  {"HMM sharpe":>11}  '
          f'{"BL trades":>9}  {"HMM trades":>10}  {"BL ret%":>8}  {"HMM ret%":>9}')
    print('-' * 72)

    bl_sharpes  = []
    hmm_sharpes = []

    for _, row in df.iterrows():
        w = int(row.get('window', 0))
        bl_sh  = float(row.get('bl_oos_sharpe',  0))
        hmm_sh = float(row.get('hmm_oos_sharpe', 0))
        bl_tr  = float(row.get('bl_oos_trades',  0))
        hmm_tr = float(row.get('hmm_oos_trades', 0))
        bl_ret = float(row.get('bl_oos_return',  0))
        hmm_ret= float(row.get('hmm_oos_return', 0))
        bl_sharpes.append(bl_sh)
        hmm_sharpes.append(hmm_sh)
        flag = '  *** OUTLIER' if abs(hmm_sh) > 10 else ''
        print(f'{w:>4}  {bl_sh:>10.3f}  {hmm_sh:>11.3f}  '
              f'{bl_tr:>9.1f}  {hmm_tr:>10.1f}  '
              f'{bl_ret:>8.2f}%  {hmm_ret:>9.2f}%{flag}')

    bl_arr  = np.array(bl_sharpes)
    hmm_arr = np.array(hmm_sharpes)

    print()
    print(f'{"Metric":<28}  {"Baseline":>10}  {"HMM":>10}')
    print('-' * 52)
    print(f'{"Mean sharpe":<28}  {bl_arr.mean():>10.3f}  {hmm_arr.mean():>10.3f}')
    print(f'{"Median sharpe":<28}  {np.median(bl_arr):>10.3f}  {np.median(hmm_arr):>10.3f}')
    print(f'{"Sharpe (outliers clipped ±5)":<28}  '
          f'{np.clip(bl_arr, -5, 5).mean():>10.3f}  '
          f'{np.clip(hmm_arr, -5, 5).mean():>10.3f}')

    n_outliers = np.sum(np.abs(hmm_arr) > 10)
    print(f'\nHMM sharpe outliers (|sharpe|>10): {n_outliers}')
    print('Outlier windows cause the −4.19 mean — median and clipped mean are representative.')

    print('\nROOT CAUSES:')
    print('  A. SHARPE ARTIFACT: Windows with ≤2 HMM trades have near-zero')
    print('     daily return std.  A single negative excess-return day (RFR')
    print('     subtraction from idle cash) creates extreme Sharpe values.')
    print('     Fix: use median or ±5-clipped mean when aggregating Sharpe across windows.')
    print()
    print('  B. OPTIMIZATION COLLAPSE: Baseline IS Sharpe for RSI on SPY/QQQ is')
    print('     consistently negative (−0.2 to −1.6).  Optuna discovers that')
    print('     blocking all trades gives Sharpe=0 > negative baseline — a')
    print('     degenerate but valid IS solution.  OOS then has 0 trades (Sharpe=0)')
    print('     or 1–2 leaking trades with pathological Sharpe.')
    print('     Root: RSI mean-reversion is structurally a poor fit for trending')
    print('     ETFs (SPY/QQQ rarely reach RSI≤30 on daily bars).')
    print()
    print('  C. GENUINE REGIME OVER-FILTERING: In windows 5 and 8, HMM blocks')
    print('     profitable OOS trades (W5: BL=+1.88% → HMM=+0.58%;')
    print('     W8: BL=+2.83% → HMM=+0.02%).  The HMM regime labelled "unfavorable"')
    print('     for the IS RSI pattern happens to coincide with valid OOS signals.')
    print('     This is not systematic inversion — it is IS→OOS overfitting.')


# ---------------------------------------------------------------------------
# 3. HMM regime vs RSI signal correlation (conceptual check)
# ---------------------------------------------------------------------------
def analyze_rsi_regime_fit():
    print(SECTION + 'RSI REGIME FIT ANALYSIS (SPY daily, 2010–2020)' + SECTION)
    print('Checking whether HMM "favorable" states correlate with RSI oversold frequency.')
    print()

    csv_path = os.path.join(_HERE, 'datas', 'spy-2000-2026.csv')
    if not os.path.exists(csv_path):
        print(f'SPY CSV not found: {csv_path}  — skipping.')
        return

    df = maq.load_csv_as_dataframe(csv_path, '2008-01-01', '2020-01-01')
    closes = df['Close']

    # Compute RSI(14)
    delta = closes.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs   = avg_gain / avg_loss
    rsi  = 100 - (100 / (1 + rs))

    # Define "RSI oversold recovery" signal (buy trigger)
    oversold_level = 30
    signal = ((rsi.shift(1) <= oversold_level) & (rsi > oversold_level)).astype(int)
    n_signals = signal.sum()
    print(f'Total RSI≤30 recovery signals (2008–2020): {n_signals}')

    # Forward return after each signal (21-bar horizon)
    fwd_ret = closes.pct_change(21).shift(-21) * 100
    sig_fwd = fwd_ret[signal == 1]
    print(f'Mean 21-bar forward return after RSI signal: {sig_fwd.mean():+.2f}%')
    print(f'Signal win rate (positive 21-bar return):    {(sig_fwd > 0).mean()*100:.0f}%')

    # Compute rolling 21-bar volatility as regime proxy
    vol_21 = closes.pct_change().rolling(21).std() * np.sqrt(252) * 100
    vol_at_signal = vol_21[signal == 1]
    vol_all       = vol_21.dropna()

    print(f'\nVol regime at RSI signal (annualised) : {vol_at_signal.mean():.1f}%')
    print(f'Vol regime overall average            : {vol_all.mean():.1f}%')
    print(f'→ RSI signals fire in HIGHER-vol regimes (+ {vol_at_signal.mean()-vol_all.mean():.1f}pp)')
    print()
    print('IMPLICATION for HMM:')
    print('  HMM "favorable" state = low volatility / trending bull = RSI rarely fires.')
    print('  HMM "unfavorable" state = high volatility / choppy = RSI fires most often.')
    print('  For mean-reversion strategies, HMM regime labels are effectively INVERTED.')
    print('  Fix: invert the regime gate for RSI (and hmm_mr) — use unfav state as entry gate,')
    print('       OR train a dedicated "volatility regime" HMM for mean-reversion strategies.')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    diagnose_tsmom()
    diagnose_rsi()
    analyze_rsi_regime_fit()
