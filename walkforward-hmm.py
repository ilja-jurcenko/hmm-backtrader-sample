#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
Walk-Forward HMM Hyperparameter Optimisation
=============================================
Slides an expanding window over a long history, running the full
IS-optimise → OOS-validate pipeline at each step.

Window design
-------------
  full window : wf_start + i*step  →  wf_start + i*step + is_years + oos_years
  in-sample   : window_start       →  window_start + is_years
  out-of-sample: window_start + is_years → window_end

The window advances by --step years each iteration.

Usage
-----
  python walkforward-hmm.py
  python walkforward-hmm.py --ticker SPY --n-trials 60
  python walkforward-hmm.py --ticker SPY QQQ --wf-start 2019-01-01 \\
        --wf-end 2026-01-01 --is-years 3 --oos-years 2 --step 1 --n-trials 50
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import os
import sys
import time
import types
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Load optimize-hmm module (handles the hyphen in the filename)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

def _load_opt():
    """Load optimize-hmm.py as a module named 'opt'."""
    path = os.path.join(_HERE, 'optimize-hmm.py')
    spec = importlib.util.spec_from_file_location('opt', path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules['opt'] = mod
    spec.loader.exec_module(mod)
    return mod

opt = _load_opt()   # exposes: backtest, make_objective, print_results_table, maq


# ---------------------------------------------------------------------------
# Walk-forward window generator
# ---------------------------------------------------------------------------

def generate_windows(wf_start: str, wf_end: str,
                     is_years: int, oos_years: int,
                     step_years: int) -> list[dict]:
    """
    Yield dicts describing each walk-forward window.

    Each dict has keys: win_from, split, win_to  (all str YYYY-MM-DD).
    """
    start = datetime.strptime(wf_start, '%Y-%m-%d')
    end   = datetime.strptime(wf_end,   '%Y-%m-%d')

    windows = []
    cursor  = start
    while True:
        win_from = cursor
        split    = cursor + relativedelta(years=is_years)
        win_to   = split  + relativedelta(years=oos_years)
        if win_to > end:
            break
        windows.append(dict(
            win_from = win_from.strftime('%Y-%m-%d'),
            split    = split.strftime('%Y-%m-%d'),
            win_to   = win_to.strftime('%Y-%m-%d'),
        ))
        cursor += relativedelta(years=step_years)
    return windows


# ---------------------------------------------------------------------------
# Single window runner (wraps optimize-hmm logic)
# ---------------------------------------------------------------------------

def run_window(window: dict, tickers: list[str], cfg) -> dict:
    """
    Run IS Optuna optimisation + OOS and full-period validation for
    one walk-forward window.

    Returns a result dict with all metrics for the window.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    IS_FROM   = window['win_from']
    IS_TO     = window['split']
    OOS_FROM  = window['split']
    OOS_TO    = window['win_to']

    label = f'{IS_FROM} → {IS_TO} | OOS {OOS_FROM} → {OOS_TO}'
    print(f'\n{"=" * 70}')
    print(f'  Window : {label}')
    print(f'  Trials : {cfg.n_trials}')
    print(f'{"=" * 70}')

    common = dict(
        tickers        = tickers,
        strategy       = getattr(cfg, 'strategy', 'sma'),
        fast           = cfg.fast,
        slow           = cfg.slow,
        rsi_period     = getattr(cfg, 'rsi_period',         14),
        rsi_oversold   = getattr(cfg, 'rsi_oversold',       30),
        rsi_overbought = getattr(cfg, 'rsi_overbought',     70),
        macd_fast      = getattr(cfg, 'macd_fast',          12),
        macd_slow      = getattr(cfg, 'macd_slow',          26),
        macd_signal    = getattr(cfg, 'macd_signal',         9),
        hmm_mr_z_threshold = getattr(cfg, 'hmm_mr_z_threshold', 0.0),
        adx_period     = getattr(cfg, 'adx_period',     14),
        adx_threshold  = getattr(cfg, 'adx_threshold',  25.0),
        channel_period = getattr(cfg, 'channel_period', 20),
        donchian_entry = getattr(cfg, 'donchian_entry', 20),
        donchian_exit  = getattr(cfg, 'donchian_exit',  10),
        ichimoku_tenkan = getattr(cfg, 'ichimoku_tenkan', 9),
        ichimoku_kijun  = getattr(cfg, 'ichimoku_kijun',  26),
        ichimoku_senkou = getattr(cfg, 'ichimoku_senkou', 52),
        psar_af        = getattr(cfg, 'psar_af',        0.02),
        psar_max_af    = getattr(cfg, 'psar_max_af',    0.20),
        tsmom_lookback = getattr(cfg, 'tsmom_lookback', 252),
        tsmom_skip     = getattr(cfg, 'tsmom_skip',     21),
        turtle_entry   = getattr(cfg, 'turtle_entry',   20),
        turtle_exit    = getattr(cfg, 'turtle_exit',    10),
        turtle_atr     = getattr(cfg, 'turtle_atr',     20),
        turtle_atr_mult= getattr(cfg, 'turtle_atr_mult',2.0),
        vol_period     = getattr(cfg, 'vol_period',     20),
        vol_atr_period = getattr(cfg, 'vol_atr_period', 14),
        vol_atr_mult   = getattr(cfg, 'vol_atr_mult',   1.5),
        hmm_features   = getattr(cfg, 'hmm_features', None),
        hmm_pca        = getattr(cfg, 'hmm_pca', None),
        regime_mode    = getattr(cfg, 'regime_mode', 'strict'),
        unfav_fraction = getattr(cfg, 'unfav_fraction', 0.25),
        state_positions = getattr(cfg, 'state_positions', None),
        stake          = cfg.stake,
        cash           = cfg.cash,
        commission     = cfg.commission,
        hmm_favourable      = getattr(cfg, 'hmm_favourable', None),
        hmm_max_pos_size    = getattr(cfg, 'hmm_max_pos_size', 2.0),
        hmm_min_pos_size    = getattr(cfg, 'hmm_min_pos_size', 0.0),
        hmm_dynamic_scoring = getattr(cfg, 'hmm_dynamic_scoring', False),
        hmm_dynamic_window  = getattr(cfg, 'hmm_dynamic_window', 0),
        stop_loss_perc      = getattr(cfg, 'stop_loss_perc',   0.0),
        take_profit_perc    = getattr(cfg, 'take_profit_perc', 0.0),
    )

    # Baselines --------------------------------------------------------------
    bl_is  = opt.backtest(fromdate=IS_FROM,  todate=IS_TO,  **common)
    bl_oos = opt.backtest(fromdate=OOS_FROM, todate=OOS_TO, **common)
    print(f'  Baseline IS  return : {bl_is["total_return"]:+.2f}%  '
          f'cagr={bl_is["annual_return"]:+.2f}%  '
          f'sharpe={bl_is["sharpe"]:.3f}  '
          f'calmar={bl_is["calmar"]:.3f}  '
          f'maxdd={bl_is["max_drawdown"]:.2f}%  '
          f'trades={bl_is.get("trade_count", 0)}  '
          f'({bl_is["time_taken"]:.1f}s)')
    print(f'  Baseline OOS return : {bl_oos["total_return"]:+.2f}%  '
          f'cagr={bl_oos["annual_return"]:+.2f}%  '
          f'sharpe={bl_oos["sharpe"]:.3f}  '
          f'calmar={bl_oos["calmar"]:.3f}  '
          f'maxdd={bl_oos["max_drawdown"]:.2f}%  '
          f'trades={bl_oos.get("trade_count", 0)}  '
          f'({bl_oos["time_taken"]:.1f}s)')

    # IS optimisation --------------------------------------------------------
    print(f'\n  Optimising on IS period ({cfg.n_trials} trials) …')
    sampler  = optuna.samplers.TPESampler(seed=cfg.seed)
    study    = optuna.create_study(direction='maximize', sampler=sampler)
    fixed_st = getattr(cfg, 'hmm_score_threshold', None)
    fixed_hc = getattr(cfg, 'hmm_components', None)
    objective = opt.make_objective(tickers, IS_FROM, IS_TO, cfg.fast, cfg.slow,
                                   strategy           = getattr(cfg, 'strategy', 'sma'),
                                   rsi_period         = getattr(cfg, 'rsi_period',         14),
                                   rsi_oversold       = getattr(cfg, 'rsi_oversold',       30),
                                   rsi_overbought     = getattr(cfg, 'rsi_overbought',     70),
                                   macd_fast          = getattr(cfg, 'macd_fast',          12),
                                   macd_slow          = getattr(cfg, 'macd_slow',          26),
                                   macd_signal        = getattr(cfg, 'macd_signal',         9),
                                   hmm_mr_z_threshold = getattr(cfg, 'hmm_mr_z_threshold', 0.0),
                                   adx_period         = getattr(cfg, 'adx_period',     14),
                                   adx_threshold      = getattr(cfg, 'adx_threshold',  25.0),
                                   channel_period     = getattr(cfg, 'channel_period', 20),
                                   donchian_entry     = getattr(cfg, 'donchian_entry', 20),
                                   donchian_exit      = getattr(cfg, 'donchian_exit',  10),
                                   ichimoku_tenkan    = getattr(cfg, 'ichimoku_tenkan', 9),
                                   ichimoku_kijun     = getattr(cfg, 'ichimoku_kijun',  26),
                                   ichimoku_senkou    = getattr(cfg, 'ichimoku_senkou', 52),
                                   psar_af            = getattr(cfg, 'psar_af',        0.02),
                                   psar_max_af        = getattr(cfg, 'psar_max_af',    0.20),
                                   tsmom_lookback     = getattr(cfg, 'tsmom_lookback', 252),
                                   tsmom_skip         = getattr(cfg, 'tsmom_skip',     21),
                                   turtle_entry       = getattr(cfg, 'turtle_entry',   20),
                                   turtle_exit        = getattr(cfg, 'turtle_exit',    10),
                                   turtle_atr         = getattr(cfg, 'turtle_atr',     20),
                                   turtle_atr_mult    = getattr(cfg, 'turtle_atr_mult',2.0),
                                   vol_period         = getattr(cfg, 'vol_period',     20),
                                   vol_atr_period     = getattr(cfg, 'vol_atr_period', 14),
                                   vol_atr_mult       = getattr(cfg, 'vol_atr_mult',   1.5),
                                   hmm_features       = getattr(cfg, 'hmm_features', None),
                                   hmm_pca            = getattr(cfg, 'hmm_pca', None),
                                   regime_mode        = getattr(cfg, 'regime_mode', 'strict'),
                                   unfav_fraction     = getattr(cfg, 'unfav_fraction', 0.25),
                                   fixed_score_threshold=fixed_st,
                                   fixed_hmm_components=fixed_hc,
                                   objective_metric=getattr(cfg, 'objective_metric', 'total_return'),
                                   state_positions    = getattr(cfg, 'state_positions', None),
                                   search_state_positions = getattr(cfg, 'search_state_positions', False),
                                   hmm_favourable      = getattr(cfg, 'hmm_favourable', None),
                                   hmm_max_pos_size    = getattr(cfg, 'hmm_max_pos_size', 2.0),
                                   hmm_min_pos_size    = getattr(cfg, 'hmm_min_pos_size', 0.0),
                                   hmm_dynamic_scoring = getattr(cfg, 'hmm_dynamic_scoring', False),
                                   hmm_dynamic_window  = getattr(cfg, 'hmm_dynamic_window', 0),
                                   stop_loss_perc      = getattr(cfg, 'stop_loss_perc',   0.0),
                                   take_profit_perc    = getattr(cfg, 'take_profit_perc', 0.0))

    obj_metric = getattr(cfg, 'objective_metric', 'total_return')
    completed = [0]
    best_box  = [bl_is[obj_metric]]

    def _cb(study, trial):
        completed[0] += 1
        val = trial.value if trial.value is not None else float('nan')
        if val > best_box[0]:
            best_box[0] = val
        pct    = 100 * completed[0] / cfg.n_trials
        filled = int(pct / 100 * 25)
        bar    = '█' * filled + '░' * (25 - filled)
        print(f'\r  [{bar}] {pct:5.1f}%  trial={completed[0]:>3}/{cfg.n_trials}  '
              f'best={best_box[0]:+.2f}%  this={val:+.2f}%',
              end='', flush=True)

    study.optimize(objective, n_trials=cfg.n_trials, callbacks=[_cb])
    print()

    best_params    = study.best_trial.params
    best_is_return = study.best_trial.value

    print(f'  Best IS {obj_metric}  : {best_is_return:+.4f}  '
          f'(Δ vs baseline: {best_is_return - bl_is[obj_metric]:+.4f})')

    # OOS validation ---------------------------------------------------------
    best_score_threshold = (
        fixed_st if fixed_st is not None
        else best_params['hmm_score_threshold']
    )
    best_hmm_components = (
        fixed_hc if fixed_hc is not None
        else best_params['hmm_components']
    )
    # Extract best feature subset from trial params (feat_* keys)
    best_features = [k[5:] for k, v in best_params.items()
                     if k.startswith('feat_') and v is True]
    if not best_features:
        best_features = getattr(cfg, 'hmm_features', None)
    # Extract best unfav_fraction if optimised (regime_mode=size)
    best_unfav_fraction = best_params.get(
        'unfav_fraction', getattr(cfg, 'unfav_fraction', None) or 0.25)
    # Extract best state_positions if searched by Optuna
    cfg_state_positions = getattr(cfg, 'state_positions', None)
    if cfg_state_positions is not None:
        best_state_positions = cfg_state_positions
        best_regime_mode = 'score'
    elif getattr(cfg, 'search_state_positions', False):
        best_state_positions = [
            best_params[f'state_pos_{i}'] for i in range(best_hmm_components)
        ]
        best_regime_mode = 'score'
    else:
        best_state_positions = getattr(cfg, 'state_positions', None)
        best_regime_mode = getattr(cfg, 'regime_mode', 'strict')
    oos_common = {k: v for k, v in common.items()
                  if k not in ('hmm_features', 'unfav_fraction',
                               'state_positions', 'regime_mode')}
    hmm_oos = opt.backtest(
        fromdate            = OOS_FROM,
        todate              = OOS_TO,
        hmm                 = True,
        hmm_components      = best_hmm_components,
        hmm_score_threshold = best_score_threshold,
        hmm_threshold       = best_params['hmm_threshold'],
        hmm_gate            = best_params['hmm_gate'],
        hmm_train_years     = best_params['hmm_train_years'],
        hmm_find_best_rs    = True,
        hmm_features        = best_features,
        unfav_fraction      = best_unfav_fraction,
        regime_mode         = best_regime_mode,
        state_positions     = best_state_positions,
        **oos_common,
    )

    oos_delta = hmm_oos['total_return'] - bl_oos['total_return']
    verdict   = '✓ IMPROVEMENT' if oos_delta > 0 else '✗ DEGRADATION'
    print(f'  OOS HMM  return : {hmm_oos["total_return"]:+.2f}%  '
          f'cagr={hmm_oos["annual_return"]:+.2f}%  '
          f'sharpe={hmm_oos["sharpe"]:.3f}  '
          f'calmar={hmm_oos["calmar"]:.3f}  '
          f'maxdd={hmm_oos["max_drawdown"]:.2f}%  '
          f'trades={hmm_oos.get("trade_count", 0)}  '
          f'({hmm_oos["time_taken"]:.1f}s)')
    print(f'  OOS Δ return    : {oos_delta:+.2f}%  {verdict}')

    return dict(
        win_from        = IS_FROM,
        split           = IS_TO,
        win_to          = OOS_TO,
        # IS
        bl_is_return    = bl_is['total_return'],
        bl_is_sharpe    = bl_is['sharpe'],
        hmm_is_return   = best_is_return,
        is_delta        = best_is_return - bl_is['total_return'],
        # OOS – baseline
        bl_oos_return   = bl_oos['total_return'],
        bl_oos_annual   = bl_oos['annual_return'],
        bl_oos_sharpe   = bl_oos['sharpe'],
        bl_oos_dd       = bl_oos['max_drawdown'],
        bl_oos_calmar   = bl_oos['calmar'],
        bl_oos_time     = bl_oos['time_taken'],
        bl_oos_trades   = bl_oos.get('trade_count', 0),
        # OOS – HMM
        hmm_oos_return  = hmm_oos['total_return'],
        hmm_oos_annual  = hmm_oos['annual_return'],
        hmm_oos_sharpe  = hmm_oos['sharpe'],
        hmm_oos_dd      = hmm_oos['max_drawdown'],
        hmm_oos_calmar  = hmm_oos['calmar'],
        hmm_oos_time    = hmm_oos['time_taken'],
        hmm_oos_trades  = hmm_oos.get('trade_count', 0),
        # Deltas
        oos_delta       = oos_delta,
        improved        = oos_delta > 0,
        best_params     = best_params,
        best_score_threshold = best_score_threshold,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

# Metric definitions for the comparison table.
# Each entry: (key_suffix, label, fmt_spec, higher_is_better)
# key_suffix is used to build  'bl_oos_<key>'  and  'hmm_oos_<key>'.
_METRICS = [
    ('return', 'Total Return (%)', '+.2f', True),
    ('annual', 'CAGR (%)',         '+.2f', True),
    ('sharpe', 'Sharpe',           '.3f',  True),
    ('calmar', 'Calmar',           '.3f',  True),
    ('dd',     'Max DrawDown (%)', '.2f',  False),   # lower is better
    ('trades', 'Trades',           '.0f',  False),   # informational
    ('time',   'Time (s)',         '.1f',  False),   # lower is better (informational)
]


def _std(values: list[float]) -> float:
    """Population standard deviation (returns 0 for a single-element list)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5


def _ir(deltas: list[float]) -> str:
    """Information Ratio = mean(Δ) / std(Δ).  Returns 'n/a' when std is 0."""
    if len(deltas) < 2:
        return 'n/a'
    mean = sum(deltas) / len(deltas)
    sd   = _std(deltas)
    if sd == 0.0:
        return 'n/a'
    return f'{mean / sd:+.3f}'


def print_report(results: list[dict], cfg):
    """Print a comprehensive walk-forward summary with all metrics and IR."""
    import pandas as pd

    n = len(results)
    W = 130   # report width

    print('\n\n' + '=' * W)
    print('  WALK-FORWARD SUMMARY REPORT')
    print('=' * W)

    # ------------------------------------------------------------------
    # Per-window delta table
    # Columns: IS Start | OOS Start | ΔRet% | ΔCAGR% | ΔSharpe | ΔCalmar | ΔMaxDD% | Result
    # ------------------------------------------------------------------
    col_fmt = '{:<12}  {:<12}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>7}  {:>7}  {:>8}'
    hdr = col_fmt.format(
        'IS Start', 'OOS Start',
        'Δ Ret%', 'Δ CAGR%', 'Δ Sharpe', 'Δ Calmar', 'Δ MaxDD%',
        'BL Trd', 'HMM Trd', 'Result')
    print(f'\n{hdr}')
    print('-' * W)

    for r in results:
        verdict    = '✓ WIN' if r['improved'] else '✗ LOSS'
        d_return   = r['hmm_oos_return'] - r['bl_oos_return']
        d_annual   = r['hmm_oos_annual'] - r['bl_oos_annual']
        d_sharpe   = r['hmm_oos_sharpe'] - r['bl_oos_sharpe']
        d_calmar   = r['hmm_oos_calmar'] - r['bl_oos_calmar']
        d_dd       = r['hmm_oos_dd']     - r['bl_oos_dd']      # positive = worse DD
        bl_trades  = r.get('bl_oos_trades', 0)
        hmm_trades = r.get('hmm_oos_trades', 0)
        print(col_fmt.format(
            r['win_from'], r['split'],
            f'{d_return:+.2f}',
            f'{d_annual:+.2f}',
            f'{d_sharpe:+.3f}',
            f'{d_calmar:+.3f}',
            f'{d_dd:+.2f}',
            f'{bl_trades}',
            f'{hmm_trades}',
            verdict,
        ))

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    n_wins    = sum(1 for r in results if r['improved'])
    win_rate  = 100 * n_wins / n if n else 0

    ret_deltas = [r['hmm_oos_return'] - r['bl_oos_return'] for r in results]
    mean_delta = sum(ret_deltas) / n if n else 0
    best_window  = max(results, key=lambda r: r['oos_delta'])
    worst_window = min(results, key=lambda r: r['oos_delta'])

    print('\n' + '=' * W)
    print(f'  Total windows     : {n}')
    print(f'  HMM wins (OOS)    : {n_wins} / {n}  ({win_rate:.1f}%)  [based on total return]')
    print(f'  Mean Δ return     : {mean_delta:+.2f}%  per window')
    print(f'  Best  window      : {best_window["win_from"]} → {best_window["win_to"]}  '
          f'Δ={best_window["oos_delta"]:+.2f}%')
    print(f'  Worst window      : {worst_window["win_from"]} → {worst_window["win_to"]}  '
          f'Δ={worst_window["oos_delta"]:+.2f}%')

    # ------------------------------------------------------------------
    # Full metrics comparison + IR
    # Columns: Metric | Baseline | HMM | Δ | Improved? | IR
    # IR = mean(Δ) / std(Δ)  across all windows for each metric.
    # ------------------------------------------------------------------
    print(f'\n  OOS METRIC COMPARISON  (mean across {n} windows)')
    print(f'  {"Metric":<22}  {"Baseline":>12}  {"HMM":>12}  {"Δ (mean)":>10}  '
          f'{"Improved?":>10}  {"IR":>8}')
    print(f'  {"-"*22}  {"-"*12}  {"-"*12}  {"-"*10}  {"-"*10}  {"-"*8}')

    improvements = []
    for key, label, fmt, higher_better in _METRICS:
        bl_vals  = [r[f'bl_oos_{key}']  for r in results]
        hmm_vals = [r[f'hmm_oos_{key}'] for r in results]
        deltas   = [h - b for h, b in zip(hmm_vals, bl_vals)]
        mean_bl  = sum(bl_vals)  / n
        mean_hmm = sum(hmm_vals) / n
        mean_d   = sum(deltas)   / n

        if key in ('time', 'trades'):
            improved_flag = '(info)'
            improved      = None
            ir_str        = '(info)'
        elif higher_better:
            improved      = mean_d > 0
            improved_flag = '✓ YES' if improved else '✗ NO'
            ir_str        = _ir(deltas)
        else:   # lower is better (max drawdown)
            improved      = mean_d < 0
            improved_flag = '✓ YES' if improved else '✗ NO'
            # Flip sign so IR is positive when HMM reduces drawdown
            ir_str        = _ir([-d for d in deltas])

        if improved:
            improvements.append(label)

        bl_str  = format(mean_bl,  fmt)
        hmm_str = format(mean_hmm, fmt)
        d_str   = ('+' if mean_d >= 0 else '') + format(mean_d, fmt)
        print(f'  {label:<22}  {bl_str:>12}  {hmm_str:>12}  {d_str:>10}  '
              f'{improved_flag:>10}  {ir_str:>8}')

    print(f'\n  Metrics improved by HMM : {len(improvements)} / {len(_METRICS)-1}  '
          f'→  {" | ".join(improvements) if improvements else "none"}')
    print(f'\n  IR > 0.5 = consistent edge  |  IR > 1.0 = strong edge  '
          f'|  IR < 0 = HMM consistently hurts  |  n/a = single window')
    print('=' * W)

    # Best params frequency --------------------------------------------------
    print('\n  MOST COMMON BEST-HMM PARAMS ACROSS WINDOWS:')
    from collections import Counter
    fixed_hc = getattr(cfg, 'hmm_components', None)
    for key in ['hmm_components', 'hmm_gate']:
        fixed_val = {'hmm_components': fixed_hc}.get(key)
        if fixed_val is not None:
            print(f'    {key:<22}: {fixed_val} (fixed)')
            continue
        vals = [r['best_params'].get(key) for r in results if r['best_params']]
        cnt  = Counter(vals).most_common(3)
        print(f'    {key:<22}: ' + '  '.join(f'{v}×{c}' for v, c in cnt))
    thresholds = [r['best_params'].get('hmm_threshold') for r in results
                  if r['best_params']]
    if thresholds:
        print(f'    {"hmm_threshold":<22}: '
              f'mean={sum(thresholds)/len(thresholds):.3f}  '
              f'min={min(thresholds):.3f}  max={max(thresholds):.3f}')
    score_thresholds = [r.get('best_score_threshold') for r in results
                        if r.get('best_score_threshold') is not None]
    if score_thresholds:
        print(f'    {"hmm_score_threshold":<22}: '
              f'mean={sum(score_thresholds)/len(score_thresholds):.3f}  '
              f'min={min(score_thresholds):.3f}  max={max(score_thresholds):.3f}')

    # Save CSV ---------------------------------------------------------------
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != 'best_params'}
        bp  = r.get('best_params', {})
        for k, v in bp.items():
            row[f'param_{k}'] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path = os.path.join(_HERE, 'walkforward-results.csv')
    df.to_csv(out_path, index=False)
    print(f'\n  Full results saved → {out_path}')
    print('=' * W)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Walk-forward HMM optimisation using optimize-hmm pipeline')

    p.add_argument('--ticker', nargs='+', default=['SPY'],
        help='Portfolio tickers (space-separated)')
    p.add_argument('--wf-start',  default='2019-01-01', dest='wf_start',
        help='Walk-forward overall start date')
    p.add_argument('--wf-end',    default='2026-01-01', dest='wf_end',
        help='Walk-forward overall end date')
    p.add_argument('--is-years',  type=int, default=3, dest='is_years',
        help='In-sample window length in years')
    p.add_argument('--oos-years', type=int, default=2, dest='oos_years',
        help='Out-of-sample window length in years')
    p.add_argument('--step',      type=int, default=1,
        help='Step size in years between windows')
    p.add_argument('--n-trials',  type=int, default=40, dest='n_trials',
        help='Optuna trials per window')
    p.add_argument('--strategy', default='sma',
        choices=['sma', 'dema', 'rsi', 'macd', 'hmm_mr',
                 'adx_dm', 'channel_breakout', 'donchian', 'ichimoku',
                 'parabolic_sar', 'tsmom', 'turtle', 'vol_adj'],
        help='Strategy to use for all windows')
    p.add_argument('--fast',      type=int, default=10)
    p.add_argument('--slow',      type=int, default=30)
    p.add_argument('--rsi-period',     type=int, default=14,  dest='rsi_period')
    p.add_argument('--rsi-oversold',   type=int, default=30,  dest='rsi_oversold')
    p.add_argument('--rsi-overbought', type=int, default=70,  dest='rsi_overbought')
    p.add_argument('--macd-fast',   type=int, default=12, dest='macd_fast')
    p.add_argument('--macd-slow',   type=int, default=26, dest='macd_slow')
    p.add_argument('--macd-signal', type=int, default=9,  dest='macd_signal')
    p.add_argument('--hmm-mr-z-threshold', type=float, default=0.0, dest='hmm_mr_z_threshold',
        help='HMM-MR: std-devs below state mean required before entry (0=any dip)')
    p.add_argument('--adx-period',    type=int,   default=14,   dest='adx_period')
    p.add_argument('--adx-threshold', type=float, default=25.0, dest='adx_threshold')
    p.add_argument('--channel-period',  type=int, default=20, dest='channel_period')
    p.add_argument('--donchian-entry',  type=int, default=20, dest='donchian_entry')
    p.add_argument('--donchian-exit',   type=int, default=10, dest='donchian_exit')
    p.add_argument('--ichimoku-tenkan', type=int, default=9,  dest='ichimoku_tenkan')
    p.add_argument('--ichimoku-kijun',  type=int, default=26, dest='ichimoku_kijun')
    p.add_argument('--ichimoku-senkou', type=int, default=52, dest='ichimoku_senkou')
    p.add_argument('--psar-af',     type=float, default=0.02, dest='psar_af')
    p.add_argument('--psar-max-af', type=float, default=0.20, dest='psar_max_af')
    p.add_argument('--tsmom-lookback', type=int, default=252, dest='tsmom_lookback')
    p.add_argument('--tsmom-skip',     type=int, default=21,  dest='tsmom_skip')
    p.add_argument('--turtle-entry',    type=int,   default=20,  dest='turtle_entry')
    p.add_argument('--turtle-exit',     type=int,   default=10,  dest='turtle_exit')
    p.add_argument('--turtle-atr',      type=int,   default=20,  dest='turtle_atr')
    p.add_argument('--turtle-atr-mult', type=float, default=2.0, dest='turtle_atr_mult')
    p.add_argument('--vol-period',      type=int,   default=20,  dest='vol_period')
    p.add_argument('--vol-atr-period',  type=int,   default=14,  dest='vol_atr_period')
    p.add_argument('--vol-atr-mult',    type=float, default=1.5, dest='vol_atr_mult')
    p.add_argument('--hmm-features', nargs='+', default=None, dest='hmm_features',
        metavar='FEAT',
        help='HMM input features (e.g. log_ret vol_short vol_long atr_norm)')
    p.add_argument('--hmm-pca', type=int, default=None, dest='hmm_pca',
        metavar='N',
        help='Apply PCA after scaling, reducing features to N components (omit to skip)')
    p.add_argument('--regime-mode', default='strict', dest='regime_mode',
        choices=['strict', 'size', 'score', 'linear'],
        help='strict = block trades in unfav regime; size = reduce position; score = score-weighted')
    p.add_argument('--unfav-fraction', type=float, default=None, dest='unfav_fraction',
        help='Fraction of stake in unfavourable regime (regime_mode=size); omit to let Optuna search')
    p.add_argument('--stake',     type=int, default=100)
    p.add_argument('--cash',      type=float, default=100_000.0)
    p.add_argument('--commission',type=float, default=0.001)
    p.add_argument('--seed',      type=int,   default=42)
    p.add_argument('--stop-loss', type=float, default=0.02,
        dest='stop_loss_perc', metavar='FRAC',
        help='Stop-loss  as fraction of entry price (e.g. 0.02 = 2%%; 0 = disabled)')
    p.add_argument('--take-profit', type=float, default=0.10,
        dest='take_profit_perc', metavar='FRAC',
        help='Take-profit as fraction of entry price (e.g. 0.10 = 10%%; 0 = disabled)')
    p.add_argument('--hmm-score-threshold', type=float, default=None,
        dest='hmm_score_threshold',
        metavar='SCORE',
        help='Fix hmm_score_threshold instead of letting Optuna search it '
             '(range [0, 3] with default weights); omit to include in search')
    p.add_argument('--hmm-components', type=int, default=None,
        dest='hmm_components',
        metavar='K',
        help='Fix number of HMM hidden states instead of letting Optuna search '
             '(range 3-6); omit to include in search')
    p.add_argument('--objective-metric', default='total_return',
        dest='objective_metric',
        choices=['total_return', 'sharpe', 'calmar'],
        help='Metric that Optuna maximises during IS optimisation')
    p.add_argument('--max-workers', type=int, default=0, dest='max_workers',
        help='Max parallel window workers (0 = auto = number of windows, '
             '1 = sequential)')
    p.add_argument('--window-log-dir', default=None, dest='window_log_dir',
        help='Directory for per-window progress log files '
             '(default: auto-created alongside output)')
    p.add_argument('--state-positions', type=float, nargs='+', default=None,
        dest='state_positions',
        metavar='POS',
        help='Fixed position sizes per HMM state (score-ranked, best\u2192worst). '
             'Number of values should match hmm_components (K)')
    p.add_argument('--search-state-positions', action='store_true', default=False,
        dest='search_state_positions',
        help='Let Optuna search for optimal per-state position sizes [0,1]. '
             'Forces regime_mode=score. Ignored if --state-positions is provided.')
    p.add_argument('--hmm-favourable', type=int, default=None, dest='hmm_favourable',
        help='Only trade in the top-N best HMM states; others get pos_size=0.')
    p.add_argument('--hmm-max-pos-size', type=float, default=2.0, dest='hmm_max_pos_size',
        help='Position size multiplier for top states (default 2.0)')
    p.add_argument('--hmm-min-pos-size', type=float, default=0.0, dest='hmm_min_pos_size',
        help='Position size multiplier for worst states (default 0.0 = blocked)')
    p.add_argument('--hmm-dynamic-scoring', action='store_true', default=False,
        dest='hmm_dynamic_scoring',
        help='Re-score HMM states bar-by-bar using expanding window')
    p.add_argument('--hmm-dynamic-window', type=int, default=0, dest='hmm_dynamic_window',
        help='Window size for dynamic scoring (0 = expanding)')

    return p.parse_args()


def _run_window_worker(i, n_total, window, tickers, cfg, log_path=None):
    """Run a single window in a worker process, writing output to a log file."""
    # Pin BLAS threads to 1 to avoid oversubscription in parallel workers
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(var, '1')

    result = None
    if log_path:
        fh = open(log_path, 'w')
    else:
        fh = io.StringIO()

    try:
        with redirect_stdout(fh):
            print(f'\n[Window {i}/{n_total}]')
            try:
                result = run_window(window, tickers, cfg)
            except Exception as exc:
                print(f'  [WARN] Window failed: {exc}')
    finally:
        if log_path:
            fh.close()

    output = ''
    if log_path:
        with open(log_path) as f:
            output = f.read()
    else:
        output = fh.getvalue()

    return i, output, result


def main():
    cfg = parse_args()

    try:
        import optuna
        import dateutil  # noqa – ensure installed
    except ImportError as e:
        sys.exit(f'[ERROR] Missing dependency: {e}\n'
                 f'Install with:  pip install optuna python-dateutil')

    tickers = cfg.ticker
    windows = generate_windows(cfg.wf_start, cfg.wf_end,
                               cfg.is_years, cfg.oos_years, cfg.step)

    if not windows:
        sys.exit('[ERROR] No walk-forward windows fit within the date range. '
                 'Reduce --is-years / --oos-years or widen the date range.')

    n_workers = cfg.max_workers if cfg.max_workers > 0 else len(windows)

    portfolio_label = ', '.join(tickers)
    print('=' * 70)
    print(f'  Walk-Forward HMM Optimisation  –  {portfolio_label}')
    print(f'  Overall   : {cfg.wf_start}  →  {cfg.wf_end}')
    print(f'  IS window : {cfg.is_years} yr   OOS window : {cfg.oos_years} yr   '
          f'Step : {cfg.step} yr')
    print(f'  Windows   : {len(windows)}   Trials/window : {cfg.n_trials}'
          f'   Workers : {n_workers}')
    print('=' * 70)

    t_start = time.perf_counter()

    if n_workers == 1:
        # Sequential mode – direct stdout, no buffering
        results = []
        for i, window in enumerate(windows, 1):
            print(f'\n[Window {i}/{len(windows)}]')
            try:
                res = run_window(window, tickers, cfg)
                results.append(res)
            except Exception as exc:
                print(f'  [WARN] Window failed: {exc}')
    else:
        # Parallel mode – run windows concurrently, write per-window log files
        strategy = getattr(cfg, 'strategy', 'sma')
        log_dir = cfg.window_log_dir or os.path.join(_HERE, 'wf_window_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_paths = {
            idx: os.path.join(log_dir, f'{strategy}_window_{idx+1}.txt')
            for idx in range(len(windows))
        }

        print(f'\n  Launching {n_workers} parallel window workers …')
        print(f'  Per-window logs → {os.path.abspath(log_dir)}/')
        for idx, lp in sorted(log_paths.items()):
            print(f'    Window {idx+1}: {os.path.basename(lp)}')
        print()

        ordered_results = [None] * len(windows)
        ordered_outputs = [''] * len(windows)

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for idx, window in enumerate(windows):
                fut = pool.submit(_run_window_worker,
                                  idx + 1, len(windows), window, tickers, cfg,
                                  log_path=log_paths[idx])
                futures[fut] = idx

            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    win_idx, output, result = fut.result()
                    ordered_outputs[idx] = output
                    ordered_results[idx] = result
                    status = '✓' if result is not None else '✗ FAILED'
                    print(f'  [Window {win_idx}/{len(windows)}] {status}')
                except Exception as exc:
                    print(f'  [Window {idx + 1}/{len(windows)}] ✗ ERROR: {exc}')

        # Print buffered output in window order
        for output in ordered_outputs:
            if output:
                print(output, end='')

        results = [r for r in ordered_results if r is not None]

    total_elapsed = time.perf_counter() - t_start
    print(f'\n  Total walk-forward time: {total_elapsed:.1f}s')

    if results:
        print_report(results, cfg)
    else:
        print('[ERROR] No windows completed successfully.')


if __name__ == '__main__':
    main()
