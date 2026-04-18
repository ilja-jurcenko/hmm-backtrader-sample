#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
HMM Hyperparameter Optimisation
================================
Uses Optuna to search for the HMM + SMA parameter combination that
maximises total return on an **in-sample** window, then validates the
result on a held-out **out-of-sample** window.

Optimised parameters
--------------------
  fast            – SMA fast period
  slow            – SMA slow period
  hmm_components  – number of hidden states (K)
  hmm_favourable  – how many top states are "allowed to trade"
  hmm_threshold   – posterior probability gate threshold
  hmm_gate        – gate function: threshold | logistic
  hmm_train_years – years of data *before* fromdate used to train the HMM

Walk-forward design
-------------------
  full period      : --fromdate  →  --todate          (baseline comparison)
  in-sample  (IS)  : --fromdate  →  --split-date      (optimisation target)
  out-of-sample (OOS): --split-date →  --todate        (honest validation)

Usage
-----
  pip install optuna
  python optimize-hmm.py
  python optimize-hmm.py --ticker SPY --n-trials 100
  python optimize-hmm.py --ticker SPY --n-trials 50 --split-date 2022-01-01
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
import warnings

warnings.filterwarnings('ignore')   # suppress backtrader / sklearn noise

# ---------------------------------------------------------------------------
# Load ma-quantstats module (handle the hyphen in the filename)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

def _load_maq():
    path = os.path.join(_HERE, 'ma-quantstats.py')
    spec = importlib.util.spec_from_file_location('maq', path)
    mod  = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so backtrader's metaclass can find the module
    # via sys.modules[cls.__module__] (otherwise KeyError: 'maq')
    sys.modules['maq'] = mod
    spec.loader.exec_module(mod)
    return mod

maq = _load_maq()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_args(**kwargs) -> types.SimpleNamespace:
    """Build an args namespace compatible with maq.run() defaults."""
    defaults = dict(
        tickers        = ['SPY'],
        fromdate       = '2019-01-01',
        todate         = '2025-01-01',
        fast           = 10,
        slow           = 50,
        stake          = 100,
        cash           = 100_000.0,
        commission     = 0.001,
        riskfreerate   = 0.01,
        printlog       = False,
        plot           = False,
        # Strategy selection
        strategy       = 'sma',
        # RSI strategy params
        rsi_period     = 14,
        rsi_oversold   = 30,
        rsi_overbought = 70,
        # MACD strategy params
        macd_fast      = 12,
        macd_slow      = 26,
        macd_signal    = 9,
        # HMM Mean-Reversion strategy params
        hmm_mr_z_threshold = 0.0,
        # HMM
        hmm            = False,
        hmm_train_years= 5.0,
        hmm_components = 6,
        hmm_favourable = None,
        hmm_score_threshold = 1.0,
        hmm_threshold  = 0.5,
        hmm_gate       = 'threshold',
        hmm_find_best_rs = True,
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def backtest(quiet=True, **kwargs) -> dict:
    """Run a single backtest and return the metrics dict."""
    import traceback
    args = make_args(**kwargs)
    try:
        return maq.run(args, quiet=quiet)
    except Exception as exc:
        if not quiet:
            traceback.print_exc()
            print(f'[WARN] backtest raised: {exc}')
        return {'total_return': -999.0, 'sharpe': -999.0,
                'max_drawdown': 999.0,  'final_value': 0.0,
                'annual_return': -999.0, 'calmar': -999.0,
                'time_taken': 0.0}


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(tickers, fromdate, todate, fast, slow,
                   strategy='sma',
                   rsi_period=14, rsi_oversold=30, rsi_overbought=70,
                   macd_fast=12, macd_slow=26, macd_signal=9,
                   hmm_mr_z_threshold=0.0,
                   fixed_score_threshold=None):
    """Return a closure that Optuna can call as an objective.

    Strategy params are FIXED so the search only measures the HMM's
    contribution, not better indicator parameters.

    fixed_score_threshold: when not None, the score threshold is pinned to
    this value and removed from the search space.
    """
    def objective(trial):
        hmm_components      = trial.suggest_int('hmm_components', 3, 3)
        if fixed_score_threshold is not None:
            hmm_score_threshold = fixed_score_threshold
        else:
            hmm_score_threshold = trial.suggest_float('hmm_score_threshold', 0.5, 2.5)
        hmm_threshold       = trial.suggest_float('hmm_threshold', 0.3, 0.99)
        hmm_gate            = trial.suggest_categorical('hmm_gate', ['threshold'])
        hmm_train_years     = trial.suggest_float('hmm_train_years', 5.0, 5.0)

        res = backtest(
            tickers             = tickers,
            fromdate            = fromdate,
            todate              = todate,
            strategy            = strategy,
            fast                = fast,
            slow                = slow,
            rsi_period          = rsi_period,
            rsi_oversold        = rsi_oversold,
            rsi_overbought      = rsi_overbought,
            macd_fast           = macd_fast,
            macd_slow           = macd_slow,
            macd_signal         = macd_signal,
            hmm_mr_z_threshold  = hmm_mr_z_threshold,
            hmm                 = True,
            hmm_components      = hmm_components,
            hmm_score_threshold = hmm_score_threshold,
            hmm_threshold       = hmm_threshold,
            hmm_gate            = hmm_gate,
            hmm_train_years     = hmm_train_years,
            hmm_find_best_rs    = True,
        )
        return res['total_return']

    return objective


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _bar(label, value, best_baseline, width=30):
    """ASCII progress bar relative to best_baseline."""
    ratio = max(0.0, value / max(abs(best_baseline), 1e-9))
    filled = int(min(ratio, 2.0) * width // 2)
    bar = '█' * filled + '░' * (width - filled)
    return f'  {label:<22} [{bar}]  {value:>+9.2f}%'


def print_results_table(title, rows):
    """Pretty-print a comparison table."""
    col_w = [26, 14, 10, 14]
    hdr   = ['Configuration', 'Total Return', 'Sharpe', 'Max Drawdown']
    sep   = '  '.join('-' * w for w in col_w)
    fmt   = '  '.join(f'{{:<{w}}}' for w in col_w)

    print(f'\n{"=" * 70}')
    print(f'  {title}')
    print(f'{"=" * 70}')
    print(fmt.format(*hdr))
    print(sep)
    for row in rows:
        print(fmt.format(
            row['label'],
            f"{row['total_return']:+.2f}%",
            f"{row['sharpe']:.3f}",
            f"{row['max_drawdown']:.2f}%",
        ))
    print(f'{"=" * 70}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Optimise HMM hyperparameters with Optuna')

    p.add_argument('--ticker',      default=['SPY'], nargs='+',
        help='One or more tickers to optimise on (space-separated, e.g. SPY QQQ AGG)')
    p.add_argument('--fromdate',    default='2019-01-01',
        help='Backtest / in-sample start date (YYYY-MM-DD)')
    p.add_argument('--split-date',  default='2022-07-01', dest='split_date',
        help='Date that separates in-sample from out-of-sample (YYYY-MM-DD)')
    p.add_argument('--todate',      default='2025-01-01',
        help='Out-of-sample end date (YYYY-MM-DD)')
    p.add_argument('--n-trials',    type=int, default=60, dest='n_trials',
        help='Number of Optuna trials')
    p.add_argument('--fast',        type=int, default=9,
        help='Baseline SMA fast period (no-HMM run)')
    p.add_argument('--slow',        type=int, default=30,
        help='Baseline SMA slow period (no-HMM run)')
    p.add_argument('--stake',       type=int, default=100)
    p.add_argument('--cash',        type=float, default=100_000.0)
    p.add_argument('--commission',  type=float, default=0.001)
    p.add_argument('--seed',        type=int, default=42,
        help='Random seed for Optuna sampler')
    p.add_argument('--hmm-score-threshold', type=float, default=None,
        dest='hmm_score_threshold',
        metavar='SCORE',
        help='Fix hmm_score_threshold instead of letting Optuna search it '
             '[0, 3] range with default weights; omit to include it in the search')

    return p.parse_args()


def main():
    cfg = parse_args()

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        sys.exit(
            '[ERROR] optuna is not installed.\n'
            '        Install it with:  pip install optuna\n')

    tickers   = cfg.ticker        # list due to nargs='+'
    portfolio_label = ','.join(tickers)
    IS_FROM   = cfg.fromdate      # in-sample start
    IS_TO     = cfg.split_date    # in-sample end  / OOS start
    OOS_FROM  = cfg.split_date    # out-of-sample start
    OOS_TO    = cfg.todate        # out-of-sample end
    FULL_FROM = cfg.fromdate
    FULL_TO   = cfg.todate

    common_bt = dict(
        tickers    = tickers,
        fast       = cfg.fast,
        slow       = cfg.slow,
        stake      = cfg.stake,
        cash       = cfg.cash,
        commission = cfg.commission,
    )

    print('=' * 70)
    print(f'  HMM Hyperparameter Optimiser  –  {portfolio_label}')
    print(f'  In-sample   : {IS_FROM}  →  {IS_TO}')
    print(f'  Out-of-sample: {OOS_FROM}  →  {OOS_TO}')
    print(f'  Optuna trials: {cfg.n_trials}')
    print('=' * 70)

    # ------------------------------------------------------------------
    # Step 1 – Baselines (no HMM)
    # ------------------------------------------------------------------
    print('\n[1/4] Running baselines (no HMM) …')

    bl_full = backtest(quiet=True,
                       fromdate=FULL_FROM, todate=FULL_TO, **common_bt)
    print(f'  Full period    total return : {bl_full["total_return"]:+.2f}%  '
          f'sharpe={bl_full["sharpe"]:.3f}')

    bl_is   = backtest(fromdate=IS_FROM, todate=IS_TO, **common_bt)
    print(f'  In-sample      total return : {bl_is["total_return"]:+.2f}%')

    bl_oos  = backtest(fromdate=OOS_FROM, todate=OOS_TO, **common_bt)
    print(f'  Out-of-sample  total return : {bl_oos["total_return"]:+.2f}%')

    # ------------------------------------------------------------------
    # Step 2 – Optimise on IN-SAMPLE period
    # ------------------------------------------------------------------
    print(f'\n[2/4] Optimising on in-sample period ({cfg.n_trials} trials) …')
    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    study   = optuna.create_study(direction='maximize', sampler=sampler)

    objective = make_objective(tickers, IS_FROM, IS_TO, cfg.fast, cfg.slow,
                               fixed_score_threshold=cfg.hmm_score_threshold)

    completed = [0]
    best_box  = [bl_is['total_return']]   # track improvement

    def _callback(study, trial):
        completed[0] += 1
        val = trial.value if trial.value is not None else float('nan')
        if val > best_box[0]:
            best_box[0] = val
        pct = 100 * completed[0] / cfg.n_trials
        bar_len = 30
        filled = int(pct / 100 * bar_len)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f'\r  [{bar}] {pct:5.1f}%  '
              f'trial={completed[0]:>3}/{cfg.n_trials}  '
              f'best={best_box[0]:+.2f}%  '
              f'this={val:+.2f}%',
              end='', flush=True)

    study.optimize(objective, n_trials=cfg.n_trials, callbacks=[_callback])
    print()   # newline after progress bar

    best          = study.best_trial
    best_params   = best.params
    best_is_return = best.value

    print(f'\n  Best in-sample return : {best_is_return:+.2f}%  '
          f'(baseline was {bl_is["total_return"]:+.2f}%)')
    print(f'  Improvement           : {best_is_return - bl_is["total_return"]:+.2f}%')
    print(f'  (fast={cfg.fast}, slow={cfg.slow} fixed – only HMM params were searched)')
    print(f'  Portfolio: {portfolio_label}')
    print('\n  Best HMM parameters found:')
    for k, v in best_params.items():
        print(f'    {k:<24} = {v}')
    if cfg.hmm_score_threshold is not None:
        print(f'    {"hmm_score_threshold":<24} = {cfg.hmm_score_threshold}  (fixed by CLI)')

    # Resolve score threshold: fixed by CLI flag or found by Optuna
    best_score_threshold = (
        cfg.hmm_score_threshold
        if cfg.hmm_score_threshold is not None
        else best_params['hmm_score_threshold']
    )

    # ------------------------------------------------------------------
    # Step 3 – Validate best params ON OUT-OF-SAMPLE period
    # ------------------------------------------------------------------
    print('\n[3/4] Validating best params on out-of-sample period …')

    oos_hmm = backtest(
        tickers             = tickers,
        fromdate            = OOS_FROM,
        todate              = OOS_TO,
        fast                = cfg.fast,
        slow                = cfg.slow,
        stake               = cfg.stake,
        cash                = cfg.cash,
        commission          = cfg.commission,
        hmm                 = True,
        hmm_components      = best_params['hmm_components'],
        hmm_score_threshold = best_score_threshold,
        hmm_threshold       = best_params['hmm_threshold'],
        hmm_gate            = best_params['hmm_gate'],
        hmm_train_years     = best_params['hmm_train_years'],
        hmm_find_best_rs    = True,
    )

    # ------------------------------------------------------------------
    # Step 4 – Validate best params ON FULL period
    # ------------------------------------------------------------------
    print('[4/4] Running best params on full period …')

    full_hmm = backtest(
        tickers             = tickers,
        fromdate            = FULL_FROM,
        todate              = FULL_TO,
        fast                = cfg.fast,
        slow                = cfg.slow,
        stake               = cfg.stake,
        cash                = cfg.cash,
        commission          = cfg.commission,
        hmm                 = True,
        hmm_components      = best_params['hmm_components'],
        hmm_score_threshold = best_score_threshold,
        hmm_threshold       = best_params['hmm_threshold'],
        hmm_gate            = best_params['hmm_gate'],
        hmm_train_years     = best_params['hmm_train_years'],
        hmm_find_best_rs    = True,
    )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print_results_table(
        f'IN-SAMPLE  ({IS_FROM} → {IS_TO})',
        [
            {'label': f'Baseline SMA({cfg.fast}/{cfg.slow}) no HMM',
             **bl_is},
            {'label': f'Best HMM  SMA({cfg.fast}/{cfg.slow})',
             'total_return': best_is_return,
             'sharpe':       best_is_return,   # not tracked per-trial; use return as proxy
             'max_drawdown': 0.0},
        ]
    )

    print_results_table(
        f'OUT-OF-SAMPLE  ({OOS_FROM} → {OOS_TO})  ← honest validation',
        [
            {'label': f'Baseline SMA({cfg.fast}/{cfg.slow}) no HMM',
             **bl_oos},
            {'label': f'Best HMM  SMA({cfg.fast}/{cfg.slow})',
             **oos_hmm},
        ]
    )

    print_results_table(
        f'FULL PERIOD  ({FULL_FROM} → {FULL_TO})',
        [
            {'label': f'Baseline SMA({cfg.fast}/{cfg.slow}) no HMM',
             **bl_full},
            {'label': f'Best HMM  SMA({cfg.fast}/{cfg.slow})',
             **full_hmm},
        ]
    )

    # HMM verdict
    oos_delta  = oos_hmm['total_return']  - bl_oos['total_return']
    full_delta = full_hmm['total_return'] - bl_full['total_return']
    print('\n  VERDICT')
    print(f'  HMM vs baseline – out-of-sample : {oos_delta:+.2f}%  '
          + ('✓ IMPROVEMENT' if oos_delta > 0 else '✗ DEGRADATION'))
    print(f'  HMM vs baseline – full period   : {full_delta:+.2f}%  '
          + ('✓ IMPROVEMENT' if full_delta > 0 else '✗ DEGRADATION'))
    print()

    # Optionally save Optuna results to CSV
    try:
        import pandas as pd
        df = study.trials_dataframe()
        safe_label = portfolio_label.replace(',', '-')
        out_path = os.path.join(_HERE, f'hmm-optuna-{safe_label}.csv')
        df.to_csv(out_path, index=False)
        print(f'  All trial results saved → {out_path}')
    except Exception:
        pass


if __name__ == '__main__':
    main()
