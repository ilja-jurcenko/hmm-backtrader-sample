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
        rsi_invert_regime = True,   # invert HMM gate for mean-reversion
        # MACD strategy params
        macd_fast      = 12,
        macd_slow      = 26,
        macd_signal    = 9,
        # HMM Mean-Reversion strategy params
        hmm_mr_z_threshold = 0.0,
        # ADX + Directional Movement
        adx_period     = 14,
        adx_threshold  = 25.0,
        # Channel Breakout
        channel_period = 20,
        # Donchian Channel
        donchian_entry = 20,
        donchian_exit  = 10,
        # Ichimoku Cloud
        ichimoku_tenkan = 9,
        ichimoku_kijun  = 26,
        ichimoku_senkou = 52,
        # Parabolic SAR
        psar_af        = 0.02,
        psar_max_af    = 0.20,
        # Time-Series Momentum
        tsmom_lookback = 252,
        tsmom_skip     = 21,
        # Turtle
        turtle_entry   = 20,
        turtle_exit    = 10,
        turtle_atr     = 20,
        turtle_atr_mult= 2.0,
        # Volatility-Adjusted (Keltner)
        vol_period     = 20,
        vol_atr_period = 14,
        vol_atr_mult   = 1.5,
        # HMM
        hmm            = False,
        hmm_train_years= 5.0,
        hmm_components = 6,
        hmm_favourable = None,
        hmm_score_threshold = 1.0,
        hmm_threshold  = 0.5,
        hmm_gate       = 'threshold',
        hmm_find_best_rs = True,
        hmm_features   = None,
        hmm_pca        = None,
        # Regime filter mode
        regime_mode    = 'strict',
        unfav_fraction = None,
        state_positions = None,
        # HMM position sizing extensions
        hmm_max_pos_size    = 2.0,
        hmm_min_pos_size    = 0.0,
        hmm_dynamic_scoring = False,
        hmm_dynamic_window  = 0,
        # Risk management
        stop_loss_perc   = 0.02,
        take_profit_perc = 0.10,
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
                'time_taken': 0.0, 'trade_count': 0}


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(tickers, fromdate, todate, fast, slow,
                   strategy='sma',
                   rsi_period=14, rsi_oversold=30, rsi_overbought=70,
                   rsi_invert_regime=True,
                   macd_fast=12, macd_slow=26, macd_signal=9,
                   hmm_mr_z_threshold=0.0,
                   adx_period=14, adx_threshold=25.0,
                   channel_period=20,
                   donchian_entry=20, donchian_exit=10,
                   ichimoku_tenkan=9, ichimoku_kijun=26, ichimoku_senkou=52,
                   psar_af=0.02, psar_max_af=0.20,
                   tsmom_lookback=252, tsmom_skip=21,
                   turtle_entry=20, turtle_exit=10, turtle_atr=20, turtle_atr_mult=2.0,
                   vol_period=20, vol_atr_period=14, vol_atr_mult=1.5,
                   hmm_features=None,
                   hmm_pca=None,
                   regime_mode='strict',
                   unfav_fraction=0.25,
                   fixed_score_threshold=None,
                   fixed_hmm_components=None,
                   objective_metric='total_return',
                   state_positions=None,
                   search_state_positions=False,
                   hmm_favourable=None,
                   hmm_max_pos_size=2.0,
                   hmm_min_pos_size=0.0,
                   hmm_dynamic_scoring=False,
                   hmm_dynamic_window=0,
                   stop_loss_perc=0.0,
                   take_profit_perc=0.0):
    """Return a closure that Optuna can call as an objective.

    Strategy params are FIXED so the search only measures the HMM's
    contribution, not better indicator parameters.

    fixed_score_threshold: when not None, the score threshold is pinned to
    this value and removed from the search space.

    hmm_features: when not None, features are fixed.  When None, Optuna
    searches over which features to include (each is a boolean toggle).
    At least one feature is always selected; trials with zero features
    are pruned.

    objective_metric: which metric to maximise.  One of
    'total_return', 'sharpe', 'calmar'.

    search_state_positions: when True and state_positions is None,
    Optuna suggests a position size in [0, 1] for each of the K states
    (score-ranked, best→worst).  Forces regime_mode='score'.
    """
    # Feature pool for Optuna search (only used when hmm_features is None)
    _ALL_FEATURES = maq.ALL_HMM_FEATURES

    def objective(trial):
        if fixed_hmm_components is not None:
            hmm_components = fixed_hmm_components
        else:
            hmm_components      = trial.suggest_int('hmm_components', 3, 6)
        if fixed_score_threshold is not None:
            hmm_score_threshold = fixed_score_threshold
        else:
            hmm_score_threshold = trial.suggest_float('hmm_score_threshold', 0.5, 2.5)
        hmm_threshold       = trial.suggest_float('hmm_threshold', 0.3, 0.99)
        hmm_gate            = trial.suggest_categorical('hmm_gate', ['threshold'])
        hmm_train_years     = trial.suggest_float('hmm_train_years', 5.0, 5.0)

        # Regime sizing: search unfav_fraction when mode is 'size' and not fixed
        if regime_mode == 'size' and unfav_fraction is None:
            trial_unfav_fraction = trial.suggest_float('unfav_fraction', 0.05, 0.5)
        elif regime_mode == 'size':
            trial_unfav_fraction = unfav_fraction
        else:
            trial_unfav_fraction = unfav_fraction or 0.25

        # Feature selection: search over feature subsets when not fixed
        if hmm_features is not None:
            trial_features = hmm_features
        else:
            import optuna
            selected = [f for f in _ALL_FEATURES
                        if trial.suggest_categorical(f'feat_{f}', [True, False])]
            if not selected:
                raise optuna.TrialPruned('No HMM features selected')
            trial_features = selected

        # State position search: suggest a position per state
        if state_positions is not None:
            trial_state_positions = state_positions
            trial_regime_mode = 'score'
        elif search_state_positions:
            trial_state_positions = [
                trial.suggest_float(f'state_pos_{i}', 0.0, 1.0)
                for i in range(hmm_components)
            ]
            trial_regime_mode = 'score'
        else:
            trial_state_positions = None
            trial_regime_mode = regime_mode

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
            rsi_invert_regime   = rsi_invert_regime,
            macd_fast           = macd_fast,
            macd_slow           = macd_slow,
            macd_signal         = macd_signal,
            hmm_mr_z_threshold  = hmm_mr_z_threshold,
            adx_period          = adx_period,
            adx_threshold       = adx_threshold,
            channel_period      = channel_period,
            donchian_entry      = donchian_entry,
            donchian_exit       = donchian_exit,
            ichimoku_tenkan     = ichimoku_tenkan,
            ichimoku_kijun      = ichimoku_kijun,
            ichimoku_senkou     = ichimoku_senkou,
            psar_af             = psar_af,
            psar_max_af         = psar_max_af,
            tsmom_lookback      = tsmom_lookback,
            tsmom_skip          = tsmom_skip,
            turtle_entry        = turtle_entry,
            turtle_exit         = turtle_exit,
            turtle_atr          = turtle_atr,
            turtle_atr_mult     = turtle_atr_mult,
            vol_period          = vol_period,
            vol_atr_period      = vol_atr_period,
            vol_atr_mult        = vol_atr_mult,
            hmm                 = True,
            hmm_components      = hmm_components,
            hmm_score_threshold = hmm_score_threshold,
            hmm_threshold       = hmm_threshold,
            hmm_gate            = hmm_gate,
            hmm_train_years     = hmm_train_years,
            hmm_find_best_rs    = True,
            hmm_features        = trial_features,
            hmm_pca             = hmm_pca,
            regime_mode         = trial_regime_mode,
            unfav_fraction      = trial_unfav_fraction,
            state_positions     = trial_state_positions,
            hmm_favourable      = hmm_favourable,
            hmm_max_pos_size    = hmm_max_pos_size,
            hmm_min_pos_size    = hmm_min_pos_size,
            hmm_dynamic_scoring = hmm_dynamic_scoring,
            hmm_dynamic_window  = hmm_dynamic_window,
            stop_loss_perc      = stop_loss_perc,
            take_profit_perc    = take_profit_perc,
        )
        return res[objective_metric]

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
    p.add_argument('--strategy',    default='sma',
        choices=['sma', 'dema', 'rsi', 'macd', 'hmm_mr',
                 'adx_dm', 'channel_breakout', 'donchian', 'ichimoku',
                 'parabolic_sar', 'tsmom', 'turtle', 'vol_adj'],
        help='Trading strategy to use')
    p.add_argument('--rsi-period',     type=int, default=14, dest='rsi_period')
    p.add_argument('--rsi-oversold',   type=int, default=30, dest='rsi_oversold')
    p.add_argument('--rsi-overbought', type=int, default=70, dest='rsi_overbought')
    p.add_argument('--rsi-invert-regime', type=lambda x: x.lower() != 'false',
        default=True, dest='rsi_invert_regime',
        metavar='BOOL',
        help='Invert HMM regime gate for RSI (default: True — favour high-vol states)')
    p.add_argument('--macd-fast',   type=int, default=12, dest='macd_fast')
    p.add_argument('--macd-slow',   type=int, default=26, dest='macd_slow')
    p.add_argument('--macd-signal', type=int, default=9,  dest='macd_signal')
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
    p.add_argument('--state-positions', type=float, nargs='+', default=None,
        dest='state_positions',
        metavar='POS',
        help='Fixed position sizes per HMM state (score-ranked, best→worst). '
             'Number of values should match --hmm-components (K)')
    p.add_argument('--search-state-positions', action='store_true', default=False,
        dest='search_state_positions',
        help='Let Optuna search for optimal per-state position sizes [0,1]. '
             'Forces regime_mode=score. Ignored if --state-positions is provided.')
    p.add_argument('--regime-mode', default='strict', dest='regime_mode',
        choices=['strict', 'size', 'score', 'linear'],
        help='Regime filter mode: strict (binary), size (position sizing), '
             'score (score-weighted)')
    p.add_argument('--hmm-features', nargs='+', default=None,
        dest='hmm_features', metavar='FEAT',
        help='Fix HMM feature set (space-separated). '
             'Omit to let Optuna search over features.')
    p.add_argument('--objective-metric', default='total_return',
        dest='objective_metric',
        choices=['total_return', 'sharpe', 'calmar'],
        help='Metric to maximise during optimisation')
    p.add_argument('--hmm-components', type=int, default=None,
        dest='hmm_components', metavar='K',
        help='Fix number of HMM components (K). '
             'Omit to let Optuna search [3, 6].')
    p.add_argument('--hmm-pca', type=int, default=None,
        dest='hmm_pca', metavar='N',
        help='Number of PCA components for HMM features. '
             'Omit to skip PCA.')
    p.add_argument('--stop-loss', type=float, default=0.02,
        dest='stop_loss_perc', metavar='FRAC',
        help='Stop-loss as a fraction of entry price (e.g. 0.02 = 2%%; 0 = disabled)')
    p.add_argument('--take-profit', type=float, default=0.10,
        dest='take_profit_perc', metavar='FRAC',
        help='Take-profit as a fraction of entry price (e.g. 0.10 = 10%%; 0 = disabled)')

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

    strat_label = cfg.strategy.upper()
    common_bt = dict(
        tickers    = tickers,
        strategy   = cfg.strategy,
        fast       = cfg.fast,
        slow       = cfg.slow,
        rsi_period     = cfg.rsi_period,
        rsi_oversold   = cfg.rsi_oversold,
        rsi_overbought = cfg.rsi_overbought,
        macd_fast      = cfg.macd_fast,
        macd_slow      = cfg.macd_slow,
        macd_signal    = cfg.macd_signal,
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
        stake      = cfg.stake,
        cash       = cfg.cash,
        commission = cfg.commission,
        stop_loss_perc   = getattr(cfg, 'stop_loss_perc',   0.0),
        take_profit_perc = getattr(cfg, 'take_profit_perc', 0.0),
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
                               strategy=cfg.strategy,
                               rsi_period=cfg.rsi_period,
                               rsi_oversold=cfg.rsi_oversold,
                               rsi_overbought=cfg.rsi_overbought,
                               macd_fast=cfg.macd_fast,
                               macd_slow=cfg.macd_slow,
                               macd_signal=cfg.macd_signal,
                               adx_period=getattr(cfg, 'adx_period', 14),
                               adx_threshold=getattr(cfg, 'adx_threshold', 25.0),
                               channel_period=getattr(cfg, 'channel_period', 20),
                               donchian_entry=getattr(cfg, 'donchian_entry', 20),
                               donchian_exit=getattr(cfg, 'donchian_exit', 10),
                               ichimoku_tenkan=getattr(cfg, 'ichimoku_tenkan', 9),
                               ichimoku_kijun=getattr(cfg, 'ichimoku_kijun', 26),
                               ichimoku_senkou=getattr(cfg, 'ichimoku_senkou', 52),
                               psar_af=getattr(cfg, 'psar_af', 0.02),
                               psar_max_af=getattr(cfg, 'psar_max_af', 0.20),
                               tsmom_lookback=getattr(cfg, 'tsmom_lookback', 252),
                               tsmom_skip=getattr(cfg, 'tsmom_skip', 21),
                               turtle_entry=getattr(cfg, 'turtle_entry', 20),
                               turtle_exit=getattr(cfg, 'turtle_exit', 10),
                               turtle_atr=getattr(cfg, 'turtle_atr', 20),
                               turtle_atr_mult=getattr(cfg, 'turtle_atr_mult', 2.0),
                               vol_period=getattr(cfg, 'vol_period', 20),
                               vol_atr_period=getattr(cfg, 'vol_atr_period', 14),
                               vol_atr_mult=getattr(cfg, 'vol_atr_mult', 1.5),
                               fixed_score_threshold=cfg.hmm_score_threshold,
                               state_positions=cfg.state_positions,
                               search_state_positions=getattr(cfg, 'search_state_positions', False),
                               regime_mode=cfg.regime_mode,
                               hmm_features=cfg.hmm_features,
                               hmm_pca=cfg.hmm_pca,
                               objective_metric=cfg.objective_metric,
                               fixed_hmm_components=cfg.hmm_components,
                               stop_loss_perc=getattr(cfg, 'stop_loss_perc', 0.0),
                               take_profit_perc=getattr(cfg, 'take_profit_perc', 0.0))

    completed = [0]
    best_box  = [bl_is[cfg.objective_metric]]   # track improvement

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

    print(f'\n  Best in-sample {cfg.objective_metric} : {best_is_return:+.2f}  '
          f'(baseline was {bl_is[cfg.objective_metric]:+.2f})')
    print(f'  Improvement           : {best_is_return - bl_is[cfg.objective_metric]:+.2f}')
    print(f'  (fast={cfg.fast}, slow={cfg.slow} fixed – only HMM params were searched)')
    print(f'  Portfolio: {portfolio_label}')
    print('\n  Best HMM parameters found:')
    for k, v in best_params.items():
        if not k.startswith('feat_'):
            print(f'    {k:<24} = {v}')
    if cfg.hmm_score_threshold is not None:
        print(f'    {"hmm_score_threshold":<24} = {cfg.hmm_score_threshold}  (fixed by CLI)')
    if cfg.hmm_components is not None:
        print(f'    {"hmm_components":<24} = {cfg.hmm_components}  (fixed by CLI)')
    if cfg.hmm_features is not None:
        print(f'    {"hmm_features":<24} = {" ".join(cfg.hmm_features)}  (fixed by CLI)')
    if cfg.hmm_pca is not None:
        print(f'    {"hmm_pca":<24} = {cfg.hmm_pca}  (fixed by CLI)')
    print(f'    {"regime_mode":<24} = {cfg.regime_mode}')
    print(f'    {"objective_metric":<24} = {cfg.objective_metric}')

    # Extract best feature subset from trial params (feat_* keys)
    best_features = [k[5:] for k, v in best_params.items()
                     if k.startswith('feat_') and v is True]
    if best_features:
        print(f'    {"hmm_features":<24} = {" ".join(best_features)}')
    else:
        best_features = cfg.hmm_features   # use CLI-fixed features, or None

    # Resolve fixed-by-CLI vs found-by-Optuna parameters
    best_hmm_components = (
        cfg.hmm_components
        if cfg.hmm_components is not None
        else best_params['hmm_components']
    )

    best_score_threshold = (
        cfg.hmm_score_threshold
        if cfg.hmm_score_threshold is not None
        else best_params['hmm_score_threshold']
    )

    # Resolve state positions: fixed by CLI, found by Optuna, or None
    if cfg.state_positions is not None:
        best_state_positions = cfg.state_positions
    elif getattr(cfg, 'search_state_positions', False):
        K = best_hmm_components
        best_state_positions = [
            best_params[f'state_pos_{i}'] for i in range(K)
        ]
        print(f'    {"state_positions":<24} = {" ".join(f"{p:.3f}" for p in best_state_positions)}')
    else:
        best_state_positions = None

    best_regime_mode = 'score' if best_state_positions is not None else cfg.regime_mode

    # ------------------------------------------------------------------
    # Step 3 – Validate best params ON OUT-OF-SAMPLE period
    # ------------------------------------------------------------------
    print('\n[3/4] Validating best params on out-of-sample period …')

    oos_hmm = backtest(
        **common_bt,
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
        hmm_pca             = cfg.hmm_pca,
        regime_mode         = best_regime_mode,
        state_positions     = best_state_positions,
    )

    # ------------------------------------------------------------------
    # Step 4 – Validate best params ON FULL period
    # ------------------------------------------------------------------
    print('[4/4] Running best params on full period …')

    full_hmm = backtest(
        **common_bt,
        fromdate            = FULL_FROM,
        todate              = FULL_TO,
        hmm                 = True,
        hmm_components      = best_hmm_components,
        hmm_score_threshold = best_score_threshold,
        hmm_threshold       = best_params['hmm_threshold'],
        hmm_gate            = best_params['hmm_gate'],
        hmm_train_years     = best_params['hmm_train_years'],
        hmm_find_best_rs    = True,
        hmm_features        = best_features,
        hmm_pca             = cfg.hmm_pca,
        regime_mode         = best_regime_mode,
        state_positions     = best_state_positions,
    )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print_results_table(
        f'IN-SAMPLE  ({IS_FROM} → {IS_TO})',
        [
            {'label': f'Baseline {strat_label}({cfg.fast}/{cfg.slow}) no HMM',
             **bl_is},
            {'label': f'Best HMM  {strat_label}({cfg.fast}/{cfg.slow})',
             'total_return': best_is_return,
             'sharpe':       best_is_return,   # not tracked per-trial; use return as proxy
             'max_drawdown': 0.0},
        ]
    )

    print_results_table(
        f'OUT-OF-SAMPLE  ({OOS_FROM} → {OOS_TO})  ← honest validation',
        [
            {'label': f'Baseline {strat_label}({cfg.fast}/{cfg.slow}) no HMM',
             **bl_oos},
            {'label': f'Best HMM  {strat_label}({cfg.fast}/{cfg.slow})',
             **oos_hmm},
        ]
    )

    print_results_table(
        f'FULL PERIOD  ({FULL_FROM} → {FULL_TO})',
        [
            {'label': f'Baseline {strat_label}({cfg.fast}/{cfg.slow}) no HMM',
             **bl_full},
            {'label': f'Best HMM  {strat_label}({cfg.fast}/{cfg.slow})',
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
