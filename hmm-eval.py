#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
HMM Regime Accuracy Evaluator
==============================

Standalone tool to evaluate HMM regime detection accuracy against
manually-defined ground-truth labels derived from price action.

Features:
  - Reuses HMM feature engineering, training and plotting from ma-quantstats.py
  - Generates manual "ground truth" regime labels from a simple rule:
    the market is favourable when a smoothed return indicator is positive
  - Compares HMM predicted labels with ground truth via confusion matrix,
    accuracy, precision, recall, F1
  - Sweeps key HMM hyper-parameters (n_components, threshold, PCA dims,
    feature subsets, score weights) and reports accuracy for each

Usage:
    python hmm-eval.py --tickers SPY
    python hmm-eval.py --tickers AAPL --sweep
    python hmm-eval.py --tickers SPY --plot
    python hmm-eval.py --tickers SPY --gt-method sma --gt-window 60
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Reuse helpers from ma-quantstats.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib.machinery import SourceFileLoader

_mq = SourceFileLoader('ma_quantstats',
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'ma-quantstats.py')).load_module()

# Import all reusable pieces
prepare_hmm_features   = _mq.prepare_hmm_features
get_favourable_states  = _mq.get_favourable_states
regime_gate            = _mq.regime_gate
find_best_random_state = _mq.find_best_random_state
compute_bic            = _mq.compute_bic
find_best_n_components = _mq.find_best_n_components
plot_hmm_regimes       = _mq.plot_hmm_regimes
ensure_csv             = _mq.ensure_csv
load_csv_as_dataframe  = _mq.load_csv_as_dataframe
ALL_HMM_FEATURES       = _mq.ALL_HMM_FEATURES
DEFAULT_HMM_FEATURES   = _mq.DEFAULT_HMM_FEATURES


# ---------------------------------------------------------------------------
# Ground-truth labelling
# ---------------------------------------------------------------------------

def label_ground_truth(df: pd.DataFrame,
                       method: str = 'sma',
                       window: int = 40,
                       threshold: float = 0.0) -> pd.Series:
    """
    Create binary ground-truth labels for "favourable" (1) vs
    "unfavourable" (0) regimes based on observable price action.

    Methods:
        'sma'       – 1 when Close > SMA(window), else 0.
                       Classic trend-following rule.
        'returns'   – 1 when rolling mean of daily returns (window) > threshold.
        'drawdown'  – 1 when current drawdown from rolling max < threshold
                       (default -5%).  Marks crash/recovery periods as unfav.
        'combined'  – majority vote of sma + returns + drawdown.

    Returns a pd.Series of int {0, 1} aligned to df.index.
    """
    close = df['Close'].astype(float)

    if method == 'sma':
        sma = close.rolling(window).mean()
        labels = (close >= sma).astype(int)

    elif method == 'returns':
        rets = close.pct_change()
        roll_mean = rets.rolling(window).mean()
        labels = (roll_mean > threshold).astype(int)

    elif method == 'drawdown':
        roll_max = close.rolling(window, min_periods=1).max()
        dd = (close - roll_max) / roll_max
        dd_thresh = threshold if threshold != 0.0 else -0.05
        labels = (dd > dd_thresh).astype(int)

    elif method == 'combined':
        sma_lbl = (close >= close.rolling(window).mean()).astype(int)
        ret_lbl = (close.pct_change().rolling(window).mean() > 0).astype(int)
        roll_max = close.rolling(window, min_periods=1).max()
        dd = (close - roll_max) / roll_max
        dd_lbl = (dd > -0.05).astype(int)
        labels = ((sma_lbl + ret_lbl + dd_lbl) >= 2).astype(int)

    else:
        raise ValueError(f'Unknown ground-truth method: {method!r}')

    labels = labels.fillna(0).astype(int)
    labels.name = 'gt_regime'
    return labels


# ---------------------------------------------------------------------------
# HMM prediction on test set
# ---------------------------------------------------------------------------

def predict_hmm_regime(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        n_components: int = 6,
        score_threshold: float = 1.0,
        covariance_type: str = 'diag',
        n_iter: int = 1000,
        gate_threshold: float = 0.5,
        gate_type: str = 'threshold',
        find_best_rs: bool = True,
        n_random_states: int = 10,
        hmm_features: list[str] | None = None,
        hmm_pca: int | None = None,
        mean_weight: float = 1.0,
        vol_weight: float = 1.0,
        up_ratio_weight: float = 0.0,
        regime_mode: str = 'strict',
        verbose: bool = False,
) -> dict:
    """
    Train HMM on df_train, predict regime on df_test.

    Returns a dict with:
        'signal'          : pd.Series of float (0/1 or continuous) aligned to df_test
        'feat_test_index' : DatetimeIndex of rows that survived feature prep
        'model'           : fitted GaussianHMM
        'favourable'      : list of favourable state indices
        'state_pos_sizes' : dict state→pos_size
        'train_states'    : np.ndarray of predicted states on train
        'test_states'     : np.ndarray of predicted states on test
        'bic'             : BIC of the model on train
    """
    from sklearn.preprocessing import StandardScaler
    from hmmlearn.hmm import GaussianHMM

    features = list(hmm_features or DEFAULT_HMM_FEATURES)

    feat_train = prepare_hmm_features(df_train)[features]
    feat_test  = prepare_hmm_features(df_test)[features]
    feat_returns = prepare_hmm_features(df_train)['Returns']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(feat_train.values)
    X_test  = scaler.transform(feat_test.values)

    # Optional PCA
    pca_obj = None
    pca_explained = None
    if hmm_pca is not None and hmm_pca > 0:
        from sklearn.decomposition import PCA
        n_pca = min(hmm_pca, X_train.shape[1])
        pca_obj = PCA(n_components=n_pca)
        X_train = pca_obj.fit_transform(X_train)
        X_test  = pca_obj.transform(X_test)
        pca_explained = pca_obj.explained_variance_ratio_.sum() * 100
        if verbose:
            print(f'[HMM] PCA: {len(features)} → {n_pca} dims '
                  f'({pca_explained:.1f}% variance)')

    # Best random seed
    rs = 42
    if find_best_rs:
        rs = find_best_random_state(X_train, n_components=n_components,
                                    n_random_states=n_random_states,
                                    verbose=verbose)

    model = GaussianHMM(n_components=n_components, covariance_type=covariance_type,
                        n_iter=n_iter, random_state=rs)
    model.fit(X_train)

    bic, logL = compute_bic(model, X_train)

    train_states = model.predict(X_train)
    test_states  = model.predict(X_test)

    favourable, state_pos_sizes = get_favourable_states(
        train_states, feat_returns.values,
        n_favourable=None,
        score_threshold=score_threshold,
        mean_weight=mean_weight,
        vol_weight=vol_weight,
        up_ratio_weight=up_ratio_weight,
        regime_mode=regime_mode,
        verbose=verbose)

    # Generate test signal
    if not favourable:
        signals = np.ones(len(feat_test), dtype=float)
    elif regime_mode in ('score', 'linear'):
        state_proba = model.predict_proba(X_test)
        n_st = state_proba.shape[1]
        pos_vec = np.array([state_pos_sizes.get(k, 0.0) for k in range(n_st)])
        signals = state_proba @ pos_vec
    else:
        state_proba = model.predict_proba(X_test)
        p_fav = state_proba[:, favourable].sum(axis=1)
        signals = regime_gate(p_fav, gate_type=gate_type, tau=gate_threshold)

    signal_series = pd.Series(0.0, index=df_test.index, dtype=float)
    signal_series.loc[feat_test.index] = signals

    return {
        'signal':          signal_series,
        'feat_test_index': feat_test.index,
        'feat_train_index': feat_train.index,
        'model':           model,
        'favourable':      favourable,
        'state_pos_sizes': state_pos_sizes,
        'train_states':    train_states,
        'test_states':     test_states,
        'bic':             bic,
        'pca_explained':   pca_explained,
    }


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute classification metrics for binary regime labels.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy  = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = (2 * precision * recall / max(precision + recall, 1e-9))

    return {
        'accuracy':  round(accuracy, 4),
        'precision': round(precision, 4),
        'recall':    round(recall, 4),
        'f1':        round(f1, 4),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def print_confusion_matrix(metrics: dict, label: str = ''):
    """Pretty-print a 2×2 confusion matrix."""
    tp, tn, fp, fn = metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']
    if label:
        print(f'\n=== {label} ===')
    print(f'              Pred FAV   Pred UNFAV')
    print(f'  True FAV    {tp:>8}   {fn:>10}')
    print(f'  True UNFAV  {fp:>8}   {tn:>10}')
    print(f'  Accuracy : {metrics["accuracy"]:.4f}')
    print(f'  Precision: {metrics["precision"]:.4f}  (of predicted FAV, how many were truly FAV)')
    print(f'  Recall   : {metrics["recall"]:.4f}  (of truly FAV, how many were detected)')
    print(f'  F1       : {metrics["f1"]:.4f}')


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------

def evaluate_once(
        ticker: str,
        csv_path: str,
        train_start: datetime.datetime,
        train_end: datetime.datetime,
        test_start: datetime.datetime,
        test_end: datetime.datetime,
        gt_method: str = 'sma',
        gt_window: int = 40,
        n_components: int = 6,
        score_threshold: float = 1.0,
        gate_threshold: float = 0.5,
        gate_type: str = 'threshold',
        hmm_features: list[str] | None = None,
        hmm_pca: int | None = None,
        mean_weight: float = 1.0,
        vol_weight: float = 1.0,
        up_ratio_weight: float = 0.0,
        regime_mode: str = 'strict',
        find_best_rs: bool = True,
        verbose: bool = True,
        plot: bool = False,
        plot_save: str | None = None,
) -> dict:
    """
    Run a single HMM evaluation against ground-truth labels.

    Returns a dict of metrics + config.
    """
    df_train = load_csv_as_dataframe(csv_path, train_start, train_end)
    df_test  = load_csv_as_dataframe(csv_path, test_start, test_end)

    if df_train.empty or df_test.empty:
        if verbose:
            print(f'[WARN] Insufficient data for {ticker}')
        return {}

    # Ground truth on test set
    gt = label_ground_truth(df_test, method=gt_method, window=gt_window)

    # HMM prediction
    result = predict_hmm_regime(
        df_train=df_train, df_test=df_test,
        n_components=n_components,
        score_threshold=score_threshold,
        gate_threshold=gate_threshold,
        gate_type=gate_type,
        find_best_rs=find_best_rs,
        hmm_features=hmm_features,
        hmm_pca=hmm_pca,
        mean_weight=mean_weight,
        vol_weight=vol_weight,
        up_ratio_weight=up_ratio_weight,
        regime_mode=regime_mode,
        verbose=verbose,
    )

    pred = result['signal']

    # Align: only compare on dates present in both
    common = gt.index.intersection(pred.index)
    if common.empty:
        if verbose:
            print(f'[WARN] No overlapping dates for {ticker}')
        return {}

    y_true = gt.loc[common].values.astype(int)
    y_pred = (pred.loc[common].values >= 0.5).astype(int)

    metrics = compute_metrics(y_true, y_pred)

    if verbose:
        print_confusion_matrix(metrics, label=f'{ticker}  K={n_components}  '
              f'gt={gt_method}(w={gt_window})  '
              f'τ={gate_threshold}  PCA={hmm_pca}')
        gt_fav_pct = y_true.mean() * 100
        pred_fav_pct = y_pred.mean() * 100
        print(f'  GT favourable  : {gt_fav_pct:.1f}%  |  '
              f'Pred favourable: {pred_fav_pct:.1f}%')

    # ---- Plot (reuse ma-quantstats plotting function) ----
    if plot or plot_save:
        # Build regime series for train period (using predicted train states)
        train_regime = pd.Series(0, index=df_train.index, dtype=int)
        if result['favourable']:
            feat_train_idx = result['feat_train_index']
            fav_mask = np.isin(result['train_states'],
                               result['favourable']).astype(int)
            train_regime.loc[feat_train_idx] = fav_mask

        # Test regime = HMM prediction (binary)
        test_regime_hmm = (pred >= 0.5).astype(int)

        # Also plot ground truth as a separate figure
        test_regime_gt = gt.reindex(df_test.index).fillna(0).astype(int)

        # Plot 1: HMM predicted regimes
        plot_hmm_regimes(
            train_close  = df_train['Close'],
            test_close   = df_test['Close'],
            train_regime = train_regime,
            test_regime  = test_regime_hmm,
            ticker       = f'{ticker} – HMM Predicted (K={n_components})',
            show         = plot and not plot_save,
            save_path    = (plot_save.replace('.png', f'_hmm_pred.png')
                           if plot_save else None),
        )

        # Plot 2: Ground truth regimes
        gt_train = label_ground_truth(df_train, method=gt_method,
                                      window=gt_window)
        plot_hmm_regimes(
            train_close  = df_train['Close'],
            test_close   = df_test['Close'],
            train_regime = gt_train.reindex(df_train.index).fillna(0).astype(int),
            test_regime  = test_regime_gt,
            ticker       = f'{ticker} – Ground Truth ({gt_method}, w={gt_window})',
            show         = plot and not plot_save,
            save_path    = (plot_save.replace('.png', f'_gt.png')
                           if plot_save else None),
        )

    return {
        **metrics,
        'ticker':          ticker,
        'n_components':    n_components,
        'score_threshold': score_threshold,
        'gate_threshold':  gate_threshold,
        'gate_type':       gate_type,
        'gt_method':       gt_method,
        'gt_window':       gt_window,
        'hmm_pca':         hmm_pca,
        'n_features':      len(hmm_features or DEFAULT_HMM_FEATURES),
        'features':        ','.join(hmm_features or DEFAULT_HMM_FEATURES),
        'mean_weight':     mean_weight,
        'vol_weight':      vol_weight,
        'up_ratio_weight': up_ratio_weight,
        'regime_mode':     regime_mode,
        'bic':             result.get('bic'),
        'pca_explained':   result.get('pca_explained'),
    }


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def run_sweep(args) -> pd.DataFrame:
    """
    Sweep over key HMM parameters and report accuracy for each config.
    """
    datas_dir = os.path.join(os.path.dirname(__file__), 'datas')
    tickers = args.tickers

    # Ensure data
    csv_paths = {}
    for ticker in tickers:
        csv_path = os.path.join(datas_dir, f'{ticker.lower()}-2000-2026.csv')
        ensure_csv(csv_path, ticker)
        csv_paths[ticker] = csv_path

    train_end   = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    train_start = train_end - datetime.timedelta(
        days=int(args.hmm_train_years * 365.25))
    test_start  = train_end
    test_end    = datetime.datetime.strptime(args.todate, '%Y-%m-%d')

    # Parameter grids
    components_grid   = [3, 4, 5, 6, 8, 10, 12]
    threshold_grid    = [0.3, 0.5, 0.7, 0.9, 0.99]
    pca_grid          = [None, 2, 3, 4, 5, 6]
    gt_method_grid    = ['sma', 'returns', 'drawdown', 'combined']
    gt_window_grid    = [20, 40, 60]

    # Feature subsets to test
    feature_sets = {
        'default_8':    DEFAULT_HMM_FEATURES,
        'all_13':       ALL_HMM_FEATURES,
        'vol_only':     ['vol_short', 'vol_long', 'atr_norm', 'vol_of_vol',
                         'downside_vol', 'vol_z'],
        'return_only':  ['Returns', 'r5', 'r20', 'log_ret'],
        'minimal':      ['log_ret', 'vol_short', 'atr_norm'],
    }

    score_weight_combos = [
        (1.0, 1.0, 0.0),   # mean + vol
        (1.0, 0.0, 0.0),   # mean only
        (0.0, 1.0, 0.0),   # vol only
        (1.0, 1.0, 1.0),   # all three equal
        (2.0, 1.0, 0.5),   # mean-heavy
    ]

    results = []
    total_combos = 0

    # --- Sweep 1: n_components -------------------------------------------------
    print('\n' + '=' * 70)
    print('SWEEP 1: Number of HMM components (K)')
    print('=' * 70)
    for ticker in tickers:
        for K in components_grid:
            r = evaluate_once(
                ticker=ticker, csv_path=csv_paths[ticker],
                train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end,
                n_components=K, gt_method=args.gt_method,
                gt_window=args.gt_window,
                hmm_features=None, hmm_pca=None,
                verbose=False,
            )
            if r:
                r['sweep'] = 'n_components'
                results.append(r)
                total_combos += 1
                print(f'  K={K:>3}  acc={r["accuracy"]:.4f}  '
                      f'f1={r["f1"]:.4f}  BIC={r.get("bic", 0):>12.1f}  '
                      f'[{ticker}]')

    # --- Sweep 2: Gate threshold τ --------------------------------------------
    print('\n' + '=' * 70)
    print('SWEEP 2: Gate threshold (τ)')
    print('=' * 70)
    for ticker in tickers:
        for tau in threshold_grid:
            r = evaluate_once(
                ticker=ticker, csv_path=csv_paths[ticker],
                train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end,
                gate_threshold=tau,
                gt_method=args.gt_method, gt_window=args.gt_window,
                verbose=False,
            )
            if r:
                r['sweep'] = 'gate_threshold'
                results.append(r)
                total_combos += 1
                print(f'  τ={tau:.2f}  acc={r["accuracy"]:.4f}  '
                      f'f1={r["f1"]:.4f}  [{ticker}]')

    # --- Sweep 3: PCA dimensions -----------------------------------------------
    print('\n' + '=' * 70)
    print('SWEEP 3: PCA dimensionality reduction')
    print('=' * 70)
    for ticker in tickers:
        for pca_n in pca_grid:
            r = evaluate_once(
                ticker=ticker, csv_path=csv_paths[ticker],
                train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end,
                hmm_pca=pca_n,
                gt_method=args.gt_method, gt_window=args.gt_window,
                verbose=False,
            )
            if r:
                r['sweep'] = 'pca'
                results.append(r)
                total_combos += 1
                pca_label = f'{pca_n}' if pca_n else 'None'
                var_label = (f' ({r["pca_explained"]:.1f}% var)'
                             if r.get('pca_explained') else '')
                print(f'  PCA={pca_label:>4}{var_label}  '
                      f'acc={r["accuracy"]:.4f}  f1={r["f1"]:.4f}  [{ticker}]')

    # --- Sweep 4: Feature subsets -----------------------------------------------
    print('\n' + '=' * 70)
    print('SWEEP 4: Feature subsets')
    print('=' * 70)
    for ticker in tickers:
        for name, feats in feature_sets.items():
            r = evaluate_once(
                ticker=ticker, csv_path=csv_paths[ticker],
                train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end,
                hmm_features=feats,
                gt_method=args.gt_method, gt_window=args.gt_window,
                verbose=False,
            )
            if r:
                r['sweep'] = 'features'
                r['feature_set'] = name
                results.append(r)
                total_combos += 1
                print(f'  {name:<14}  ({len(feats)} feats)  '
                      f'acc={r["accuracy"]:.4f}  f1={r["f1"]:.4f}  [{ticker}]')

    # --- Sweep 5: Score weights -------------------------------------------------
    print('\n' + '=' * 70)
    print('SWEEP 5: Score weights (mean, vol, up_ratio)')
    print('=' * 70)
    for ticker in tickers:
        for mw, vw, uw in score_weight_combos:
            r = evaluate_once(
                ticker=ticker, csv_path=csv_paths[ticker],
                train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end,
                mean_weight=mw, vol_weight=vw, up_ratio_weight=uw,
                gt_method=args.gt_method, gt_window=args.gt_window,
                verbose=False,
            )
            if r:
                r['sweep'] = 'score_weights'
                results.append(r)
                total_combos += 1
                print(f'  w=({mw:.1f},{vw:.1f},{uw:.1f})  '
                      f'acc={r["accuracy"]:.4f}  f1={r["f1"]:.4f}  [{ticker}]')

    # --- Sweep 6: Ground-truth method ------------------------------------------
    print('\n' + '=' * 70)
    print('SWEEP 6: Ground-truth labelling method & window')
    print('=' * 70)
    for ticker in tickers:
        for gtm in gt_method_grid:
            for gtw in gt_window_grid:
                r = evaluate_once(
                    ticker=ticker, csv_path=csv_paths[ticker],
                    train_start=train_start, train_end=train_end,
                    test_start=test_start, test_end=test_end,
                    gt_method=gtm, gt_window=gtw,
                    verbose=False,
                )
                if r:
                    r['sweep'] = 'gt_method'
                    results.append(r)
                    total_combos += 1
                    print(f'  {gtm:<10} w={gtw:>3}  '
                          f'acc={r["accuracy"]:.4f}  f1={r["f1"]:.4f}  [{ticker}]')

    df = pd.DataFrame(results)
    print(f'\n[INFO] Total configs evaluated: {total_combos}')
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Evaluate HMM regime detection accuracy against '
                    'ground-truth price-action labels')

    parser.add_argument('--tickers', nargs='+', default=['SPY'],
        help='Tickers to evaluate')
    parser.add_argument('--fromdate', default='2019-01-01',
        help='Test period start (HMM training uses data before this)')
    parser.add_argument('--todate', default='2025-01-01',
        help='Test period end')
    parser.add_argument('--hmm-train-years', type=float, default=5.0,
        dest='hmm_train_years',
        help='Years of history before fromdate for HMM training')

    # HMM params
    parser.add_argument('--hmm-components', type=int, default=7,
        dest='hmm_components')
    parser.add_argument('--hmm-threshold', type=float, default=0.5,
        dest='hmm_threshold',
        help='Posterior probability threshold τ')
    parser.add_argument('--hmm-gate', default='threshold',
        dest='hmm_gate', choices=['threshold', 'linear', 'logistic'])
    parser.add_argument('--hmm-score-threshold', type=float, default=1.0,
        dest='hmm_score_threshold')
    parser.add_argument('--hmm-features', nargs='+', default=None,
        dest='hmm_features', choices=ALL_HMM_FEATURES)
    parser.add_argument('--hmm-pca', type=int, default=None,
        dest='hmm_pca')
    parser.add_argument('--hmm-find-best-rs', action='store_true', default=False,
        dest='hmm_find_best_rs')
    parser.add_argument('--regime-mode', default='strict',
        dest='regime_mode', choices=['strict', 'size', 'score', 'linear'])

    # Score weights
    parser.add_argument('--score-mean-weight', type=float, default=1.0,
        dest='score_mean_weight')
    parser.add_argument('--score-vol-weight', type=float, default=1.0,
        dest='score_vol_weight')
    parser.add_argument('--score-upratio-weight', type=float, default=0.0,
        dest='score_upratio_weight')

    # Ground truth
    parser.add_argument('--gt-method', default='sma', dest='gt_method',
        choices=['sma', 'returns', 'drawdown', 'combined'],
        help='Ground-truth labelling method')
    parser.add_argument('--gt-window', type=int, default=40, dest='gt_window',
        help='Look-back window for ground-truth labelling')

    # Mode
    parser.add_argument('--sweep', action='store_true', default=False,
        help='Run full parameter sweep (takes a while)')
    parser.add_argument('--plot', action='store_true', default=False,
        help='Show regime comparison plots')
    parser.add_argument('--plot-save', type=str, default=None, dest='plot_save',
        help='Save plots to this path (use {ticker} placeholder)')
    parser.add_argument('--output-csv', type=str, default=None, dest='output_csv',
        help='Save sweep results to CSV')

    args = parser.parse_args()

    datas_dir = os.path.join(os.path.dirname(__file__), 'datas')

    if args.sweep:
        df = run_sweep(args)
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f'[INFO] Sweep results saved → {args.output_csv}')
        else:
            csv_out = 'hmm-eval-sweep.csv'
            df.to_csv(csv_out, index=False)
            print(f'[INFO] Sweep results saved → {csv_out}')

        # Print best configs per sweep dimension
        print('\n' + '=' * 70)
        print('BEST CONFIGS PER SWEEP')
        print('=' * 70)
        for sweep_name in df['sweep'].unique():
            sub = df[df['sweep'] == sweep_name]
            best = sub.loc[sub['f1'].idxmax()]
            print(f'\n  [{sweep_name}]  Best F1={best["f1"]:.4f}  '
                  f'acc={best["accuracy"]:.4f}')
            for col in ['n_components', 'gate_threshold', 'hmm_pca',
                        'feature_set', 'mean_weight', 'vol_weight',
                        'up_ratio_weight', 'gt_method', 'gt_window']:
                if col in best and pd.notna(best.get(col)):
                    print(f'    {col}: {best[col]}')

    else:
        # Single evaluation per ticker
        train_end   = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
        train_start = train_end - datetime.timedelta(
            days=int(args.hmm_train_years * 365.25))
        test_start  = train_end
        test_end    = datetime.datetime.strptime(args.todate, '%Y-%m-%d')

        for ticker in args.tickers:
            csv_path = os.path.join(datas_dir,
                                    f'{ticker.lower()}-2000-2026.csv')
            ensure_csv(csv_path, ticker)

            print(f'\n{"=" * 70}')
            print(f'Evaluating {ticker}')
            print(f'  Train: {train_start.date()} → {train_end.date()}')
            print(f'  Test : {test_start.date()} → {test_end.date()}')
            print(f'{"=" * 70}')

            plot_path = None
            if args.plot_save:
                plot_path = args.plot_save.replace('{ticker}', ticker)

            evaluate_once(
                ticker=ticker, csv_path=csv_path,
                train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end,
                n_components=args.hmm_components,
                score_threshold=args.hmm_score_threshold,
                gate_threshold=args.hmm_threshold,
                gate_type=args.hmm_gate,
                hmm_features=args.hmm_features,
                hmm_pca=args.hmm_pca,
                mean_weight=args.score_mean_weight,
                vol_weight=args.score_vol_weight,
                up_ratio_weight=args.score_upratio_weight,
                regime_mode=args.regime_mode,
                find_best_rs=args.hmm_find_best_rs,
                verbose=True,
                plot=args.plot,
                plot_save=plot_path,
            )


if __name__ == '__main__':
    main()
