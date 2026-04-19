#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
Multi-Asset Moving Average Crossover Strategy with QuantStats Analysis
and Optional HMM Regime Filter
=======================================================================

This example demonstrates:
  - Accepting an arbitrary list of tickers as a portfolio
  - Auto-downloading OHLCV data from Yahoo Finance (cached as CSV)
  - A Simple Moving Average (SMA) crossover strategy applied
    independently to each instrument
  - Optional Hidden Markov Model (HMM) regime filter that gates trades
    only when the market is in a statistically favourable hidden state
  - Using backtrader's built-in PyFolio analyzer to extract returns
  - Generating a full QuantStats HTML tearsheet report

Usage:
    python ma-quantstats.py
    python ma-quantstats.py --tickers SPY AGG QQQ GLD
    python ma-quantstats.py --tickers AAPL MSFT GOOG --fast 20 --slow 100
    python ma-quantstats.py --plot

    # With HMM regime filter:
    python ma-quantstats.py --hmm
    python ma-quantstats.py --hmm --hmm-components 6 --hmm-favourable 3
    python ma-quantstats.py --hmm --hmm-threshold 0.6 --hmm-gate logistic
"""
from __future__ import (absolute_import, annotations, division,
                        print_function, unicode_literals)

import argparse
import datetime
import os
import sys

import time

import numpy as np
import pandas as pd

import backtrader as bt

# Strategy registry – all concrete strategies live in the strategies/ package
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from strategies import REGISTRY  # noqa: E402


# ---------------------------------------------------------------------------
# HMM helper functions  (ported from hmm_test_with_posterior_prob.py)
# ---------------------------------------------------------------------------

# All features computed by prepare_hmm_features()
ALL_HMM_FEATURES = [
    'Returns', 'Range', 'r5', 'r20', 'vol',
    'log_ret', 'vol_short', 'vol_long', 'atr_norm',
    'vol_of_vol', 'vol_lag1', 'downside_vol', 'vol_z',
]

# Default feature subset used when --hmm-features is not specified
DEFAULT_HMM_FEATURES = [
    'log_ret', 'r5', 'r20', 'vol_short', 'atr_norm',
]

def prepare_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute HMM input features from a raw OHLCV DataFrame.

    Features:
        Returns       – daily pct change of Close
        Range         – (High / Low) - 1   (intraday range proxy)
        r5            – 5-day log return
        r20           – 20-day log return
        vol           – normalised intraday range
        log_ret       – daily log return
        vol_short     – 5-day rolling volatility (std of log returns)
        vol_long      – 20-day rolling volatility
        atr_norm      – 14-day normalised Average True Range
        vol_of_vol    – 20-day rolling std of vol_short (vol-of-vol)
        vol_lag1      – lagged vol_short by 1 day
        downside_vol  – 20-day rolling downside volatility (std of negative log returns)
        vol_z         – z-scored vol_short relative to its 60-day rolling mean/std

    The DataFrame is returned with NaN rows dropped.
    """
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['Range']   = df['High'] / df['Low'] - 1
    log_close = np.log(df['Close'])
    df['r5']   = log_close.diff(5)
    df['r20']  = log_close.diff(20)
    df['vol'] = (df['High'] - df['Low']) / df['Close']

    # --- Volatility features -------------------------------------------------
    df['log_ret'] = log_close.diff(1)

    # Rolling volatility (short / long)
    df['vol_short'] = df['log_ret'].rolling(5).std()
    df['vol_long']  = df['log_ret'].rolling(20).std()

    # Normalised ATR (14-day)
    high_low   = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close  = (df['Low']  - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_norm'] = true_range.rolling(14).mean() / df['Close']

    # Vol-of-vol: rolling std of short-term volatility
    df['vol_of_vol'] = df['vol_short'].rolling(20).std()

    # Lagged volatility
    df['vol_lag1'] = df['vol_short'].shift(1)

    # Downside volatility: std of negative log returns only (20-day)
    neg_ret = df['log_ret'].clip(upper=0)
    df['downside_vol'] = neg_ret.rolling(20).std()

    # Z-scored volatility (vol_short vs its 60-day rolling stats)
    vol_roll_mean = df['vol_short'].rolling(60).mean()
    vol_roll_std  = df['vol_short'].rolling(60).std()
    df['vol_z'] = (df['vol_short'] - vol_roll_mean) / vol_roll_std.replace(0, np.nan)

    df.dropna(inplace=True)
    return df


def get_favourable_states(
        hidden_states,
        returns,
        n_favourable: int | None = None,
        score_threshold: float = 1.0,
        mean_weight: float = 1.0,
        vol_weight: float = 1.0,
        up_ratio_weight: float = 1.0,
        state_positions: list[float] | None = None,
        min_mean=None,
        max_volatility=None,
        min_up_ratio=None,
        regime_mode: str = 'strict',
        verbose: bool = True) -> list:
    """
    Rank HMM hidden states by a composite score of mean return, low
    volatility and upside frequency, then return those whose score meets
    *score_threshold*.

    Args:
        hidden_states  : array-like, predicted state label per bar
        returns        : array-like, matching daily returns
        n_favourable   : optional hard cap on the number of states returned
                         (None = no cap, keep all states above the threshold)
        score_threshold: minimum composite score for a state to be considered
                         favourable.  Scores are in [0, sum-of-weights];
                         with default weights of 1.0 each the range is [0, 3].
        *_weight       : relative weight for each score component
        min_*          : optional hard-filter thresholds (states that fail
                         are excluded before scoring)
        verbose        : print per-state statistics table

    Returns:
        List of state indices (integers) ranked best-first.
    """
    hs   = np.asarray(hidden_states).reshape(-1).astype(int)
    rets = np.asarray(returns, dtype=float).reshape(-1)

    if len(hs) != len(rets):
        raise ValueError(
            f'Length mismatch: hidden_states={len(hs)}, returns={len(rets)}')

    stats = []
    for state in np.unique(hs):
        mask       = hs == state
        state_rets = rets[mask]
        mean_ret   = state_rets.mean()
        volatility = state_rets.std()
        up_days    = int((state_rets > 0).sum())
        down_days  = int((state_rets < 0).sum())
        up_ratio   = up_days / len(state_rets) if len(state_rets) > 0 else 0.0
        stats.append(dict(
            state=state, mean=mean_ret, volatility=volatility,
            up_days=up_days, down_days=down_days,
            up_ratio=up_ratio, count=int(mask.sum())))

    # Hard filters
    filtered = [
        s for s in stats
        if (min_mean       is None or s['mean']       >= min_mean)
        and (max_volatility is None or s['volatility'] <= max_volatility)
        and (min_up_ratio   is None or s['up_ratio']   >= min_up_ratio)
    ]
    if not filtered:
        if verbose:
            print('[HMM] No states passed hard filters – returning empty list.')
        return []

    # Normalise each metric to [0, 1]
    def _norm(arr):
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / rng if rng > 0 else np.ones_like(arr) * 0.5

    means  = np.array([s['mean']       for s in filtered])
    vols   = np.array([s['volatility'] for s in filtered])
    ratios = np.array([s['up_ratio']   for s in filtered])

    mean_scores     = _norm(means)
    vol_scores      = 1.0 - _norm(vols)   # lower vol → higher score
    up_ratio_scores = _norm(ratios)

    combined = (mean_weight     * mean_scores
                + vol_weight    * vol_scores
                + up_ratio_weight * up_ratio_scores)

    for i, s in enumerate(filtered):
        s['score']      = round(float(combined[i]), 4)
        s['sc_mean']    = round(float(mean_scores[i]), 4)
        s['sc_vol']     = round(float(vol_scores[i]), 4)
        s['sc_upratio'] = round(float(up_ratio_scores[i]), 4)
        s['mean']       = round(s['mean'],       6)
        s['volatility'] = round(s['volatility'], 6)
        s['up_ratio']   = round(s['up_ratio'],   3)

    filtered.sort(key=lambda s: s['score'], reverse=True)

    # Compute position size per state
    if regime_mode == 'linear':
        # Linear rank-based sizing: best state → 1.0, worst → 0.0
        K = len(filtered)
        for i, s in enumerate(filtered):
            s['pos_size'] = round(1.0 - i / max(K - 1, 1), 4)
    elif state_positions is not None:
        # User-supplied fixed position sizes (score-ranked order, best→worst)
        if len(state_positions) != len(filtered):
            if verbose:
                print(f'[HMM] WARNING: --state-positions has {len(state_positions)} '
                      f'values but model has {len(filtered)} states. '
                      f'Padding/truncating to match.')
            # Pad with last value or truncate
            if len(state_positions) < len(filtered):
                pad_val = state_positions[-1] if state_positions else 0.1
                state_positions = list(state_positions) + [pad_val] * (len(filtered) - len(state_positions))
            else:
                state_positions = list(state_positions[:len(filtered)])
        for i, s in enumerate(filtered):
            s['pos_size'] = round(float(state_positions[i]), 4)
    else:
        min_score = min(s['score'] for s in filtered) if filtered else 0.0
        MIN_POS = 0.1   # floor for the worst state
        for s in filtered:
            if s['score'] >= score_threshold:
                s['pos_size'] = 1.0
            else:
                denom = score_threshold - min_score
                if denom < 1e-8:
                    s['pos_size'] = MIN_POS
                else:
                    ratio = (s['score'] - min_score) / denom
                    s['pos_size'] = round(MIN_POS + (1.0 - MIN_POS) * ratio, 4)

    if verbose:
        hdr = (f"{'State':>6} {'Mean':>10} {'Volatility':>12} {'Up Ratio':>10} "
               f"{'SC_Mean':>8} {'SC_Vol':>8} {'SC_UpR':>8} "
               f"{'Score':>8} {'Pos Size':>10} {'Count':>7}")
        print(hdr)
        print('-' * 105)
        for s in filtered:
            print(f"{s['state']:>6} {s['mean']:>10.6f} {s['volatility']:>12.6f} "
                  f"{s['up_ratio']:>10.3f} "
                  f"{s['sc_mean']:>8.4f} {s['sc_vol']:>8.4f} {s['sc_upratio']:>8.4f} "
                  f"{s['score']:>8.4f} {s['pos_size']:>10.4f} {s['count']:>7}")

    # Select states whose composite score meets the threshold
    above = [s for s in filtered if s['score'] >= score_threshold]
    if not above:
        if verbose:
            print(f'[HMM] No states met score threshold {score_threshold:.3f} '
                  f'– keeping best state as fallback.')
        above = filtered[:1]   # always return at least one state

    # Optional hard cap
    if n_favourable is not None:
        above = above[:n_favourable]

    favourable = [s['state'] for s in above]

    # Build state → pos_size mapping for all states
    state_pos_sizes = {s['state']: s['pos_size'] for s in filtered}
    # Build state → composite score mapping
    state_scores = {s['state']: s['score'] for s in filtered}

    if verbose:
        print(f'\n[HMM] Favourable states (score ≥ {score_threshold:.3f}): {favourable}')
    return favourable, state_pos_sizes, state_scores


def regime_gate(p: np.ndarray, gate_type: str = 'threshold',
               tau: float = 0.5, k: float = 10.0) -> np.ndarray:
    """
    Map posterior probability p ∈ [0,1] → binary integer 0 or 1.

    gate_type:
        'threshold' / 'linear' – 1 if p ≥ τ, else 0
        'logistic'             – smooth sigmoid gate (equivalent threshold at τ)

    Returns an integer array of the same shape as *p*.
    """
    p = np.asarray(p, dtype=float)
    if gate_type in ('threshold', 'linear'):
        return (p >= tau).astype(int)
    elif gate_type == 'logistic':
        sigmoid = 1.0 / (1.0 + np.exp(-k * (p - tau)))
        return (sigmoid >= 0.5).astype(int)
    else:
        raise ValueError(f'Unknown gate_type: {gate_type!r}. '
                         f'Choose from threshold, linear, logistic.')


def compute_bic(hmm_model, X: np.ndarray) -> tuple:
    """
    Bayesian Information Criterion for a fitted GaussianHMM.

    BIC = -2·logL + k·log(T)

    Lower BIC indicates a better bias-variance trade-off.

    Returns:
        (bic, logL) tuple.
    """
    from hmmlearn.hmm import GaussianHMM  # local import – optional dep
    K = hmm_model.n_components
    T, d = X.shape
    logL = hmm_model.score(X)
    # Parameters: start probs (K-1), transition rows K*(K-1), emission means+vars K*2d
    k_params = (K - 1) + K * (K - 1) + K * (2 * d)
    bic = -2.0 * logL + k_params * np.log(T)
    return bic, logL


def find_best_n_components(
        X_train_scaled: np.ndarray,
        state_range=range(2, 10),
        n_random_seeds: int = 5,
        covariance_type: str = 'diag',
        n_iter: int = 1000,
        tol: float = 1e-3,
        random_state_base: int = 42,
        verbose: bool = True) -> dict:
    """
    Grid-search the optimal number of HMM hidden states using BIC.

    Returns a dict with keys: best_n_states, best_model, best_bic,
    best_logL, all_results.
    """
    from hmmlearn.hmm import GaussianHMM

    if X_train_scaled.ndim != 2 or X_train_scaled.size == 0:
        raise ValueError('X_train_scaled must be a non-empty 2-D array.')

    results = []
    best = {'bic': np.inf, 'model': None, 'K': None, 'logL': None}

    if verbose:
        print(f'[HMM] Searching over K ∈ {list(state_range)} '
              f'with {n_random_seeds} seeds each …')

    for K in state_range:
        best_bic_k, best_model_k, best_logL_k = np.inf, None, None
        for seed in range(random_state_base, random_state_base + n_random_seeds):
            try:
                m = GaussianHMM(n_components=K, covariance_type=covariance_type,
                                n_iter=n_iter, tol=tol, random_state=seed,
                                verbose=False)
                m.fit(X_train_scaled)
                bic, logL = compute_bic(m, X_train_scaled)
                if bic < best_bic_k:
                    best_bic_k, best_model_k, best_logL_k = bic, m, logL
            except Exception:
                continue
        if best_model_k is None:
            continue
        results.append({'K': K, 'bic': best_bic_k, 'logL': best_logL_k})
        if verbose:
            print(f'  K={K:>3}  BIC={best_bic_k:>12.2f}  logL={best_logL_k:>12.2f}')
        if best_bic_k < best['bic']:
            best.update(bic=best_bic_k, model=best_model_k,
                        K=K, logL=best_logL_k)

    if best['model'] is None:
        raise RuntimeError('[HMM] No HMM converged during component search.')

    if verbose:
        print(f'[HMM] Best K={best["K"]} (BIC={best["bic"]:.2f})')

    return dict(best_n_states=best['K'], best_model=best['model'],
                best_bic=best['bic'], best_logL=best['logL'],
                all_results=results)


def find_best_random_state(
        X_train_scaled: np.ndarray,
        n_components: int = 4,
        n_random_states: int = 10,
        n_iter: int = 100,
        covariance_type: str = 'diag',
        tol: float = 1e-4,
        verbose: bool = False) -> int:
    """
    Try *n_random_states* different random seeds, return the seed that
    yields the highest training log-likelihood.
    """
    from hmmlearn.hmm import GaussianHMM

    best_score, best_rs = -np.inf, 0
    for rs in range(n_random_states):
        try:
            m = GaussianHMM(n_components=n_components,
                            covariance_type=covariance_type,
                            n_iter=n_iter, tol=tol,
                            random_state=rs, verbose=False)
            m.fit(X_train_scaled)
            score = m.score(X_train_scaled)
            if score > best_score:
                best_score, best_rs = score, rs
        except Exception:
            continue
    if verbose:
        print(f'[HMM] Best random_state={best_rs}  (logL={best_score:.4f})')
    return best_rs


def train_hmm_and_get_signals(
        df_train: pd.DataFrame,
        df_test:  pd.DataFrame,
        n_components: int = 4,
        n_favourable: int | None = None,
        score_threshold: float = 1.0,
        covariance_type: str = 'diag',
        n_iter: int = 1000,
        random_state: int = 42,
        threshold: float = 0.5,
        gate_type: str = 'threshold',
        find_best_rs: bool = True,
        n_random_states: int = 10,
        hmm_features: list[str] | None = None,
        hmm_pca: int | None = None,
        regime_mode: str = 'strict',
        mean_weight: float = 1.0,
        vol_weight: float = 1.0,
        up_ratio_weight: float = 1.0,
        state_positions: list[float] | None = None,
        verbose: bool = True,
        plot: bool = False,
        plot_ticker: str = '',
        plot_save_path: str | None = None,
        plot_show: bool = True) -> np.ndarray:
    """
    Full HMM regime pipeline:

    1. Prepare features (Returns, Range, r5, r20) for train and test sets.
    2. Fit StandardScaler on train, transform both.
    3. Optionally search for the best random initialisation.
    4. Fit GaussianHMM on the training set.
    5. Identify *n_favourable* states from training predictions.
    6. Compute posterior P(favourable | test data) via predict_proba.
    7. Apply regime_gate → binary int array aligned to *df_test*.

    Returns:
        np.ndarray of shape (len(df_test_features),) with values 0 or 1.
        Rows that were dropped during feature prep are assigned 0 so the
        caller can align the array back to df_test using the index.
    """
    from sklearn.preprocessing import StandardScaler

    HMM_FEATURES = list(hmm_features or DEFAULT_HMM_FEATURES)

    # Prepare features
    feat_train = prepare_hmm_features(df_train)[HMM_FEATURES]
    feat_test  = prepare_hmm_features(df_test)[HMM_FEATURES]

    feat_returns = prepare_hmm_features(df_train)['Returns']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(feat_train.values)
    X_test  = scaler.transform(feat_test.values)

    # Optional PCA dimensionality reduction
    if hmm_pca is not None and hmm_pca > 0:
        from sklearn.decomposition import PCA
        n_components_pca = min(hmm_pca, X_train.shape[1])
        pca = PCA(n_components=n_components_pca)
        X_train = pca.fit_transform(X_train)
        X_test  = pca.transform(X_test)
        explained = pca.explained_variance_ratio_.sum() * 100
        print(f'[HMM] PCA: {feat_train.shape[1]} features → {n_components_pca} components '
            f'({explained:.1f}% variance explained)')

    # Find best random state if requested
    rs = random_state
    if find_best_rs:
        if verbose:
            print(f'[HMM] Searching best random state over {n_random_states} seeds …')
        rs = find_best_random_state(X_train, n_components=n_components,
                                    n_random_states=n_random_states,
                                    verbose=verbose)

    # Fit model
    from hmmlearn.hmm import GaussianHMM
    model = GaussianHMM(n_components=n_components,
                        covariance_type=covariance_type,
                        n_iter=n_iter, random_state=rs)
    model.fit(X_train)
    if verbose:
        print(f'[HMM] Training logL={model.score(X_train):.4f}')

    # Derive favourable states from training predictions
    train_states = model.predict(X_train)
    favourable, state_pos_sizes, state_scores = get_favourable_states(
        train_states, feat_returns.values,
        n_favourable=n_favourable,
        score_threshold=score_threshold,
        mean_weight=mean_weight,
        vol_weight=vol_weight,
        up_ratio_weight=up_ratio_weight,
        state_positions=state_positions,
        regime_mode=regime_mode,
        verbose=verbose)

    # Predict dominant state for every test bar
    test_states = model.predict(X_test)

    # Build per-bar composite score from dominant state
    test_scores = np.array([state_scores.get(s, 0.0) for s in test_states])

    if not favourable:
        if verbose:
            print('[HMM] Warning: no favourable states found – disabling filter.')
        # Return all-ones (no filtering) aligned to test feature index
        signals = np.ones(len(feat_test), dtype=float)
    elif regime_mode in ('score', 'linear'):
        # Continuous sizing: per-bar pos_size = Σ P(state_k) * pos_size_k
        state_proba = model.predict_proba(X_test)
        n_states = state_proba.shape[1]
        pos_size_vec = np.array([state_pos_sizes.get(k, 0.0) for k in range(n_states)])
        signals = state_proba @ pos_size_vec  # shape (n_bars,), values in [0, 1]
        if verbose:
            mean_ps = signals.mean()
            label = 'Linear' if regime_mode == 'linear' else 'Score-based'
            print(f'[HMM] {label} sizing: mean pos_size={mean_ps:.3f}  '
                  f'min={signals.min():.3f}  max={signals.max():.3f}')
    else:
        # Binary gate via posterior probabilities (strict / size modes)
        state_proba = model.predict_proba(X_test)
        p_fav = state_proba[:, favourable].sum(axis=1)
        signals = regime_gate(p_fav, gate_type=gate_type, tau=threshold)

    if regime_mode not in ('score', 'linear'):
        if verbose:
            active = signals.sum()
            print(f'[HMM] Gate: {gate_type}  τ={threshold}  '
                  f'Active bars: {active}/{len(signals)} '
                  f'({100*active/max(len(signals),1):.1f}%)')

    # Build a Series aligned to df_test's index (0 for rows outside feature prep)
    signal_series = pd.Series(0.0, index=df_test.index, dtype=float)
    signal_series.loc[feat_test.index] = signals

    # Build predicted-state series (-1 for rows outside feature prep)
    state_series = pd.Series(-1, index=df_test.index, dtype=int)
    state_series.loc[feat_test.index] = test_states

    # Build per-bar composite score series (0.0 for rows outside feature prep)
    score_series = pd.Series(0.0, index=df_test.index, dtype=float)
    score_series.loc[feat_test.index] = test_scores

    # ---- Optional debug plot -----------------------------------------------
    if plot:
        # Training regime: 1 where predicted state is favourable, else 0
        train_regime = pd.Series(0, index=df_train.index, dtype=int)
        if favourable:
            fav_mask = np.isin(train_states, favourable).astype(int)
            train_regime.loc[feat_train.index] = fav_mask

        # Test p_fav series (NaN outside feature-prep rows)
        p_fav_plot: pd.Series | None = None
        if favourable and regime_mode not in ('score', 'linear'):
            p_fav_full = pd.Series(np.nan, index=df_test.index, dtype=float)
            p_fav_full.loc[feat_test.index] = p_fav
            p_fav_plot = p_fav_full
        elif favourable and regime_mode in ('score', 'linear'):
            # Use the continuous score signal as the probability plot
            p_fav_full = pd.Series(np.nan, index=df_test.index, dtype=float)
            p_fav_full.loc[feat_test.index] = signals
            p_fav_plot = p_fav_full

        plot_hmm_regimes(
            train_close   = df_train['Close'],
            test_close    = df_test['Close'],
            train_regime  = train_regime,
            test_regime   = signal_series,
            ticker        = plot_ticker,
            p_fav_series  = p_fav_plot,
            threshold     = threshold,
            show          = plot_show,
            save_path     = plot_save_path,
        )

    return signal_series, state_series, score_series


def train_hmm_price_model(
        df_train: pd.DataFrame,
        df_test:  pd.DataFrame,
        n_components: int = 4,
        covariance_type: str = 'diag',
        n_iter: int = 1000,
        random_state: int = 42,
        find_best_rs: bool = True,
        n_random_states: int = 10,
        verbose: bool = True) -> pd.DataFrame:
    """
    Train a GaussianHMM on **closing prices** and return a DataFrame with
    per-bar state statistics aligned to *df_test*.

    Each hidden state k has a Gaussian(μ_k, σ_k) distribution over price.
    The returned DataFrame has two columns:
      state_mean – μ of the predicted state at that bar.
      state_std  – σ of the predicted state at that bar.

    Used exclusively by the HmmMeanReversionStrategy ('hmm_mr').
    """
    from hmmlearn.hmm import GaussianHMM

    X_train = df_train['Close'].values.reshape(-1, 1).astype(float)
    X_test  = df_test['Close'].values.reshape(-1, 1).astype(float)

    # Optionally find the best random seed
    rs = random_state
    if find_best_rs:
        if verbose:
            print(f'[HMM-MR] Searching best random state over {n_random_states} seeds …')
        rs = find_best_random_state(X_train, n_components=n_components,
                                    n_random_states=n_random_states,
                                    verbose=verbose)

    model = GaussianHMM(n_components=n_components,
                        covariance_type=covariance_type,
                        n_iter=n_iter, random_state=rs)
    model.fit(X_train)
    if verbose:
        print(f'[HMM-MR] Training logL={model.score(X_train):.4f}')

    # Extract per-state statistics (single feature → flatten is safe)
    state_means = model.means_.flatten()                     # shape (K,)
    # covars_ for 'diag' + 1 feature: shape (K, 1); flatten → (K,)
    state_stds  = np.sqrt(np.abs(model.covars_.reshape(n_components, -1)[:, 0]))

    # Predict states for the test window
    predicted_states = model.predict(X_test)                 # shape (len(test),)

    if verbose:
        for k in range(n_components):
            cnt = int((predicted_states == k).sum())
            print(f'[HMM-MR]   State {k}: '
                  f'mean={state_means[k]:.4f}  '
                  f'std={state_stds[k]:.4f}  '
                  f'bars={cnt}')

    result = pd.DataFrame(index=df_test.index)
    result['state_mean'] = state_means[predicted_states]
    result['state_std']  = state_stds[predicted_states]
    return result


# ---------------------------------------------------------------------------
# HMM regime debug plot  (fully decoupled – no dependency on HMM training)
# ---------------------------------------------------------------------------

def plot_hmm_regimes(
        train_close: pd.Series,
        test_close: pd.Series,
        train_regime: pd.Series,
        test_regime: pd.Series,
        ticker: str = '',
        p_fav_series: pd.Series | None = None,
        threshold: float = 0.5,
        show: bool = True,
        save_path: str | None = None,
) -> None:
    """
    Plot close prices over training and test periods with HMM regime shading.

    Favourable bars (regime == 1) are shaded green; unfavourable (regime == 0)
    are shaded red.  A vertical dashed line marks the train / test split.
    An optional second panel shows P(favourable) with the threshold line.

    This function is intentionally self-contained: it only depends on pandas
    Series / numpy arrays and matplotlib – not on any HMM or backtrader code.

    Args:
        train_close   : pd.Series of close prices indexed by date (train period).
        test_close    : pd.Series of close prices indexed by date (OOS period).
        train_regime  : pd.Series of int 0/1 aligned to train_close.index.
                        1 = favourable state, 0 = unfavourable.
        test_regime   : pd.Series of int 0/1 aligned to test_close.index.
        ticker        : Label used in the plot title.
        p_fav_series  : Optional pd.Series of P(favourable | data) for the test
                        period. When provided, a second panel is drawn.
        threshold     : Gate threshold drawn as a horizontal dashed line on the
                        probability panel.
        show          : Call plt.show() after drawing. Set to False when saving
                        to a file in a headless environment.
        save_path     : File path to save the figure (e.g. 'regimes.png').
                        Skipped when None.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print('[WARN] matplotlib not installed – skipping HMM regime plot.')
        return

    # ------------------------------------------------------------------
    # Helper: shade contiguous runs of the same regime value
    # ------------------------------------------------------------------
    def _shade_regimes(ax, close_series, regime_series,
                       fav_color, unfav_color, alpha=0.25):
        """Draw axvspan for each contiguous block of identical regime values."""
        aligned = regime_series.reindex(close_series.index).fillna(0).astype(int)
        if aligned.empty:
            return
        dates  = aligned.index
        values = aligned.values
        # Locate start indices of every run
        run_starts = np.where(np.diff(values, prepend=values[0] - 1) != 0)[0]
        run_starts = np.append(run_starts, len(values))   # sentinel
        for i in range(len(run_starts) - 1):
            si, ei = run_starts[i], run_starts[i + 1] - 1
            color = fav_color if values[si] == 1 else unfav_color
            ax.axvspan(dates[si], dates[ei], alpha=alpha, color=color, lw=0)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    n_panels = 2 if p_fav_series is not None else 1
    height_ratios = [3, 1] if n_panels == 2 else [1]
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(14, 4 * n_panels),
        sharex=True,
        gridspec_kw={'height_ratios': height_ratios},
        squeeze=False,
    )
    ax_price = axes[0, 0]

    # ------------------------------------------------------------------
    # Price panel
    # ------------------------------------------------------------------
    all_close = pd.concat([train_close, test_close]).sort_index()
    ax_price.plot(all_close.index, all_close.values,
                  color='black', lw=0.8, zorder=3, label='Close')

    _shade_regimes(ax_price, train_close, train_regime,
                   fav_color='#2ecc71', unfav_color='#e74c3c')
    _shade_regimes(ax_price, test_close, test_regime,
                   fav_color='#27ae60', unfav_color='#c0392b')

    # Train / test split marker
    if not test_close.empty:
        ax_price.axvline(test_close.index[0], color='navy',
                         lw=1.2, ls='--', zorder=4, label='Train / Test split')

    legend_handles = [
        mpatches.Patch(color='#2ecc71', alpha=0.5, label='Favourable – train'),
        mpatches.Patch(color='#e74c3c', alpha=0.5, label='Unfavourable – train'),
        mpatches.Patch(color='#27ae60', alpha=0.5, label='Favourable – OOS'),
        mpatches.Patch(color='#c0392b', alpha=0.5, label='Unfavourable – OOS'),
    ]
    ax_price.legend(handles=legend_handles, loc='upper left', fontsize=8)
    title = f'HMM Regime Filter – {ticker}' if ticker else 'HMM Regime Filter'
    ax_price.set_title(title, fontsize=12)
    ax_price.set_ylabel('Close Price')
    ax_price.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Probability panel (optional)
    # ------------------------------------------------------------------
    if p_fav_series is not None:
        ax_prob = axes[1, 0]
        ax_prob.plot(p_fav_series.index, p_fav_series.values,
                     color='steelblue', lw=0.9, label='P(favourable)')
        ax_prob.axhline(threshold, color='darkorange', ls='--', lw=1.2,
                        label=f'τ = {threshold}')
        ax_prob.fill_between(p_fav_series.index, p_fav_series.values,
                             threshold,
                             where=(p_fav_series.values >= threshold),
                             interpolate=True,
                             color='#27ae60', alpha=0.25)
        ax_prob.fill_between(p_fav_series.index, p_fav_series.values,
                             threshold,
                             where=(p_fav_series.values < threshold),
                             interpolate=True,
                             color='#c0392b', alpha=0.25)
        ax_prob.set_ylim(0, 1)
        ax_prob.set_ylabel('P(fav)')
        ax_prob.legend(loc='upper left', fontsize=8)
        ax_prob.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'[HMM] Regime plot saved → {save_path}')
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Custom data feed – OHLCV + pre-computed HMM regime signal
# ---------------------------------------------------------------------------

class HMMRegimeFeed(bt.feeds.PandasData):
    """
    Extends PandasData with extra lines:
      regime    – pre-computed regime signal (0/1 or float for score mode)
      hmm_state – dominant HMM component index at each bar
      hmm_score – composite score of the dominant state
    """
    lines = ('regime', 'hmm_state', 'hmm_score',)
    params = (
        ('regime', -1),      # -1 → auto-detect column named 'regime'
        ('hmm_state', -1),   # -1 → auto-detect column named 'hmm_state'
        ('hmm_score', -1),   # -1 → auto-detect column named 'hmm_score'
    )


class HMMStateFeed(bt.feeds.PandasData):
    """
    Extends PandasData with two extra lines for the HMM Mean-Reversion
    strategy:
      state_mean – Gaussian mean of the predicted HMM state (same units as price)
      state_std  – Gaussian std  of the predicted HMM state

    Both are pre-computed by ``train_hmm_price_model()`` before cerebro.run().
    The strategy accesses them as ``data.state_mean[0]`` and ``data.state_std[0]``.
    """
    lines = ('state_mean', 'state_std')
    params = (
        ('state_mean', -1),   # -1 → auto-detect column named 'state_mean'
        ('state_std',  -1),   # -1 → auto-detect column named 'state_std'
    )


# ---------------------------------------------------------------------------
# Strategy  – concrete implementations live in strategies/ (see REGISTRY)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Download helper  (only runs when the CSV does not yet exist)
# ---------------------------------------------------------------------------
def ensure_csv(csv_path, ticker, quiet=False):
    """Download data from Yahoo Finance if the CSV is not present."""
    if os.path.exists(csv_path):
        if not quiet:
            print(f'[INFO] Using existing data file: {csv_path}')
        return

    print(f'[INFO] Downloading {ticker} data from Yahoo Finance …')
    try:
        import yfinance as yf

        df = yf.download(ticker, start='2000-01-01', end='2026-12-31', auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(csv_path)
        print(f'[INFO] Saved {len(df)} rows → {csv_path}')
    except Exception as exc:
        sys.exit(f'[ERROR] Could not download {ticker} data: {exc}\n'
                 f'Install yfinance first:  pip install yfinance')


def load_csv_as_dataframe(csv_path: str, fromdate, todate) -> pd.DataFrame:
    """
    Load a yfinance-style CSV into a pandas DataFrame with OHLCV columns
    and a DatetimeIndex, clipped to [fromdate, todate].
    """
    # yfinance CSV column order: Date, Close, High, Low, Open, Volume
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    # Normalise column names produced by different yfinance versions
    df.columns = [c.strip().capitalize() for c in df.columns]
    # Rename 'Price' header artifact if present (yfinance ≥ 0.2 writes extra rows)
    df = df[df.index.notna()]
    # Keep only numeric rows
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)
    # Clip to requested date range
    df = df.loc[(df.index >= pd.Timestamp(fromdate)) &
                (df.index <= pd.Timestamp(todate))]
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args=None, quiet=False):
    """
    Run the backtest and return a metrics dict::

        {
            'total_return': float,   # % total return
            'final_value':  float,   # final portfolio value
            'sharpe':       float,   # annualised Sharpe ratio
            'max_drawdown': float,   # % max drawdown (positive = loss)
            'annual_return': float,  # CAGR % over the full period
            'calmar':       float,   # CAGR / |max_drawdown| (0 if no DD)
            'time_taken':   float,   # wall-clock seconds for cerebro.run()
        }

    Pass ``quiet=True`` to suppress all console output (useful during
    optimisation loops).
    """
    _p = (lambda *a, **kw: None) if quiet else print

    # Accept either a list of CLI strings, a pre-built namespace, or None
    if args is None or isinstance(args, list):
        args = parse_args(args)

    datas_dir = os.path.join(os.path.dirname(__file__), 'datas')
    tickers = args.tickers

    # Ensure CSV files exist for every ticker
    csv_paths = {}
    for ticker in tickers:
        csv_path = os.path.join(datas_dir,
                                f'{ticker.lower()}-2000-2026.csv')
        ensure_csv(csv_path, ticker, quiet=quiet)
        csv_paths[ticker] = csv_path

    # ---- Cerebro setup ------------------------------------------------------
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(args.cash)
    cerebro.broker.setcommission(commission=args.commission)

    # ---- Date ranges ---------------------------------------------------------
    fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    todate   = datetime.datetime.strptime(args.todate,   '%Y-%m-%d')

    # ---- Strategy key – determined early so HMM wiring can branch on it -----
    strategy_key = getattr(args, 'strategy', 'sma')

    # ---- HMM regime pre-computation (per ticker) ----------------------------
    hmm_signals:  dict = {}   # ticker → pd.Series of 0/1 (regime filter)
    hmm_states:   dict = {}   # ticker → pd.Series of int (dominant HMM state)
    hmm_scores:   dict = {}   # ticker → pd.Series of float (composite score)
    hmm_mr_state: dict = {}   # ticker → DataFrame(state_mean, state_std)

    if args.hmm and strategy_key != 'hmm_mr':
        _p('\n[HMM] Building regime signals …')
        train_end   = fromdate
        train_start = train_end - datetime.timedelta(days=int(args.hmm_train_years * 365.25))
        _p(f'[HMM] Training window : {train_start.date()} → {train_end.date()}')
        _p(f'[HMM] Test window     : {fromdate.date()} → {todate.date()}')

        for ticker in tickers:
            _p(f'\n[HMM] Processing {ticker} …')
            df_train_all = load_csv_as_dataframe(csv_paths[ticker],
                                                 train_start, train_end)
            df_test_all  = load_csv_as_dataframe(csv_paths[ticker],
                                                 fromdate, todate)

            if df_train_all.empty:
                _p(f'[HMM] WARNING: No training data for {ticker} – '
                   f'regime filter disabled for this ticker.')
                hmm_signals[ticker] = pd.Series(1, index=df_test_all.index)
                hmm_states[ticker]  = pd.Series(-1, index=df_test_all.index)
                hmm_scores[ticker]  = pd.Series(0.0, index=df_test_all.index)
                continue

            hmm_plot      = getattr(args, 'hmm_plot', False)
            hmm_plot_save = getattr(args, 'hmm_plot_save', None)
            signals, states, scores = train_hmm_and_get_signals(
                df_train=df_train_all,
                df_test=df_test_all,
                n_components=args.hmm_components,
                n_favourable=args.hmm_favourable,
                score_threshold=args.hmm_score_threshold,
                covariance_type='diag',
                n_iter=1000,
                random_state=42,
                threshold=args.hmm_threshold,
                gate_type=args.hmm_gate,
                find_best_rs=args.hmm_find_best_rs,
                n_random_states=10,
                hmm_features=getattr(args, 'hmm_features', None),
                hmm_pca=getattr(args, 'hmm_pca', None),
                regime_mode=getattr(args, 'regime_mode', 'strict'),
                mean_weight=getattr(args, 'score_mean_weight', 1.0),
                vol_weight=getattr(args, 'score_vol_weight', 1.0),
                up_ratio_weight=getattr(args, 'score_upratio_weight', 1.0),
                state_positions=getattr(args, 'state_positions', None),
                verbose=not quiet,
                plot=hmm_plot and not quiet,
                plot_ticker=ticker,
                plot_save_path=(
                    hmm_plot_save.format(ticker=ticker)
                    if hmm_plot_save else None
                ),
                plot_show=hmm_plot,
            )
            hmm_signals[ticker] = signals
            hmm_states[ticker]  = states
            hmm_scores[ticker]  = scores

    # ---- HMM price model for the Mean-Reversion strategy --------------------
    if strategy_key == 'hmm_mr':
        _p('\n[HMM-MR] Training price-level state model …')
        train_end   = fromdate
        train_start = train_end - datetime.timedelta(
            days=int(args.hmm_train_years * 365.25))
        _p(f'[HMM-MR] Training window : {train_start.date()} → {train_end.date()}')
        _p(f'[HMM-MR] Test window     : {fromdate.date()} → {todate.date()}')
        for ticker in tickers:
            _p(f'\n[HMM-MR] Processing {ticker} …')
            df_train_mr = load_csv_as_dataframe(csv_paths[ticker],
                                                train_start, train_end)
            df_test_mr  = load_csv_as_dataframe(csv_paths[ticker],
                                                fromdate, todate)
            if df_train_mr.empty:
                _p(f'[HMM-MR] WARNING: no training data for {ticker} – '
                   'state mean/std will be 0. Strategy will not trade.')
                hmm_mr_state[ticker] = pd.DataFrame(
                    {'state_mean': 0.0, 'state_std': 1.0},
                    index=df_test_mr.index)
                continue
            hmm_mr_state[ticker] = train_hmm_price_model(
                df_train        = df_train_mr,
                df_test         = df_test_mr,
                n_components    = args.hmm_components,
                covariance_type = 'diag',
                n_iter          = 1000,
                random_state    = 42,
                find_best_rs    = args.hmm_find_best_rs,
                n_random_states = 10,
                verbose         = not quiet,
            )

    # ---- Data feeds ----------------------------------------------------------
    if strategy_key == 'hmm_mr':
        # Use HMMStateFeed: OHLCV + state_mean + state_std lines
        for ticker in tickers:
            df_raw = load_csv_as_dataframe(csv_paths[ticker], fromdate, todate)
            mr_df  = hmm_mr_state[ticker].reindex(df_raw.index)
            df_raw['state_mean'] = mr_df['state_mean'].ffill().fillna(df_raw['Close'])
            df_raw['state_std']  = mr_df['state_std'].ffill().fillna(df_raw['Close'].std())
            df_bt = df_raw[['Open', 'High', 'Low', 'Close', 'Volume',
                            'state_mean', 'state_std']].copy()
            data = HMMStateFeed(
                dataname    = df_bt,
                fromdate    = fromdate,
                todate      = todate,
                open        = 'Open',
                high        = 'High',
                low         = 'Low',
                close       = 'Close',
                volume      = 'Volume',
                openinterest= -1,
                state_mean  = 'state_mean',
                state_std   = 'state_std',
            )
            cerebro.adddata(data, name=ticker)
    elif args.hmm:
        # Use HMMRegimeFeed (PandasData + regime + hmm_state lines) for each ticker
        for ticker in tickers:
            df_raw = load_csv_as_dataframe(csv_paths[ticker], fromdate, todate)
            regime_vals = hmm_signals[ticker].reindex(df_raw.index).fillna(0.0)
            if getattr(args, 'regime_mode', 'strict') != 'score':
                regime_vals = regime_vals.astype(int)
            df_raw['regime'] = regime_vals
            state_vals = hmm_states[ticker].reindex(df_raw.index).fillna(-1).astype(int)
            df_raw['hmm_state'] = state_vals
            score_vals = hmm_scores[ticker].reindex(df_raw.index).fillna(0.0)
            df_raw['hmm_score'] = score_vals
            df_bt = df_raw[['Open', 'High', 'Low', 'Close', 'Volume',
                            'regime', 'hmm_state', 'hmm_score']].copy()
            data = HMMRegimeFeed(
                dataname=df_bt,
                fromdate=fromdate,
                todate=todate,
                open='Open',
                high='High',
                low='Low',
                close='Close',
                volume='Volume',
                openinterest=-1,
                regime='regime',
                hmm_state='hmm_state',
                hmm_score='hmm_score',
            )
            cerebro.adddata(data, name=ticker)
    else:
        # Standard CSV feed (no HMM)
        # yfinance CSV columns: Date(0), Close(1), High(2), Low(3), Open(4), Volume(5)
        csv_kwargs = dict(
            fromdate=fromdate,
            todate=todate,
            dtformat='%Y-%m-%d',
            datetime=0,
            open=4,
            high=2,
            low=3,
            close=1,
            volume=5,
            openinterest=-1,
        )
        for ticker in tickers:
            data = bt.feeds.GenericCSVData(dataname=csv_paths[ticker], **csv_kwargs)
            cerebro.adddata(data, name=ticker)

    _p(f'\n[INFO] Portfolio: {", ".join(tickers)}  ({len(tickers)} assets)')
    if strategy_key == 'hmm_mr':
        _p(f'[INFO] HMM price model   : ON  '
           f'(K={args.hmm_components}, train={args.hmm_train_years}y, '
           f'z_threshold={getattr(args, "hmm_mr_z_threshold", 0.0)})')
    elif args.hmm:
        _p(f'[INFO] HMM regime filter : ON  '
           f'(K={args.hmm_components}, favourable={args.hmm_favourable}, '
           f'gate={args.hmm_gate}, τ={args.hmm_threshold})')
    else:
        _p('[INFO] HMM regime filter : OFF')

    # ---- Strategy ------------------------------------------------------------
    if strategy_key not in REGISTRY:
        raise ValueError(f'Unknown strategy: {strategy_key!r}. Choose from {list(REGISTRY)}')
    _strat_entry = REGISTRY[strategy_key]
    _strat_cls   = _strat_entry['cls']
    _strat_kw    = _strat_entry['build_kwargs'](args)
    _p(f'[INFO] Strategy          : {_strat_entry["label"]}')
    cerebro.addstrategy(_strat_cls, **_strat_kw)

    # ---- Analyzers -----------------------------------------------------------
    # PyFolio analyzer collects daily returns, positions, transactions
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    # Standard analysers for a quick summary
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                        riskfreerate=args.riskfreerate,
                        timeframe=bt.TimeFrame.Days,
                        annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annualreturn')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # ---- Run -----------------------------------------------------------------
    starting_cash = cerebro.broker.getvalue()
    _p('=' * 60)
    _p(f'Starting Portfolio Value : {starting_cash:.2f}')
    _t0 = time.perf_counter()
    results = cerebro.run()
    time_taken = time.perf_counter() - _t0
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - starting_cash) / starting_cash * 100
    _p(f'Final   Portfolio Value : {final_value:.2f}')
    _p(f'Total Return           : {total_return:.2f}%')
    _p(f'Time taken             : {time_taken:.2f}s')
    _p('=' * 60)

    # ---- Extract analyzer results -------------------------------------------
    sharpe_ana = strat.analyzers.sharpe.get_analysis()
    sharpe_val = sharpe_ana.get('sharperatio') or 0.0
    dd_ana     = strat.analyzers.drawdown.get_analysis()
    max_dd     = dd_ana.max.drawdown
    annual     = strat.analyzers.annualreturn.get_analysis()
    ta         = strat.analyzers.trades.get_analysis()

    # CAGR and Calmar
    n_days   = (todate - fromdate).days
    n_years  = n_days / 365.25
    cagr     = (((1.0 + total_return / 100.0) ** (1.0 / max(n_years, 1e-6))) - 1.0) * 100.0
    calmar   = cagr / abs(max_dd) if max_dd else 0.0

    # ---- Print built-in analyzer summaries -----------------------------------
    _p('\n--- Sharpe Ratio ---')
    _p(f"  Sharpe Ratio : {sharpe_val}")

    _p('\n--- Drawdown ---')
    _p(f"  Max Drawdown : {max_dd:.2f}%")
    _p(f"  Max Money    : {dd_ana.max.moneydown:.2f}")

    _p('\n--- Annual Returns ---')
    for year, ret in annual.items():
        _p(f"  {year}: {ret * 100:.2f}%")
    _p(f'  CAGR         : {cagr:.2f}%')
    _p(f'  Calmar       : {calmar:.4f}')

    _p('\n--- Trade Analysis ---')
    total_t = ta.get('total', {})
    _p(f"  Total trades : {total_t.get('total', 0)}")
    _p(f"  Won          : {total_t.get('open', 0) if not ta.get('won') else ta.won.total}")
    _p(f"  Lost         : {total_t.get('closed', 0) if not ta.get('lost') else ta.lost.total}")

    # ---- QuantStats tearsheet (skipped in quiet / optimisation mode) --------
    if not quiet:
        try:
            import quantstats as qs
            pyfoliozer = strat.analyzers.getbyname('pyfolio')
            returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
            returns.index = returns.index.tz_localize(None)
            qs.extend_pandas()
            print('\nQuantStats – Portfolio vs SPY benchmark')
            qs.reports.metrics(returns, mode='full', display=True)
            strategy_key = getattr(args, 'strategy', 'sma')
            strat_label  = REGISTRY.get(strategy_key, {}).get('label', strategy_key.upper())
            hmm_tag      = '_hmm' if args.hmm else ''
            report_path  = os.path.join(
                datas_dir, f'ma-quantstats-report-{strategy_key}{hmm_tag}.html')
            title = (strat_label + (' + HMM Regime' if args.hmm else '') +
                     f' on {", ".join(tickers)}')
            benchmark = tickers[0] if len(tickers) > 0 else 'SPY'
            qs.reports.html(returns, benchmark=benchmark, output=report_path,
                            title=title)
            _p(f'\n[INFO] Full HTML report saved → {report_path}')
        except (ImportError, Exception) as qs_err:
            _p(f'[WARN] QuantStats skipped: {qs_err}')

    # ---- Optional plot (skipped in quiet mode) -------------------------------
    plot_save = getattr(args, 'plot_save', None)
    if (args.plot or plot_save) and not quiet:
        import matplotlib
        if plot_save:
            matplotlib.use('Agg')
        figs = cerebro.plot(style='candle', barup='green', bardown='red',
                            volume=True, numfigs=1)
        if plot_save and figs:
            for i, fig_list in enumerate(figs):
                for j, fig in enumerate(fig_list):
                    path = plot_save if (i == 0 and j == 0) else (
                        f'{os.path.splitext(plot_save)[0]}_{i}_{j}'
                        f'{os.path.splitext(plot_save)[1]}')
                    fig.savefig(path, dpi=150, bbox_inches='tight')
                    _p(f'[INFO] Backtrader plot saved → {path}')

    return {
        'total_return':  total_return,
        'final_value':   final_value,
        'sharpe':        float(sharpe_val),
        'max_drawdown':  float(max_dd),
        'annual_return': float(cagr),
        'calmar':        float(calmar),
        'time_taken':    float(time_taken),
        'trade_count':   int(total_t.get('total', 0)),
    }


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Multi-strategy backtester with QuantStats analysis')

    # Data
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'AGG'],
        help='List of Yahoo Finance tickers to trade (e.g. SPY AGG QQQ GLD)')

    parser.add_argument('--fromdate', default='2019-01-01',
        help='Start date in YYYY-MM-DD')

    parser.add_argument('--todate', default='2025-01-01',
        help='End date in YYYY-MM-DD')

    # Strategy selector
    parser.add_argument('--strategy', default='sma',
        choices=list(REGISTRY),
        help='Trading strategy to run: sma | dema | rsi | macd')

    # SMA / DEMA params (shared)
    parser.add_argument('--fast', type=int, default=10,
        help='Fast period for SMA / DEMA crossover strategies')

    parser.add_argument('--slow', type=int, default=50,
        help='Slow period for SMA / DEMA crossover strategies')

    # HMM Mean-Reversion strategy params
    parser.add_argument('--hmm-mr-z-threshold', type=float, default=0.0,
        dest='hmm_mr_z_threshold',
        help='HMM-MR: std-devs below state mean before entry triggers. '
             '0=any dip, 0.5=half-std dip, 1.0=one-std dip (more conservative)')

    # RSI params
    parser.add_argument('--rsi-period', type=int, default=14,
        dest='rsi_period',
        help='RSI look-back window')

    parser.add_argument('--rsi-oversold', type=int, default=30,
        dest='rsi_oversold',
        help='RSI level that triggers a buy signal (oversold threshold)')

    parser.add_argument('--rsi-overbought', type=int, default=70,
        dest='rsi_overbought',
        help='RSI level that triggers a sell signal (overbought threshold)')

    # MACD params
    parser.add_argument('--macd-fast', type=int, default=12,
        dest='macd_fast',
        help='MACD fast EMA period')

    parser.add_argument('--macd-slow', type=int, default=26,
        dest='macd_slow',
        help='MACD slow EMA period')

    parser.add_argument('--macd-signal', type=int, default=9,
        dest='macd_signal',
        help='MACD signal (smoothing) EMA period')

    parser.add_argument('--stake', type=int, default=100,
        help='Shares per trade')

    parser.add_argument('--cash', type=float, default=100000.0,
        help='Starting cash')

    parser.add_argument('--commission', type=float, default=0.001,
        help='Broker commission (fraction, e.g. 0.001 = 0.1%%)')

    parser.add_argument('--riskfreerate', type=float, default=0.01,
        help='Annual risk-free rate for Sharpe calculation')

    # Regime filter mode
    parser.add_argument('--regime-mode', default='strict',
        dest='regime_mode',
        choices=['strict', 'size', 'score', 'linear'],
        help='HMM regime filter mode: strict = block trades in unfavourable '
             'regime; size = reduce position size by unfav-fraction; '
             'score = continuous position sizing based on HMM state scores; '
             'linear = rank-based linear position sizing (no search needed)')

    parser.add_argument('--unfav-fraction', type=float, default=None,
        dest='unfav_fraction',
        help='Fraction of stake used during unfavourable regime '
             '(only used when --regime-mode=size, e.g. 0.25 = 25%% of stake; omit to let Optuna search)')

    # Output
    parser.add_argument('--printlog', action='store_true', default=False,
        help='Print trade log to stdout')

    parser.add_argument('--plot', '-p', action='store_true', default=False,
        help='Plot the result')

    parser.add_argument('--plot-save', type=str, default=None,
        dest='plot_save',
        metavar='PATH',
        help='Save the backtrader plot to a PNG file (e.g. bt-plot.png). '
             'Implies --plot.')

    # HMM regime filter
    parser.add_argument('--hmm', action='store_true', default=False,
        help='Enable Hidden Markov Model regime filter')

    parser.add_argument('--hmm-train-years', type=float, default=5.0,
        dest='hmm_train_years',
        help='Years of history before --fromdate used to train the HMM')

    parser.add_argument('--hmm-components', type=int, default=7,
        dest='hmm_components',
        help='Number of hidden states in the HMM (K)')

    parser.add_argument('--hmm-favourable', type=int, default=None,
        dest='hmm_favourable',
        metavar='N',
        help='Hard cap on the number of favourable states (default: no cap, '
             'use --hmm-score-threshold instead)')

    parser.add_argument('--score-mean-weight', type=float, default=1.0,
        dest='score_mean_weight',
        help='Weight for the mean-return component in the composite score (0 to disable)')
    parser.add_argument('--score-vol-weight', type=float, default=1.0,
        dest='score_vol_weight',
        help='Weight for the low-volatility component in the composite score (0 to disable)')
    parser.add_argument('--score-upratio-weight', type=float, default=0.0,
        dest='score_upratio_weight',
        help='Weight for the up-ratio component in the composite score (0 to disable)')
    parser.add_argument('--state-positions', type=float, nargs='+', default=None,
        dest='state_positions',
        metavar='POS',
        help='Fixed position sizes per state in score-ranked order (best to worst). '
             'Overrides the linear scoring formula. Number of values must match '
             '--hmm-components (e.g. --state-positions 1.0 1.0 0.6 0.5 0.1 0.1)')

    parser.add_argument('--hmm-score-threshold', type=float, default=1.0,
        dest='hmm_score_threshold',
        help='Minimum composite score [0, sum-of-weights] for a state to be '
             'considered favourable. With default weights the range is [0, 3]. '
             'States scoring below this are excluded (default: 1.0)')

    parser.add_argument('--hmm-threshold', type=float, default=0.99,
        dest='hmm_threshold',
        help='Posterior probability threshold τ for the regime gate')

    parser.add_argument('--hmm-gate', type=str, default='threshold',
        dest='hmm_gate',
        choices=['threshold', 'linear', 'logistic'],
        help='Gate function: threshold | linear | logistic')

    parser.add_argument('--hmm-find-best-rs', action='store_true', default=False,
        dest='hmm_find_best_rs',
        help='Search multiple random seeds and pick the best HMM initialisation')

    parser.add_argument('--hmm-features', nargs='+', default=None,
        dest='hmm_features',
        metavar='FEAT',
        help='HMM input features (choose from: %(choices)s). '
             'Default: ' + ' '.join(DEFAULT_HMM_FEATURES),
        choices=ALL_HMM_FEATURES)

    parser.add_argument('--hmm-pca', type=int, default=None,
        dest='hmm_pca',
        metavar='N',
        help='Apply PCA after scaling, reducing features to N components. '
             'Omit or 0 to skip PCA.')

    parser.add_argument('--hmm-plot', action='store_true', default=False,
        dest='hmm_plot',
        help='Show a regime debug plot for each ticker after HMM training')

    parser.add_argument('--hmm-plot-save', type=str, default=None,
        dest='hmm_plot_save',
        metavar='PATH',
        help=(
            'Save the regime plot to this path instead of (or in addition to) '
            'showing it. Use {ticker} as a placeholder, e.g. '
            'regimes-{ticker}.png'))

    return parser.parse_args(pargs)


if __name__ == '__main__':
    run()
