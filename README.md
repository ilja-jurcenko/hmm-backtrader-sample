# HMM Backtrader Sample

A collection of algorithmic trading scripts built on [Backtrader](https://www.backtrader.com/) that use Hidden Markov Models (HMM) as a regime filter, with Optuna-powered hyperparameter optimisation and walk-forward validation.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Strategies

All strategies live in `strategies/` and share the same HMM regime gate, stop-loss and take-profit hooks from `BaseRegimeStrategy`.

| Key | Class | Description |
|---|---|---|
| `sma` | `SmaCrossOver` | Simple Moving Average fast/slow crossover |
| `dema` | `DemaCrossOver` | Double EMA fast/slow crossover |
| `rsi` | `RsiStrategy` | RSI oversold-bounce entry, overbought/midline exit |
| `macd` | `MacdStrategy` | MACD line × signal-line crossover |
| `hmm_mr` | `HmmMeanReversionStrategy` | HMM price-level model; buy below state mean |
| `adx_dm` | `AdxDmStrategy` | +DI/−DI crossover gated by ADX trend-strength filter |
| `channel_breakout` | `ChannelBreakoutStrategy` | N-period high/low price-channel breakout |
| `donchian` | `DonchianStrategy` | Dual-period Donchian channel (separate entry/exit lookbacks) |
| `ichimoku` | `IchimokuStrategy` | Tenkan/Kijun cross confirmed by price above Kumo cloud |
| `parabolic_sar` | `ParabolicSarStrategy` | Parabolic SAR bullish/bearish flip |
| `tsmom` | `TsmomStrategy` | 12-1 month time-series momentum (sign of past return) |
| `turtle` | `TurtleStrategy` | Turtle System 1: N-day entry channel, M-day exit + ATR stop |
| `vol_adj` | `VolatilityAdjustedStrategy` | Keltner Channel: EMA ± ATR-scaled bands breakout |

---

## Scripts

### `ma-quantstats.py` — Backtest + QuantStats Tearsheet

Runs any strategy on a multi-asset portfolio with an optional HMM regime filter and produces a full QuantStats HTML report.

#### Data & universe

| Argument | Default | Description |
|---|---|---|
| `--tickers` | `SPY AGG` | Space-separated list of Yahoo Finance tickers |
| `--fromdate` | `2019-01-01` | Backtest start date |
| `--todate` | `2025-01-01` | Backtest end date |

#### Strategy selector

| Argument | Default | Choices |
|---|---|---|
| `--strategy` | `sma` | `sma dema rsi macd hmm_mr adx_dm channel_breakout donchian ichimoku parabolic_sar tsmom turtle vol_adj` |

#### Strategy-specific parameters

**SMA / DEMA**

| Argument | Default | Description |
|---|---|---|
| `--fast` | `10` | Fast MA period |
| `--slow` | `50` | Slow MA period |

**RSI**

| Argument | Default | Description |
|---|---|---|
| `--rsi-period` | `14` | RSI look-back window |
| `--rsi-oversold` | `30` | Buy threshold (RSI rises back above this) |
| `--rsi-overbought` | `70` | Sell threshold (RSI crosses above this) |

**MACD**

| Argument | Default | Description |
|---|---|---|
| `--macd-fast` | `12` | Fast EMA period |
| `--macd-slow` | `26` | Slow EMA period |
| `--macd-signal` | `9` | Signal EMA period |

**HMM Mean-Reversion (`hmm_mr`)**

| Argument | Default | Description |
|---|---|---|
| `--hmm-mr-z-threshold` | `0.0` | Std-devs below state mean required before entry (0 = any dip) |

**ADX + Directional Movement (`adx_dm`)**

| Argument | Default | Description |
|---|---|---|
| `--adx-period` | `14` | ADX / DI look-back period |
| `--adx-threshold` | `25.0` | Minimum ADX value to allow entry (trend-strength gate) |

**Channel Breakout (`channel_breakout`)**

| Argument | Default | Description |
|---|---|---|
| `--channel-period` | `20` | Look-back for highest high / lowest low |

**Donchian Channel (`donchian`)**

| Argument | Default | Description |
|---|---|---|
| `--donchian-entry` | `20` | Entry channel look-back (breakout high) |
| `--donchian-exit` | `10` | Exit channel look-back (pullback low) |

**Ichimoku Cloud (`ichimoku`)**

| Argument | Default | Description |
|---|---|---|
| `--ichimoku-tenkan` | `9` | Tenkan-sen (conversion line) period |
| `--ichimoku-kijun` | `26` | Kijun-sen (base line) period |
| `--ichimoku-senkou` | `52` | Senkou Span B (cloud) period |

**Parabolic SAR (`parabolic_sar`)**

| Argument | Default | Description |
|---|---|---|
| `--psar-af` | `0.02` | Acceleration factor (initial value and increment) |
| `--psar-max-af` | `0.20` | Maximum acceleration factor |

**Time-Series Momentum (`tsmom`)**

| Argument | Default | Description |
|---|---|---|
| `--tsmom-lookback` | `252` | Total return look-back in trading days (≈ 12 months) |
| `--tsmom-skip` | `21` | Most-recent bars to skip (≈ 1 month short-term reversal window) |

**Turtle (`turtle`)**

| Argument | Default | Description |
|---|---|---|
| `--turtle-entry` | `20` | Entry channel: N-day highest high |
| `--turtle-exit` | `10` | Exit channel: M-day lowest low |
| `--turtle-atr` | `20` | ATR period for trailing stop |
| `--turtle-atr-mult` | `2.0` | ATR multiplier for trailing stop (0 = disabled) |

**Volatility-Adjusted / Keltner (`vol_adj`)**

| Argument | Default | Description |
|---|---|---|
| `--vol-period` | `20` | EMA period for channel midline |
| `--vol-atr-period` | `14` | ATR period for band width |
| `--vol-atr-mult` | `1.5` | ATR multiplier for upper/lower bands |

#### Risk management

| Argument | Default | Description |
|---|---|---|
| `--stop-loss` | `0.02` | Stop-loss as fraction of entry price (0 = disabled) |
| `--take-profit` | `0.10` | Take-profit as fraction of entry price (0 = disabled) |
| `--stake` | `100` | Shares per trade |
| `--cash` | `100000` | Starting cash |
| `--commission` | `0.001` | Commission rate (fraction, e.g. 0.001 = 0.1%) |
| `--riskfreerate` | `0.01` | Annual risk-free rate for Sharpe calculation |

#### HMM regime filter

| Argument | Default | Description |
|---|---|---|
| `--hmm` | off | Enable HMM regime filter |
| `--hmm-train-years` | `5.0` | Years of history before `--fromdate` used to train the HMM |
| `--hmm-components` | `7` | Number of hidden states K |
| `--hmm-favourable` | _(no cap)_ | Hard cap on favourable states; omit to use score threshold instead |
| `--hmm-score-threshold` | `1.0` | Minimum composite score [0, sum-of-weights] for a state to be favourable |
| `--hmm-threshold` | `0.99` | Posterior probability threshold τ for the binary regime gate |
| `--hmm-gate` | `threshold` | Gate function: `threshold`, `logistic` |
| `--hmm-find-best-rs` | off | Search multiple random seeds and keep the best HMM initialisation |
| `--hmm-features` | _(default set)_ | HMM input features (see [HMM Features](#hmm-features)) |
| `--hmm-pca` | _(off)_ | Reduce features to N PCA components before fitting |
| `--regime-mode` | `strict` | Regime filter mode (see [HMM Regime Modes](#hmm-regime-modes)) |
| `--unfav-fraction` | _(search)_ | Position fraction in unfavourable regime (`size` mode only) |
| `--score-mean-weight` | `1.0` | Weight for mean-return component in state scoring |
| `--score-vol-weight` | `1.0` | Weight for low-volatility component in state scoring |
| `--score-upratio-weight` | `0.0` | Weight for up-ratio component in state scoring |
| `--state-positions` | _(computed)_ | Fixed position sizes per state, score-ranked best→worst |
| `--hmm-max-pos-size` | `2.0` | Position size multiplier for best-scored states |
| `--hmm-min-pos-size` | `0.01` | Position size multiplier for worst-scored states |
| `--hmm-dynamic-scoring` | off | Re-score states bar-by-bar using expanding/rolling window |
| `--hmm-dynamic-window` | `0` | Rolling window for dynamic scoring (0 = expanding) |
| `--hmm-plot` | off | Show regime debug plot after training |
| `--hmm-plot-save` | _(off)_ | Save regime plot to file (use `{ticker}` placeholder) |

#### Output

| Argument | Default | Description |
|---|---|---|
| `--printlog` | off | Print per-trade log to stdout |
| `--plot` | off | Show backtrader chart |
| `--plot-save` | _(off)_ | Save backtrader chart to PNG |

#### Quick-start examples

```bash
# Defaults (SMA on SPY+AGG, 2019–2025)
python ma-quantstats.py

# Ichimoku on multiple tickers with HMM filter
python ma-quantstats.py --strategy ichimoku --tickers SPY QQQ --hmm

# Donchian with 5% stop-loss and 15% take-profit
python ma-quantstats.py --strategy donchian \
    --donchian-entry 20 --donchian-exit 10 \
    --stop-loss 0.05 --take-profit 0.15

# Channel breakout with score-based HMM sizing
python ma-quantstats.py --strategy channel_breakout \
    --hmm --regime-mode score --hmm-components 6

# Turtle with ATR stop, custom periods
python ma-quantstats.py --strategy turtle \
    --turtle-entry 40 --turtle-exit 20 --turtle-atr-mult 3.0

# Time-series momentum, 6-month lookback
python ma-quantstats.py --strategy tsmom \
    --tsmom-lookback 126 --tsmom-skip 21

# Volatility-adjusted (Keltner) with wider bands
python ma-quantstats.py --strategy vol_adj \
    --vol-atr-mult 2.0 --vol-period 30
```

---

### `optimize-hmm.py` — Single-Period HMM Hyperparameter Optimisation

Uses Optuna to search HMM hyperparameters that maximise a chosen metric on an in-sample window, then validates on a held-out out-of-sample window. Strategy indicator params are **fixed**; only the HMM parameters are searched.

**Optuna search space:** `hmm_components`, `hmm_score_threshold`, `hmm_threshold`, `hmm_gate`, `hmm_train_years`, `unfav_fraction` (when `--regime-mode size`), feature subset (when `--hmm-features` is not fixed), per-state position sizes (when `--search-state-positions`).

#### Data & period

| Argument | Default | Description |
|---|---|---|
| `--ticker` | `SPY` | One or more tickers (space-separated) |
| `--fromdate` | `2019-01-01` | Start of full period |
| `--split-date` | `2022-07-01` | IS / OOS split date |
| `--todate` | `2025-01-01` | End of full period |

#### Optimisation control

| Argument | Default | Description |
|---|---|---|
| `--n-trials` | `60` | Optuna trials |
| `--seed` | `42` | Random seed for Optuna sampler |
| `--objective-metric` | `total_return` | Metric to maximise: `total_return`, `sharpe`, `calmar` |
| `--hmm-score-threshold` | _(search)_ | Pin score threshold; omit to include in search |
| `--hmm-components` | _(search)_ | Pin K; omit to search over [3, 6] |
| `--hmm-features` | _(search)_ | Pin feature set; omit to search over all combinations |
| `--hmm-pca` | _(off)_ | Fix PCA components applied after feature scaling |
| `--search-state-positions` | off | Let Optuna search per-state position sizes |
| `--state-positions` | _(computed)_ | Fix per-state position sizes (overrides search) |

#### Broker & risk

| Argument | Default | Description |
|---|---|---|
| `--fast` | `9` | SMA/DEMA fast period (fixed during HMM search) |
| `--slow` | `30` | SMA/DEMA slow period |
| `--stake` | `100` | Shares per trade |
| `--cash` | `100000` | Starting cash |
| `--commission` | `0.001` | Commission rate |
| `--stop-loss` | `0.02` | Stop-loss fraction (0 = disabled) |
| `--take-profit` | `0.10` | Take-profit fraction (0 = disabled) |

All strategy-specific params (`--rsi-*`, `--macd-*`, `--adx-*`, `--donchian-*`, `--ichimoku-*`, `--psar-*`, `--tsmom-*`, `--turtle-*`, `--vol-*`) are accepted with the same defaults as `ma-quantstats.py`.

#### HMM & regime

All `--hmm-*` and `--regime-mode` arguments from `ma-quantstats.py` are supported. The `--hmm` flag is implied — the script always runs with HMM enabled.

```bash
# Defaults
python optimize-hmm.py

# Channel breakout, optimise on Sharpe
python optimize-hmm.py --ticker SPY --strategy channel_breakout \
    --objective-metric sharpe --n-trials 80

# Ichimoku with score-based sizing, search state positions
python optimize-hmm.py --ticker SPY --strategy ichimoku \
    --regime-mode score --search-state-positions --hmm-components 6

# Donchian, fix features and components
python optimize-hmm.py --ticker SPY --strategy donchian \
    --hmm-features log_ret vol_short atr_norm --hmm-components 5

# Multi-asset portfolio
python optimize-hmm.py --ticker SPY QQQ AGG --strategy sma
```

---

### `walkforward-hmm.py` — Walk-Forward HMM Optimisation

Slides a rolling window over the full history, running IS-optimise → OOS-validate at each step and accumulating out-of-sample results.

**Window design:**
```
wf_start ──[IS: is_years]──[OOS: oos_years]── wf_start + is_years + oos_years
             step ──[IS]──[OOS]──
                    step ──[IS]──[OOS]── ... wf_end
```

#### Walk-forward window

| Argument | Default | Description |
|---|---|---|
| `--ticker` | `SPY` | One or more tickers |
| `--wf-start` | `2019-01-01` | Walk-forward overall start date |
| `--wf-end` | `2026-01-01` | Walk-forward overall end date |
| `--is-years` | `3` | In-sample window length (years) |
| `--oos-years` | `2` | Out-of-sample window length (years) |
| `--step` | `1` | Years to advance the window per iteration |
| `--n-trials` | `40` | Optuna trials per window |
| `--max-workers` | `0` | Parallel window workers (0 = auto, 1 = sequential) |
| `--window-log-dir` | _(auto)_ | Directory for per-window progress log files |
| `--objective-metric` | `total_return` | Metric to maximise: `total_return`, `sharpe`, `calmar` |
| `--seed` | `42` | Random seed |

#### Strategy

| Argument | Default | Choices |
|---|---|---|
| `--strategy` | `sma` | `sma dema rsi macd hmm_mr adx_dm channel_breakout donchian ichimoku parabolic_sar tsmom turtle vol_adj` |

All strategy-specific params (`--fast`, `--slow`, `--rsi-*`, `--macd-*`, `--adx-*`, `--donchian-*`, `--ichimoku-*`, `--psar-*`, `--tsmom-*`, `--turtle-*`, `--vol-*`, `--hmm-mr-z-threshold`) are accepted with the same defaults as `ma-quantstats.py`.

#### Broker & risk

| Argument | Default | Description |
|---|---|---|
| `--stake` | `100` | Shares per trade |
| `--cash` | `100000` | Starting cash |
| `--commission` | `0.001` | Commission rate |
| `--stop-loss` | `0.02` | Stop-loss fraction (0 = disabled) |
| `--take-profit` | `0.10` | Take-profit fraction (0 = disabled) |

#### HMM & regime

All `--hmm-*`, `--regime-mode`, `--hmm-score-threshold`, `--hmm-components`, `--hmm-features`, `--hmm-pca`, `--state-positions`, `--search-state-positions`, `--hmm-favourable`, `--hmm-max-pos-size`, `--hmm-min-pos-size`, `--hmm-dynamic-scoring`, `--hmm-dynamic-window` arguments from `ma-quantstats.py` are supported.

```bash
# Minimal
python walkforward-hmm.py --ticker SPY

# Donchian walk-forward, strict HMM filter
python walkforward-hmm.py --ticker SPY --strategy donchian \
    --wf-start 2015-01-01 --wf-end 2025-01-01 \
    --is-years 3 --oos-years 1 --n-trials 50

# Ichimoku, optimise for Calmar, score-based HMM sizing
python walkforward-hmm.py --ticker SPY --strategy ichimoku \
    --regime-mode score --objective-metric calmar --hmm-components 6

# Turtle with fixed features and wider stop
python walkforward-hmm.py --ticker SPY --strategy turtle \
    --stop-loss 0.05 --hmm-features log_ret vol_short atr_norm

# Channel breakout, parallel workers
python walkforward-hmm.py --ticker SPY --strategy channel_breakout \
    --max-workers 4 --n-trials 60
```

---

### `walkforward-compare.py` — Parallel Multi-Strategy Walk-Forward Comparison

Runs `walkforward-hmm.py` for each chosen strategy **in parallel** (one subprocess per strategy), writes results to `.txt` files, then prints a unified side-by-side OOS summary.

#### Comparison control

| Argument | Default | Description |
|---|---|---|
| `--strategies` | `sma dema rsi macd` | Strategies to compare (space-separated, any subset of all 13) |
| `--out-dir` | `./wf_results` | Directory for output `.txt` files |
| `--wf-max-workers` | `0` | Max parallel window workers **inside** each strategy subprocess |

#### Walk-forward window (passed through to each subprocess)

Same arguments as `walkforward-hmm.py`: `--ticker`, `--wf-start`, `--wf-end`, `--is-years`, `--oos-years`, `--step`, `--n-trials`, `--seed`, `--objective-metric`.

#### Broker & risk (passed through)

`--stake`, `--cash`, `--commission`, `--stop-loss`, `--take-profit`

#### Strategy params (passed through)

All strategy-specific params are forwarded: `--fast`, `--slow`, `--rsi-*`, `--macd-*`, `--adx-*`, `--channel-period`, `--donchian-*`, `--ichimoku-*`, `--psar-*`, `--tsmom-*`, `--turtle-*`, `--vol-*`, `--hmm-mr-z-threshold`.

#### HMM & regime (passed through)

`--regime-mode`, `--unfav-fraction`, `--hmm-score-threshold`, `--hmm-components`, `--hmm-features`, `--hmm-pca`, `--state-positions`, `--search-state-positions`, `--hmm-favourable`, `--hmm-max-pos-size`, `--hmm-min-pos-size`, `--hmm-dynamic-scoring`, `--hmm-dynamic-window`.

```bash
# Compare the four classic strategies
python walkforward-compare.py \
    --ticker SPY \
    --wf-start 2015-01-01 --wf-end 2025-01-01 \
    --is-years 3 --oos-years 1 --n-trials 50 \
    --strategies sma dema rsi macd \
    --out-dir ./wf_results

# Compare trend-following strategies with wider stop-loss
python walkforward-compare.py \
    --strategies channel_breakout donchian turtle ichimoku \
    --stop-loss 0.05 --take-profit 0.10 \
    --out-dir ./wf_results_trend

# Compare all 13 strategies with score-based HMM sizing
python walkforward-compare.py \
    --strategies sma dema rsi macd adx_dm channel_breakout donchian \
                 ichimoku parabolic_sar tsmom turtle vol_adj hmm_mr \
    --regime-mode score --hmm-components 6 \
    --out-dir ./wf_results_all
```

Output files are saved as `<out-dir>/<strategy>.txt`.

---

## HMM Regime Modes

The `--regime-mode` flag controls how the HMM signal is converted into position sizing.

| Mode | Flag | Description |
|---|---|---|
| **Strict** | `--regime-mode strict` | Binary gate: no trades in unfavourable states; full stake in favourable states |
| **Size** | `--regime-mode size` | Reduced stake (`--unfav-fraction`) in unfavourable states, full stake in favourable states |
| **Score** | `--regime-mode score` | Continuous sizing: each bar's position = Σ P(state_k) × pos_size_k. State pos_sizes are derived from composite scores or set via `--state-positions` |
| **Linear** | `--regime-mode linear` | Rank-based linear sizing: best state → `--hmm-max-pos-size`, worst → `--hmm-min-pos-size`, intermediate states interpolated linearly |

### Choosing a mode

- **`strict`** — most conservative; eliminates all trades when the HMM sees a bad regime. Best when regime prediction is highly reliable.
- **`size`** — softer version of strict; keeps a toe in the market during uncertain periods. Tune with `--unfav-fraction` (e.g. `0.25` = 25% of normal stake).
- **`score`** — treats the HMM as a continuous signal; naturally handles uncertainty. Combine with `--hmm-max-pos-size 2.0` to allow leveraged sizing in the best state.
- **`linear`** — simplest continuous mode; no Optuna search needed (rank order is deterministic). Good baseline for comparing against `score`.

### State scoring

States are ranked by a composite score:

```
score = mean_weight × norm(mean_return)
      + vol_weight  × (1 − norm(volatility))
      + up_ratio_weight × norm(up_ratio)
```

Weights are controlled by `--score-mean-weight`, `--score-vol-weight`, `--score-upratio-weight` (defaults 1.0 / 1.0 / 0.0).

### Dynamic scoring

`--hmm-dynamic-scoring` re-scores states bar-by-bar using the expanding (or rolling, via `--hmm-dynamic-window N`) window of observed test returns, updating position sizes as new data arrives.

---

## HMM Features

The `--hmm-features` argument selects which features are fed to the Gaussian HMM. Omit it to let Optuna search over all combinations.

| Feature | Description |
|---|---|
| `log_ret` | Daily log return of Close |
| `r5` | 5-day log return |
| `r20` | 20-day log return |
| `vol_short` | 5-day rolling volatility (std of log returns) |
| `vol_long` | 20-day rolling volatility |
| `atr_norm` | 14-day normalised Average True Range |
| `Returns` | Daily percentage change of Close |
| `Range` | (High / Low) − 1  (intraday range proxy) |
| `vol` | (High − Low) / Close |
| `vol_of_vol` | 20-day rolling std of `vol_short` |
| `vol_lag1` | `vol_short` lagged by 1 day |
| `downside_vol` | 20-day rolling std of negative log returns only |
| `vol_z` | Z-scored `vol_short` vs its 60-day rolling mean/std |

**Default feature set** (used when `--hmm-features` is omitted and Optuna is not searching):
`log_ret r5 r20 vol_short atr_norm`

**Example fixed feature sets:**

```bash
# Volatility-focused (good with trend-following strategies)
--hmm-features log_ret vol_short atr_norm vol_of_vol

# Return-momentum focused
--hmm-features log_ret r5 r20

# Full feature set (slower, may overfit on short histories)
--hmm-features log_ret r5 r20 vol_short vol_long atr_norm vol_of_vol downside_vol vol_z
```

---

## HMM Variants — Testing Reference

Common configurations for exploring HMM behaviour:

```bash
# --- Gate types ---

# Hard threshold gate (default): trade only when P(favourable) ≥ 0.99
python ma-quantstats.py --hmm --hmm-gate threshold --hmm-threshold 0.99

# Softer threshold: allow trades when P(favourable) ≥ 0.60
python ma-quantstats.py --hmm --hmm-gate threshold --hmm-threshold 0.60

# Logistic (smooth) gate around τ=0.70
python ma-quantstats.py --hmm --hmm-gate logistic --hmm-threshold 0.70

# --- Number of states ---

# Fewer states (coarser regimes)
python ma-quantstats.py --hmm --hmm-components 3

# More states (finer regimes, needs more training data)
python ma-quantstats.py --hmm --hmm-components 8 --hmm-train-years 10

# Auto-search K via BIC (walkforward / optimize)
python walkforward-hmm.py --ticker SPY  # --hmm-components omitted → Optuna searches [3,6]

# --- Regime modes ---

# Binary strict gate
python ma-quantstats.py --hmm --regime-mode strict

# Half-stake in unfavourable regimes
python ma-quantstats.py --hmm --regime-mode size --unfav-fraction 0.5

# Score-based continuous sizing
python ma-quantstats.py --hmm --regime-mode score \
    --hmm-max-pos-size 2.0 --hmm-min-pos-size 0.01

# Linear rank-based sizing (no search needed)
python ma-quantstats.py --hmm --regime-mode linear \
    --hmm-max-pos-size 1.5 --hmm-min-pos-size 0.1

# --- State selection ---

# Restrict to the single best state
python ma-quantstats.py --hmm --hmm-favourable 1

# Top-3 states, score threshold 1.5
python ma-quantstats.py --hmm --hmm-favourable 3 --hmm-score-threshold 1.5

# Manual position sizes per state (score-ranked best→worst, K=6)
python ma-quantstats.py --hmm --hmm-components 6 \
    --state-positions 1.0 0.8 0.5 0.2 0.05 0.0

# --- Feature sets ---

# Fix features (skip Optuna feature search)
python ma-quantstats.py --hmm --hmm-features log_ret vol_short atr_norm

# PCA compression (3 components)
python ma-quantstats.py --hmm --hmm-pca 3

# --- State scoring weights ---

# Prioritise low volatility over returns
python ma-quantstats.py --hmm \
    --score-mean-weight 0.5 --score-vol-weight 2.0 --score-upratio-weight 0.5

# Return-only scoring
python ma-quantstats.py --hmm \
    --score-mean-weight 1.0 --score-vol-weight 0.0 --score-upratio-weight 0.0

# --- Dynamic scoring ---

# Expanding-window dynamic scoring
python ma-quantstats.py --hmm --regime-mode score --hmm-dynamic-scoring

# 60-bar rolling-window dynamic scoring
python ma-quantstats.py --hmm --regime-mode score \
    --hmm-dynamic-scoring --hmm-dynamic-window 60

# --- Combined examples ---

# Channel breakout + score HMM, optimise for Calmar
python optimize-hmm.py --ticker SPY --strategy channel_breakout \
    --regime-mode score --objective-metric calmar \
    --hmm-components 6 --n-trials 100

# Ichimoku walk-forward, strict HMM, top-2 states, 8y training
python walkforward-hmm.py --ticker SPY --strategy ichimoku \
    --regime-mode strict --hmm-favourable 2 --hmm-train-years 8 \
    --wf-start 2015-01-01 --wf-end 2025-01-01

# Turtle vs Donchian comparison with score-based sizing
python walkforward-compare.py \
    --strategies turtle donchian \
    --regime-mode score --hmm-components 5 \
    --stop-loss 0.05 --take-profit 0.10 \
    --out-dir ./wf_results_td
```

---

## Data

Market data is auto-downloaded from Yahoo Finance and cached as CSV files in `datas/`. Pre-downloaded CSVs for S&P 500 constituents and common ETFs (SPY, QQQ, AGG, GLD, AAPL, NVDA, GOOGL, etc.) from 2000–2026 are included.
