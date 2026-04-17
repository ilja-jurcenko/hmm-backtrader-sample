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

## Scripts

### `ma-quantstats.py` — SMA Crossover + HMM Filter + QuantStats Report

Runs a multi-asset SMA crossover strategy with an optional HMM regime filter and produces a full QuantStats HTML tearsheet.

```bash
# Defaults (SPY)
python ma-quantstats.py

# Multiple tickers
python ma-quantstats.py --tickers SPY AGG QQQ GLD

# Custom MA periods
python ma-quantstats.py --tickers AAPL MSFT GOOG --fast 20 --slow 100

# Show backtrader plot
python ma-quantstats.py --plot

# With HMM regime filter (gates trades to favourable hidden states)
python ma-quantstats.py --hmm
python ma-quantstats.py --hmm --hmm-components 6 --hmm-favourable 3
python ma-quantstats.py --hmm --hmm-threshold 0.6 --hmm-gate logistic
```

---

### `optimize-hmm.py` — Single-Period HMM Hyperparameter Optimisation

Uses Optuna to search HMM + SMA parameters that maximise return on an in-sample window, then validates on a held-out out-of-sample window.

**Optimised parameters:** `fast`, `slow`, `hmm_components`, `hmm_favourable`, `hmm_threshold`, `hmm_gate`, `hmm_train_years`

| Argument | Default | Description |
|---|---|---|
| `--ticker` | `SPY` | One or more tickers (space-separated) |
| `--fromdate` | `2019-01-01` | Start of full period |
| `--split-date` | `2022-07-01` | IS / OOS split |
| `--todate` | `2025-01-01` | End of full period |
| `--n-trials` | `60` | Optuna trials |
| `--fast` | `9` | Initial SMA fast period |
| `--slow` | `30` | Initial SMA slow period |
| `--stake` | `100` | Number of shares per trade |
| `--cash` | `100000` | Starting cash |
| `--commission` | `0.001` | Commission rate |
| `--seed` | `42` | Random seed |
| `--hmm-score-threshold` | _(search)_ | Fix HMM score threshold instead of optimising it |

```bash
# Defaults
python optimize-hmm.py

# Custom ticker and trials
python optimize-hmm.py --ticker SPY --n-trials 100

# Custom split date
python optimize-hmm.py --ticker SPY --n-trials 50 --split-date 2022-01-01

# Multi-asset portfolio
python optimize-hmm.py --ticker SPY QQQ AGG
```

---

### `walkforward-hmm.py` — Walk-Forward HMM Optimisation

Slides an expanding window over the full history, running IS-optimise → OOS-validate at each step.

**Window design:**
- Full window: `wf_start + i*step` → `wf_start + i*step + is_years + oos_years`
- In-sample: `window_start` → `window_start + is_years`
- Out-of-sample: `window_start + is_years` → `window_end`

| Argument | Default | Description |
|---|---|---|
| `--ticker` | `SPY` | One or more tickers (space-separated) |
| `--wf-start` | `2019-01-01` | Walk-forward overall start date |
| `--wf-end` | `2026-01-01` | Walk-forward overall end date |
| `--is-years` | `3` | In-sample window length (years) |
| `--oos-years` | `2` | Out-of-sample window length (years) |
| `--step` | `1` | Years to advance the window each iteration |
| `--n-trials` | `40` | Optuna trials per window |
| `--strategy` | `sma` | Strategy: `sma`, `dema`, `rsi`, `macd`, `hmm_mr` |
| `--fast` | `10` | SMA/DEMA fast period |
| `--slow` | `30` | SMA/DEMA slow period |
| `--rsi-period` | `14` | RSI period |
| `--rsi-oversold` | `30` | RSI oversold threshold |
| `--rsi-overbought` | `70` | RSI overbought threshold |
| `--macd-fast` | `12` | MACD fast period |
| `--macd-slow` | `26` | MACD slow period |
| `--macd-signal` | `9` | MACD signal period |
| `--hmm-mr-z-threshold` | `0.0` | HMM-MR: std-devs below state mean required before entry |
| `--stake` | `100` | Shares per trade |
| `--cash` | `100000` | Starting cash |
| `--commission` | `0.001` | Commission rate |
| `--seed` | `42` | Random seed |
| `--hmm-score-threshold` | _(search)_ | Fix HMM score threshold instead of optimising it |

```bash
# Minimal
python walkforward-hmm.py --ticker SPY

# Full example
python walkforward-hmm.py \
  --ticker SPY QQQ \
  --wf-start 2019-01-01 \
  --wf-end 2026-01-01 \
  --is-years 3 \
  --oos-years 2 \
  --step 1 \
  --n-trials 50

# Different strategy
python walkforward-hmm.py --ticker SPY --strategy rsi --n-trials 60

# HMM mean-reversion strategy
python walkforward-hmm.py --ticker SPY --strategy hmm_mr --hmm-mr-z-threshold 1.0
```

---

### `walkforward-compare.py` — Parallel Multi-Strategy Walk-Forward Comparison

Runs walk-forward optimisation for multiple strategies **in parallel** (each as a subprocess), saves results to `.txt` files, then prints a side-by-side summary.

| Argument | Default | Description |
|---|---|---|
| `--strategies` | _(all)_ | Strategies to compare (space-separated) |
| `--out-dir` | `./wf_results` | Directory for output `.txt` files |
| `--ticker` | `SPY` | One or more tickers |
| `--wf-start` | `2015-01-01` | Walk-forward start date |
| `--wf-end` | `2025-01-01` | Walk-forward end date |
| `--is-years` | `3` | In-sample window length (years) |
| `--oos-years` | `1` | Out-of-sample window length (years) |
| `--step` | `1` | Step size in years |
| `--n-trials` | `40` | Optuna trials per window |
| `--fast` | `10` | SMA/DEMA fast period |
| `--slow` | `30` | SMA/DEMA slow period |
| `--rsi-period` | `14` | RSI period |
| `--rsi-oversold` | `30` | RSI oversold threshold |
| `--rsi-overbought` | `70` | RSI overbought threshold |
| `--macd-fast` | `12` | MACD fast period |
| `--macd-slow` | `26` | MACD slow period |
| `--macd-signal` | `9` | MACD signal period |
| `--stake` | `100` | Shares per trade |
| `--cash` | `100000` | Starting cash |
| `--commission` | `0.001` | Commission rate |

```bash
# Compare all strategies on SPY
python walkforward-compare.py \
    --ticker SPY \
    --wf-start 2015-01-01 --wf-end 2025-01-01 \
    --is-years 3 --oos-years 1 --step 1 --n-trials 50 \
    --strategies sma dema rsi macd \
    --out-dir ./wf_results

# Quick comparison with fewer trials
python walkforward-compare.py --ticker SPY --n-trials 20 --strategies sma rsi macd
```

Output files are saved to `--out-dir` as `sma.txt`, `dema.txt`, etc.

---

## Strategies

| Name | Description |
|---|---|
| `sma` | Simple Moving Average crossover |
| `dema` | Double Exponential Moving Average crossover |
| `rsi` | RSI mean-reversion |
| `macd` | MACD signal crossover |
| `hmm_mr` | HMM-gated mean reversion |

---

## Data

Market data is auto-downloaded from Yahoo Finance and cached as CSV files in `datas/`. Pre-downloaded CSVs for common tickers (SPY, QQQ, AGG, GLD, AAPL, NVDA, GOOGL, DELL, TSM) from 2000–2026 are included.
