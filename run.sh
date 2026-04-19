#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

.venv/bin/python walkforward-compare.py \
    --ticker AAPL MSFT NVDA AMZN BRK-B XOM UNH WMT BA KO \
    --wf-start 2010-01-01 --wf-end 2026-04-10 \
    --is-years 2 --oos-years 1 --step 1 --n-trials 20 \
    --strategies sma dema rsi macd \
    --out-dir ./wf_results_pca3 \
    --regime-mode size \
    --hmm-features Returns Range r5 r20 vol log_ret vol_short vol_long atr_norm vol_of_vol vol_lag1 downside_vol vol_z \
    --hmm-pca 4 \
    --objective-metric sharpe