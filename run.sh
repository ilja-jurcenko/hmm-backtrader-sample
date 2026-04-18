#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

.venv/bin/python walkforward-compare.py \
    --ticker AAPL MSFT NVDA AMZN BRK-B XOM UNH WMT BA KO \
    --wf-start 2010-01-01 --wf-end 2025-01-01 \
    --is-years 2 --oos-years 1 --step 1 --n-trials 30 \
    --strategies sma dema rsi macd \
    --out-dir ./3_wf_results_2010_2025_AAPL_MSFT_NVDA_AMZN_BRK-B_XOM_UNH_WMT_BA_KO
