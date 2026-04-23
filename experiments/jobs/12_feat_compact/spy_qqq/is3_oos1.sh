#!/bin/bash
#SBATCH --job-name=12_feat_compact__spy_qqq__is3_oos1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/Users/user/Dev/hmm-backtrader-sample/logs/12_feat_compact__spy_qqq__is3_oos1_%j.out
#SBATCH --error=/Users/user/Dev/hmm-backtrader-sample/logs/12_feat_compact__spy_qqq__is3_oos1_%j.out

# --- environment ---
cd "/Users/user/Dev/hmm-backtrader-sample"
source .venv/bin/activate

# --- run ---
python walkforward-compare.py \
    --strategies sma dema rsi macd adx_dm channel_breakout donchian ichimoku parabolic_sar tsmom turtle vol_adj hmm_mr \
    --ticker SPY QQQ \
    --wf-start 2010-01-01 \
    --wf-end 2026-04-10 \
    --is-years 3 \
    --oos-years 1 \
    --step 1 \
    --n-trials 20 \
    --objective-metric sharpe \
    --seed 42 \
    --stake 100 \
    --cash 100000 \
    --commission 0.001 \
    --stop-loss 0.0 \
    --take-profit 0.0 \
    --wf-max-workers 16 \
    --out-dir "/Users/user/Dev/hmm-backtrader-sample/results/12_feat_compact/spy_qqq/is3_oos1" \
    --regime-mode size \
    --hmm-components 3 \
    --hmm-features Returns Range vol log_ret
