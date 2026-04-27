#!/bin/bash
#SBATCH --job-name=14_best_multiasset__spy_qqq__is2_oos1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/lustre/home/ilju3280/hmm-backtrader-sample/logs/14_best_multiasset__spy_qqq__is2_oos1_%j.out
#SBATCH --error=/scratch/lustre/home/ilju3280/hmm-backtrader-sample/logs/14_best_multiasset__spy_qqq__is2_oos1_%j.out

# --- environment ---
cd "/scratch/lustre/home/ilju3280/hmm-backtrader-sample"
source .venv/bin/activate

# Prevent OpenBLAS/MKL/OMP from spawning extra threads per worker process.
# Without this, each of the 16 parallel workers tries to create 16 BLAS threads,
# quickly exhausting RLIMIT_NPROC (1000) and causing KeyboardInterrupt in threadpoolctl.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- run ---
python walkforward-compare.py \
    --strategies sma dema rsi macd adx_dm channel_breakout donchian ichimoku parabolic_sar tsmom tsmom_fast turtle vol_adj bollinger kama false_breakout composite_trend \
    --ticker SPY QQQ \
    --wf-start 2010-01-01 \
    --wf-end 2026-04-10 \
    --is-years 2 \
    --oos-years 1 \
    --step 1 \
    --n-trials 20 \
    --objective-metric sharpe \
    --seed 42 \
    --stake 100 \
    --cash 100000 \
    --commission 0.001 \
    --stop-loss 0.05 \
    --take-profit 0.1 \
    --wf-max-workers 1 \
    --out-dir "/scratch/lustre/home/ilju3280/hmm-backtrader-sample/results/phase_5/14_best_multiasset/spy_qqq/is2_oos1" \
    --regime-mode score \
    --hmm-components 4 \
    --hmm-features Returns Range r5 r20 vol log_ret vol_short vol_long atr_norm vol_of_vol vol_lag1 downside_vol vol_z
