#!/bin/bash
#SBATCH --job-name=02_size__spy_qqq__is3_oos1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/lustre/home/ilju3280/hmm-backtrader-sample/logs/02_size__spy_qqq__is3_oos1_%j.out
#SBATCH --error=/scratch/lustre/home/ilju3280/hmm-backtrader-sample/logs/02_size__spy_qqq__is3_oos1_%j.out

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
    --is-years 3 \
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
    --out-dir "/scratch/lustre/home/ilju3280/hmm-backtrader-sample/results/phase_1/02_size/spy_qqq/is3_oos1" \
    --regime-mode size
