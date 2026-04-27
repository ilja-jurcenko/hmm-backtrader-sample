#!/bin/bash
# Quick SL/TP comparison test — runs two walk-forward jobs via srun.
#
# Combos tested:
#   A) SL=5%  TP=10%
#   B) SL=5%  TP=15%
#
# Uses a smaller window (2018-2026), 5 Optuna trials, and 8 CPUs to keep
# each run fast.  All 16 strategies, SPY+QQQ, IS=2/OOS=1.
#
# Usage:
#   cd /scratch/lustre/home/ilju3280/hmm-backtrader-sample
#   bash experiments/test_sltp.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

STRATEGIES="sma dema rsi macd adx_dm channel_breakout donchian ichimoku parabolic_sar tsmom tsmom_fast turtle vol_adj bollinger kama false_breakout composite_trend"
TICKERS="SPY QQQ"
WF_START="2018-01-01"
WF_END="2026-04-10"
IS_YEARS=2
OOS_YEARS=1
STEP=1
N_TRIALS=5
CPUS=8
CASH=100000
STAKE=100
COMMISSION=0.001

run_combo() {
    local sl="$1"
    local tp="$2"
    local tag="sl${sl/./}_tp${tp/./}"   # e.g. sl005_tp010
    local out_dir="$REPO/results/sltp_test/${tag}"
    mkdir -p "$out_dir"

    echo ""
    echo "============================================================"
    echo "  Combo: SL=${sl} ($(echo "$sl * 100" | bc -l | xargs printf '%.0f')%)  TP=${tp} ($(echo "$tp * 100" | bc -l | xargs printf '%.0f')%)"
    echo "  Output: results/sltp_test/${tag}"
    echo "============================================================"

    srun --ntasks=1 --cpus-per-task="$CPUS" \
        python walkforward-compare.py \
            --strategies $STRATEGIES \
            --ticker $TICKERS \
            --wf-start "$WF_START" \
            --wf-end "$WF_END" \
            --is-years "$IS_YEARS" \
            --oos-years "$OOS_YEARS" \
            --step "$STEP" \
            --n-trials "$N_TRIALS" \
            --objective-metric sharpe \
            --seed 42 \
            --stake "$STAKE" \
            --cash "$CASH" \
            --commission "$COMMISSION" \
            --stop-loss "$sl" \
            --take-profit "$tp" \
            --wf-max-workers "$CPUS" \
            --out-dir "$out_dir" \
            --regime-mode score \
            --hmm-components 4

    echo "  Done: $tag"
}

# Combo A: SL=5%  TP=10%
run_combo 0.05 0.10

# Combo B: SL=5%  TP=15%
run_combo 0.05 0.15

echo ""
echo "All combos complete. Results in results/sltp_test/"
