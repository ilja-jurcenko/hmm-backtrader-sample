#!/bin/bash
# Submit all Phase 5 jobs

sbatch "/scratch/lustre/home/ilju3280/hmm-backtrader-sample/experiments/jobs/14_best_multiasset/sp500_10/is2_oos1.sh"
sbatch "/scratch/lustre/home/ilju3280/hmm-backtrader-sample/experiments/jobs/14_best_multiasset/sp500_10/is3_oos1.sh"
sbatch "/scratch/lustre/home/ilju3280/hmm-backtrader-sample/experiments/jobs/14_best_multiasset/spy_qqq/is2_oos1.sh"
sbatch "/scratch/lustre/home/ilju3280/hmm-backtrader-sample/experiments/jobs/14_best_multiasset/spy_qqq/is3_oos1.sh"
