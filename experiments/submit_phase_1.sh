#!/bin/bash
# Submit all Phase 1 jobs

sbatch "/scratch/lustre/home/ilju3280/hmm-backtrader-sample/experiments/jobs/01_strict/spy_qqq/is2_oos1.sh"
sbatch "/scratch/lustre/home/ilju3280/hmm-backtrader-sample/experiments/jobs/01_strict/spy_qqq/is3_oos1.sh"
sbatch "/scratch/lustre/home/ilju3280/hmm-backtrader-sample/experiments/jobs/02_size/spy_qqq/is2_oos1.sh"
sbatch "/scratch/lustre/home/ilju3280/hmm-backtrader-sample/experiments/jobs/02_size/spy_qqq/is3_oos1.sh"
sbatch "/scratch/lustre/home/ilju3280/hmm-backtrader-sample/experiments/jobs/03_score/spy_qqq/is2_oos1.sh"
sbatch "/scratch/lustre/home/ilju3280/hmm-backtrader-sample/experiments/jobs/03_score/spy_qqq/is3_oos1.sh"
sbatch "/scratch/lustre/home/ilju3280/hmm-backtrader-sample/experiments/jobs/04_linear/spy_qqq/is2_oos1.sh"
sbatch "/scratch/lustre/home/ilju3280/hmm-backtrader-sample/experiments/jobs/04_linear/spy_qqq/is3_oos1.sh"
