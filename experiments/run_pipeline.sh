#!/bin/bash
# run_pipeline.sh
# ================
# Submit all 5 phases with SLURM job dependencies so each phase
# starts automatically once every job in the previous phase succeeds.
#
# Usage:
#   bash experiments/run_pipeline.sh
#
# To submit specific phases only:
#   bash experiments/run_pipeline.sh 1 2
#
# After Phase 1 completes, review results with:
#   python experiments/report_phase1.py results/phase_1/
# (and similarly for each subsequent phase)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

# Which phases to submit (default: all)
PHASES=("${@:-1 2 3 4 5}")
# If called with no args, expand "1 2 3 4 5" properly
if [[ $# -eq 0 ]]; then
    PHASES=(1 2 3 4 5)
fi

# Map phase number -> list of job scripts (extracted from submit_phase_N.sh)
declare -A PHASE_SCRIPTS
for p in "${PHASES[@]}"; do
    submit="$HERE/submit_phase_${p}.sh"
    if [[ ! -f "$submit" ]]; then
        echo "ERROR: $submit not found. Run: python experiments/gen_jobs.py" >&2
        exit 1
    fi
    # Extract the sbatch script paths from the submit file
    mapfile -t scripts < <(grep 'sbatch' "$submit" | sed 's/.*sbatch "\(.*\)"/\1/')
    PHASE_SCRIPTS[$p]="${scripts[*]}"
done

# Submit phases, chaining dependencies
prev_ids=()

for p in "${PHASES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Phase $p"
    echo "============================================================"

    # Build dependency string
    dep_flag=""
    if [[ ${#prev_ids[@]} -gt 0 ]]; then
        dep_str=$(printf ":%s" "${prev_ids[@]}")
        dep_str="${dep_str:1}"   # strip leading ':'
        dep_flag="--dependency=afterok:${dep_str}"
        echo "  Waiting for job IDs: ${prev_ids[*]}"
    fi

    current_ids=()
    IFS=' ' read -ra scripts <<< "${PHASE_SCRIPTS[$p]}"
    for script in "${scripts[@]}"; do
        if [[ -n "$dep_flag" ]]; then
            result=$(sbatch $dep_flag "$script")
        else
            result=$(sbatch "$script")
        fi
        jid=$(echo "$result" | awk '{print $NF}')
        current_ids+=("$jid")
        echo "  Submitted: $(basename "$(dirname "$script")")/$(basename "$script")  → job $jid"
    done

    prev_ids=("${current_ids[@]}")

    echo "  Phase $p job IDs: ${current_ids[*]}"
done

echo ""
echo "============================================================"
echo "  All phases submitted."
echo "  Monitor with:  squeue -u \$USER"
echo "  Phase logs in: $ROOT/logs/"
echo "============================================================"
echo ""
echo "  Phase result directories:"
for p in "${PHASES[@]}"; do
    echo "    Phase $p → $ROOT/results/phase_$p/"
done
echo ""
echo "  Review results after each phase completes:"
for p in "${PHASES[@]}"; do
    echo "    python experiments/report_phase${p}.py results/phase_${p}/"
done
