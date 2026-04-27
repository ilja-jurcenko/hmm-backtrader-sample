#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh
# =============================================================================
# Master pipeline driver for the 5-phase HMM walk-forward experiment.
#
# For each phase (1 → 5):
#   1. Submit all SLURM jobs via experiments/submit_phase_N.sh
#   2. Poll squeue until every submitted job has finished
#   3. Run experiments/aggregate.py --phase N to produce master_results.csv
#   4. Run experiments/report_phaseN.py to produce the HTML report
#   5. Move on to the next phase
#
# Usage:
#   bash run_pipeline.sh [--start-phase N] [--poll-interval SECONDS]
#
# Options:
#   --start-phase N       Resume from phase N (default: 1)
#   --poll-interval N     Seconds between squeue checks (default: 60)
#   --dry-run             Print commands but do not execute them
#
# Requirements:
#   - Run from the repository root (hmm-backtrader-sample/)
#   - Python venv at .venv/ with all deps installed
#   - SLURM (sbatch, squeue) available in PATH
# =============================================================================

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
START_PHASE=1
POLL_INTERVAL=60
DRY_RUN=false

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv/bin/activate"
EXPERIMENTS="$ROOT/experiments"
RESULTS="$ROOT/results"
LOG_DIR="$ROOT/logs"
PIPELINE_LOG="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-phase)   START_PHASE="$2";  shift 2 ;;
        --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=true;      shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── helpers ───────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
TOTAL_PHASES=5

# Tee all output to both stdout and the log file
exec > >(tee -a "$PIPELINE_LOG") 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_cmd() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY-RUN] $*"
    else
        log ">>> $*"
        eval "$@"
    fi
}

# Submit jobs for a phase and return the space-separated list of job IDs
submit_phase() {
    local phase=$1
    local submit_script="$EXPERIMENTS/submit_phase_${phase}.sh"

    if [[ ! -f "$submit_script" ]]; then
        log "ERROR: submit script not found: $submit_script"
        exit 1
    fi

    log "Submitting Phase $phase jobs..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY-RUN] Would run: bash $submit_script"
        echo "DRY_RUN_JOB_1 DRY_RUN_JOB_2"
        return
    fi

    # Capture sbatch output to extract job IDs
    local job_ids=()
    while IFS= read -r line; do
        # Each sbatch call emits: "Submitted batch job 12345"
        if [[ "$line" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
            job_ids+=("${BASH_REMATCH[1]}")
            log "  Submitted job ${BASH_REMATCH[1]}"
        else
            log "  sbatch: $line"
        fi
    done < <(bash "$submit_script" 2>&1)

    if [[ ${#job_ids[@]} -eq 0 ]]; then
        log "ERROR: No jobs were submitted for Phase $phase."
        exit 1
    fi

    log "Phase $phase: submitted ${#job_ids[@]} job(s): ${job_ids[*]}"
    echo "${job_ids[*]}"
}

# Poll squeue until all given job IDs have left the queue (completed/failed)
wait_for_jobs() {
    local phase=$1
    shift
    local job_ids=("$@")

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY-RUN] Would wait for jobs: ${job_ids[*]}"
        return
    fi

    log "Waiting for ${#job_ids[@]} Phase $phase job(s) to finish..."
    log "  Polling every ${POLL_INTERVAL}s. Job IDs: ${job_ids[*]}"

    local failed_jobs=()

    while true; do
        local pending=()
        for jid in "${job_ids[@]}"; do
            # squeue -j returns a line for the job while it is still running/pending
            if squeue -j "$jid" --noheader 2>/dev/null | grep -q "$jid"; then
                pending+=("$jid")
            fi
        done

        if [[ ${#pending[@]} -eq 0 ]]; then
            log "All Phase $phase jobs have left the queue."
            break
        fi

        log "  Still running: ${pending[*]} — checking again in ${POLL_INTERVAL}s"
        sleep "$POLL_INTERVAL"
    done

    # Check sacct for failed jobs (state FAILED or TIMEOUT)
    log "Checking job exit states via sacct..."
    for jid in "${job_ids[@]}"; do
        local state
        state=$(sacct -j "$jid" --noheader --format=State --parsable2 2>/dev/null \
                | head -1 | tr -d '[:space:]')
        if [[ "$state" == "FAILED" || "$state" == "TIMEOUT" || "$state" == "CANCELLED" ]]; then
            failed_jobs+=("$jid ($state)")
        fi
    done

    if [[ ${#failed_jobs[@]} -gt 0 ]]; then
        log "WARNING: ${#failed_jobs[@]} job(s) did not complete successfully:"
        for f in "${failed_jobs[@]}"; do
            log "  $f"
        done
        log "Pipeline continuing — partial results may affect reports."
    else
        log "All Phase $phase jobs completed successfully."
    fi
}

# Aggregate per-window CSVs and produce master_results.csv for a phase
aggregate_phase() {
    local phase=$1
    local out_csv="$RESULTS/master_results_phase_${phase}.csv"
    run_cmd "source '$VENV' && python '$EXPERIMENTS/aggregate.py' \
        --phase '$phase' \
        --out '$out_csv'"
    log "Aggregated CSV written to: $out_csv"
}

# Generate HTML report for a phase
report_phase() {
    local phase=$1
    local report_script="$EXPERIMENTS/report_phase${phase}.py"
    local out_html="$RESULTS/phase_${phase}_report.html"

    if [[ ! -f "$report_script" ]]; then
        log "WARNING: report script not found: $report_script — skipping report"
        return
    fi

    run_cmd "source '$VENV' && python '$report_script' \
        --master '$RESULTS/master_results_phase_${phase}.csv' \
        --out '$out_html'"
    log "Report written to: $out_html"
}

# ── main ──────────────────────────────────────────────────────────────────────
log "========================================================"
log "  HMM Walk-Forward Pipeline"
log "  Phases $START_PHASE → $TOTAL_PHASES"
log "  Poll interval : ${POLL_INTERVAL}s"
log "  Dry-run       : $DRY_RUN"
log "  Log file      : $PIPELINE_LOG"
log "========================================================"

# Activate venv once to validate it exists
if [[ "$DRY_RUN" == "false" ]]; then
    if [[ ! -f "$VENV" ]]; then
        log "ERROR: venv not found at $VENV"
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$VENV"
    log "Python: $(python --version)"
fi

for phase in $(seq "$START_PHASE" "$TOTAL_PHASES"); do
    log ""
    log "════════════════════════════════════════════════════"
    log "  PHASE $phase / $TOTAL_PHASES"
    log "════════════════════════════════════════════════════"

    # 1. Submit jobs
    job_id_str=$(submit_phase "$phase")
    read -ra job_ids <<< "$job_id_str"

    # 2. Wait for all jobs to finish
    wait_for_jobs "$phase" "${job_ids[@]}"

    # 3. Aggregate results
    log "Aggregating Phase $phase results..."
    aggregate_phase "$phase"

    # 4. Generate HTML report
    log "Generating Phase $phase report..."
    report_phase "$phase"

    log "Phase $phase complete."
done

log ""
log "========================================================"
log "  Pipeline complete — all $TOTAL_PHASES phases finished."
log "  Reports: $RESULTS/phase_*_report.html"
log "========================================================"
