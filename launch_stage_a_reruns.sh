#!/bin/bash
#==============================================================================
# Stage A: launch all Phase-2.5 reruns with prediction logging.
#
# Submits one SLURM job per experiment via submit_job.sh. Each job runs
# its experiment with --log-predictions (and --deterministic for the
# experiments that have it) so we get patient-level OOF prediction JSON
# files for #25 (pooled-AUC bootstrap), #27 (sens/spec), and the
# cross-cohort half of #31 (DeLong).
#
# Usage:
#   bash launch_stage_a_reruns.sh           # submit all 8 to SLURM
#   bash launch_stage_a_reruns.sh --dry-run # show commands, don't submit
#   bash launch_stage_a_reruns.sh --local   # run sequentially on this machine
#   bash launch_stage_a_reruns.sh --only exp1,exp4  # subset
#
# After all jobs complete, run:
#   python -m thesisStandalone.analysis.consume_stage_a_predictions
#==============================================================================

# NB: do NOT set -e here. We want one bad sbatch invocation (e.g. an
# invalid partition name on a particular cluster) to log an error and
# move on to the remaining experiments, not abort the whole batch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=false
LOCAL=false
ONLY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --local)   LOCAL=true;   shift ;;
        --only)    ONLY="$2";    shift 2 ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Experiment list and per-experiment args.
# Each row: <exp_id>|<extra_args>|<wall_time>|<requires_gpu>
EXPERIMENTS=(
    "exp1|--log-predictions --deterministic|2:00:00|true"
    "exp2|--log-predictions --deterministic|4:00:00|true"
    "exp3|--log-predictions --deterministic|3:00:00|true"
    "exp4|--log-predictions --deterministic|0:30:00|false"
    "exp5|--log-predictions --deterministic|2:00:00|true"
    "exp6|--log-predictions --deterministic|3:00:00|true"
    "exp7|--deterministic|3:00:00|true"
    "exp9|--log-predictions --deterministic|5:00:00|true"
    "exp11|--log-predictions --deterministic|3:00:00|true"
)

if [ -n "$ONLY" ]; then
    IFS=',' read -ra ONLY_LIST <<< "$ONLY"
    FILTERED=()
    for row in "${EXPERIMENTS[@]}"; do
        EXP_ID="${row%%|*}"
        for want in "${ONLY_LIST[@]}"; do
            if [ "$EXP_ID" = "$want" ]; then
                FILTERED+=("$row")
            fi
        done
    done
    EXPERIMENTS=("${FILTERED[@]}")
fi

echo "============================================================"
echo "Stage A rerun launcher"
echo "Mode: $([ "$LOCAL" = "true" ] && echo "local" || echo "SLURM")"
echo "Dry run: $DRY_RUN"
echo "Experiments to launch: ${#EXPERIMENTS[@]}"
echo "============================================================"

JOB_IDS=()
for row in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r EXP_ID EXP_ARGS WALL_TIME NEEDS_GPU <<< "$row"

    if [ "$LOCAL" = "true" ]; then
        CMD="bash submit_job.sh -e $EXP_ID -a \"$EXP_ARGS\""
        echo ""
        echo ">> $EXP_ID (local): $CMD"
        if [ "$DRY_RUN" = "false" ]; then
            bash submit_job.sh -e "$EXP_ID" -a "$EXP_ARGS" 2>&1 | tee "logs/stage_a_${EXP_ID}_local.out"
        fi
    else
        SBATCH_FLAGS="--time=$WALL_TIME --job-name=stageA_${EXP_ID}"
        if [ "$NEEDS_GPU" = "false" ]; then
            # M3 has no 'cpu' partition; 'short' is CPU-only with a 30 min
            # cap (fine for exp4). Explicitly request zero GPUs to
            # override the --gres=gpu:1 SBATCH directive in submit_job.sh.
            SBATCH_FLAGS="$SBATCH_FLAGS --partition=short --gres=gpu:0 --mem=16G --cpus-per-task=4"
        fi
        CMD="sbatch $SBATCH_FLAGS submit_job.sh -e $EXP_ID -a \"$EXP_ARGS\""
        echo ""
        echo ">> $EXP_ID: $CMD"
        if [ "$DRY_RUN" = "false" ]; then
            if JOB_OUTPUT=$(sbatch $SBATCH_FLAGS submit_job.sh -e "$EXP_ID" -a "$EXP_ARGS" 2>&1); then
                echo "  $JOB_OUTPUT"
                JOB_ID=$(echo "$JOB_OUTPUT" | grep -oE '[0-9]+')
                JOB_IDS+=("$JOB_ID:$EXP_ID")
            else
                echo "  FAILED: $JOB_OUTPUT"
                echo "  (continuing with remaining experiments)"
            fi
        fi
    fi
done

if [ "$LOCAL" = "false" ] && [ "$DRY_RUN" = "false" ] && [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "============================================================"
    echo "Submitted ${#JOB_IDS[@]} jobs:"
    for entry in "${JOB_IDS[@]}"; do
        echo "  Job ${entry%%:*}  ($EXP_ID)"
    done
    echo ""
    echo "Monitor with:    squeue -u \$USER"
    echo "Cancel one with: scancel <job_id>"
    echo "Cancel all:      scancel ${JOB_IDS[*]%%:*}"
    echo "============================================================"
fi
