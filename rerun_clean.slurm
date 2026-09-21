#!/bin/bash
# M3 array wrapper for rerun_clean.sh: array index i runs the i-th work item of
# `bash rerun_clean.sh list`. Tasks are independent and skip themselves once
# done, so a failed or timed-out index can simply be resubmitted.
#
# Submit from the experiments repo root on M3 (after pulling both repos):
#     bash rerun_clean.sh list | cat -n              # index <-> task
#     sbatch rerun_clean.slurm                        # every item (24 at a time)
#     sbatch --array=0,5,14 rerun_clean.slurm         # a subset
#     sbatch --array=22 --time=24:00:00 rerun_clean.slurm   # a long one
# then, once the array has finished:
#     bash rerun_clean.sh verify
#SBATCH --job-name=clean_rerun
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-139%24
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/clean_rerun_%A_%a.out
#SBATCH --error=logs/clean_rerun_%A_%a.err

set -uo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"
mkdir -p logs

TASK="$(bash rerun_clean.sh list | sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p")"
if [[ -z "$TASK" ]]; then
    echo "no task for array index ${SLURM_ARRAY_TASK_ID:-0}" >&2
    exit 2
fi
echo "== array index ${SLURM_ARRAY_TASK_ID:-0} -> $TASK =="
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash rerun_clean.sh "$TASK"
