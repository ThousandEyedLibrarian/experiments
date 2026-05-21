#!/bin/bash
# Exp15: REVE quad-modal overnight sweep
# 5 seeds x 3 ASM-balance variants = 15 SLURM jobs
#
# Usage (on M3): bash launch_exp15_overnight.sh

set -euo pipefail

SEEDS=(42 137 2025 7 314)
VARIANTS=(none weighted stratified_batch)

mkdir -p logs

submitted=0
for seed in "${SEEDS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        echo "Submitting: seed=${seed} variant=${variant}"
        sbatch submit_job.sh -e exp15 \
            -a "--seed ${seed} --asm-balance ${variant} --mode predictions --deterministic"
        submitted=$((submitted + 1))
        sleep 1  # avoid SLURM submission rate limits
    done
done

echo ""
echo "Submitted ${submitted} jobs (5 seeds x 3 variants)."
echo "Queue status:"
squeue -u "$USER"
