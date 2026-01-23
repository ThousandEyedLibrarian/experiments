#!/bin/bash
#==============================================================================
# M3 HPC Job Submission Script for ASM Experiments
#==============================================================================
# Usage:
#   1. Edit the CONFIGURATION section below
#   2. Edit SBATCH directives if needed (time, memory, GPU type)
#   3. Test with: DRY_RUN=true bash submit_job.sh
#   4. Submit with: sbatch submit_job.sh
#   5. Monitor with: squeue -u $USER
#
# Override SBATCH settings via command line:
#   sbatch --time=8:00:00 submit_job.sh
#   sbatch --gres=gpu:A40:1 submit_job.sh
#==============================================================================

#==============================================================================
# SBATCH DIRECTIVES - Edit these as needed - Keep first hash
#==============================================================================
#SBATCH --job-name=asm_exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/asm_%j.out
#SBATCH --error=logs/asm_%j.err

# Uncomment for specific GPU type:
#SBATCH --gres=gpu:A100:1
##SBATCH --gres=gpu:A40:1

# Uncomment for email notifications:
##SBATCH --mail-user=your.email@monash.edu
##SBATCH --mail-type=BEGIN,END,FAIL

#==============================================================================
# CONFIGURATION - Edit these values
#==============================================================================

# Experiment to run: "exp1" or "exp2"
EXPERIMENT="exp1"

# Extra arguments for the experiment script
# exp1: "--dry-run", "--exp1a", "--exp1b", "--quiet"
# exp2: "--dry-run", "--eeg-encoder simplecnn", "--smiles-model chemberta"
EXTRA_ARGS=""

# Dry run mode: set to "true" to test without running experiments
# Usage: DRY_RUN=true bash submit_job.sh
DRY_RUN="${DRY_RUN:-false}"

#==============================================================================
# JOB EXECUTION - No need to edit below
#==============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "ASM Experiment Job"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Start time:  $(date)"
echo "Directory:   ${SCRIPT_DIR}"
echo "Experiment:  ${EXPERIMENT}"
echo "Extra args:  ${EXTRA_ARGS:-none}"
echo "============================================================"

# Create logs directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/logs"

# Change to project directory
cd "${SCRIPT_DIR}"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv-others/bin/activate

# Verify GPU access
echo ""
echo "GPU Status:"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

# Check nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "nvidia-smi:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
fi

# Determine the command to run
if [ "$EXPERIMENT" = "exp1" ]; then
    CMD="python -m exp1_fusion.run_experiments ${EXTRA_ARGS}"
elif [ "$EXPERIMENT" = "exp2" ]; then
    CMD="python -m exp2_fusion.run_experiments ${EXTRA_ARGS}"
else
    echo "ERROR: Unknown experiment '${EXPERIMENT}'. Use 'exp1' or 'exp2'."
    exit 1
fi

# Run the experiment (or show what would run in dry-run mode)
echo ""
echo "============================================================"
if [ "$DRY_RUN" = "true" ]; then
    echo "DRY RUN MODE - Not executing, just showing configuration"
    echo "============================================================"
    echo ""
    echo "Would execute: ${CMD}"
    echo ""
    echo "Environment check complete. To run for real:"
    echo "  sbatch submit_job.sh"
    EXIT_CODE=0
else
    echo "Starting ${EXPERIMENT}..."
    echo "============================================================"
    echo ""
    echo "Command: ${CMD}"
    echo ""
    ${CMD}
    EXIT_CODE=$?
fi

echo ""
echo "============================================================"
echo "Job completed"
echo "Exit code:   ${EXIT_CODE}"
echo "End time:    $(date)"
echo "============================================================"

exit ${EXIT_CODE}
