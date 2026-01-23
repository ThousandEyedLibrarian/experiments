#!/bin/bash
#==============================================================================
# M3 HPC Job Submission Script for ASM Experiments
#==============================================================================
# Usage:
#   sbatch submit_job.sh -e exp2                    # Run experiment 2
#   sbatch submit_job.sh -e exp1 -a "--dry-run"    # Exp 1 dry run
#   bash submit_job.sh -e exp2 -d                   # Local dry run test
#
# Options:
#   -e, --experiment EXP   Experiment to run: exp1 or exp2 (default: exp1)
#   -a, --args ARGS        Extra arguments for the experiment script
#   -d, --dry-run          Test mode - show config without executing
#   -h, --help             Show help message
#
# Override SBATCH settings via command line:
#   sbatch --time=8:00:00 submit_job.sh -e exp2
#   sbatch --gres=gpu:A40:1 submit_job.sh -e exp2
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
# CLI ARGUMENT PARSING
#==============================================================================

# Default values (can also be set via environment variables)
EXPERIMENT="${EXPERIMENT:-exp1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-false}"

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --experiment EXP   Experiment to run: exp1 or exp2 (default: exp1)"
    echo "  -a, --args ARGS        Extra arguments for the experiment script"
    echo "  -d, --dry-run          Test mode - show config without executing"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Extra args examples:"
    echo "  exp1: --dry-run, --exp1a, --exp1b, --quiet"
    echo "  exp2: --dry-run, --eeg-encoder simplecnn, --smiles-model chemberta"
    echo ""
    echo "Examples:"
    echo "  sbatch $0 -e exp2"
    echo "  sbatch $0 -e exp2 -a '--eeg-encoder simplecnn'"
    echo "  bash $0 -e exp1 -d"
    echo "  EXPERIMENT=exp2 sbatch $0"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        -a|--args)
            EXTRA_ARGS="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate experiment choice
if [[ "$EXPERIMENT" != "exp1" && "$EXPERIMENT" != "exp2" ]]; then
    echo "ERROR: Invalid experiment '$EXPERIMENT'. Must be 'exp1' or 'exp2'."
    exit 1
fi

#==============================================================================
# JOB EXECUTION
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

# Run environment validation
echo ""
echo "============================================================"
echo "Environment Validation"
echo "============================================================"
if [ "$EXPERIMENT" = "exp1" ]; then
    python check_environment.py --exp1
    ENV_CHECK=$?
elif [ "$EXPERIMENT" = "exp2" ]; then
    python check_environment.py --exp2
    ENV_CHECK=$?
else
    python check_environment.py --all
    ENV_CHECK=$?
fi

if [ $ENV_CHECK -ne 0 ]; then
    echo ""
    echo "WARNING: Environment check reported issues (exit code $ENV_CHECK)"
    echo "The experiment may fail. Check the output above for details."
    echo ""
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
    echo "  sbatch submit_job.sh -e ${EXPERIMENT}"
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
