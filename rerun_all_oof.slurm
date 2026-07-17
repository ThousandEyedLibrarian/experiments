#!/bin/bash
# M3 GPU wrapper around rerun_all_oof.sh. On a GPU node every experiment
# auto-selects CUDA, so exp11's EEG2Vec encoder (the CPU long-pole) runs fast.
# The core script (archive prior preds -> rerun none+weighted -> verify_oof
# gate) is unchanged; this only requests the GPU and packages the fresh OOF
# files for scp back to the laptop.
#
# Submit from the experiments repo root on M3:
#     sbatch rerun_all_oof.slurm
# Override directives on the CLI if needed, e.g.
#     sbatch --gres=gpu:A100:1 --time=16:00:00 rerun_all_oof.slurm
#SBATCH --job-name=rerun_oof
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/rerun_oof_%j.out
#SBATCH --error=logs/rerun_oof_%j.err

set -uo pipefail   # not -e: rerun_all_oof.sh handles per-config failure itself

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"
mkdir -p logs

echo "== GPU =="
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || echo "(no nvidia-smi)"

# Run the portable core (archive + rerun none+weighted + gate). Capture its
# exit code (0 == gate PASS) but keep going so the fresh files are packaged
# regardless, for inspection.
bash rerun_all_oof.sh
GATE_RC=$?

# Package every fresh OOF file for transfer back to the laptop.
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="rerun_all_oof_predictions_${STAMP}.tar.gz"
echo ""
echo "== Prediction files being packaged (relative to outputs/) =="
( cd outputs && find . -path './_prediction_archive_*' -prune -o -name 'predictions_*.json' -print | sort )
( cd outputs && find . -path './_prediction_archive_*' -prune -o -name 'predictions_*.json' -print0 \
    | tar -czf "$REPO_DIR/$ARCHIVE" --null -T - )

echo ""
echo "=================================================================="
echo "gate exit code: $GATE_RC   (0 == PASS)"
echo "Packaged -> $REPO_DIR/$ARCHIVE"
echo ""
echo "COPY BACK TO THE LAPTOP (run locally in a new terminal):"
echo "  scp carterfs@m3.massive.org.au:$REPO_DIR/$ARCHIVE ~/Downloads/"
echo "  tar -xzf ~/Downloads/$ARCHIVE -C /Users/carter/carter_massive/experiments/outputs"
echo "=================================================================="
exit $GATE_RC
