#!/bin/bash
# Follow-up: refresh the two EEG-encoder configs that errored in the extended
# run (EEGNet + LaBraM needed braindecode, which was missing). Both are
# standalone EEG-only (n=147) and inherit exp2's patient-level de-duplication
# via prepare_data, so they are leakage-free on the corrected cohort.
#
# Submit from the experiments repo root on M3:  sbatch rerun_encoders.slurm
#SBATCH --job-name=enc_rerun
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/enc_rerun_%j.out
#SBATCH --error=logs/enc_rerun_%j.err

set -uo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"
mkdir -p logs
source .venv-others/bin/activate

echo "== GPU =="
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || echo "(no nvidia-smi)"

# braindecode provides the EEGNet + LaBraM encoders. --no-cache dodges the
# corrupt cached wheel (tabulate) that fails a normal uv install on M3.
echo "== installing braindecode =="
uv pip install --no-cache braindecode || echo "WARN: braindecode install returned nonzero"

FAILED=()
for enc in encoder_eegnet encoder_labram; do
    echo ""
    echo "== exp9 $enc --log-predictions --deterministic =="
    python -m exp9_eeg_investigation.run_experiments \
        --experiment "$enc" --log-predictions --deterministic \
        || FAILED+=("$enc")
done

echo ""
echo "== files produced =="
ls -1 outputs/exp9_predictions/predictions_oof_exp9_encoder_{eegnet,labram}.json 2>/dev/null

# Gate the EEG-only cohort (147) for the fresh files.
echo ""
echo "== verify_oof (exp9 -> 147) =="
python -m shared.verify_oof outputs || true

# Package just the two new files for scp back.
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="encoder_rerun_${STAMP}.tar.gz"
tar -czf "$REPO_DIR/$ARCHIVE" \
    outputs/exp9_predictions/predictions_oof_exp9_encoder_eegnet.json \
    outputs/exp9_predictions/predictions_oof_exp9_encoder_labram.json 2>/dev/null || true

echo ""
echo "=================================================================="
(( ${#FAILED[@]} )) && echo "STILL FAILED: ${FAILED[*]}" || echo "both encoders OK"
echo "packaged -> $REPO_DIR/$ARCHIVE"
echo "scp back:  scp m3:'$REPO_DIR/$ARCHIVE' ~/Downloads/  then"
echo "           tar -xzf ~/Downloads/$ARCHIVE -C /Users/carter/carter_massive/experiments/outputs"
echo "=================================================================="
