#!/bin/bash
# Regenerate the Exp7a headline-model predictions needed to satisfy Duong's
# review (Fig 3, Fig 4, counterfactual, and the headline operating-point row),
# for both the unbalanced (vanilla) and inverse-frequency-weighted configs.
#
# Weighted = the "best class-balancing strategy" per thesis Section 3.4.4.
# stratified_batch is deliberately NOT run (it over-corrected; rejected).
#
# Run from the experiments repo root on M3:
#     sbatch rerun_dn_figures.slurm
# Then follow the scp instructions printed at the end of logs/dn_rerun_*.out.
#
# SBATCH directives copied from submit_job.sh; override on the CLI if needed,
# e.g.  sbatch --gres=gpu:A100:1 --time=6:00:00 rerun_dn_figures.slurm
#SBATCH --job-name=dn_rerun
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/dn_rerun_%j.out
#SBATCH --error=logs/dn_rerun_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"
mkdir -p logs outputs/exp7_predictions

if [[ ! -f .venv-others/bin/activate ]]; then
    echo "ERROR: .venv-others not found in $REPO_DIR. See README 'Environment Setup'." >&2
    exit 1
fi
source .venv-others/bin/activate

echo "== GPU =="
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || echo "(no nvidia-smi)"

run_variant () {
    local mode="$1"
    echo ""
    echo "=================================================================="
    echo "== Exp7a predictions  --asm-balance ${mode}"
    echo "=================================================================="
    python -m exp7_all_modalities.run_experiments \
        --mode predictions \
        --asm-balance "${mode}" \
        --deterministic \
        --output_dir outputs/exp7_predictions
}

run_variant none
run_variant weighted

# ------------------------------------------------------------------
# Verify the four files exist, then package for transfer.
# ------------------------------------------------------------------
OUT=outputs/exp7_predictions
EXPECTED=(
    "$OUT/predictions_oof.json"
    "$OUT/predictions_in_sample.json"
    "$OUT/predictions_oof_asmweighted.json"
    "$OUT/predictions_in_sample_asmweighted.json"
)
missing=0
for f in "${EXPECTED[@]}"; do
    if [[ -s "$f" ]]; then
        echo "OK   $f ($(wc -c < "$f") bytes)"
    else
        echo "MISSING or empty: $f" >&2
        missing=1
    fi
done
[[ $missing -eq 0 ]] || { echo "ERROR: some prediction files are missing; not packaging." >&2; exit 1; }

# Package the exp7 figure/counterfactual predictions PLUS every other config's
# logged OOF predictions (outputs/exp*_predictions/predictions_*.json), so the
# same archive also unblocks the operating-point (precision) table locally.
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="dn_rerun_predictions_${STAMP}.tar.gz"
echo ""
echo "== Prediction files being packaged (relative to outputs/) =="
( cd outputs && find . -name 'predictions_*.json' -print )
( cd outputs && find . -name 'predictions_*.json' -print0 | tar -czf "$REPO_DIR/$ARCHIVE" --null -T - )
echo ""
echo "Packaged -> $REPO_DIR/$ARCHIVE"
echo ""
echo "=================================================================="
echo "COPY BACK TO YOUR LAPTOP (run this locally in a new terminal):"
echo "  scp carterfs@m3.massive.org.au:$REPO_DIR/$ARCHIVE ~/Downloads/"
echo "Then extract into the experiments outputs dir:"
echo "  mkdir -p ~/Sync/Academics/University/postgrad/MonashHonours/thesis/experiments/outputs"
echo "  tar -xzf ~/Downloads/$ARCHIVE -C ~/Sync/Academics/University/postgrad/MonashHonours/thesis/experiments/outputs"
echo "=================================================================="
