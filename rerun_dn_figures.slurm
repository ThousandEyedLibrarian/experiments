#!/bin/bash
# Regenerate the predictions needed to satisfy Duong's review.
#
# Part 1 (figures): Exp7a headline model, unbalanced (vanilla) + inverse-
#   frequency-weighted, with per-ASM counterfactual logging. Feeds Fig 3,
#   Fig 4, the counterfactual, and the headline operating-point row.
#   Weighted = the "best class-balancing strategy" per thesis Section 3.4.4.
#   stratified_batch is deliberately NOT run (it over-corrected; rejected).
#
# Part 2 (operating-point table): baseline (unbalanced) OOF predictions for
#   the other configurations in the results table, so precision (and bootstrap
#   sensitivity/specificity) can be computed for the whole table. The original
#   per-config prediction files were never committed (outputs/ is gitignored)
#   and are absent from a fresh clone, so they must be regenerated.
#
#   Config -> experiment (from analysis/runs_manifest.json):
#     Clinical / Clinical+Text / Clinical+EEG (drug fixed) -> exp5, exp6, exp11
#     Text / EEG / Text+EEG (embedding only)               -> exp1, exp2, exp3
#     Clinical(MLP) ablation                               -> exp4
#   Each experiment defaults to running all its sub-configs and encodes the
#   config in the prediction filename (predictions_oof_<config>.json), so we
#   run the default sweep and match files by name afterwards.
#
#   NOTE: these are fresh deterministic reruns. As with the headline (fresh
#   0.804 vs published 0.790), derived metrics may differ slightly from the
#   published table. Decide on the full-table re-derivation once files land.
#
# Run from the experiments repo root on M3:
#     sbatch rerun_dn_figures.slurm
# Then follow the scp instructions printed at the end of logs/dn_rerun_*.out.
#
# SBATCH directives copied from submit_job.sh; override on the CLI if needed,
# e.g.  sbatch --gres=gpu:A100:1 --time=10:00:00 rerun_dn_figures.slurm
#SBATCH --job-name=dn_rerun
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
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

FAILED=()

# ------------------------------------------------------------------
# Part 1: Exp7a headline predictions (figures + headline row).
# ------------------------------------------------------------------
run_variant () {
    local mode="$1"
    echo ""
    echo "== Exp7a predictions  --asm-balance ${mode} =="
    if ! python -m exp7_all_modalities.run_experiments \
            --mode predictions --asm-balance "${mode}" --deterministic \
            --output_dir outputs/exp7_predictions; then
        FAILED+=("exp7:${mode}")
    fi
}
run_variant none
run_variant weighted

# ------------------------------------------------------------------
# Part 2: baseline OOF predictions for the other table configs.
# Each row: "<module> <extra args>". --log-predictions --deterministic added.
# A failure in one config is logged and skipped, not fatal, so the figure
# predictions still come back.
# ------------------------------------------------------------------
run_config () {
    local module="$1"; shift
    echo ""
    echo "== ${module}  $* --log-predictions --deterministic =="
    if ! python -m "${module}.run_experiments" "$@" --log-predictions --deterministic; then
        FAILED+=("${module}")
    fi
}
run_config exp1_fusion
run_config exp2_fusion
run_config exp3_fusion --fusion mlp
run_config exp4_baseline
run_config exp5_clinical_fusion
run_config exp6_clinical_triple
run_config exp11_eeg_upgrade

# ------------------------------------------------------------------
# Verify the figure predictions, then package everything for transfer.
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
[[ $missing -eq 0 ]] || { echo "ERROR: figure predictions missing; not packaging." >&2; exit 1; }

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "WARNING: these runs failed and are absent from the archive: ${FAILED[*]}" >&2
fi

# Package every logged prediction file (outputs/exp*_predictions/predictions_*.json):
# the exp7 figure/counterfactual set plus the per-config baseline OOF predictions.
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="dn_rerun_predictions_${STAMP}.tar.gz"
echo ""
echo "== Prediction files being packaged (relative to outputs/) =="
( cd outputs && find . -name 'predictions_*.json' -print | sort )
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
