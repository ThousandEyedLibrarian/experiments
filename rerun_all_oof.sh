#!/bin/bash
# Regenerate ONE consistent OOF prediction set for the whole results table,
# unbalanced (none) + weighted (inverse-sqrt), then gate on shared.verify_oof
# (per-fold dedup, no cross-fold leakage, cohort counts 198/117/147/107).
#
# Successor to rerun_dn_figures.slurm. Runs LOCALLY on the Mac: every table
# config is a cached-embedding MLP (text/SMILES/EEG features are precomputed
# under outputs/), so CPU is sufficient and no SLURM/scp is needed. The one
# exception is exp11, which trains an EEG2Vec encoder and is slow on CPU
# (minutes/fold); it runs last so the fast configs finish first, and
# --skip-exp11 offloads it (run that single module on M3 by hand if needed).
#
#   bash rerun_all_oof.sh              # archive old preds, rerun all, gate
#   bash rerun_all_oof.sh --skip-exp11 # everything except the slow encoder
#   bash rerun_all_oof.sh --extended   # also: exp7b quad-MoE, exp7a stratified,
#                                      #        exp9 encoder sweep, exp15 REVE
#
# The --extended set refreshes the previously un-rerun supplementary analyses
# (quad mixture-of-experts, the standalone EEG-encoder comparison table, the
# REVE-base substitution, and stratified-batch balancing). These configs train
# EEG/foundation-model encoders and are GPU-heavy - run --extended on M3.
#
# NOT set -e: a single bad config is logged and skipped, not fatal, so one
# failure never costs the whole sweep. The gate's exit code is the verdict.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

SKIP_EXP11=0
EXTENDED=0
for arg in "$@"; do
    case "$arg" in
        --skip-exp11) SKIP_EXP11=1 ;;
        --extended)   EXTENDED=1 ;;
    esac
done

if [[ ! -x .venv-others/bin/python ]]; then
    echo "ERROR: .venv-others not found in $REPO_DIR. See README 'Environment Setup'." >&2
    exit 1
fi
PY=.venv-others/bin/python
OUT=outputs
STAMP="$(date +%Y%m%d_%H%M%S)"

# 1. Archive every prior prediction set so the gate sees only fresh files.
#    Non-table experiments (exp9/exp15/...) are archived and not regenerated.
#    The archive lives one level deeper than the gate's glob, so it is ignored.
ARCHIVE="$OUT/_prediction_archive_$STAMP"
shopt -s nullglob
old=("$OUT"/exp*_predictions)
shopt -u nullglob
if (( ${#old[@]} )); then
    mkdir -p "$ARCHIVE"
    echo "== Archiving ${#old[@]} prior prediction dir(s) -> $ARCHIVE =="
    mv "${old[@]}" "$ARCHIVE"/
fi

FAILED=()

# 2. exp7 headline: figures + Quad table row + counterfactual + in-sample.
for mode in none weighted; do
    echo ""
    echo "== exp7a predictions  --asm-balance $mode =="
    $PY -m exp7_all_modalities.run_experiments \
        --mode predictions --asm-balance "$mode" --deterministic \
        --output_dir "$OUT/exp7_predictions" || FAILED+=("exp7:$mode")
done

# 3. Per-config OOF for every other table experiment, none + weighted.
#    exp2/exp3 omit --fusion so both the mlp (a) and fusemoe (b) sub-configs run.
run_config () {
    local module="$1"; shift
    for mode in none weighted; do
        echo ""
        echo "== $module  $* --asm-balance $mode --log-predictions --deterministic =="
        $PY -m "$module.run_experiments" "$@" \
            --asm-balance "$mode" --log-predictions --deterministic \
            || FAILED+=("$module:$mode")
    done
}
run_config exp1_fusion
run_config exp2_fusion
run_config exp3_fusion
run_config exp4_baseline
run_config exp5_clinical_fusion
run_config exp6_clinical_triple
(( SKIP_EXP11 )) || run_config exp11_eeg_upgrade

# 3b. Extended set (GPU-heavy; refreshes the caveated supplementary analyses).
if (( EXTENDED )); then
    # exp7b quad mixture-of-experts OOF (parametrised fusion in the predictions
    # path); lands as exp7_predictions/predictions_oof_7b*.json, gated at 107.
    for mode in none weighted; do
        echo ""
        echo "== exp7b (quad MoE) predictions  --asm-balance $mode =="
        $PY -m exp7_all_modalities.run_experiments \
            --mode predictions --exp 7b --asm-balance "$mode" --deterministic \
            --output_dir "$OUT/exp7_predictions" || FAILED+=("exp7b:$mode")
    done

    # exp7a stratified-batch balancing (the third balancing arm the appendix cites).
    echo ""
    echo "== exp7a predictions  --asm-balance stratified_batch =="
    $PY -m exp7_all_modalities.run_experiments \
        --mode predictions --asm-balance stratified_batch --deterministic \
        --output_dir "$OUT/exp7_predictions" || FAILED+=("exp7:stratbatch")

    # exp15 REVE-base quad (routed through shared.cohort; refreshes the REVE
    # substitution + recommendation-distribution appendix items).
    for mode in none weighted; do
        echo ""
        echo "== exp15 (REVE quad) predictions  --asm-balance $mode =="
        $PY -m exp15_reve_quad_mlp.run_experiments \
            --mode predictions --asm-balance "$mode" --seed 42 \
            --output-dir "$OUT/exp15_predictions" || FAILED+=("exp15:$mode")
    done

    # exp9 standalone EEG-encoder sweep (inherits exp2's dedup via prepare_data;
    # refreshes the encoder-comparison table). Needs iterative-stratification.
    echo ""
    echo "== exp9 (EEG encoder sweep) --log-predictions =="
    $PY -m exp9_eeg_investigation.run_experiments \
        --log-predictions --deterministic || FAILED+=("exp9")
fi

# 4. Gate. PASS == every fresh file is deduped, leak-free, right cohort size.
echo ""
echo "== verify_oof gate =="
if $PY -m shared.verify_oof "$OUT"; then GATE="PASS"; else GATE="FAIL"; fi

# verify_oof only checks files that EXIST; a config that errored produced none
# (and its prior file was archived away), so a silent per-config failure would
# otherwise slip through. Any FAILED entry forces the verdict to FAIL.
if (( ${#FAILED[@]} )); then GATE="FAIL"; fi

echo ""
echo "=================================================================="
echo "rerun complete. gate: $GATE"
(( ${#FAILED[@]} )) && echo "configs that errored (absent from the fresh set): ${FAILED[*]}"
(( SKIP_EXP11 )) && echo "exp11 skipped (--skip-exp11); Clinical+EEG row needs it."
(( EXTENDED )) && echo "extended set run: exp7b, exp7a stratified, exp15 REVE, exp9 encoders."
echo "prior predictions archived under: $ARCHIVE"
echo "=================================================================="
[[ "$GATE" == "PASS" ]]
