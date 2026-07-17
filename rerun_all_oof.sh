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
#
# NOT set -e: a single bad config is logged and skipped, not fatal, so one
# failure never costs the whole sweep. The gate's exit code is the verdict.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

SKIP_EXP11=0
[[ "${1:-}" == "--skip-exp11" ]] && SKIP_EXP11=1

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

# 4. Gate. PASS == every fresh file is deduped, leak-free, right cohort size.
echo ""
echo "== verify_oof gate =="
if $PY -m shared.verify_oof "$OUT"; then GATE="PASS"; else GATE="FAIL"; fi

echo ""
echo "=================================================================="
echo "rerun complete. gate: $GATE"
(( ${#FAILED[@]} )) && echo "configs that errored (absent from the fresh set): ${FAILED[*]}"
(( SKIP_EXP11 )) && echo "exp11 skipped (--skip-exp11); Clinical+EEG row needs it."
echo "prior predictions archived under: $ARCHIVE"
echo "=================================================================="
[[ "$GATE" == "PASS" ]]
