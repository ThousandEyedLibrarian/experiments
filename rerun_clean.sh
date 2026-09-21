#!/bin/bash
# Clean-CV rerun (docs/analysis_plan_clean_rerun_exp18.md): one work item per call.
#
#   bash rerun_clean.sh preflight         # check env, data, caches and repos before submitting
#   bash rerun_clean.sh list              # work items "task:seed", in array-index order
#   bash rerun_clean.sh <task>:<seed>     # run one item (skipped if already done)
#   bash rerun_clean.sh smoke <task>      # exp18 1-fold / 2-epoch dry run, output to /tmp
#   bash rerun_clean.sh verify            # gate: verify_oof + expected files + exp18 + HEP
#   sbatch rerun_clean.slurm              # every item as a slurm array (see that file)
#
# Every experiment runs with the clean protocol (--splitter multilabel
# --inner-val 0.2) under each repeated-CV seed 42-46 (--cv-seed; the plan's
# 2026-09-21 deviation), writing files suffixed _sp-multilabel_iv20_s<seed>
# next to the legacy ones, so nothing needs archiving and nothing legacy is
# overwritten. exp18 keeps its own seed sets (EEG configurations 42-44). A
# finished item writes outputs/_clean_rerun/<task>_s<seed>.done; rerunning a
# done item is a no-op unless FORCE=1. Set ASM_EXPERIMENTS_DIR if
# thesisStandalone is not cloned inside this repo.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
PY="$REPO_DIR/.venv-others/bin/python"
OUT=outputs
DONE="$OUT/_clean_rerun"
THESIS="$REPO_DIR/thesisStandalone"
SEEDS=(42 43 44 45 46)
EXP18_EEG_SEEDS=" 42 43 44 "
export ASM_EXPERIMENTS_DIR="${ASM_EXPERIMENTS_DIR:-$REPO_DIR}"

TASKS=(
    exp4 exp5 exp6 exp1 exp2 exp3
    exp7a exp7b exp7a_stratbatch exp11 exp9 exp15 exp16 exp17
    exp4_decomp
    hep_forward hep_eeg hep_reverse hep_focal hep_reduced reve
    exp18_Exp4a exp18_Exp5a exp18_Exp5b exp18_Exp5c exp18_Exp6b exp18_Exp7a exp18_noRMH
)

# Both balance modes, as in the legacy rerun. CV is set per item (seed).
balanced () {
    local module="$1"; shift
    for mode in none weighted; do
        "$PY" -m "$module.run_experiments" "$@" --asm-balance "$mode" \
            --log-predictions --deterministic "${CV[@]}" || return 1
    done
}

run_task () {
    local task="$1" seed="$2"
    CV=(--splitter multilabel --inner-val 0.2 --cv-seed "$seed")
    case "$task" in
        exp1) balanced exp1_fusion ;;
        exp2) balanced exp2_fusion \
              && balanced exp2_fusion --eeg-encoder eeg2vec --smiles-model chemberta --fusion mlp ;;
        exp3) balanced exp3_fusion ;;
        exp4) balanced exp4_baseline ;;
        exp5) balanced exp5_clinical_fusion ;;
        exp6) balanced exp6_clinical_triple ;;
        exp11) balanced exp11_eeg_upgrade ;;
        exp9) "$PY" -m exp9_eeg_investigation.run_experiments --log-predictions --deterministic "${CV[@]}" ;;
        exp7a)
            for mode in none weighted; do
                "$PY" -m exp7_all_modalities.run_experiments --mode predictions --asm-balance "$mode" \
                    --deterministic --output_dir "$OUT/exp7_predictions" "${CV[@]}" || return 1
            done ;;
        exp7b)
            for mode in none weighted; do
                "$PY" -m exp7_all_modalities.run_experiments --mode predictions --exp 7b --asm-balance "$mode" \
                    --deterministic --output_dir "$OUT/exp7_predictions" "${CV[@]}" || return 1
            done ;;
        exp7a_stratbatch)
            "$PY" -m exp7_all_modalities.run_experiments --mode predictions --asm-balance stratified_batch \
                --deterministic --output_dir "$OUT/exp7_predictions" "${CV[@]}" ;;
        exp15)
            for mode in none weighted; do
                "$PY" -m exp15_reve_quad_mlp.run_experiments --mode predictions --asm-balance "$mode" \
                    --seed "$seed" --output-dir "$OUT/exp15_predictions" "${CV[@]}" || return 1
            done ;;
        exp16) "$PY" -m exp16_reduced_capacity.run_experiments --mode predictions --seed "$seed" \
                   --output-dir "$OUT/exp16_predictions" "${CV[@]}" ;;
        exp17) "$PY" -m exp17_focal_only.run_experiments --mode predictions --seed "$seed" \
                   --output-dir "$OUT/exp17_predictions" "${CV[@]}" ;;
        exp4_decomp)
            # 2x2 splitter x early-stopping on exp4a, all on this host, kept out
            # of exp4_predictions so the legacy cell cannot replace the archived file.
            for cell in "legacy 0" "legacy 0.2" "multilabel 0" "multilabel 0.2"; do
                set -- $cell
                "$PY" -m exp4_baseline.run_experiments --model mlp --log-predictions --deterministic \
                    --splitter "$1" --inner-val "$2" --cv-seed "$seed" \
                    --predictions-dir "$OUT/exp4_decomposition" \
                    --output "$OUT/exp4_decomposition/results_$1_$2_s$seed.json" || return 1
            done ;;
        hep_forward) (cd "$THESIS" && "$PY" analysis/hep_external_validation.py "${CV[@]}") ;;
        hep_eeg) (cd "$THESIS" && "$PY" -m analysis.hep_external_validation_eeg "${CV[@]}") ;;
        hep_reverse) (cd "$THESIS" && "$PY" analysis/hep_reverse_validation.py "${CV[@]}") ;;
        hep_focal) (cd "$THESIS" && "$PY" analysis/hep_focal_external_validation.py "${CV[@]}") ;;
        hep_reduced) (cd "$THESIS" && "$PY" -m analysis.hep_reduced_external_validation "${CV[@]}") ;;
        reve) (cd "$THESIS" && "$PY" analysis/reve_standalone.py "${CV[@]}" \
                   --log-predictions "$REPO_DIR/$OUT/exp9_predictions") ;;
        exp18_noRMH) "$PY" -m exp18_mixed_cohort.run_experiments --config Exp4a Exp5a Exp5b \
                         --exclude-rmh --seeds "$seed" ;;
        exp18_Exp5c|exp18_Exp6b|exp18_Exp7a)
            if [[ "$EXP18_EEG_SEEDS" != *" $seed "* ]]; then
                echo "   (exp18 EEG configurations use seeds${EXP18_EEG_SEEDS}only; nothing to do)"; return 0
            fi
            "$PY" -m exp18_mixed_cohort.run_experiments --config "${task#exp18_}" --seeds "$seed" ;;
        exp18_*) "$PY" -m exp18_mixed_cohort.run_experiments --config "${task#exp18_}" --seeds "$seed" ;;
        *) echo "unknown task: $task (see: bash rerun_clean.sh list)" >&2; return 2 ;;
    esac
}

verify () {
    local rc=0
    echo "== verify_oof (all prediction files + expected clean files) =="
    "$PY" -m shared.verify_oof "$OUT" --expect clean_rerun_expected.txt || rc=1
    echo ""
    echo "== task completion =="
    local n_done=0
    for item in $(items); do
        if [[ -f "$DONE/${item%%:*}_s${item##*:}.done" ]]; then n_done=$((n_done + 1))
        else echo "MISSING $item"; rc=1; fi
    done
    echo "$n_done/$(items | wc -l) work items done"
    echo ""
    echo "== clean HEP outputs =="
    for f in hep_external_summary hep_external_summary_eeg hep_reverse_summary \
             hep_focal_external_summary hep_reduced_external_summary; do
        for seed in "${SEEDS[@]}"; do
            [[ -f "$THESIS/analysis/output/${f}_sp-multilabel_iv20_s${seed}.csv" ]] || { echo "MISSING ${f} s${seed}"; rc=1; }
        done
    done
    echo ""
    echo "== exp18 analysis =="
    "$PY" -m exp18_mixed_cohort.analyse || rc=1
    echo ""
    if (( rc == 0 )); then echo "GATE: PASS"; else echo "GATE: FAIL"; fi
    return $rc
}

preflight () {
    local rc=0
    check () { if eval "$2" >/dev/null 2>&1; then echo "ok      $1"; else echo "MISSING $1"; rc=1; fi; }
    check "python env (.venv-others)" "[[ -x '$PY' ]]"
    check "packages (torch, iterstrat, sklearn, scipy, pandas)" \
        "'$PY' -c 'import torch, iterstrat, sklearn, scipy, pandas'"
    check "braindecode (exp9 EEGNet/LaBraM encoders)" "'$PY' -c 'import braindecode'"
    check "CUDA visible (expected on a GPU node only)" "'$PY' -c 'import torch; assert torch.cuda.is_available()'"
    check "asm_data (clinical CSVs)" "'$PY' -c 'from shared.hep_cohort import ALFRED_CSV, HEP_CSV; assert ALFRED_CSV.exists() and HEP_CSV.exists()'"
    check "27-channel EEG cache (exp2-7, 9, 11, 16, 17)" "ls $OUT/eeg_cache/processed_eeg*.pkl | grep -v std19"
    check "19-channel EEG caches (HEP EEG, exp18)" \
        "[[ -f $OUT/eeg_cache/processed_eeg_std19_alfred.pkl && -f $OUT/eeg_cache/processed_eeg_std19_hep.pkl ]]"
    check "text + SMILES embeddings" "[[ -f $OUT/bert_alfred_1stregimen_eeg_embeddings.npy && -f $OUT/hep_clinicalbert_eeg_embeddings.npy && -f $OUT/chemberta_asm_embeddings.npy ]]"
    check "REVE features (exp15, reve)" "ls $OUT/reve_features_alfred*.npz"
    check "thesisStandalone clone" "[[ -f '$THESIS/analysis/hep_external_validation.py' ]]"
    check "expected-files manifest" "[[ -f clean_rerun_expected.txt ]]"
    echo "experiments $(git rev-parse --short HEAD)  thesisStandalone $(git -C "$THESIS" rev-parse --short HEAD 2>/dev/null)"
    echo "(compare both commits with the laptop before submitting)"
    return $rc
}

items () {
    local t s
    for t in "${TASKS[@]}"; do for s in "${SEEDS[@]}"; do echo "$t:$s"; done; done
}

case "${1:-}" in
    list) items ;;
    preflight) preflight ;;
    verify) verify ;;
    smoke)
        # A quick end-to-end pass of one task on this host's data. Only exp18
        # has a native smoke mode; the others are exercised with the real
        # flags, so run them via slurm with a short --time instead.
        [[ "${2:-}" == exp18_* ]] || { echo "smoke supports exp18_* tasks only" >&2; exit 2; }
        "$PY" -m exp18_mixed_cohort.run_experiments --config "${2#exp18_}" --seeds 42 --smoke \
            --out-dir "/tmp/exp18_smoke_$$" ;;
    "") echo "usage: bash rerun_clean.sh {preflight|list|verify|smoke <task>|<task>:<seed>}" >&2; exit 2 ;;
    *:*)
        task="${1%%:*}"; seed="${1##*:}"
        marker="$DONE/${task}_s${seed}.done"
        mkdir -p "$DONE"
        if [[ -f "$marker" && "${FORCE:-0}" != 1 ]]; then
            echo "== $task seed $seed already done ($marker); FORCE=1 to rerun =="; exit 0
        fi
        echo "== $task seed $seed  (host $(hostname), $(date -Is)) =="
        start=$(date +%s)
        if run_task "$task" "$seed"; then
            echo "$(date -Is) $(( $(date +%s) - start ))s $(git rev-parse --short HEAD)" > "$marker"
            echo "== $task seed $seed done in $(( $(date +%s) - start ))s =="
        else
            echo "== $task seed $seed FAILED ==" >&2; exit 1
        fi ;;
    *) echo "expected <task>:<seed>, e.g. exp4:42 (see: bash rerun_clean.sh list)" >&2; exit 2 ;;
esac
