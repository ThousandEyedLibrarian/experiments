# Analysis plan: clean-CV rerun and exp18 mixed-cohort experiment

Frozen on 2026-09-21, before any clean-CV or mixed-cohort result was computed.
The commit that adds this file is the reference point. Any later change is
listed under "Deviations" with its date and reason; nothing above that section
is edited after the first clean result exists.

## 1. Why

Every CV loop in exp1-17 and the HEP1 external-validation scripts chose the
early-stopping epoch, the restored weights, the Youden threshold (and, in exp1
and exp2, the `ReduceLROnPlateau` steps) using AUC on the outer held-out fold,
then reported that same fold. Table rows also used the best encoder
combination per row, again chosen on those folds. Both make the published
internal AUCs optimistic. This plan fixes the protocol for a full rerun and
for the new mixed-cohort experiment the supervisor requested.

## 2. Clean CV protocol (every single-cohort experiment)

- **Outer split:** 5 folds, iterative multi-label stratification on
  `outcome`, `focal`, `sex` (`shared.cv_splits.outer_splits(mode="multilabel")`),
  seed 42. This matches what the Methods already state.
- **Inner split:** 20% of each outer training fold, stratified on `outcome`,
  seed `42 + fold` (`shared.cv_splits.inner_val_split`). It is used only for
  early stopping, LR scheduling and the Youden threshold.
- **Early stopping:** criterion and patience unchanged from each experiment's
  config (inner-validation AUC). Best weights are restored, and the outer test
  fold is scored once with those weights.
- **Operating point:** Youden-J threshold chosen on the inner-validation
  predictions, logged with each fold, and applied unchanged to the outer test
  fold.
- **Hyperparameters:** frozen at the values in each `expN/config.py` at this
  commit. No further tuning.
- **Determinism:** `shared.determinism.enable_determinism` with seed 42 (per
  fold where the experiment already does so).
- **Output naming:** clean files carry the suffix `_sp-multilabel_iv20`.

## 3. Headline table rows: pre-specified encoders

Every row uses the headline encoder set: ChemBERTa (drug), ClinicalBERT
(mean-pooled, frozen text), EEG2Vec 256-d with the two-layer transformer
aggregator (EEG), late-fusion MLP.

| Row | Config |
|---|---|
| Clinical | `exp5a_chemberta` |
| Clinical + Text | `exp6a_clinicalbert_chemberta` |
| Clinical + EEG | `exp6b_eeg2vec_chemberta` (new config: EEG2Vec in place of SimpleCNN) |
| Clinical + Text + EEG (full model) | `exp7a` |
| Text | `exp1a_clinicalbert_chemberta` |
| EEG | `exp2_eeg2vec_chemberta_mlp` (new config) |
| Text + EEG | `exp3a_clinicalbert_chemberta` with EEG2Vec (new config) |
| Clinical, drug removed | `exp4a_mlp` |
| Clinical + Text, drug removed | `exp5b_clinicalbert` |
| Clinical + EEG, drug removed | `exp5c_eeg2vec` |

All other encoder, aggregator and fusion variants (SimpleCNN, EEG2Vec-128,
PubMedBERT, SMILES Transformer, FuseMoE, REVE-base) are reported only in the
Supplementary and labelled exploratory.

## 4. Metrics and tests (single-cohort)

- **Primary metric:** patient-level AUC pooled across folds by the existing
  random-effects method (per-fold bootstrap variance, method-of-moments tau2,
  Knapp-Hartung t-interval).
- **Secondary:** sensitivity, specificity, precision and balanced accuracy at
  the inner-validation threshold, pooled the same way.
- **Pairwise:** all-pairs DeLong on out-of-fold predictions, BH-corrected,
  exploratory (as before).
- **Decomposition (exp4a only):** a 2x2 of splitter (legacy outcome-only vs
  multilabel) x early stopping (outer fold vs inner 20%). It attributes the
  old-vs-clean change to the splitter, the leak fix and the 20% training-data
  loss. Descriptive only.

## 5. HEP1 external validation

Same six configurations as the current Table 3, with the same encoder set as
Section 3 (19-channel EEG2Vec for every EEG configuration), trained under the
clean protocol and applied to HEP1 as a five-fold ensemble. External AUC with a
patient-level percentile bootstrap CI (2000 resamples, seed 42). Reverse,
focal-matched and reduced-capacity analyses are rerun under the same protocol.

## 6. exp18 mixed-cohort experiment

**Cohorts:** Melbourne (deduplicated, n = 198) and HEP1 (n = 438), minus any
cross-cohort duplicates confirmed by the overlap audit (kept in Melbourne
only). pids are prefixed `MEL_` / `HEP_`.

**Configurations:** the six Table 3 configurations (Exp4a, Exp5a, Exp5b,
Exp5c, Exp6b, Exp7a) on the portable pipeline: ChemBERTa, mean-pooled
ClinicalBERT, 19-channel EEG2Vec.

**Splits:**
- Outer: 5-fold `StratifiedKFold` on the joint outcome x cohort key.
- Inner: 20% of each arm's training data, stratified on the same key.
- Seeds: 42-46 for the clinical and text configurations (Exp4a, Exp5a,
  Exp5b); 42-44 for the EEG configurations.

**Arms** (every arm scores every outer test patient in both cohorts):
- `mixed`: trained on the full outer training fold.
- `mel_only`: trained on its Melbourne patients only.
- `hep_only`: trained on its HEP1 patients only.
- `mixed_sizematched` (Exp4a only): for each test cohort, the mixed training
  fold subsampled (cohort proportions preserved) to the size of that cohort's
  own-only arm, 10 draws.

Preprocessors and any normalisation are fitted on each arm's own fit set.

**Primary metrics:**
- Overall: the cohort-stratified AUC, i.e. the probability that a seizure-free
  patient outranks a non-seizure-free patient **from the same cohort**. This
  equals the pair-weighted mean of the within-cohort AUCs and cannot be
  inflated by learning cohort base rates.
- Per cohort: out-of-fold AUC on that cohort's test patients, averaged over
  seeds, with a DeLong 95% CI per seed.

**Secondary metrics:**
- The whole-fold AUC requested by the supervisor (per fold, mean and SD),
  always reported next to the **cohort-only floor**: the AUC of a score equal to
  the training-fold seizure-free rate of the patient's cohort.
- Calibration per cohort: Brier score, calibration-in-the-large and
  calibration slope.

**Primary hypothesis tests** (two tests, Holm-adjusted):
1. Exp4a, Melbourne test patients: `mixed` vs `mel_only`.
2. Exp4a, HEP1 test patients: `mixed` vs `hep_only`.

Each uses the Nadeau-Bengio corrected resampled t-test on the per-(seed, fold)
differences in within-cohort AUC: J = 25 differences, variance inflation
`1/J + n_test/n_train`, df = J - 1.

**Exploratory:** every other contrast (other configurations, `mixed` vs the
other-cohort arm, size-matched draws), with paired DeLong per seed on
out-of-fold predictions. Reported without claims of significance.

**Sensitivity:** Exp4a, Exp5a and Exp5b, all arms, excluding all 29 HEP1
patients from the Royal Melbourne Hospital site (`RMH####`), whatever the
overlap audit finds.

**Sanity checks** (not tests):
- `mel_only` on Melbourne should be close to the clean single-cohort Exp4a.
- `mel_only` on HEP1 should be near chance, as in the existing external result.
- The cohort-only floor should be near 0.57 on the clinical tier.

## Deviations

(none yet)
