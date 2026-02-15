# Experiment 11: EEG2Vec 128D Upgrade

**Date:** 15 February 2026
**Dataset:** Varies by base experiment (n=107 for exp3a, n=151 for exp6b)

---

## Objective

Test EEG2Vec encoder with 128D embeddings as a drop-in replacement for SimpleCNN across existing base experiments, validating the exp9 ablation findings (128D optimal, MeanMax competitive) in multi-modal settings.

Two aggregator types tested per configuration:
- **Transformer** (2-layer, 4 heads) - standard from exp9 baseline
- **MeanMax** (concatenated mean + max pooling) - best alternative from exp9 ablation

---

## Base Experiments

| Base | Modalities | Dataset | Configs |
|------|------------|---------|---------|
| Exp3a | LLM + EEG + SMILES (triple MLP) | n=107 | 8 (2 text x 2 SMILES x 2 aggregators) |
| Exp6b | Clinical + SMILES + EEG | n=151 | 4 (2 SMILES x 2 aggregators) |
| Exp7a | Clinical + LLM + EEG + SMILES (quad MLP) | n=107 | 4 (2 text x 2 aggregators) - **DID NOT COMPLETE** |

---

## EEG Configuration

| Parameter | Value |
|-----------|-------|
| Encoder | EEG2Vec |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Max windows | 120 |
| Window chunk size | 32 |
| Channels | 27 |
| Samples per window | 2000 (10s at 200Hz) |

---

## Results: Exp3a Base (Triple MLP, n=107)

| Text Model | SMILES Model | Aggregator | AUC | Bal Acc Tuned | F1 Tuned |
|------------|--------------|-----------|-----|---------------|----------|
| **ClinicalBERT** | **SMILES-Trf** | **MeanMax** | **0.736 +/- 0.036** | 0.752 +/- 0.034 | **0.779 +/- 0.020** |
| ClinicalBERT | SMILES-Trf | Transformer | 0.733 +/- 0.087 | **0.777 +/- 0.056** | 0.764 +/- 0.085 |
| ClinicalBERT | ChemBERTa | Transformer | 0.729 +/- 0.078 | 0.756 +/- 0.084 | 0.751 +/- 0.110 |
| ClinicalBERT | ChemBERTa | MeanMax | 0.721 +/- 0.104 | 0.742 +/- 0.095 | 0.723 +/- 0.133 |
| PubMedBERT | SMILES-Trf | MeanMax | 0.658 +/- 0.086 | 0.710 +/- 0.070 | 0.730 +/- 0.084 |
| PubMedBERT | SMILES-Trf | Transformer | 0.654 +/- 0.082 | 0.696 +/- 0.065 | 0.681 +/- 0.071 |
| PubMedBERT | ChemBERTa | MeanMax | 0.646 +/- 0.071 | 0.680 +/- 0.061 | 0.683 +/- 0.074 |
| PubMedBERT | ChemBERTa | Transformer | 0.625 +/- 0.073 | 0.697 +/- 0.075 | 0.723 +/- 0.050 |

Previous exp3a best (SimpleCNN): AUC 0.687 (ClinicalBERT + ChemBERTa + MLP).
**EEG2Vec upgrade: +0.049 AUC** (0.687 -> 0.736).

---

## Results: Exp6b Base (Clinical + SMILES + EEG, n=151)

| SMILES Model | Aggregator | AUC | Bal Acc Tuned | F1 Tuned |
|--------------|-----------|-----|---------------|----------|
| **ChemBERTa** | **Transformer** | **0.697 +/- 0.070** | 0.694 +/- 0.050 | 0.672 +/- 0.096 |
| ChemBERTa | MeanMax | 0.693 +/- 0.051 | **0.712 +/- 0.044** | 0.684 +/- 0.037 |
| SMILES-Trf | Transformer | 0.685 +/- 0.089 | 0.722 +/- 0.084 | **0.742 +/- 0.073** |
| SMILES-Trf | MeanMax | 0.654 +/- 0.065 | 0.685 +/- 0.052 | 0.639 +/- 0.162 |

Previous exp6b best (SimpleCNN): AUC 0.647 (SimpleCNN + SMILES-Trf).
**EEG2Vec upgrade: +0.050 AUC** (0.647 -> 0.697).

---

## Results: Exp7a Base (Quad MLP)

**Did not complete.** The exp7a EEG2Vec configs (4 configurations) did not return results. Re-submission required.

---

## Key Observations

1. **EEG2Vec upgrade improves triple MLP by +0.049 AUC** (0.687 -> 0.736) - the largest single-component improvement observed across all experiments

2. **ClinicalBERT consistently outperforms PubMedBERT** across all combinations (0.721-0.736 vs 0.625-0.658 for exp3a)

3. **MeanMax aggregator achieves best AUC (0.736) with lowest variance** (std 0.036) for exp3a, confirming the exp9 ablation finding that MeanMax is competitive with transformers

4. **Transformer and MeanMax perform comparably for exp6b** (0.697 vs 0.693), though MeanMax has lower variance (std 0.051 vs 0.070)

5. **EEG2Vec upgrades exp6b by +0.050 AUC** (0.647 -> 0.697), nearly closing the gap between EEG fusion (0.697) and text fusion (exp6a, 0.702)

6. **SMILES-Trf slightly outperforms ChemBERTa** when paired with MeanMax for exp3a (0.736 vs 0.721), though ChemBERTa is better for exp6b (0.697 vs 0.685)

7. **MeanMax F1 tuned is remarkably stable** (std 0.020 for best config) - lowest F1 variance observed across all experiments

---

## Files

- `exp11_eeg_upgrade/__init__.py` - Package marker
- `exp11_eeg_upgrade/config.py` - 16 experiment configurations (8 exp3a + 4 exp6b + 4 exp7a)
- `exp11_eeg_upgrade/models.py` - TripleMLPv2, ClinicalEEGFusionv2, QuadMLPv2 with configurable EEG encoder and aggregator
- `exp11_eeg_upgrade/run_experiments.py` - Entry point with base/aggregator filters

---

## Next Steps

1. Re-submit exp7a EEG2Vec configs (quad modality) - most important missing result
2. If exp7a improves with EEG2Vec, it would set a new overall best (current: AUC 0.798 with SimpleCNN)
3. Test EEG2Vec in exp2 (EEG + SMILES) base experiment
4. Consider combining best aggregator (MeanMax) with best text model (ClinicalBERT) and best SMILES model (context-dependent) for final optimised configuration
