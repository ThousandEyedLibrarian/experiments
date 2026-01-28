# ASM Outcome Prediction: Architecture Documentation

**Date:** 28 January 2026
**Dataset:** 205 patients total (varies by modality availability)
**Diagrams:** Attached as `.drawio` files (open with [app.diagrams.net](https://app.diagrams.net))

---

## Overview

Five experiment sets predict anti-seizure medication (ASM) treatment response:

| Experiment | Inputs | Fusion Methods | Dataset |
|------------|--------|----------------|---------|
| **Exp1** | Text reports (LLM) + Drug structure (SMILES) | (a) ConcatMLP, (b) FuseMoE | 121 |
| **Exp2** | EEG signals + Drug structure (SMILES) | (a) ConcatMLP, (b) FuseMoE | 151 |
| **Exp3** | Text + EEG + SMILES (triple modality) | (a) ConcatMLP, (b) FuseMoE | 107 |
| **Exp4** | Clinical features only (baseline) | (a) MLP, (b) Attention | 205 |
| **Exp5** | Clinical + single modality | (a) +SMILES, (b) +Text, (c) +EEG | varies |

---

## Experiment 1: LLM + SMILES Fusion

### Input Embeddings

| Modality | Models | Dimension |
|----------|--------|-----------|
| Text Reports | ClinicalBERT, PubMedBERT | 768 |
| Drug Structure | ChemBERTa, SMILES Transformer | 768 / 256 |

### Exp1a: ConcatMLP (~953K params)
Concatenate embeddings → 4-layer MLP classifier
**Diagram:** `exp1a_concat_mlp.drawio`

| Layer | Dims | Activation | Regularisation |
|-------|------|------------|----------------|
| FC1 | 1536→512 | ReLU | LayerNorm, Dropout(0.3) |
| FC2 | 512→256 | ReLU | LayerNorm, Dropout(0.3) |
| FC3 | 256→128 | ReLU | LayerNorm, Dropout(0.2) |
| Output | 128→2 | - | - |

### Exp1b: FuseMoE (~2.6M params)
Modality projections → learnable tokens → MoE fusion layers
**Diagram:** `exp1b_fusemoe.drawio`

| Component | Configuration |
|-----------|---------------|
| Projections | 768→256 per modality |
| Self-Attention | 4 heads, dim=256 |
| MoE Layers | 2 layers, 4 experts, top-2 routing |
| Expert FFN | 256→256→256 with GELU |

---

## Experiment 2: EEG + SMILES Fusion

### EEG Preprocessing
**Diagram:** `eeg_preprocessing.drawio`

| Parameter | Value |
|-----------|-------|
| Sampling Rate | 200 Hz |
| Bandpass | 0.1–75 Hz |
| Notch Filter | 50 Hz |
| Duration | Skip 5 min, use next 20 min |
| Windowing | 10s windows (max 120) |
| Channels | 27 (10-20 standard) |

### SimpleCNN Encoder
**Diagram:** `simplecnn_encoder.drawio`

| Layer | Channels | Kernel | Pool |
|-------|----------|--------|------|
| Conv1 | 27→64 | 7 | MaxPool(4) |
| Conv2 | 64→128 | 5 | MaxPool(4) |
| Conv3 | 128→256 | 3 | AdaptiveAvgPool(1) |

### Exp2a: EEG-MLP (~1.2M params)
SimpleCNN → Transformer aggregator (2L, 4H) → concat with SMILES → MLP
**Diagram:** `exp2a_eeg_mlp.drawio`

### Exp2b: EEG-FuseMoE (~2.8M params)
Same EEG path → cross-modal MoE fusion instead of concatenation
**Diagram:** `exp2b_eeg_fusemoe.drawio`

---

## Experiment 3: Triple Modality Fusion (LLM + EEG + SMILES)

### Input Embeddings

| Modality | Models | Dimension |
|----------|--------|-----------|
| Text Reports | ClinicalBERT, PubMedBERT | 768 |
| EEG Signals | SimpleCNN + Transformer | 256 |
| Drug Structure | ChemBERTa, SMILES Transformer | 768 / 256 |

### Exp3a: TripleMLP (~2.5M params)
Project all modalities → 256D → concatenate → MLP classifier
**Diagram:** `exp3a_triple_mlp.drawio`

| Component | Configuration |
|-----------|---------------|
| Text Projection | 768→256 + LayerNorm |
| EEG Encoder | SimpleCNN → Transformer(2L, 4H) → 256 |
| SMILES Projection | dim→256 + LayerNorm |
| Fusion | Concat (768D) → MLP |

### Exp3b: TripleFuseMoE (~4.7M params)
Learnable modality tokens → self-attention → 2× MoE layers → classifier
**Diagram:** `exp3b_triple_fusemoe.drawio`

| Component | Configuration |
|-----------|---------------|
| Modality Projections | All → 256D |
| Learnable Tokens | 3 tokens (text, EEG, SMILES) |
| Self-Attention | 4 heads, dim=256 |
| MoE Layers | 2 layers, 4 experts, top-2 routing |
| Expert FFN | 256→512→256 with GELU |

---

## Experiment 4: Clinical Features Baseline

### Input Features (20D after encoding)

| Category | Features | Encoding |
|----------|----------|----------|
| Binary (13) | sex, pretrt_sz_5, focal, fam_hx, febrile, ci, birth_t, head, drug, alcohol, cvd, psy, ld | 0/1 |
| Numeric (1) | age_init | Z-score normalised |
| Categorical (2) | lesion (3 levels), eeg_cat (3 levels) | One-hot (6D) |

### Exp4a: Clinical MLP (~3.7K params)
**Diagram:** `exp4a_clinical_mlp.drawio`

| Layer | Dims | Activation | Regularisation |
|-------|------|------------|----------------|
| FC1 | 20→64 | ReLU | LayerNorm, Dropout(0.3) |
| FC2 | 64→32 | ReLU | LayerNorm, Dropout(0.3) |
| Output | 32→2 | - | - |

### Exp4b: Clinical Attention (~104K params)
**Diagram:** `exp4b_clinical_attention.drawio`

| Component | Configuration |
|-----------|---------------|
| Feature Embeddings | 20 features × 64D |
| Positional Embeddings | Learnable, 21 positions |
| CLS Token | Learnable, 64D |
| Transformer Encoder | 2 layers, 4 heads |
| Classifier | 64→32→2 |

---

## Experiment 5: Clinical + Single Modality Fusion

### Architecture Pattern (Late Fusion)

All Exp5 variants use the same late fusion architecture:
```
Clinical (20D) → Encoder → 64D ─┐
                                 ├→ Concat (128D) → Classifier → 2
Modality (xD)  → Encoder → 64D ─┘
```

### Exp5a: Clinical + SMILES
**Diagram:** `exp5a_clinical_smiles.drawio`

| Modality | Input Dim | Encoder | Output |
|----------|-----------|---------|--------|
| Clinical | 20 | Linear+ReLU+LN+Dropout | 64D |
| SMILES | 768/256 | Linear+ReLU+LN+Dropout | 64D |

### Exp5b: Clinical + Text
**Diagram:** `exp5b_clinical_llm.drawio`

| Modality | Input Dim | Encoder | Output |
|----------|-----------|---------|--------|
| Clinical | 20 | Linear+ReLU+LN+Dropout | 64D |
| Text | 768 | Linear+ReLU+LN+Dropout | 64D |

### Exp5c: Clinical + EEG
**Diagram:** `exp5c_clinical_eeg.drawio`

| Modality | Input | Encoder | Output |
|----------|-------|---------|--------|
| Clinical | 20D | Linear+ReLU+LN+Dropout | 64D |
| EEG | (windows, 27, 2000) | SimpleCNN→Transformer→MeanPool | 64D |

---

## Training Configuration

| Parameter | Exp1a | Exp1b | Exp2a | Exp2b | Exp3a | Exp3b | Exp4a | Exp4b | Exp5 |
|-----------|-------|-------|-------|-------|-------|-------|-------|-------|------|
| Learning Rate | 1e-4 | 5e-5 | 1e-4 | 1e-4 | 1e-4 | 5e-5 | 1e-3 | 5e-4 | 1e-3 |
| Batch Size | 16 | 16 | 8 | 8 | 8 | 8 | 16 | 8 | 16 |
| Max Epochs | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Early Stopping | 15 | 20 | 20 | 20 | 20 | 20 | 15 | 15 | 15 |
| Optimizer | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW |
| CV Folds | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |

---

## Results Summary

| Experiment | Description | Params | Best AUC | Best Bal Acc |
|------------|-------------|--------|----------|--------------|
| Exp1a | LLM+SMILES MLP | 953K | 0.640 | - |
| Exp1b | LLM+SMILES FuseMoE | 2.6M | 0.658 | - |
| Exp2a | EEG+SMILES MLP | 1.2M | 0.668 | - |
| Exp2b | EEG+SMILES FuseMoE | 2.8M | 0.608 | - |
| Exp3a | Triple MLP | 2.5M | 0.672 | 0.706 |
| Exp3b | Triple FuseMoE | 4.7M | 0.662 | 0.696 |
| **Exp4a** | **Clinical MLP** | **3.7K** | **0.664** | **0.675** |
| Exp4b | Clinical Attention | 104K | 0.636 | 0.673 |
| **Exp5a** | **Clinical+SMILES** | **59K** | **0.689** | **0.682** |
| Exp5b | Clinical+Text | 59K | 0.676 | 0.708 |
| Exp5c | Clinical+EEG | 1.9M | 0.644 | 0.690 |

**Key finding:** Clinical + SMILES fusion (Exp5a) achieves best AUC (0.689), outperforming both clinical-only baseline and embedding-only approaches.

---

## Attached Diagrams

| Diagram | File |
|---------|------|
| Exp1a ConcatMLP | `exp1a_concat_mlp.drawio` |
| Exp1b FuseMoE | `exp1b_fusemoe.drawio` |
| Exp2a EEG-MLP | `exp2a_eeg_mlp.drawio` |
| Exp2b EEG-FuseMoE | `exp2b_eeg_fusemoe.drawio` |
| Exp3a Triple-MLP | `exp3a_triple_mlp.drawio` |
| Exp3b Triple-FuseMoE | `exp3b_triple_fusemoe.drawio` |
| Exp4a Clinical MLP | `exp4a_clinical_mlp.drawio` |
| Exp4b Clinical Attention | `exp4b_clinical_attention.drawio` |
| Exp5a Clinical+SMILES | `exp5a_clinical_smiles.drawio` |
| Exp5b Clinical+Text | `exp5b_clinical_llm.drawio` |
| Exp5c Clinical+EEG | `exp5c_clinical_eeg.drawio` |
| EEG Preprocessing | `eeg_preprocessing.drawio` |
| SimpleCNN Encoder | `simplecnn_encoder.drawio` |
