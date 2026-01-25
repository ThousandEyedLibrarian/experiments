# ASM Outcome Prediction: Architecture Documentation

**Date:** 25 January 2026
**Dataset:** 151 patients with EEG recordings and ASM outcomes (107 with all modalities)
**Diagrams:** Attached as `.drawio` files (open with [app.diagrams.net](https://app.diagrams.net))

---

## Overview

Three experiment sets predict anti-seizure medication (ASM) treatment response using multimodal fusion:

| Experiment | Inputs | Fusion Methods |
|------------|--------|----------------|
| **Exp1** | Text reports (LLM) + Drug structure (SMILES) | (a) ConcatMLP, (b) FuseMoE |
| **Exp2** | EEG signals + Drug structure (SMILES) | (a) ConcatMLP, (b) FuseMoE |
| **Exp3** | Text + EEG + SMILES (triple modality) | (a) ConcatMLP, (b) FuseMoE |

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

## Training Configuration

| Parameter | Exp1a | Exp1b | Exp2a | Exp2b | Exp3a | Exp3b |
|-----------|-------|-------|-------|-------|-------|-------|
| Learning Rate | 1e-4 | 5e-5 | 1e-4 | 1e-4 | 1e-4 | 5e-5 |
| Batch Size | 16 | 16 | 8 | 8 | 8 | 8 |
| Max Epochs | 100 | 100 | 100 | 100 | 100 | 100 |
| Early Stopping | 15 | 20 | 20 | 20 | 20 | 20 |
| Optimizer | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW |
| CV Folds | 5 | 5 | 5 | 5 | 5 | 5 |

---

## Results Summary

| Model | Fusion | Params | Best AUC |
|-------|--------|--------|----------|
| Exp1a | Concat+MLP | 953K | 0.640 |
| Exp1b | FuseMoE | 2.6M | 0.658 |
| **Exp2a** | **Concat+MLP** | **1.2M** | **0.668** |
| Exp2b | FuseMoE | 2.8M | 0.608 |
| Exp3a | Concat+MLP | 2.5M | *pending* |
| Exp3b | FuseMoE | 4.7M | *pending* |

**Current best:** Exp2a (EEG + SMILES with MLP fusion) achieved AUC 0.668.
**Exp3 hypothesis:** Triple modality fusion may improve over dual modality baselines.

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
| EEG Preprocessing | `eeg_preprocessing.drawio` |
| SimpleCNN Encoder | `simplecnn_encoder.drawio` |
