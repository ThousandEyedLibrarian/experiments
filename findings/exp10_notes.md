# Experiment 10: Direct LLM Text Modality

**Date:** 12 February 2026
**Objective:** Run LLM inference at training time instead of pre-computed embeddings, enabling end-to-end fine-tuning of the text encoder

---

## Motivation

Previous experiments (Exp1, Exp5b) used pre-computed text embeddings from frozen LLM encoders. This approach:
- Cannot adapt the text representation to the downstream task
- Requires separate embedding extraction before training
- Limits exploration of different encoding strategies

Exp10 wraps HuggingFace transformer models directly in the training pipeline, allowing:
- Frozen encoder mode (feature extraction, matching previous approach)
- Fine-tuned encoder mode (backprop through last N transformer layers)
- Easy comparison across different biomedical LLMs

---

## Architecture

### Late Fusion Pattern (matching Exp5b)

```
Clinical (19D) -> ModalityEncoder -> 64D --+
                                            |-> Concat (128D) -> Classifier -> 2
Raw Text -> LLM Tokeniser -> LLM Encoder -> embed_dim -> ModalityEncoder -> 64D --+
```

### LLM Encoder

The LLMEncoder class wraps any HuggingFace `AutoModel` and extracts [CLS] token embeddings:

1. Pre-tokenise all EEG reports at dataset creation (deterministic, no gradients)
2. Store input_ids and attention_masks as tensors in the Dataset
3. During training, pass tokenised inputs through the LLM encoder
4. Extract [CLS] token (first token) from last hidden state

### Models Tested

| Model | HuggingFace ID | Embed Dim | Domain |
|-------|----------------|-----------|--------|
| PubMedBERT | NeuML/pubmedbert-base-embeddings | 768 | Biomedical literature |
| ClinicalBERT | medicalai/ClinicalBERT | 768 | Clinical text |
| Qwen 2.5 0.5B | Qwen/Qwen2.5-0.5B | 896 | General-purpose multilingual |

### Experiment Matrix

**Frozen encoder (Phase 1):**
- `exp10_pubmedbert_frozen` - PubMedBERT + clinical features
- `exp10_clinicalbert_frozen` - ClinicalBERT + clinical features
- `exp10_qwen_frozen` - Qwen 2.5 + clinical features

**Fine-tuned encoder (Phase 2 - after identifying best frozen model):**
- `exp10_pubmedbert_finetune` - PubMedBERT, last 2 layers unfrozen
- `exp10_clinicalbert_finetune` - ClinicalBERT, last 2 layers unfrozen

---

## Training Configuration

### Frozen Encoder

| Parameter | Value |
|-----------|-------|
| Batch Size | 8 |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Max Epochs | 100 |
| Early Stopping | 20 epochs |
| Dropout | 0.3 |

### Fine-tuned Encoder

| Parameter | Value |
|-----------|-------|
| Batch Size | 4 (smaller for backprop through LLM) |
| Encoder LR | 2e-5 (low for pre-trained weights) |
| Head LR | 1e-3 (higher for classification layers) |
| Weight Decay | 1e-4 |
| Max Epochs | 50 |
| Early Stopping | 15 epochs |
| Unfreeze Layers | Last 2 transformer layers |

Differential learning rates prevent catastrophic forgetting of pre-trained representations while allowing the classification head to converge faster.

---

## Key Design Decisions

1. **Pre-tokenisation**: Text is tokenised once per experiment run rather than per batch. This is valid because tokenisation is deterministic and parameter-free. It avoids redundant computation across epochs.

2. **[CLS] token extraction**: Using the [CLS] token rather than mean pooling. This is standard for BERT-family models and matches the pre-training objective.

3. **Pad token handling**: Models without a native pad token (Qwen, GPT-style) use eos_token as pad_token, following HuggingFace convention.

4. **Selective layer unfreezing**: Only the last N transformer layers are unfrozen during fine-tuning. Earlier layers capture general language features; later layers are more task-specific.

---

## Cross-Validation

Uses multilabel stratification on outcome + focal + sex + age_group (matching Exp8 methodology). Falls back to outcome-only stratification if `iterative-stratification` is not installed.

5-fold CV with shuffle, random_state=42.

---

## Results: Frozen Encoder (Phase 1)

**HPC Job:** 51362383 | **Node:** m3n102 (A100 80GB) | **Runtime:** ~20 minutes | **Date:** 12 February 2026, 21:46-22:05

### Frozen Encoder Comparison

| Model | AUC | Bal Acc Tuned | F1 Tuned |
|-------|-----|---------------|----------|
| **Qwen 2.5 0.5B** | **0.689 +/- 0.088** | **0.717 +/- 0.073** | 0.666 +/- 0.119 |
| ClinicalBERT | 0.644 +/- 0.121 | 0.695 +/- 0.092 | 0.671 +/- 0.130 |
| PubMedBERT | 0.635 +/- 0.096 | 0.671 +/- 0.075 | 0.674 +/- 0.058 |

### Per-Fold AUC

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|-------|--------|--------|--------|--------|--------|
| Qwen 2.5 0.5B | 0.769 | 0.790 | 0.671 | 0.542 | 0.671 |
| ClinicalBERT | 0.788 | 0.629 | 0.580 | 0.458 | 0.762 |
| PubMedBERT | 0.744 | 0.685 | 0.643 | 0.458 | 0.643 |

### Per-Fold Balanced Accuracy (Tuned)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|-------|--------|--------|--------|--------|--------|
| Qwen 2.5 0.5B | 0.728 | 0.794 | 0.710 | 0.583 | 0.769 |
| ClinicalBERT | 0.801 | 0.647 | 0.636 | 0.583 | 0.808 |
| PubMedBERT | 0.766 | 0.717 | 0.664 | 0.542 | 0.664 |

### Key Observations

1. **Qwen 2.5 0.5B (general-purpose) outperforms both biomedical models** in frozen mode - surprising given PubMedBERT and ClinicalBERT were pre-trained on biomedical/clinical text
2. **Fold 4 consistently weakest** across all 3 models (AUC 0.458-0.542), suggesting a data composition issue in that fold rather than model-specific weakness
3. **ClinicalBERT has highest variance** (AUC std 0.121) - performance ranges from 0.458 to 0.788
4. **PubMedBERT most stable on F1** (std 0.058) despite lower AUC - fewer catastrophic fold failures
5. **Frozen results comparable to Exp5b pre-computed embeddings** (ClinicalBERT frozen 0.644 vs Exp5b pre-computed 0.676) - small gap suggests pre-computed pipeline was well-calibrated
6. **Runtime efficient:** All 3 models completed in ~20 minutes on A100 80GB

### Comparison with Exp5b (Pre-computed Embeddings)

| Model | Exp10 Frozen AUC | Exp5b Pre-computed AUC | Delta |
|-------|------------------|------------------------|-------|
| ClinicalBERT | 0.644 | 0.676 | -0.032 |
| PubMedBERT | 0.635 | 0.620 | +0.015 |

The small differences confirm that frozen encoder mode produces comparable results to pre-computed embeddings, validating the Exp10 pipeline.

---

## Results: Fine-tuned Encoder (Phase 2)

**HPC Job:** 51370915 | **Node:** A100 80GB | **Date:** 13 February 2026

Last 2 transformer layers unfrozen with differential learning rates (encoder: 2e-5, head: 1e-3). Batch size reduced to 4 for backprop through LLM layers.

### Fine-tuned Encoder Comparison

| Model | AUC | Bal Acc Tuned | F1 Tuned |
|-------|-----|---------------|----------|
| **ClinicalBERT (fine-tuned)** | **0.691 +/- 0.081** | **0.723 +/- 0.057** | **0.698 +/- 0.106** |
| PubMedBERT (fine-tuned) | 0.638 +/- 0.144 | 0.690 +/- 0.084 | 0.674 +/- 0.101 |

### Per-Fold AUC (Fine-tuned)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|-------|--------|--------|--------|--------|--------|
| ClinicalBERT (fine-tuned) | 0.737 | 0.776 | 0.643 | 0.556 | 0.741 |
| PubMedBERT (fine-tuned) | 0.801 | 0.594 | 0.720 | 0.382 | 0.692 |

### Per-Fold Balanced Accuracy (Fine-tuned)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|-------|--------|--------|--------|--------|--------|
| ClinicalBERT (fine-tuned) | 0.756 | 0.769 | 0.692 | 0.625 | 0.773 |
| PubMedBERT (fine-tuned) | 0.801 | 0.692 | 0.696 | 0.542 | 0.717 |

### Frozen vs Fine-tuned Comparison

| Model | Frozen AUC | Fine-tuned AUC | Delta |
|-------|-----------|---------------|-------|
| ClinicalBERT | 0.644 | 0.691 | **+0.047** |
| PubMedBERT | 0.635 | 0.638 | +0.003 |

### Key Observations

1. **ClinicalBERT benefits substantially from fine-tuning** (+0.047 AUC, +0.028 Bal Acc) - task-specific adaptation of later layers captures clinical nuance
2. **PubMedBERT barely improves with fine-tuning** (+0.003 AUC) - already well-suited to clinical text in frozen mode, or possibly more sensitive to overfitting with small data
3. **Fine-tuned ClinicalBERT (0.691) marginally outperforms frozen Qwen 2.5 0.5B (0.689)** - fine-tuning a smaller domain model can match a larger general-purpose model
4. **Fold 4 remains weakest** for both models (AUC 0.382-0.556) - consistent with frozen results, confirming data composition issue
5. **PubMedBERT fine-tuning has very high variance** (AUC std 0.144) - fold 4 AUC 0.382 is a near-complete failure
6. **Qwen 2.5 fine-tuning not yet tested** - potential next step given frozen Qwen already competitive

---

## Usage

```bash
# Run all frozen experiments
python -m exp10_direct_llm.run_experiments

# Run specific experiment
python -m exp10_direct_llm.run_experiments --experiment exp10_pubmedbert_frozen

# Run fine-tuning experiments
python -m exp10_direct_llm.run_experiments --finetune

# Specify device
python -m exp10_direct_llm.run_experiments --device cuda
```

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | LLM model configs, training params, experiment definitions |
| `data_pipeline.py` | Load raw text + clinical features, pre-tokenise, create datasets |
| `models/llm_encoder.py` | HuggingFace model wrapper with freeze/unfreeze |
| `models/fusion.py` | Clinical + LLM late fusion model |
| `training.py` | Training loop, evaluation, cross-validation |
| `run_experiments.py` | CLI entry point |
