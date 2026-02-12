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

## Results

*Pending - to be run on HPC*

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
