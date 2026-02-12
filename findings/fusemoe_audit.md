# FuseMoE Implementation Audit

Comparison of professor's reference implementation (`shared/fuse_moe.py`) against our
experiment-specific implementations.

## Summary of Differences

| Component | Professor's (`shared/`) | Ours (exp1) | Ours (exp2) |
|-----------|------------------------|-------------|-------------|
| **Gating** | Laplace (L2 distance + exp) | Softmax (linear gate) | Softmax (linear gate) |
| **Routing** | Learned expert embeddings + router embeddings | Linear gate -> top-k | Linear gate -> top-k |
| **Load Balancing** | MI loss (entropy-based) | CV-squared loss | KL-based loss |
| **Expert Architecture** | 3-layer residual blocks with LayerNorm | 2-layer MLP (ReLU) | 2-layer MLP (GELU) |
| **Temperature** | Annealing (exponential decay) | None | None |
| **Strategies** | Joint, PerModality, Disjoint | Cross-modal MoE | Sparse MoE + cross-attention |
| **Noise** | None (uses Laplace distance) | Gaussian noise on gate logits | None |
| **Missing dep** | `ProbGaussianNoise` from `commons` | None | None |

## Gating Mechanism Comparison

### Professor's: Laplace Gating

- Router produces a single embedding per sample via a 2-layer MLP with GELU + LayerNorm
- Expert embedder produces one embedding per expert per sample via learned linear map + 2-layer MLP
- Routing score = exp(-||router_emb - expert_emb||_2 / temperature)
- Top-k selection, normalised by sum of all scores
- Temperature anneals from max (1.0) to min (0.5) with decay factor 0.9995

### Ours (exp1): Softmax Gating with Noise

- Linear gate: input -> num_experts logits
- Gaussian noise (std=0.1) added during training for load balancing
- Top-k selection, re-normalised via softmax over top-k logits only
- No temperature parameter

### Ours (exp2): Softmax Gating without Noise

- Linear gate: input -> num_experts logits
- Softmax over all experts, then top-k selection
- Re-normalised by dividing top-k probs by their sum
- No noise injection

## Load Balancing Comparison

### Professor's: MI Loss

- Computes mutual information between routing probabilities
- E[H(p)] - H(E[p]) formulation (negative JSD)
- Encourages diverse expert usage across the batch
- More principled information-theoretic approach

### Ours (exp1): CV-squared

- Coefficient of variation of expert importance scores
- importance = sum of gate probs per expert across batch
- loss = (std(importance) / mean(importance))^2
- Simpler but less principled than MI-based approach

### Ours (exp2): KL-based

- KL divergence between expert usage distribution and uniform
- Directly penalises deviation from uniform expert usage
- loss = sum(expert_usage * log(uniform) - log(expert_usage) * uniform)
- Note: this formulation is not standard KL divergence - it mixes the terms

## Expert Architecture Comparison

### Professor's

- 3-layer residual blocks
- Each block: Linear -> GELU -> Dropout(0.2) -> Linear -> Dropout(0.2)
- Residual connection + LayerNorm after each block
- Total depth: init_linear + 3 residual blocks = deeper expert networks
- Uses GELU activation throughout

### Ours (exp1)

- 2-layer MLP per expert: Linear -> ReLU -> Linear
- No residual connections within experts
- No normalisation layers within experts
- Shallower but faster to train

### Ours (exp2)

- 2-layer MLP per expert: Linear -> GELU -> Dropout -> Linear
- No residual connections within experts
- Separate Expert class with cleaner structure
- Also shallower than professor's

## Strategy Comparison

### Professor's Strategies

1. **JointMoE**: Concatenate all modalities -> single router -> shared experts
2. **PerModalityRouterMoE**: Separate router per modality -> shared experts -> sum outputs
3. **DisjointMoE**: Separate router + separate experts per modality -> sum outputs

All three strategies use the same Laplace gating and MI loss.

### Our Approach (exp1)

- Cross-modal self-attention per modality (text and SMILES attend to each other)
- Joint MoE fusion after cross-attention
- MoEFusionLayer combines self-attention, MoE, and FFN in transformer-like blocks
- Multiple stacked fusion layers (default 2)

### Our Approach (exp2)

- Cross-attention: EEG queries attend to SMILES keys/values
- Concatenation of attended EEG + SMILES projections
- Single SparseMoE layer on concatenated features
- Classification head after MoE output

## Missing Dependency

The professor's implementation imports `ProbGaussianNoise` from `commons` (line 7 of
`shared/fuse_moe.py`), but this module does not exist in the repository. This import is
not used in any of the three MoE strategy classes - it may be used elsewhere in the
professor's codebase or was left as a dead import. The `shared/fuse_moe.py` will fail to
import as-is due to this missing dependency.

## Recommendations (Original)

1. **Fix the dead import** - RESOLVED
2. **Consider Laplace gating** - RESOLVED
3. **Consider temperature annealing** - RESOLVED
4. **Consider deeper experts** - RESOLVED
5. **MI loss vs CV-squared** - RESOLVED
6. **Fix exp2 KL loss** - RESOLVED

---

## Resolution (2026-02-13)

All experiment MoE implementations have been rewritten to use `shared/fuse_moe.py`
(the professor's reference implementation). The following changes were made:

### Changes Applied

| Issue | Before | After |
|-------|--------|-------|
| Gating mechanism | Softmax (linear gate) | Laplace distance-based gating |
| Load balancing loss | CV-squared (exp1b) / malformed KL (exp2b, exp3b, exp7b) | MI loss (information-theoretic) |
| Expert architecture | 2-layer MLP (no residuals) | 3-layer residual blocks with LayerNorm |
| Temperature | None | Exponential annealing (1.0 -> 0.5, decay 0.9995) |
| Dead import | `ProbGaussianNoise` commented out | Already commented out, verified importable |

### Strategy Assignments

| Experiment | Strategy | Rationale |
|------------|----------|-----------|
| exp1b | `FuseMoE("permodality", num_modalities=2)` | Text + SMILES routed separately through shared experts in each MoEFusionLayer |
| exp2b | `FuseMoE("joint", num_modalities=1)` | Single concatenated (attended_eeg + smiles) input, joint routing |
| exp3b | `FuseMoE("permodality", num_modalities=3)` | Text, EEG, SMILES modality tokens routed separately after cross-modal self-attention |
| exp7b | `FuseMoE("permodality", num_modalities=4)` | Clinical, text, EEG, SMILES modality tokens routed separately after cross-modal self-attention |

### Training Loop Updates

Temperature annealing added to all MoE training loops (exp1b, exp2b, exp3b, exp7b):
- `global_step` counter initialised at 0 per fold
- `model.update_temperature(global_step)` called after each `optimizer.step()`
- Temperature decays exponentially per batch: `max(0.5, 1.0 * 0.9995^step)`

### Files Modified

**Model files:**
- `exp1_fusion/models/fusemoe.py` - Removed `SparseGatedMoE`, use `shared.fuse_moe.FuseMoE`
- `exp2_fusion/models/fusion.py` - Removed `Expert` + `SparseMoELayer`, use `shared.fuse_moe.FuseMoE`
- `exp3_fusion/models/triple_fusemoe.py` - Removed `Expert` + `SparseMoELayer`, use `shared.fuse_moe.FuseMoE`
- `exp7_all_modalities/models.py` - Removed `Expert` + `SparseMoELayer`, use `shared.fuse_moe.FuseMoE`

**Training files:**
- `exp1_fusion/training.py` - Added `global_step` + temperature annealing
- `exp2_fusion/training.py` - Added `global_step` + temperature annealing
- `exp3_fusion/training.py` - Added `global_step` + temperature annealing
- `exp7_all_modalities/training.py` - Added `global_step` + temperature annealing
