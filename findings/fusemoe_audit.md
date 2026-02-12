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

## Recommendations

1. **Fix the dead import**: Remove `from commons import ProbGaussianNoise` from
   `shared/fuse_moe.py` or add a try/except guard, so the module can be imported
   for reference and testing.

2. **Consider Laplace gating**: The professor's Laplace gating with learned expert
   embeddings is more expressive than our linear softmax gate. Worth testing whether
   this improves expert specialisation.

3. **Consider temperature annealing**: The professor uses temperature decay to sharpen
   routing decisions over training. This could reduce expert collapse.

4. **Consider deeper experts**: The professor's 3-layer residual experts are deeper
   than our 2-layer MLPs. Given our small dataset (107-286 patients), our shallower
   experts may be more appropriate to avoid overfitting.

5. **MI loss vs CV-squared**: The MI-based load balancing is more principled. Could
   be tested as a drop-in replacement in our training loop.

6. **Fix exp2 KL loss**: The current formulation in `exp2_fusion/models/fusion.py`
   (line 210) mixes KL divergence terms - should be reviewed for correctness.
