from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Literal, Union

# ProbGaussianNoise imported from professor's commons module (not included in this repo)
# from commons import ProbGaussianNoise


def compute_mi_loss(p: torch.Tensor):
    # ============================
    # 1) TRUE batch MI computation
    # ============================
    eps = 1e-8

    # E[H(p)]
    H_each = -(p * (p + eps).log()).sum(dim=-1)
    # H_each = self._entropy(p_sel, self.eps)    # (N, G)
    EH = H_each.mean(dim=0)                    # (G,)

    # H(E[p])
    p_bar = p.mean(dim=0)                  # (G, V)
    HE = -(p_bar * (p_bar + eps).log()).sum(dim=-1)       # (G,)

    # Loss (negative JSD)
    mi_loss = EH.mean() - HE.mean()            # scalar <= 0

    # ============================
    # 2) EMA updates (diagnostics)
  
    return mi_loss

def laplace_gating_with_probs(expert_embedding, router_embedding, k, temperature=1):
    """
    Laplace gating

    Args:
        expert_embedding: [B, E, D]
        router_embedding: [B, D] - one embedding per expert
        k: int - top-k experts

    Returns:
        topk_indices: [B, k]
        topk_probs: [B, k]
        all_probs: [B, E]
    """
    router_embedding = router_embedding.unsqueeze(1)  # [B, 1, D]
    distances = torch.linalg.vector_norm(
        router_embedding - expert_embedding, dim=-1, ord=2
    )  # [B, E]
    distances = torch.exp(-distances / temperature)

    topk_scores, topk_indices = torch.topk(distances, k, dim=-1)  # [B, T, k]
    topk_probs = topk_scores / torch.sum(distances, dim=-1, keepdim=True)  # [B, T, k]
    all_probs = distances / torch.sum(distances, dim=-1, keepdim=True)  # [B, T, k]

    return topk_indices, topk_probs, all_probs



class Router(nn.Module):
    def __init__(self, num_experts, input_dim, hidden_dim):
        super(Router, self).__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.map_num_experts = nn.Linear(input_dim, hidden_dim * num_experts)
        self.expert_embedder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        expert_map = self.map_num_experts(x)
        expert_map = torch.reshape(
            expert_map, (expert_map.shape[0], self.num_experts, self.hidden_dim)
        )
        expert_embedding = self.expert_embedder(expert_map)        
        router_embedding = self.router(x)
        return router_embedding, expert_embedding


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Expert, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_linear = torch.nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.n_layers = 3
        for i in range(self.n_layers):
            self.blocks.append(torch.nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(0.2)
                    )
                )
            self.norms.append(torch.nn.LayerNorm(hidden_dim))

    def forward(self, x):
        x = self.init_linear(x)
        for i in range(self.n_layers):
            z = self.blocks[i](x)
            x = z + x
            x = self.norms[i](x)
            
        return x


class JointMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim=512, num_experts=4, k=2, num_modalities=2):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.out_dim = out_dim
        self.num_modalities = num_modalities

        self.router = Router(num_experts, input_dim, hidden_dim)
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, temperature, *inputs):
        x = torch.concat(inputs, dim=-1)
        router_embedding, expert_embedding = self.router(x)
        
        topk_indices, topk_scores, all_probs = laplace_gating_with_probs(expert_embedding, router_embedding, self.k, temperature)

        expert_embeddings = []
        for batch_idx, expert_ids in enumerate(topk_indices):
            _embedding = None
            for tok_idx, expert_idx in enumerate(expert_ids):
                out = self.experts[expert_idx](x[batch_idx]) 
                if _embedding is None:
                    _embedding = out
                else:
                    _embedding = _embedding + out

            expert_embeddings.append(_embedding)

        outs = torch.stack(expert_embeddings, dim=0)
        ent_loss = compute_mi_loss(all_probs)

        return outs, ent_loss


class PerModalityRouterMoE(nn.Module):
    def __init__(self,
                dims: int,
                 hidden_dim, out_dim=512, num_experts=4, k=2, num_modalities=2):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.num_modalities = num_modalities

        self.routers = nn.ModuleList([Router(num_experts, dims, hidden_dim) for _ in range(num_modalities)])                               
        self.experts = nn.ModuleList(
            [Expert(dims, hidden_dim) for d in range(num_experts)]
        )

    def forward(self, temperature, *inputs):
        outputs = []
        scores = []
        for input_idx, x in enumerate(inputs):
            router_embedding, expert_embedding = self.routers[input_idx](x)
            topk_indices, topk_scores, all_probs = laplace_gating_with_probs(expert_embedding, router_embedding, self.k, temperature)

            expert_embeddings = []
            for batch_idx, expert_ids in enumerate(topk_indices):
                _embedding = None
                for topk_idx, expert_idx in enumerate(expert_ids):
                    out = self.experts[expert_idx](x[batch_idx])
                    if _embedding is None:
                        _embedding = out
                    else:
                        _embedding = _embedding + out

                expert_embeddings.append(_embedding)

            expert_embeddings = torch.stack(expert_embeddings, dim=0)
            expert_embeddings = expert_embeddings.unsqueeze(1)

            outputs.append(expert_embeddings)
            scores.append(all_probs)

        ent_loss = compute_mi_loss(torch.concat(scores, dim=0))

        outs = torch.cat(outputs, dim=1)
        outs = torch.sum(outs, dim=1)
        return outs, ent_loss
        

class DisjointMoE(nn.Module):
    def __init__(
        self,
        dims: Union[int, list,tuple],
        hidden_dim,
        out_dim=512,
        num_experts=4,
        k=2,
        num_modalities=2
    ):
        super().__init__()
        self.dims = dims
        self.k = k
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.num_modalities = num_modalities

        self.experts = nn.ModuleList()
        self.routers = nn.ModuleList()
        for idx in range(num_modalities):
            d = dims if type(dims) is int else dims[idx]
            self.routers.append(Router(num_experts, d, hidden_dim))
            self.experts.append(nn.ModuleList(
                [Expert(d, hidden_dim) for _ in range(num_experts)]
            ))              

    def forward(self, temperature, *inputs):
        outputs = []
        scores = []
        for input_idx, x in enumerate(inputs):
            router_embedding, expert_embedding = self.routers[input_idx](x)
            topk_indices, topk_probs, all_probs = laplace_gating_with_probs(expert_embedding, router_embedding, self.k, temperature)

            expert_embeddings = []
            for batch_idx, expert_ids in enumerate(topk_indices):
                _embedding = None
                for topk_idx, expert_idx in enumerate(expert_ids):
                    out = self.experts[input_idx][expert_idx](x[batch_idx])
                    if _embedding is None:
                        _embedding = out
                    else:
                        _embedding = _embedding + out

                expert_embeddings.append(_embedding)
                
            expert_embeddings = torch.stack(expert_embeddings, dim=0)

            outputs.append(expert_embeddings)
            scores.append(all_probs)

        ent_loss = compute_mi_loss(torch.concat(scores, dim=0))

        outs = torch.stack(outputs, dim=0)
        outs = torch.sum(outs, dim=0)
        return outs, ent_loss


class FuseMoE(torch.nn.Module):
    def __init__(
        self,
        strategy: Literal["joint", "permodality", "disjoint"],
        input_dims: Union[int, list, tuple] = 512,
        hidden_dim=512,
        out_dim=512,
        num_experts=16,
        k=4,
        num_modalities=2,
        max_temperature=1.0,
        min_temperature=0.5,
        temperature_decay=0.9995,
    ):
        super(FuseMoE, self).__init__()
        self.strategy = strategy
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = out_dim
        self.num_experts = num_experts
        self.k = k
        self.num_modalities = num_modalities
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.current_temperature = max_temperature
        self.temperature_decay = temperature_decay

        self.fuse_norm = torch.nn.LayerNorm(hidden_dim)
        self.out_linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(out_dim),
        )

        if self.strategy == "joint":
            self.moe = JointMoE(
                input_dim=input_dims,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                out_dim=self.output_dim,
                k=k,
                num_modalities=num_modalities
            )
        elif self.strategy == "permodality":
            self.moe = PerModalityRouterMoE(
                dims=input_dims,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                k=k,
                out_dim=self.output_dim,
                num_modalities=num_modalities
            )
        elif self.strategy == "disjoint":
            self.moe = DisjointMoE(
                dims=input_dims,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                k=k,
                out_dim=self.output_dim,
                num_modalities=num_modalities
            )
        else:
            raise Exception("Invalid MoE strategy")

    def update_temperature(self, global_step):
        self.current_temperature = np.max(
            [
                self.min_temperature,
                self.max_temperature * np.pow(self.temperature_decay, global_step),
            ]
        )

    def forward(self, *inputs):
        moe_outputs, ent_loss = self.moe(self.current_temperature, *inputs)
        # outputs = self.fuse_norm(moe_outputs)
        outputs = self.out_linear(moe_outputs)
        
        return outputs, ent_loss
