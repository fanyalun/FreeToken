from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer
from freetoken.models.glm4_moe.mlp import GlmGatedMLP


class DeepseekVL2MoE(BaseOP):
    def __init__(self, config, layer_id):
        self._config = config
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        if config.deepseek_vl2_args.topk_method == "noaux_tc":
            self.e_score_correction_bias = torch.empty(config.num_experts, dtype=torch.float32)
        self.experts = make_moe_layer(config, layer_id=layer_id - config.first_k_dense_replace,
                                      renormalize=False)
        if config.n_shared_experts:
            self.shared_experts = GlmGatedMLP(config.hidden_size, config.moe_intermediate_size * config.n_shared_experts)

    def route(self, x):
        config = self._config
        args = config.deepseek_vl2_args
        logits = F.linear(x.float(), self.gate.weight.float())
        scores = logits.sigmoid() if args.scoring_func == "sigmoid" else logits.softmax(-1)
        choice = scores + self.e_score_correction_bias if args.topk_method == "noaux_tc" else scores
        if args.topk_method != "greedy" and config.n_group > 1:
            groups = choice.view(-1, config.n_group, config.num_experts // config.n_group)
            group_scores = groups.topk(2, -1).values.sum(-1) if args.topk_method == "noaux_tc" else groups.max(-1).values
            group_ids = group_scores.topk(config.topk_group, -1, sorted=False).indices
            mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, group_ids, True)
            choice = choice.masked_fill(~mask.repeat_interleave(config.num_experts // config.n_group, -1), 0.)
        ids = choice.topk(config.num_experts_per_tok, -1, sorted=False).indices
        weights = scores.gather(-1, ids)
        if config.norm_topk_prob and config.num_experts_per_tok > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return (weights * config.routed_scaling_factor).float().contiguous(), ids.int().contiguous()

    def forward(self, x):
        weights, ids = self.route(x)
        shared = self.shared_experts.forward(x) if self._config.n_shared_experts else None
        out = self.experts.routed_forward(x, weights, ids)
        return out if shared is None else out + shared
