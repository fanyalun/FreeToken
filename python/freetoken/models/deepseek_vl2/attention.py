from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.attention.base import AttentionSpec
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
from freetoken.layers.rotary import get_rope


def rope_reorder(x: torch.Tensor) -> torch.Tensor:
    # The native model changes adjacent pairs to half-split order before RoPE.
    return x.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2).contiguous()


class DeepseekVL2Attention(BaseOP):
    def __init__(self, config, layer_id):
        args = config.deepseek_vl2_args
        self._args = args
        self._layer = layer_id
        self._heads = config.num_qo_heads
        self._dim = config.head_dim
        self._spec = AttentionSpec(sm_scale=config.attn_sm_scale)
        if args.q_lora_rank is None:
            self.q_proj = LinearReplicated(config.hidden_size, self._heads * self._dim, has_bias=False)
        else:
            self.q_a_proj = LinearReplicated(config.hidden_size, args.q_lora_rank, has_bias=args.attention_bias)
            self.q_a_layernorm = RMSNorm(args.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = LinearReplicated(args.q_lora_rank, self._heads * self._dim, has_bias=False)
        self.kv_a_proj_with_mqa = LinearReplicated(config.hidden_size, args.kv_lora_rank + args.qk_rope_head_dim,
                                                  has_bias=args.attention_bias)
        self.kv_a_layernorm = RMSNorm(args.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = LinearReplicated(args.kv_lora_rank,
                                         self._heads * (args.qk_nope_head_dim + args.v_head_dim), has_bias=False)
        self.o_proj = LinearReplicated(self._heads * args.v_head_dim, config.hidden_size, has_bias=args.attention_bias)
        self.rotary = get_rope(head_dim=args.qk_rope_head_dim, rotary_dim=args.qk_rope_head_dim,
                               max_position=config.rotary_config.max_position, base=config.rotary_config.base)

    def project(self, x, positions):
        args = self._args
        q = (self.q_proj.forward(x) if args.q_lora_rank is None else
             self.q_b_proj.forward(self.q_a_layernorm.forward(self.q_a_proj.forward(x))))
        q = q.view(-1, self._heads, self._dim)
        q_nope, q_pe = q.split((args.qk_nope_head_dim, args.qk_rope_head_dim), -1)
        latent, k_pe = self.kv_a_proj_with_mqa.forward(x).split((args.kv_lora_rank, args.qk_rope_head_dim), -1)
        kv = self.kv_b_proj.forward(self.kv_a_layernorm.forward(latent))
        k_nope, v = kv.view(-1, self._heads, args.qk_nope_head_dim + args.v_head_dim).split(
            (args.qk_nope_head_dim, args.v_head_dim), -1)
        q_pe, k_pe = rope_reorder(q_pe), rope_reorder(k_pe[:, None, :])
        q_pe, k_pe = self.rotary.forward(positions, q_pe.flatten(1), k_pe.flatten(1))
        q = torch.cat((q_nope, q_pe.view(-1, self._heads, args.qk_rope_head_dim)), -1)
        k = torch.cat((k_nope, k_pe[:, None, :].expand(-1, self._heads, -1)), -1)
        return q.contiguous(), k.contiguous(), F.pad(v, (0, self._dim - args.v_head_dim)).contiguous()

    def forward(self, x):
        ctx = get_global_ctx()
        q, k, v = self.project(x, ctx.batch.get_attn_positions())
        runtime = getattr(ctx, "tablemoe_runtime", None)
        if runtime is not None and ctx.batch.is_prefill and getattr(runtime, "prefill_uses_tablemoe", True):
            from freetoken.moe.tablemoe import last_token_attention_scores

            bank_layer = self._layer - runtime.config.model_config.first_k_dense_replace
            if bank_layer >= 0:
                ctx.batch.tablemoe_attn_scores[bank_layer] = last_token_attention_scores(
                    q, k, ctx.batch.padded_reqs, sm_scale=self._spec.sm_scale)
        out = ctx.attn_backend.forward(q, k.flatten(1), v.flatten(1), self._layer, ctx.batch, attn_spec=self._spec)
        return self.o_proj.forward(out.view(-1, self._heads, self._dim)[..., :self._args.v_head_dim].reshape(
            -1, self._heads * self._args.v_head_dim))
