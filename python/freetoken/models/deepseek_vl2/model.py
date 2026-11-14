from __future__ import annotations

from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel, embed_input_ids
from freetoken.models.glm4_moe.mlp import GlmGatedMLP

from .attention import DeepseekVL2Attention
from .moe import DeepseekVL2MoE
from .vision import DeepseekVL2VisionMixin, DeepseekVL2Vision


class DeepseekVL2DecoderLayer(BaseOP):
    def __init__(self, config, layer_id):
        self.self_attn = DeepseekVL2Attention(config, layer_id)
        self.mlp = (GlmGatedMLP(config.hidden_size, config.intermediate_size)
                    if layer_id < config.first_k_dense_replace else DeepseekVL2MoE(config, layer_id))
        self.input_layernorm = RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, residual=None):
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        return self.mlp.forward(x), residual


class DeepseekVL2Model(BaseOP):
    def __init__(self, config):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList([DeepseekVL2DecoderLayer(config, i) for i in range(config.num_layers)])
        self.norm = RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, ids):
        x = embed_input_ids(self.embed_tokens, ids, get_global_ctx().batch)
        residual = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class DeepseekVLV2ForCausalLM(DeepseekVL2VisionMixin, BaseLLMModel):
    def __init__(self, config):
        if get_tp_info().size != 1:
            raise ValueError("DeepSeek-VL2 currently supports TP=1")
        from freetoken.moe import is_offload_moe_strategy

        if not is_offload_moe_strategy(config.moe_strategy):
            raise ValueError("DeepSeek-VL2 requires --moe-strategy offload, cpu, hybrid or tablemoe")
        self.model = DeepseekVL2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, tie_word_embeddings=False)
        if config.is_multimodal:
            self.visual = DeepseekVL2Vision(config.vision_config)
        super().__init__()

    def forward(self):
        return self.lm_head.forward(self.model.forward(get_global_ctx().batch.input_ids))
