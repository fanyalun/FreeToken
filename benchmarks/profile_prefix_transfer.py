"""Diagnostic NVTX ranges over the unchanged eager startup benchmark."""

from functools import wraps
import sys

import torch

from freetoken.engine import Engine
from freetoken.engine.qwen_cpu_target import QwenCpuTarget
from bench_cpu_verify_feasibility import main


def annotate(name, function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        label = name(*args, **kwargs) if callable(name) else name
        with torch.cuda.nvtx.range(label):
            return function(*args, **kwargs)
    return wrapped


def instrument_engine(self, *args, **kwargs):
    engine_init(self, *args, **kwargs)
    for index, layer in enumerate(self.model.model.layers.op_list):
        mixer = layer.linear_attn if layer._is_linear else layer.self_attn
        kind = "GDN" if layer._is_linear else "Attention"
        mixer.forward = annotate(f"GPU {kind} layer {index}", mixer.forward)
        layer.mlp.forward = annotate(f"GPU MoE layer {index}", layer.mlp.forward)
    cache = self.moe_offload_cache
    cache.prefetch_prefill_layer = annotate(lambda layer_id: f"Expert H2D submit layer {layer_id}",
                                           cache.prefetch_prefill_layer)


# Spawn executes these bindings too, so the CPU child gets actual compute ranges.
for method, label in (("_gdn", "GDN"), ("_attention", "Attention"), ("_moe", "MoE")):
    function = getattr(QwenCpuTarget, method)
    setattr(QwenCpuTarget, method, annotate(
        lambda self, layer, *args, _label=label, **kwargs: f"CPU {_label} layer {layer}", function))
engine_init = Engine.__init__
Engine.__init__ = instrument_engine


if __name__ == "__main__":
    if "--profile" not in sys.argv:
        sys.argv.append("--profile")
    main()
