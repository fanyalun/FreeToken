from __future__ import annotations

import functools

import torch

import bench_speculative_execution as benchmark
from freetoken.engine.qwen_mtp import QwenMTPRunner
from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.offload_cache import OffloadMoeCache


def main():
    original_ranges = benchmark.profile_ranges

    def profile_ranges():
        original_ranges()

        def wrap(owner, method, label):
            original = getattr(owner, method)
            @functools.wraps(original)
            def call(*args, **kwargs):
                with torch.cuda.nvtx.range(label(*args, **kwargs)):
                    return original(*args, **kwargs)
            setattr(owner, method, call)

        wrap(QwenMTPRunner, "_execute", lambda runner, phase, batch, *a, **kw:
             f"EXECUTE/{runner.three_level.profile_name if runner.three_level else 'sd'}/{phase}/rows={batch.input_ids.numel()}")
        wrap(OffloadMoeCache, "copy_missing", lambda cache, *a, **kw:
             f"BASELINE_DEMAND/layer={cache._pending_src_layer}")
        wrap(OffloadMoeCache, "ensure_experts", lambda cache, layer, *a, **kw:
             f"BASELINE_ADMISSION/layer={layer}")
        wrap(OffloadMoELayer, "_expert_gemm", lambda layer, *a, **kw:
             f"EXPERT_COMPUTE/layer={layer.layer_id}")

    benchmark.profile_ranges = profile_ranges
    benchmark.main()


if __name__ == "__main__":
    main()
