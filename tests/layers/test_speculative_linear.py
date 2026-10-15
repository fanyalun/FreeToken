from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from freetoken.core import Batch, Context
from freetoken.layers.quantization.linear.unquantized import TorchLinearKernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [2, 3, 4])
def test_bf16_verify_uses_one_batched_linear(width, monkeypatch):
    import freetoken.core as core
    generator = torch.Generator(device="cuda").manual_seed(7)
    layer = SimpleNamespace(weight=torch.randn(8192, 2048, device="cuda", dtype=torch.bfloat16, generator=generator), bias=None)
    x = torch.randn(width, 2048, device="cuda", dtype=torch.bfloat16, generator=generator)
    ctx = Context(1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    batch = Batch([], "decode")
    batch.rowwise_linear = True
    reference = F.linear(x, layer.weight)
    linear = F.linear
    calls = []

    def measured_linear(input, weight, bias=None):
        calls.append(input.shape)
        return linear(input, weight, bias)

    monkeypatch.setattr(F, "linear", measured_linear)
    with ctx.forward_batch(batch):
        actual = TorchLinearKernel().apply(layer, x)
    assert calls == [x.shape]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
