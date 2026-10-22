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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [4, 26])
def test_explicit_reference_math_has_width_invariant_first_row(width, monkeypatch):
    import freetoken.core as core
    torch.manual_seed(28)
    layer = SimpleNamespace(weight=torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16), bias=None)
    x = torch.randn(width, 4096, device="cuda", dtype=torch.bfloat16)
    ctx = Context(1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    batch = Batch([], "decode")
    batch.speculative_reference_math = True
    expected = torch.cat([F.linear(row, layer.weight) for row in x.split(1)])
    with ctx.forward_batch(batch):
        actual = TorchLinearKernel().apply(layer, x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
