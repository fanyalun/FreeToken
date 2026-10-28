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
@pytest.mark.parametrize("width", [1, 4, 15, 17, 26])
@pytest.mark.parametrize("mode", ["speculative_reference_math", "stable_target_linear"])
def test_explicit_reference_math_has_width_invariant_first_row(width, mode, monkeypatch):
    import freetoken.core as core
    torch.manual_seed(28)
    layer = SimpleNamespace(weight=torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16), bias=None)
    x = torch.randn(width, 4096, device="cuda", dtype=torch.bfloat16)
    ctx = Context(1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    batch = Batch([], "decode")
    setattr(batch, mode, True)
    expected = torch.cat([F.linear(row, layer.weight) for row in x.split(1)])
    with ctx.forward_batch(batch):
        actual = TorchLinearKernel().apply(layer, x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [1, 4, 17, 33])
@pytest.mark.parametrize("tile_rows", [None, 16])
def test_greedy_head_graph_preserves_ties_and_device_valid_rows(width, tile_rows):
    from freetoken.kernel.triton.greedy_head import greedy_head

    x = torch.ones(width, 96, device="cuda", dtype=torch.bfloat16)
    x[1::2].neg_()
    weight = torch.zeros(259, 96, device="cuda", dtype=torch.bfloat16)
    weight[3].fill_(0.125)
    weight[129].copy_(weight[3])
    count = torch.tensor(width, device="cuda", dtype=torch.int32)
    for _ in range(3):
        greedy_head(x, weight, count, tile_rows=tile_rows)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = greedy_head(x, weight, count, tile_rows=tile_rows)
    for valid in (width, 1, 0, width):
        count.fill_(valid)
        graph.replay()
        reference = F.linear(x, weight).argmax(-1)
        reference[valid:] = 0
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    x.zero_()
    graph.replay()
    torch.testing.assert_close(actual, F.linear(x, weight).argmax(-1), rtol=0, atol=0)
    weight[129].fill_(float("nan"))
    for valid in (width, 1, 0):
        count.fill_(valid)
        graph.replay()
        reference = F.linear(x, weight).argmax(-1)
        reference[valid:] = 0
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [1, 4, 17, 65])
@pytest.mark.parametrize("stable_target", [False, True])
def test_lm_head_greedy_matches_forward_and_reference_fallback(width, stable_target, monkeypatch):
    import freetoken.core as core
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.embedding import ParallelLMHead

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    with torch.device("cuda"):
        head = ParallelLMHead(32771, 96)
    head.weight = torch.zeros(32771, 96, device="cuda", dtype=torch.bfloat16)
    head.weight[3].fill_(0.125)
    head.weight[129].copy_(head.weight[3])
    x = torch.ones(width, 96, device="cuda", dtype=torch.bfloat16)
    x[1::2].neg_()
    ctx = Context(1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    batch = Batch([], "decode")
    batch.stable_target_linear = stable_target
    with ctx.forward_batch(batch):
        expected = head.forward(x).argmax(-1)
        torch.testing.assert_close(head.greedy(x), expected, rtol=0, atol=0)
        batch.speculative_reference_math = True
        torch.testing.assert_close(head.greedy(x), head.forward(x).argmax(-1), rtol=0, atol=0)
