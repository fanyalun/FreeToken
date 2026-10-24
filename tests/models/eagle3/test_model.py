import pytest
import torch
import torch.nn.functional as F

from freetoken.models.config import ModelConfig, RotaryConfig
from freetoken.models.eagle3 import Eagle3Config, Eagle3Draft


def test_target_aux_features_are_residual_complete_decoder_boundaries():
    from types import SimpleNamespace
    from freetoken.models.qwen3_moe.model import Qwen3Model

    model = object.__new__(Qwen3Model)
    model.embed_tokens = SimpleNamespace(forward=lambda ids: ids[:, None].float())
    class Layer:
        def __init__(self, boundary):
            self.boundary = boundary
        def forward(self, x, residual):
            return x.new_full(x.shape, self.boundary), x.new_full(x.shape, self.boundary * 100)
    model.layers = SimpleNamespace(op_list=[Layer(i) for i in range(1, 95)])
    model.norm = SimpleNamespace(forward=lambda x, residual: (x * 0 - 1, residual))
    model.enable_aux_hidden_capture((2, 47, 91))
    output = model.forward(torch.tensor([3, 5]))
    assert output.tolist() == [[-1.0], [-1.0]]
    assert model.last_aux_hidden.tolist() == [[202.0, 4747.0, 9191.0]] * 2


@pytest.mark.parametrize("conditioning", [True, False])
@pytest.mark.parametrize("norm_before", [True, False])
def test_eagle3_hidden_and_kv_match_dense_reference(monkeypatch, tmp_path, conditioning, norm_before):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import freetoken.core as core
    from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata
    from freetoken.core import Batch, Context, Req, SamplingParams
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.layers.quantization import finalize_quant
    from freetoken.layers.rotary import set_rope_device
    from freetoken.utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    device = torch.device("cuda:0")
    torch.manual_seed(712)
    mc = ModelConfig(1, 4, 2, 64, 128, 256, 256, 1e-6,
        RotaryConfig(64, 64, 64, 10000.0, None), "silu", False,
        0, 0, 0, False, "eagle3", ["Eagle3DraftModel"])
    config = Eagle3Config(mc, 128, 192, (2, 4, 5), norm_before)
    set_rope_device(device)
    with torch.device(device), torch_dtype(torch.bfloat16):
        model = Eagle3Draft(config, 0)
    for name, tensor in model.state_dict().items():
        tensor.copy_(torch.ones_like(tensor) if tensor.ndim == 1 else torch.randn_like(tensor) * 0.025)
    finalize_quant(model)
    raw = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    pool = MHAKVCache(2, 1, 64, 65, 1, torch.bfloat16, device)
    ctx = Context(1)
    ctx.kv_cache = pool
    ctx.page_table = torch.arange(64, device=device, dtype=torch.int32).reshape(1, -1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    ctx.attn_backend = TritonAttentionBackend(mc)
    n = 5
    reqs = [Req(torch.zeros(64, dtype=torch.int32), 0, i, 1, 0, SamplingParams(), None) for i in range(n)]
    batch = Batch(reqs, "decode")
    batch.padded_reqs = reqs
    batch.input_ids = torch.tensor([3, 19, 0, 71, 5], device=device, dtype=torch.int32)
    batch.positions = torch.arange(n, device=device, dtype=torch.int32)
    batch.out_loc = batch.positions.clone()
    offsets = torch.arange(n + 1, device=device, dtype=torch.int32)
    batch.attn_metadata = TritonMetadata(offsets, offsets * 64, ctx.page_table.flatten().repeat(n),
        offsets[:-1], batch.positions, True, batch.positions, 1)
    hidden = torch.randn(n, 384 if conditioning else 128, device=device, dtype=torch.bfloat16)
    source = hidden.clone()

    def norm(x, key):
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + mc.rms_norm_eps)
                * raw[key].float()).to(torch.bfloat16)

    def linear(x, key):
        return F.linear(x, raw[key])

    h = linear(source, "fc.weight") if conditioning else source
    normalized = norm(h, "layers.0.hidden_norm.weight")
    residual = normalized if norm_before else h
    embed = F.embedding(batch.input_ids.long(), raw["embed_tokens.weight"])
    inputs = torch.cat((norm(embed, "layers.0.input_layernorm.weight"), normalized), -1)
    q, k, v = linear(inputs, "layers.0.self_attn.qkv_proj.weight").split((256, 128, 128), -1)
    q, k, v = q.view(n, 4, 64), k.view(n, 2, 64), v.view(n, 2, 64)
    inv = 10000.0 ** (-torch.arange(0, 64, 2, device=device).float() / 64)
    angle = batch.positions.float()[:, None] * inv
    cos = torch.cat((angle.cos(), angle.cos()), -1)[:, None]
    sin = torch.cat((angle.sin(), angle.sin()), -1)[:, None]

    def rope(x):
        a, b = x.float().chunk(2, -1)
        return (x.float() * cos + torch.cat((-b, a), -1) * sin).to(x.dtype)

    q, k = rope(q), rope(k)
    attn = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0),
        k.repeat_interleave(2, 1).transpose(0, 1).unsqueeze(0),
        v.repeat_interleave(2, 1).transpose(0, 1).unsqueeze(0), is_causal=True)
    attn = attn.squeeze(0).transpose(0, 1).reshape(n, 256)
    merged = linear(attn, "layers.0.self_attn.o_proj.weight").float() + residual.float()
    x = norm(merged, "layers.0.post_attention_layernorm.weight")
    residual = merged.to(torch.bfloat16)
    gate, up = linear(x, "layers.0.mlp.gate_up_proj.weight").chunk(2, -1)
    x = linear((F.silu(gate.float()) * up.float()).to(torch.bfloat16), "layers.0.mlp.down_proj.weight")
    expected_prenorm = (x.float() + residual.float()).to(torch.bfloat16)
    expected = norm(x.float() + residual.float(), "norm.weight")
    with ctx.forward_batch(batch):
        normalized_out, recursive = model.forward(batch.input_ids, hidden)
    torch.testing.assert_close(recursive, expected_prenorm, rtol=0.015, atol=0.015)
    torch.testing.assert_close(normalized_out, expected, rtol=0.015, atol=0.015)
    torch.testing.assert_close(pool.k_cache(0)[:n, 0], k, rtol=0.008, atol=0.004)
    torch.testing.assert_close(pool.v_cache(0)[:n, 0], v, rtol=0, atol=0)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with ctx.forward_batch(batch):
            model.forward(batch.input_ids, source)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream), ctx.forward_batch(batch):
            graphed = model.forward(batch.input_ids, source)
        graph.replay()
        stream.synchronize()
    for actual, reference in zip(graphed, (normalized_out, recursive)):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    from safetensors.torch import save_file
    checkpoint = {key: value.cpu() for key, value in raw.items()}
    qkv = checkpoint.pop("layers.0.self_attn.qkv_proj.weight").split((256, 128, 128))
    for name, weight in zip(("q", "k", "v"), qkv):
        checkpoint[f"layers.0.self_attn.{name}_proj.weight"] = weight.clone()
    gate_up = checkpoint.pop("layers.0.mlp.gate_up_proj.weight").chunk(2)
    for name, weight in zip(("gate", "up"), gate_up):
        checkpoint[f"layers.0.mlp.{name}_proj.weight"] = weight.clone()
    checkpoint["d2t"] = torch.full((192,), 13, dtype=torch.int64)
    checkpoint["t2d"] = torch.zeros(256, dtype=torch.bool)
    checkpoint["t2d"][13:205] = True
    save_file(checkpoint, tmp_path / "model.safetensors")
    model.load_checkpoint(tmp_path, device)
    assert model.map_tokens(torch.tensor([0, 11, 191], device=device)).tolist() == [13, 24, 204]
