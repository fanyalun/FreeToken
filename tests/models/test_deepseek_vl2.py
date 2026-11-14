from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

from freetoken.models.deepseek_vl2.config import normalize_hf_config, parse_config
from freetoken.models.deepseek_vl2.moe import DeepseekVL2MoE
from freetoken.models.deepseek_vl2.vision import format_image_features
from freetoken.models.deepseek_vl2.weight import iter_expert_pieces, iter_weights
from freetoken.layers.quantization import QuantKind
from freetoken.mm.config import MultimodalConfig
from freetoken.mm.processors.deepseek_vl2 import DeepseekVL2MMProcessor
from freetoken.utils.hf import RawConfigShim


def config():
    return normalize_hf_config(RawConfigShim(dict(
        model_type="deepseek_vl_v2", language_config=dict(hidden_size=8, vocab_size=32,
            intermediate_size=16, n_routed_experts=4, num_experts_per_tok=2,
            moe_intermediate_size=8, first_k_dense_replace=1, n_shared_experts=2,
            norm_topk_prob=True, routed_scaling_factor=2.),
        vision_config=dict(width=16, layers=2, heads=2, mlp_ratio=2, patch_size=14),
        projector_config=dict(n_embed=8), tile_tag="2D", global_view_pos="head")))


def test_native_config_missing_architecture_and_language_defaults():
    hf = config()
    mc = parse_config(hf)
    assert hf.architectures == mc.architectures == ["DeepseekVLV2ForCausalLM"]
    assert (mc.num_layers, mc.num_moe_layers, mc.num_qo_heads, mc.head_dim) == (30, 29, 32, 192)
    assert (mc.first_k_dense_replace, mc.n_shared_experts) == (1, 2)
    assert mc.is_moe and mc.is_multimodal
    hf._data["language_config"]["rope_scaling"] = {"factor": 2.}
    with pytest.raises(ValueError, match="scaled RoPE"):
        parse_config(hf)


def test_expert_reader_maps_global_layers_and_preserves_bf16(tmp_path):
    tensors = {f"language.model.layers.1.mlp.experts.{expert}.{proj}.weight":
               torch.full(shape, expert + i, dtype=torch.bfloat16)
               for expert in range(2) for i, (proj, shape) in enumerate(
                   [("gate_proj", (3, 4)), ("up_proj", (3, 4)), ("down_proj", (4, 3))])}
    tensors["language.model.norm.weight"] = torch.ones(4, dtype=torch.bfloat16)
    tensors["language.model.layers.1.mlp.gate.e_score_correction_bias"] = torch.ones(2, dtype=torch.bfloat16)
    save_file(tensors, tmp_path / "model.safetensors")
    mc = SimpleNamespace(first_k_dense_replace=1, num_moe_layers=1, num_experts=2)
    pieces = list(iter_expert_pieces(str(tmp_path), mc, QuantKind.NONE))
    assert len(pieces) == 2
    for layer, lo, hi, piece in pieces:
        assert (layer, hi) == (0, lo + 1)
        assert piece["gate_up"].shape == (1, 6, 4)
        assert piece["down"].dtype == torch.bfloat16
        torch.testing.assert_close(piece["gate_up"][0, :3], tensors[f"language.model.layers.1.mlp.experts.{lo}.gate_proj.weight"])
    dense = dict(iter_weights(str(tmp_path), torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    assert set(dense) == {"model.norm.weight", "model.layers.1.mlp.e_score_correction_bias"}
    assert dense["model.layers.1.mlp.e_score_correction_bias"].dtype == torch.float32


@pytest.mark.parametrize("topk", [1, 2, 3])
def test_router_bias_selects_without_reweighting_sigmoid_scores(topk):
    mc = replace(parse_config(config()), num_experts_per_tok=topk)
    layer = DeepseekVL2MoE.__new__(DeepseekVL2MoE)
    layer._config = mc
    layer.gate = SimpleNamespace(weight=torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
    layer.e_score_correction_bias = torch.tensor([-.3, -.2, .8, .5])
    x = torch.tensor([[1., 2.], [3., -.5]])
    weights, ids = layer.route(x)
    scores = F.linear(x.float(), layer.gate.weight).sigmoid()
    expected_ids = (scores + layer.e_score_correction_bias).topk(topk, sorted=False).indices
    expected = scores.gather(-1, expected_ids)
    if topk > 1:
        expected = expected / expected.sum(-1, keepdim=True)
    assert torch.equal(ids.long(), expected_ids)
    torch.testing.assert_close(weights, expected * 2)


def test_router_top5_override_preserves_native_model_config():
    from freetoken.models.deepseek_vl2.config import with_router_top_k

    original = replace(parse_config(config()), num_experts=72, num_experts_per_tok=6)
    adjusted = with_router_top_k(original, 5)
    assert original.num_experts_per_tok == 6 and adjusted.num_experts_per_tok == 5
    assert adjusted.routed_scaling_factor == original.routed_scaling_factor == 2
    for invalid in (0, 73):
        with pytest.raises(ValueError, match="between"):
            with_router_top_k(original, invalid)
    with pytest.raises(ValueError, match="requires DeepSeek"):
        with_router_top_k(replace(original, deepseek_vl2_args=None), 5)


def test_context_override_grows_only_the_unscaled_rope_table():
    from freetoken.models.deepseek_vl2.config import with_context_length

    original = parse_config(config())
    adjusted = with_context_length(original, 32768)
    assert adjusted.rotary_config.max_position == 32768
    assert original.rotary_config.max_position == 2048
    assert adjusted.rotary_config.base == original.rotary_config.base
    assert adjusted.rotary_config.scaling == original.rotary_config.scaling is None
    assert with_context_length(original, 1024) is original


@pytest.mark.parametrize("bypass", [False, True])
def test_tablemoe_ranks_unbiased_route_weights_and_preserves_full_baseline(monkeypatch, bypass):
    import freetoken.models.deepseek_vl2.moe as module

    weights = torch.tensor([[.2, .8, .4, .6]])
    ids = torch.tensor([[3, 1, 0, 2]], dtype=torch.int32)
    observed = {}
    layer = DeepseekVL2MoE.__new__(DeepseekVL2MoE)
    layer._config = SimpleNamespace(moe_strategy="tablemoe", n_shared_experts=0)
    layer.route = lambda x: (weights, ids)

    def capture(x, w, selected):
        observed.update(weights=w, ids=selected)
        return x

    layer.experts = SimpleNamespace(routed_forward=capture)
    monkeypatch.setattr(module, "get_global_ctx", lambda: SimpleNamespace(tablemoe_runtime=SimpleNamespace(bypass_tables=bypass)))
    layer.forward(torch.ones(1, 2))
    assert observed['ids'].tolist() == (ids.tolist() if bypass else [[1, 2, 0, 3]])
    assert observed['weights'].sum().item() == pytest.approx(weights.sum().item())


def test_image_processor_native_crop_switch_and_feature_layout(tmp_path):
    hf = config()
    hf._data["candidate_resolutions"] = [[384, 384], [384, 768], [768, 384]]
    processor = DeepseekVL2MMProcessor(hf, str(tmp_path), MultimodalConfig())
    image = Image.new("RGB", (300, 600), (255, 0, 0))
    single = processor.process([image])[0]
    assert single.model_specific_data["crop"] == [1, 2]
    assert single.feature.shape == (3, 3, 384, 384)
    multiple = processor.process([image] * 3)
    assert all(item.model_specific_data["crop"] == [1, 1] for item in multiple)
    assert len(processor.prompt_replacement(single).full) == 631
    features = torch.arange(3 * 4 * 2).reshape(3, 4, 2).float()
    newline, separator = torch.tensor([-1., -1.]), torch.tensor([-2., -2.])
    rows = format_image_features(features, [1, 2], newline, separator)
    assert rows.shape == (19, 2)
    torch.testing.assert_close(rows[6], separator)
    torch.testing.assert_close(rows[7:9], features[1, :2])


def test_chat_template_matches_native_roles_and_bos():
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from freetoken.models.deepseek_vl2.chat import CHAT_TEMPLATE

    template = ImmutableSandboxedEnvironment().from_string(CHAT_TEMPLATE)
    text = template.render(bos_token="<bos>", eos_token="<eos>", messages=[
        dict(role="user", content=[dict(type="image"), dict(type="text", text="Describe.")])], add_generation_prompt=True)
    assert text == "<bos><|User|>: <image>\nDescribe.\n\n<|Assistant|>:"


def test_bf16_checkpoint_keeps_float32_selection_bias(tmp_path):
    from freetoken.moe.tablemoe import validate_fp16_checkpoint

    save_file({"language.model.layers.1.mlp.gate.e_score_correction_bias": torch.ones(4),
               "language.model.layers.1.mlp.gate.weight": torch.ones(4, 8, dtype=torch.bfloat16)},
              tmp_path / "model.safetensors")
    validate_fp16_checkpoint(str(tmp_path), dtype=torch.bfloat16)
    save_file({"language.model.layers.1.mlp.gate.weight": torch.ones(4, 8)}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="requires BF16"):
        validate_fp16_checkpoint(str(tmp_path), dtype=torch.bfloat16)


def test_offline_assets_use_global_layer_ids_without_content_hashes(tmp_path, monkeypatch):
    from freetoken.moe import tablemoe_artifacts
    from freetoken.moe.tablemoe import TableMoeRuntime

    pca, cache, model = [tmp_path / part for part in ("pca", "cache", "model")]
    for path in (pca / "layer_1", cache / "layer_1", model):
        path.mkdir(parents=True)
    outputs = []
    for modality in ("text", "vision"):
        torch.save(dict(mean=torch.zeros(4, dtype=torch.bfloat16), proj=torch.eye(4, 2, dtype=torch.bfloat16)),
                   pca / "layer_1" / f"L1_{modality}_pca.pt")
        path = cache / "layer_1" / f"L1_E0_{modality}_cache.pt"
        torch.save(dict(key=torch.ones(1, 2, dtype=torch.bfloat16), value=torch.ones(1, 4, dtype=torch.bfloat16)), path)
        outputs.append(path)
    tablemoe_artifacts.finish_stage(cache, {}, outputs, hash_outputs=False, dtype="bfloat16",
        provenance=dict(checkpoint=dict(path=str(model)), convert_bf16=False, processor={"native_defaults": True}),
        pca_outputs={f"layer_1/L1_{modality}_pca.pt": {} for modality in ("text", "vision")})
    original_hash = tablemoe_artifacts.file_hash

    def metadata_hash(path):
        assert path.suffix not in (".pt", ".bin", ".phase"), "asset content hashing is prohibited"
        return original_hash(path)

    monkeypatch.setattr(tablemoe_artifacts, "file_hash", metadata_hash)
    runtime = TableMoeRuntime.__new__(TableMoeRuntime)
    runtime.config = SimpleNamespace(tablemoe_pca_dir=str(pca), tablemoe_cache_dir=str(cache), model_path=str(model),
        tablemoe_convert_bf16_weights=False, model_config=SimpleNamespace(deepseek_vl2_args=object()), mm=MultimodalConfig())
    runtime.dtype = torch.bfloat16
    runtime.hidden, runtime.num_layers, runtime.num_experts = 4, 1, 1
    runtime.asset_layer_offset = 1
    runtime.pca, runtime.offline_paths = {}, {}
    runtime._load_assets()
    assert set(runtime.pca) == {(0, "text"), (0, "vision")}
    assert set(runtime.offline_paths) == {(0, 0, "text"), (0, 0, "vision")}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mla_projection_matches_native_rope_and_zero_padded_values(monkeypatch):
    from freetoken.distributed import info
    from freetoken.distributed import DistributedInfo
    from freetoken.models.deepseek_vl2.attention import DeepseekVL2Attention

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 1))
    mc = replace(parse_config(config()), num_qo_heads=2, head_dim=68,
                 deepseek_vl2_args=replace(parse_config(config()).deepseek_vl2_args,
                                           kv_lora_rank=8, qk_nope_head_dim=4, v_head_dim=4))
    with torch.device("cuda"):
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            layer = DeepseekVL2Attention(mc, 0)
        finally:
            torch.set_default_dtype(original_dtype)
        for value in layer.state_dict().values():
            value.normal_(0, .1)
        x = torch.randn(5, 8, dtype=torch.bfloat16)
        positions = torch.arange(5, dtype=torch.int32)
        q, k, v = layer.project(x, positions)
        raw_q = F.linear(x, layer.q_proj.weight).view(5, 2, 68)
        raw_latent, raw_kpe = F.linear(x, layer.kv_a_proj_with_mqa.weight).split((8, 64), -1)
        latent = (raw_latent.float() * torch.rsqrt(raw_latent.float().square().mean(-1, keepdim=True) + mc.rms_norm_eps)).to(x.dtype) * layer.kv_a_layernorm.weight
        expanded = F.linear(latent, layer.kv_b_proj.weight).view(5, 2, 8)
        frequency = positions.float()[:, None] / (10000. ** (torch.arange(0, 64, 2).float() / 64))
        cosine, sine = frequency.cos()[:, None], frequency.sin()[:, None]

        def native_rotate(t):
            even, odd = t[..., ::2], t[..., 1::2]
            return torch.cat((even * cosine - odd * sine, odd * cosine + even * sine), -1)

        torch.testing.assert_close(q[..., 4:].float(), native_rotate(raw_q[..., 4:]).float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(k[..., :4], expanded[..., :4], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(k[..., 4:].float(), native_rotate(raw_kpe[:, None]).expand(-1, 2, -1).float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(v[..., :4], expanded[..., 4:], atol=2e-2, rtol=2e-2)
        assert torch.count_nonzero(v[..., 4:]) == 0
