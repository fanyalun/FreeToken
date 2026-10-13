from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from freetoken.moe.modes import load_calibration, modes_keep_mask, prepare_modes_config, validate_calibration


def _upstream(weights, modality, valid, importance, tau):
    score = weights.half().clone()
    for mod in (0, 1):
        mask = (modality == mod)[:, None].expand_as(score)
        prod = score * float(importance[mod])
        score[mask] = prod[mask]
    skip = torch.zeros_like(score, dtype=torch.bool)
    for mod in (0, 1):
        skip |= (modality == mod)[:, None] & (score < float(tau[mod]))
    return ~skip & valid[:, None]


@pytest.mark.parametrize("importance,tau", [([1., 1.], [.25, .5]), ([0., 0.], [0., .1]),
                                            ([.123456789, .987654321], [.03, .1]),
                                            ([1., 1.], [2., 2.]), ([1., 1.], [0., 0.])])
def test_reference_matches_upstream_fp16(importance, tau):
    torch.manual_seed(4)
    weights = torch.softmax(torch.randn(8, 4), -1)
    weights[0] = .25
    modality = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    valid = torch.tensor([True] * 7 + [False])
    imp, thresholds = torch.tensor(importance), torch.tensor(tau)
    expected = _upstream(weights, modality, valid, imp, thresholds)
    assert torch.equal(modes_keep_mask(weights, modality, valid, imp, thresholds), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_device_selection_special_visual_padding_and_graph():
    import triton
    from freetoken.kernel.triton.modes import _select

    torch.manual_seed(4)
    weights = torch.softmax(torch.randn(8, 4, device="cuda"), -1)
    weights[0] = .25
    tokens = torch.tensor([7, 7, 8, 9, 9, 8, 7, 7], device="cuda", dtype=torch.int32)
    special = torch.tensor([7, 9], device="cuda", dtype=torch.int32)
    visual = torch.tensor([0, 1, 1, 1, 0, 1, 1, 1], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0] * 7 + [99], device="cuda", dtype=torch.int32)
    imp = torch.tensor([.123456789, .987654321], device="cuda")
    tau = torch.tensor([.03, .1], device="cuda")
    keep = torch.empty_like(weights, dtype=torch.bool)

    def select():
        _select[(triton.cdiv(weights.numel(), 256),)](
            weights, tokens, slots, visual, special, imp, tau, keep, 4, 2, 99, weights.numel(), 256)

    mod = torch.where((visual == 1) & torch.isin(tokens, special), 2, visual)
    expected = _upstream(weights, mod, slots != 99, imp, tau)
    select()
    assert torch.equal(keep, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select()
    for thresholds in ([0., 0.], [2., 2.], [.030001, .100001]):
        tau.copy_(torch.tensor(thresholds, device="cuda"))
        graph.replay()
        assert torch.equal(keep, _upstream(weights, mod, slots != 99, imp, tau))
    for importance, thresholds in (([0., 0.], [.1, .1]), ([0., 0.], [0., 0.]), ([1., 1.], [.25, .25])):
        imp.copy_(torch.tensor(importance, device="cuda"))
        tau.copy_(torch.tensor(thresholds, device="cuda"))
        graph.replay()
        assert torch.equal(keep, _upstream(weights, mod, slots != 99, imp, tau))


def _importer():
    path = Path(__file__).resolve().parents[2] / "scripts/import_modes_config.py"
    spec = importlib.util.spec_from_file_location("import_modes_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.import_config


def _assets(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(dict(
        architectures=["Qwen3VLMoeForConditionalGeneration"], text_config=dict(num_hidden_layers=2))))
    save_file({"weight": torch.ones(2).half()}, model / "model.safetensors")
    importance, search = tmp_path / "importance.pkl", tmp_path / "search.pkl"
    importance.write_bytes(pickle.dumps({i: dict(text=.25 + i, visual=.75 + i) for i in range(2)}))
    search.write_bytes(pickle.dumps(dict(tau=dict(text=.1, visual=.2))))
    return model, importance, search


def test_import_checkpoint_binding_manifest_and_control(tmp_path):
    model, importance, search = _assets(tmp_path)
    path = _importer()(importance, search, model, tmp_path / "imported")
    data = load_calibration(path, model, range(2))
    assert data["checkpoint_association"] == "declared_by_importer"
    assert data["layers"]["1"]["importance"]["text"] == 1.25
    control = _importer()(importance, None, model, tmp_path / "control", zero_tau=True)
    assert load_calibration(control, model, range(2))["control"]
    with pytest.raises(ValueError, match="cover"):
        load_calibration(path, model, range(3))
    save_file({"weight": torch.zeros(2).half()}, model / "model.safetensors")
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        load_calibration(path, model, range(2))
    path.write_text("{}")
    with pytest.raises(ValueError, match="changed artifact"):
        load_calibration(path, model, range(2))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.])
def test_import_rejects_invalid_values(tmp_path, value):
    model, importance, search = _assets(tmp_path)
    importance.write_bytes(pickle.dumps({i: dict(text=value, visual=.75) for i in range(2)}))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _importer()(importance, search, model, tmp_path / "bad")


@pytest.mark.parametrize("override", [dict(text=.5, visual=.2), dict(text=-1., visual=.2),
                                      dict(text=float("nan"), visual=.2), dict(text=.5)])
@pytest.mark.parametrize("field", ["decode_tau", "decode_importance"])
def test_decode_threshold_override_is_validated(tmp_path, override, field):
    model, importance, search = _assets(tmp_path)
    path = _importer()(importance, search, model, tmp_path / "imported")
    data = json.loads(path.read_text())
    data["layers"]["0"][field] = override
    if override == dict(text=.5, visual=.2):
        assert validate_calibration(data, range(2))["layers"]["0"]["tau"]["text"] == .1
    else:
        with pytest.raises(ValueError, match="requires|finite and nonnegative"):
            validate_calibration(data, range(2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("field", ["decode_tau", "decode_importance"])
def test_decode_threshold_override_preserves_prefill_and_graph_selection(field):
    from freetoken.moe.modes_runtime import ModesRuntime

    rt = object.__new__(ModesRuntime)
    rt.importance = torch.tensor([[1., 1.]], device="cuda")
    rt.tau = torch.tensor([[.1, .1]], device="cuda")
    rt.decode_tau = torch.tensor([[.1, .5]], device="cuda")
    if field == "decode_importance":
        rt.decode_tau = rt.tau
        rt.decode_importance = torch.tensor([[1., .25]], device="cuda")
    rt.special_ids = torch.tensor([7], device="cuda", dtype=torch.int32)
    rt.dummy_slot = 99
    weights = torch.tensor([[.25, .75]] * 4, device="cuda", dtype=torch.float16)
    tokens = torch.tensor([1, 7, 1, 1], device="cuda", dtype=torch.int32)
    ws = dict(request_slots=torch.tensor([0, 0, 99, 0], device="cuda", dtype=torch.int32),
              modality=torch.tensor([1, 1, 1, 0], device="cuda", dtype=torch.int32),
              keep=torch.empty_like(weights, dtype=torch.bool))
    rt._select(0, weights, tokens, ws, is_prefill=True)
    prefill = ws["keep"].clone()
    assert prefill.tolist() == [[True, True], [True, True], [False, False], [True, True]]
    rt._select(0, weights, tokens, ws)
    expected = [[False, True], [True, True], [False, False], [True, True]]
    assert ws["keep"].tolist() == expected
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rt._select(0, weights, tokens, ws)
    graph.replay()
    assert ws["keep"].tolist() == expected
    rt._select(0, weights, tokens, ws, is_prefill=True)
    assert torch.equal(ws["keep"], prefill)


def test_policy_requires_calibration_and_explicit_backend():
    with pytest.raises(ValueError, match="offload, hybrid or cpu"):
        prepare_modes_config(SimpleNamespace(moe_policy="modes", moe_strategy="tablemoe"))
    with pytest.raises(ValueError, match="modes-config"):
        prepare_modes_config(SimpleNamespace(moe_policy="modes", moe_strategy="cpu", modes_config=None))


@pytest.mark.parametrize("flag", ["--tablemoe-prefill-keep-rate", "--tablemoe-decode-keep-k"])
@pytest.mark.parametrize("policy", ["modes", "modes_tablemoe"])
def test_cli_rejects_explicit_tablemoe_selection(policy, flag, capsys):
    from freetoken.server.args import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--model", "/unused", "--moe-policy", policy, f"{flag}=1"])
    assert "keep-rate/keep-k flags conflict" in capsys.readouterr().err


@pytest.mark.parametrize("policy", ["modes", "modes_tablemoe"])
def test_cli_tablemoe_prefill_accepts_keep_rate_and_rejects_decode_keep_k(policy, capsys):
    from freetoken.server.args import parse_args

    args = ["--model", "/unused", "--dtype", "float16", "--moe-policy", policy,
            "--modes-prefill-policy", "tablemoe", "--tablemoe-prefill-keep-rate=0.6"]
    config, _ = parse_args(args)
    assert config.modes_prefill_policy == "tablemoe"
    assert config.tablemoe_prefill_keep_rate == .6
    with pytest.raises(SystemExit):
        parse_args(args + ["--tablemoe-decode-keep-k=1"])
    assert "keep-rate/keep-k flags conflict" in capsys.readouterr().err


def _prepare_config(model, path, policy, prefill_policy):
    return SimpleNamespace(
        moe_policy=policy, modes_prefill_policy=prefill_policy, moe_strategy="cpu",
        modes_config=str(path), model_path=str(model), dtype=torch.float16, tp_info=SimpleNamespace(size=1),
        model_config=SimpleNamespace(architectures=["Qwen3VLMoeForConditionalGeneration"],
                                     num_moe_layers=2, num_experts=2, num_experts_per_tok=2, expert_quant="none"),
        tablemoe_implementation="optimized", tablemoe_collect_config=None,
        cuda_graph_bs=[], cuda_graph_max_bs=0, tablemoe_skip_lookup=False, tablemoe_online_cache_size=64,
        tablemoe_decode_keep_k=1, tablemoe_prefill_keep_rate=.6, tablemoe_pca_dir=None, tablemoe_cache_dir=None,
        tablemoe_convert_bf16_weights=False, tablemoe_decode_online_cosine_threshold=None,
        tablemoe_decode_offline_cosine_threshold=None, tablemoe_offline_cosine_threshold_vision=.6,
        tablemoe_offline_cosine_threshold_text=.7, use_dummy_weight=False, moe_cache_size=0,
        moe_cache_rate=None, moe_cache_auto=False, moe_cpu_layers=None)


@pytest.mark.parametrize("policy", ["modes", "modes_tablemoe"])
def test_tablemoe_prefill_requires_assets_even_with_pure_modes_decode(tmp_path, policy):
    model, importance, search = _assets(tmp_path)
    path = _importer()(importance, search, model, tmp_path / "imported")
    config = _prepare_config(model, path, policy, "tablemoe")
    with pytest.raises(ValueError, match="tablemoe-pca-dir"):
        prepare_modes_config(config)
    assert not config.tablemoe_skip_lookup
    assert config.tablemoe_decode_keep_k == 2
    assert config.tablemoe_expert_mode == "cpu"
    pca, cache = tmp_path / "pca", tmp_path / "cache"
    pca.mkdir()
    cache.mkdir()
    config.tablemoe_pca_dir, config.tablemoe_cache_dir = str(pca), str(cache)
    prepare_modes_config(config)
    assert config.tablemoe_online_cache_size == (0 if policy == "modes" else 64)
    assert config.tablemoe_decode_online_cosine_threshold == .7
    assert config.tablemoe_decode_offline_cosine_threshold == .7


def test_prefill_policy_rejects_native_invalid_and_lookup_disabled():
    with pytest.raises(ValueError, match="requires a MoDES policy"):
        prepare_modes_config(SimpleNamespace(moe_policy="native", modes_prefill_policy="tablemoe"))
    with pytest.raises(ValueError, match="invalid --modes-prefill-policy"):
        prepare_modes_config(SimpleNamespace(moe_policy="modes", modes_prefill_policy="bad"))
    config = _prepare_config("/unused", "/unused", "modes", "tablemoe")
    config.tablemoe_skip_lookup = True
    with pytest.raises(ValueError, match="requires lookup"):
        prepare_modes_config(config)


@pytest.mark.parametrize("prefill_policy", ["modes", "tablemoe"])
@pytest.mark.parametrize("prefill", [False, True])
def test_attention_scores_only_for_tablemoe_prefill(monkeypatch, prefill_policy, prefill):
    from freetoken.models.qwen3_moe import attention as module
    from freetoken.moe import tablemoe

    reqs = [SimpleNamespace(extend_len=2)]
    batch = SimpleNamespace(is_prefill=prefill, padded_reqs=reqs, tablemoe_attn_scores={},
                            get_attn_positions=lambda: torch.arange(2))
    context = SimpleNamespace(batch=batch,
                              tablemoe_runtime=SimpleNamespace(prefill_uses_tablemoe=prefill_policy == "tablemoe"),
                              attn_backend=SimpleNamespace(forward=lambda q, k, v, layer, batch: q))
    monkeypatch.setattr(module, "get_global_ctx", lambda: context)
    scores = torch.tensor([.3, .7])
    monkeypatch.setattr(tablemoe, "last_token_attention_scores", lambda q, k, requests: scores)
    attention = SimpleNamespace(qkv_proj=SimpleNamespace(forward=lambda x: torch.zeros(2, 12)),
                                q_norm=None, k_norm=None, num_qo_heads=1, num_kv_heads=1,
                                qo_attn_dim=4, kv_attn_dim=4, head_dim=4, layer_id=0,
                                rotary=SimpleNamespace(forward=lambda pos, q, k: (q, k)),
                                o_proj=SimpleNamespace(forward=lambda x: x))
    module.Qwen3MoeAttention.forward(attention, torch.zeros(2, 4))
    assert (0 in batch.tablemoe_attn_scores) == (prefill and prefill_policy == "tablemoe")
    if batch.tablemoe_attn_scores:
        assert batch.tablemoe_attn_scores[0] is scores


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["cpu", "offload", "hybrid"])
@pytest.mark.parametrize("policy", ["modes", "modes_tablemoe"])
@pytest.mark.parametrize("prefill_policy", ["modes", "tablemoe"])
def test_modes_constructor_loads_checkpoint_and_policy_assets(tmp_path, monkeypatch, mode, policy,
                                                             prefill_policy, record_property):
    from freetoken.moe import modes_runtime
    from transformers import Qwen3VLMoeConfig

    model, importance, search = _assets(tmp_path)
    hf = Qwen3VLMoeConfig(text_config=dict(
        num_hidden_layers=1, hidden_size=128, num_attention_heads=2, num_key_value_heads=2,
        head_dim=64, intermediate_size=256, num_experts=2, num_experts_per_tok=2,
        moe_intermediate_size=64, vocab_size=128)).to_dict()
    hf["architectures"] = ["Qwen3VLMoeForConditionalGeneration"]
    (model / "config.json").write_text(json.dumps(hf))
    importance.write_bytes(pickle.dumps({0: dict(text=1., visual=1.)}))
    search.write_bytes(pickle.dumps(dict(tau=dict(text=.5, visual=.5))))
    hidden, inter = 128, 64
    torch.manual_seed(9)
    gate_up = torch.randn(2, 2 * inter, hidden).half() / 5
    down = torch.randn(2, hidden, inter).half() / 5
    save_file({"model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up.transpose(1, 2).contiguous(),
               "model.language_model.layers.0.mlp.experts.down_proj": down.transpose(1, 2).contiguous()},
              model / "model.safetensors")
    path = _importer()(importance, search, model, tmp_path / "calibration")
    mc = SimpleNamespace(architectures=hf["architectures"], hidden_size=hidden, num_experts=2,
                         num_experts_per_tok=2, num_moe_layers=1, num_layers=1,
                         moe_intermediate_size=inter, expert_quant="none")
    cfg = SimpleNamespace(
        moe_policy=policy, modes_prefill_policy=prefill_policy, moe_strategy=mode,
        modes_config=str(path), model_config=mc,
        dtype=torch.float16, tp_info=SimpleNamespace(size=1), model_path=str(model),
        use_dummy_weight=False, moe_cache_size=0 if mode == "cpu" else 2,
        moe_cache_rate=None, moe_cache_auto=False, moe_cpu_layers=None, moe_cpu_threads=2,
        moe_hybrid_max_fetch=0, moe_cache_policy="lru", max_running_req=2, max_extend_tokens=128,
        cuda_graph_bs=[], cuda_graph_max_bs=0,
        tablemoe_pca_dir=None, tablemoe_cache_dir=None, tablemoe_convert_bf16_weights=False,
        tablemoe_expert_mode=mode, tablemoe_implementation="optimized", tablemoe_skip_lookup=False,
        tablemoe_stats=True, tablemoe_report_path=None, tablemoe_collect_config=None,
        tablemoe_prefill_keep_rate=.6, tablemoe_decode_keep_k=1, tablemoe_online_cache_size=64,
        tablemoe_offline_cosine_threshold_vision=.6, tablemoe_offline_cosine_threshold_text=.7,
        tablemoe_decode_online_cosine_threshold=None, tablemoe_decode_offline_cosine_threshold=None)
    if policy == "modes_tablemoe" or prefill_policy == "tablemoe":
        pca, cache = tmp_path / "pca", tmp_path / "cache"
        (pca / "layer_0").mkdir(parents=True)
        (cache / "layer_0").mkdir(parents=True)
        for mod in ("text", "vision"):
            torch.save(dict(mean=torch.zeros(hidden), proj=torch.eye(hidden, 2)),
                       pca / "layer_0" / f"L0_{mod}_pca.pt")
        torch.save(dict(key=torch.tensor([[1., 0.]]).half(), value=torch.ones(1, hidden).half()),
                   cache / "layer_0/L0_E1_text_cache.pt")
        cfg.tablemoe_pca_dir, cfg.tablemoe_cache_dir = str(pca), str(cache)
    prepare_modes_config(cfg)
    from freetoken.models.qwen3_vl import weight as vl_weight
    from freetoken.distributed import DistributedInfo

    monkeypatch.setattr(vl_weight, "get_tp_info", lambda: DistributedInfo(0, 1))
    monkeypatch.setattr(modes_runtime, "load_tokenizer", lambda _: SimpleNamespace(all_special_ids=[7]))
    rt = modes_runtime.ModesRuntime(cfg)
    try:
        if policy == "modes" and prefill_policy == "modes":
            assert rt.pca == rt.offline_paths == {}
        else:
            assert len(rt.asset_fingerprints) == 3
            assert rt.config.tablemoe_decode_online_cosine_threshold == .7
        assert (rt.fixed_online[0]["keys"].numel() == 0) == (policy == "modes")
        assert rt.lookup_enabled(True) == (prefill_policy == "tablemoe")
        assert rt.lookup_enabled(False) == (policy == "modes_tablemoe")
        assert (rt.expert_cache is None) == (mode == "cpu")
        x = torch.randn(1, hidden, device="cuda", dtype=torch.float16) / 5
        ids = torch.tensor([[0, 1]], device="cuda")
        weights = torch.tensor([[.25, .75]], device="cuda")
        batch = SimpleNamespace(tablemoe_slots=None, is_prefill=False, mm_rows=None,
                                input_ids=torch.tensor([1], device="cuda", dtype=torch.int32),
                                padded_reqs=[SimpleNamespace(uid=1, table_idx=0, extend_len=1)])
        actual = rt.forward(0, x, weights, ids, batch)
        g, u = F.linear(x, gate_up[1].cuda()).chunk(2, -1)
        expected = (F.linear(F.silu(g) * u, down[1].cuda()).float() * .75).half()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        error = (actual.float() - expected.float()).abs()
        record_property("max_absolute_error", error.max().item())
        record_property("max_relative_error_eps_1e-5", (error / expected.float().abs().clamp_min(1e-5)).max().item())
        assert rt.resident_gpu_bytes() > 0
        rt.release_request(1)
    finally:
        rt.close()
