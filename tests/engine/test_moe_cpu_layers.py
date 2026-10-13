"""Resolver for the hybrid CPU/GPU MoE decode split (--moe-cpu-layers).

CPU-only: exercises _parse_cpu_layers_spec / _resolve_cpu_layers without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.engine.engine import _parse_cpu_layers_spec as parse
from freetoken.engine.engine import _resolve_cpu_layers as resolve

L = 40


def test_explicit_list():
    assert parse("3,7,11", L) == frozenset({3, 7, 11})
    assert parse("3, 7 ,11,", L) == frozenset({3, 7, 11})  # whitespace + trailing comma
    assert parse("5,5,5", L) == frozenset({5})  # dups collapse


def test_count_evenly_strided():
    assert parse("8", L) == frozenset({0, 5, 10, 15, 20, 25, 30, 35})
    assert parse("1", L) == frozenset({0})
    assert len(parse(str(L), L)) == L  # all layers
    assert parse("0", L) == frozenset()


def test_fraction():
    assert len(parse("0.5", L)) == L // 2
    assert len(parse("1.0", L)) == L
    assert parse("0.0", L) == frozenset()


def test_empty():
    assert parse("", L) == frozenset()
    assert parse("   ", L) == frozenset()


@pytest.mark.parametrize("spec", ["99", "40,1", "-1", "1.5"])
def test_out_of_range_raises(spec):
    with pytest.raises(ValueError):
        parse(spec, L)


def _cfg(backend, spec=None):
    return SimpleNamespace(moe_strategy=backend, moe_cpu_layers=spec)


def test_resolve_backend_dispatch():
    # cpu backend -> every layer, ignoring any spec
    assert resolve(_cfg("cpu"), L) == frozenset(range(L))
    assert resolve(_cfg("cpu", "8"), L) == frozenset(range(L))
    # offload + spec -> parsed subset
    assert len(resolve(_cfg("offload", "8"), L)) == 8
    # offload, no spec -> none (plain offload)
    assert resolve(_cfg("offload", None), L) == frozenset()
    # non-offload backend ignores the spec (validation lives in _adjust_config)
    assert resolve(_cfg("fused", "8"), L) == frozenset()


def _engine_moe_config(tmp_path, **overrides):
    import json
    import pickle
    import torch
    from safetensors.torch import save_file
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from scripts.import_modes_config import import_config

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(dict(
        architectures=["Qwen3VLMoeForConditionalGeneration"],
        text_config=dict(num_hidden_layers=2))))
    save_file({"weight": torch.ones(2).half()}, model / "model.safetensors")
    importance, search = tmp_path / "importance.pkl", tmp_path / "search.pkl"
    importance.write_bytes(pickle.dumps({i: dict(text=.5, visual=.5) for i in range(2)}))
    search.write_bytes(pickle.dumps(dict(tau=dict(text=.1, visual=.2))))
    path = import_config(importance, search, model, tmp_path / "calibration")
    config = EngineConfig(model_path=str(model), tp_info=DistributedInfo(0, 1),
                          dtype=torch.float16, attention_backend="triton", **overrides)
    object.__setattr__(config, "model_config", SimpleNamespace(
        architectures=["Qwen3VLMoeForConditionalGeneration"], is_moe=True,
        expert_quant="none", moe_weight_format=None, hidden_act="silu",
        num_moe_layers=2, num_experts=4, num_experts_per_tok=2, norm_topk_prob=False))
    return config, str(path)


@pytest.mark.parametrize("backend, expected", [("auto", "fused"), ("offload", "offload")])
def test_native_unified_memory_config_resolution(tmp_path, monkeypatch, backend, expected):
    from freetoken.engine import engine

    config, _ = _engine_moe_config(tmp_path, moe_strategy=backend, moe_cache_size=4)
    monkeypatch.setattr(engine, "_is_unified_memory_gpu", lambda: True)
    engine._adjust_config(config)
    assert config.moe_strategy == expected
    assert config.tablemoe_expert_mode == "offload"
    assert not config.tablemoe_skip_lookup


@pytest.mark.parametrize("policy", ["modes", "modes_tablemoe"])
@pytest.mark.parametrize("backend", ["offload", "hybrid", "cpu"])
@pytest.mark.parametrize("prefill_policy", ["modes", "tablemoe"])
def test_adjust_config_prepares_modes_before_runtime(tmp_path, monkeypatch, policy, backend, prefill_policy):
    from freetoken.engine import engine

    config, path = _engine_moe_config(tmp_path, moe_policy=policy, moe_strategy=backend,
                                     modes_prefill_policy=prefill_policy,
                                     moe_cache_size=0 if backend == "cpu" else 4)
    object.__setattr__(config, "modes_config", path)
    if policy == "modes_tablemoe" or prefill_policy == "tablemoe":
        for name in ("tablemoe_pca_dir", "tablemoe_cache_dir"):
            folder = tmp_path / name
            folder.mkdir()
            object.__setattr__(config, name, str(folder))
    monkeypatch.setattr(engine, "_is_unified_memory_gpu", lambda: False)
    engine._adjust_config(config)
    assert config.moe_strategy == config.tablemoe_expert_mode == backend
    assert config.tablemoe_decode_keep_k == 2
    assert config.model_config.norm_topk_prob
    assert config.tablemoe_decode_online_cosine_threshold == .7
    assert config.tablemoe_decode_offline_cosine_threshold == .7
    assert config.tablemoe_skip_lookup == (policy == "modes" and prefill_policy == "modes")
    assert config.tablemoe_online_cache_size == (0 if policy == "modes" else 64)
    if backend == "cpu":
        assert config.moe_cache_size == 0
        assert config.cuda_graph_max_bs == 0
    else:
        assert config.cuda_graph_max_bs == config.max_running_req


@pytest.mark.parametrize("overrides, error", [
    ({"moe_policy": "modes", "moe_strategy": "hybrid"}, "modes-config"),
    ({"moe_policy": "modes", "moe_strategy": "tablemoe"}, "offload, hybrid or cpu"),
    ({"moe_policy": "unknown"}, "invalid --moe-policy"),
    ({"modes_config": "/unused"}, "requires a MoDES policy"),
])
def test_adjust_config_rejects_invalid_modes_policy(tmp_path, overrides, error):
    from freetoken.engine.engine import _adjust_config

    config, _ = _engine_moe_config(tmp_path, **overrides)
    with pytest.raises(ValueError, match=error):
        _adjust_config(config)


@pytest.mark.parametrize("graph", [{"cuda_graph_max_bs": 1}, {"cuda_graph_bs": [1]}])
def test_modes_cpu_rejects_explicit_graph_at_config_time(tmp_path, graph):
    from freetoken.engine.engine import _adjust_config

    config, path = _engine_moe_config(tmp_path, moe_policy="modes", moe_strategy="cpu", **graph)
    object.__setattr__(config, "modes_config", path)
    with pytest.raises(ValueError, match="CPU backend supports eager only"):
        _adjust_config(config)


def test_adjust_config_rejects_modes_on_dense_model(tmp_path):
    from freetoken.engine.engine import _adjust_config

    config, path = _engine_moe_config(tmp_path, moe_policy="modes", moe_strategy="hybrid", moe_cache_size=4)
    object.__setattr__(config, "modes_config", path)
    config.model_config.architectures = ["Qwen3ForCausalLM"]
    config.model_config.is_moe = False
    with pytest.raises(ValueError, match="MoDES requires Qwen3VLMoeForConditionalGeneration"):
        _adjust_config(config)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
