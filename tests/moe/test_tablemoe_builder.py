from pathlib import Path

import pytest
import torch

from freetoken.moe.tablemoe_artifacts import finish_stage, read_json, verify_stage, write_json
from freetoken.moe.tablemoe_builder import (
    InputCollector, algorithm_config, cluster, expert_values, finalize_collection, kmeans, materialize, train_pca,
)


def test_disk_reservoir_resume_and_owned_snapshot(tmp_path):
    from types import SimpleNamespace

    def run(root, resume):
        collector = InputCollector(root, dict(reservoir_size=5), 4, 2, identity="input")
        x = torch.arange(80).reshape(20, 4).half()
        ids = torch.tensor([[0, 1]] * 20)
        collector.add(0, x[:10], ids[:10], torch.ones(10), "prefill")
        collector.flush(1)
        if resume:
            collector.close()
            collector = InputCollector(root, dict(reservoir_size=5), 4, 2, identity="input")
        collector.snapshot(0, x[10:], ids[10:], SimpleNamespace(is_prefill=True, mm_rows=None,
                           padded_reqs=[SimpleNamespace(uid=2, extend_len=10)]))
        x[10:].zero_()
        ids[10:].zero_()
        collector.flush(2)
        collector.close()
        finalize_collection(root, dict(sample_count=2), 1, 2)
        return torch.load(root / "layer_0/L0_E1_text_inputs.pt", weights_only=True)

    a = run(tmp_path / "continuous", False)
    b = run(tmp_path / "resumed", True)
    assert len(a["inputs"]) == 5
    assert torch.equal(a["inputs"], b["inputs"])
    assert torch.count_nonzero(a["inputs"]) > 0
    with pytest.raises(ValueError, match="mismatch"):
        InputCollector(tmp_path / "resumed", {}, 4, 2, identity="different")


def test_pca_rank_and_missing_modality_fail_explicitly():
    with pytest.raises(ValueError, match="coverage gap"):
        train_pca(torch.randn(2, 4), 2)
    with pytest.raises(ValueError, match="rank coverage gap"):
        train_pca(torch.ones(8, 4), 2)


def test_kmeans_small_samples_duplicate_centers_and_high_centroids():
    x = torch.tensor([[0., 0.], [0., 0.], [3., 3.]])
    high = torch.tensor([[1., 2., 3.], [3., 4., 5.], [7., 8., 9.]])
    out = kmeans(x, high, algorithm_config(dict(clusters=256)))
    assert len(out["key"]) == 2
    assert sorted(out["counts"].tolist()) == [1, 2]
    assert torch.equal(out["centroid"][out["key"][:, 0] == 0], torch.tensor([[2., 3., 4.]]).half())


def test_cuda_centroids_repeat_at_reservoir_capacity():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    generator = torch.Generator(device="cuda").manual_seed(42)
    high = torch.randn(30000, 32, generator=generator, device="cuda")
    low = high[:, :4]
    config = algorithm_config(dict(clusters=4, iterations=3))
    expected = kmeans(low, high, config)
    for _ in range(3):
        actual = kmeans(low, high, config)
        for key in ("key", "centroid", "counts"):
            assert torch.equal(actual[key], expected[key]), key


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_full_builder_sparse_assets_and_expert_values(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(42)
    collect = tmp_path / "collect"
    collector = InputCollector(collect, dict(reservoir_size=30), 4, 2)
    inputs = dict(sample_count=1, checkpoint={"path": "fixture"}, convert_bf16=True, processor={})
    x = torch.randn(24, 4).half()
    ids = torch.zeros(24, 1, dtype=torch.int64)
    collector.add(0, x, ids, torch.arange(24) % 2, "prefill")
    collector.flush(1)
    collector.close()
    finalize_collection(collect, inputs, 1, 2)
    config = dict(pca_dim=2, clusters=8, device=device)
    cluster(tmp_path, config)
    gate = torch.randn(2, 6, 4).bfloat16()
    down = torch.randn(2, 4, 3).bfloat16()
    pieces = [(0, 0, 2, dict(gate_up=gate, down=down))]
    materialize(tmp_path, config, pieces=pieces)
    manifest = verify_stage(tmp_path / "cache")
    assert manifest["missing_tables"] == ["L0_E1_vision", "L0_E1_text"]
    path = tmp_path / "cache/layer_0/L0_E0_text_cache.pt"
    data = torch.load(path, weights_only=True)
    centroids = torch.load(tmp_path / "cluster/layer_0/L0_E0_text_clusters.pt", weights_only=True)["centroid"]
    expected = expert_values(centroids.to(device), gate[0].to(device).half(), down[0].to(device).half())
    torch.testing.assert_close(data["value"], expected, atol=0, rtol=0)
    assert data["key"].dtype == data["value"].dtype == torch.float16
    assert data["id"].tolist() == list(range(len(data["key"])))
    assert materialize(tmp_path, config, pieces=pieces)["input_sha256"] == manifest["input_sha256"]
    write_json(tmp_path / "cluster/complete.json", dict(complete=False))
    cluster(tmp_path, config)
    write_json(tmp_path / "cache/complete.json", dict(complete=False))
    materialize(tmp_path, config, pieces=pieces)
    write_json(tmp_path / "cache/complete.json", dict(complete=False))
    with pytest.raises(ValueError, match="input mismatch"):
        materialize(tmp_path, {**config, "value_chunk": 32}, pieces=pieces)
    materialize(tmp_path, config, pieces=pieces)
    with pytest.raises(ValueError, match="mismatch"):
        cluster(tmp_path, {**config, "clusters": 4})
    path.write_bytes(b"interrupted")
    with pytest.raises(ValueError, match="changed artifact"):
        verify_stage(tmp_path / "cache")


def test_builder_ignores_undeclared_cluster_files(tmp_path):
    path = tmp_path / "cluster/layer_0/L0_E0_text_clusters.pt"
    path.parent.mkdir(parents=True)
    torch.save(dict(key=torch.zeros(1, 2).half(), centroid=torch.ones(1, 4).half()), path)
    finish_stage(tmp_path / "cluster", {}, [], source_inputs=dict(checkpoint={}, convert_bf16=False, processor={}),
                 num_layers=1, num_experts=1, missing_tables=["L0_E0_text", "L0_E0_vision"])
    materialize(tmp_path, dict(device="cpu"), pieces=[(0, 0, 1, dict(gate_up=torch.ones(1, 6, 4).half(),
                                                                  down=torch.ones(1, 4, 3).half()))])
    assert not list((tmp_path / "cache").glob("layer_*/*_cache.pt"))


def test_incomplete_collection_and_missing_modality_are_not_complete(tmp_path):
    root = tmp_path / "collect"
    collector = InputCollector(root, {}, 4, 1)
    collector.add(0, torch.randn(6, 4).half(), torch.zeros(6, 1, dtype=torch.long), torch.ones(6), "decode")
    collector.flush(0)
    collector.close()
    with pytest.raises(ValueError, match="coverage mismatch"):
        finalize_collection(root, dict(sample_count=2), 1, 1)
    finalize_collection(root, dict(sample_count=1), 1, 1)
    with pytest.raises(ValueError, match="no inputs"):
        cluster(tmp_path, dict(pca_dim=2, device="cpu"))
    assert read_json(tmp_path / "cluster/complete.json")["complete"] is False


def test_conversion_mismatch_is_not_a_complete_asset(tmp_path):
    root = tmp_path / "cluster"
    path = root / "layer_0/L0_E0_text_clusters.pt"
    path.parent.mkdir(parents=True)
    torch.save(dict(key=torch.zeros(1, 2).half(), centroid=torch.ones(1, 4).half()), path)
    finish_stage(root, {}, [path], source_inputs=dict(checkpoint={}, convert_bf16=False, processor={}),
                 num_layers=1, num_experts=1, missing_tables=["L0_E0_vision"])
    with pytest.raises(ValueError, match="conversion policy"):
        materialize(tmp_path, dict(device="cpu"), pieces=[(0, 0, 1, dict(gate_up=torch.ones(1, 6, 4).bfloat16(),
                                                                      down=torch.ones(1, 4, 3).bfloat16()))])
    assert not read_json(tmp_path / "cache/complete.json")["complete"]


def test_async_cuda_collection_owns_inputs_and_routes(tmp_path):
    from types import SimpleNamespace
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    collector = InputCollector(tmp_path, dict(reservoir_size=16), 32, 2)
    x = torch.arange(256, device="cuda", dtype=torch.float16).reshape(8, 32)
    ids = torch.ones(8, 1, device="cuda", dtype=torch.int64)
    expected = x.cpu()
    collector.snapshot(0, x, ids, SimpleNamespace(is_prefill=False, padded_reqs=[SimpleNamespace(uid=i) for i in range(8)]))
    x.zero_()
    ids.zero_()
    collector.flush(1)
    collector.close()
    finalize_collection(tmp_path, dict(sample_count=1), 1, 2)
    stored = torch.load(tmp_path / "layer_0/L0_E1_text_inputs.pt", weights_only=True)
    assert torch.equal(stored["inputs"], expected)
    assert stored["phase"].tolist() == [1] * 8


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_kmeans_sparse_duplicate_and_reproducible_centroids(device):
    from freetoken.moe.tablemoe_builder import batched_kmeans
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    low = torch.tensor([[0., 0.], [0., 0.], [3., 3.]], device=device)
    high = torch.tensor([[1., 2., 3.], [3., 4., 5.], [7., 8., 9.]], device=device)
    samples = [(low, high, "expert0"), (low[:1], high[:1], "expert1")]
    config = algorithm_config(dict(clusters=256, iterations=30))
    out = batched_kmeans(samples, config)
    repeat = batched_kmeans(samples, config)
    for name in out:
        for key in out[name]:
            assert torch.equal(out[name][key], repeat[name][key])
    assert sorted(out["expert0"]["counts"].tolist()) == [1, 2]
    assert torch.equal(out["expert0"]["centroid"][out["expert0"]["key"][:, 0] == 0], torch.tensor([[2., 3., 4.]]).half())
    assert out["expert1"]["counts"].tolist() == [1]


def test_exact_modality_budget_and_raw_only_inputs(tmp_path):
    config = dict(reservoir_size=30, vision_budget=8, vision_population=20, export_inputs=False)
    collector = InputCollector(tmp_path / "collect", config, 4, 1)
    x = torch.arange(120).reshape(30, 4).half()
    ids = torch.zeros(30, 1, dtype=torch.long)
    modality = torch.tensor([0] * 20 + [1] * 10)
    for start, end in ((0, 7), (7, 23), (23, 30)):
        for layer in (0, 1):
            collector.add(layer, x[start:end], ids[start:end], modality[start:end], "prefill")
    collector.flush(0)
    collector.close()
    manifest = finalize_collection(tmp_path / "collect", dict(sample_count=1), 2, 1)
    assert manifest["sampling"]["0"] == manifest["sampling"]["1"]
    assert manifest["sampling"]["0"]["vision"]["selected"] == 8
    assert manifest["sampling"]["0"]["text"]["selected"] == 10
    assert not list((tmp_path / "collect").rglob("*_inputs.pt"))
    cluster(tmp_path, dict(pca_dim=1, clusters=4, device="cpu"))
    verify_stage(tmp_path / "cluster")
    assert (tmp_path / "collect/layer_0/L0_E0_vision.bin").read_bytes() == (
        tmp_path / "collect/layer_1/L1_E0_vision.bin").read_bytes()


def test_incomplete_modality_population_is_rejected(tmp_path):
    collector = InputCollector(tmp_path, dict(vision_budget=4, vision_population=10), 4, 1)
    collector.add(0, torch.randn(8, 4).half(), torch.zeros(8, 1, dtype=torch.long), torch.zeros(8), "prefill")
    collector.flush(0)
    collector.close()
    with pytest.raises(ValueError, match="population or budget"):
        finalize_collection(tmp_path, dict(sample_count=1), 1, 1)


@pytest.mark.parametrize("capacity", [4, 30000])
def test_incremental_checksums_match_files_for_append_and_reservoir_replacement(tmp_path, capacity):
    from freetoken.moe.tablemoe_artifacts import file_hash
    collector = InputCollector(tmp_path, dict(reservoir_size=capacity), 4, 1)
    for start in range(0, 32, 4):
        x = torch.arange(start * 4, (start + 4) * 4).reshape(4, 4).half()
        collector.add(0, x, torch.zeros(4, 1, dtype=torch.long), torch.ones(4, dtype=torch.int8), "decode")
        collector.flush(start)
        state = read_json(tmp_path / "progress.json")["buckets"]["L0_E0_text"]
        for suffix in ("bin", "phase"):
            assert state[suffix + "_sha256"] == file_hash(tmp_path / f"layer_0/L0_E0_text.{suffix}")
    collector.close()


@pytest.mark.parametrize("args,expected", [
    ((10000, 100000, 72, [], 1, 64), 64),
    ((100, 40000, 72, [], 8, 128), 100),
    ((100, 20, 4, [400], 8, 128), 8),
    ((100, 3, 4, [400], 8, 128), 3),
    ((100, 0, 4, [], 8, 128), 0),
])
def test_original_tablemoe_adaptive_quota(args, expected):
    from freetoken.moe.tablemoe_builder import adaptive_quota
    assert adaptive_quota(*args) == expected


def test_per_sample_prefill_selection_respects_caps_chunks_layers_and_resume(tmp_path):
    from types import SimpleNamespace
    from freetoken.moe.tablemoe_builder import load_inputs

    config = dict(sampling_policy="tablemoe_default", sample_count=2, export_inputs=False,
                  max_vision_tokens=100, max_text_tokens=150)
    inputs = torch.arange(2400).reshape(600, 4).half()
    ids = torch.zeros(600, 1, dtype=torch.long)
    collector = InputCollector(tmp_path, config, 4, 1, identity="samples")

    def snapshot_chunk(collector, uid, lo, hi, admission):
        req = SimpleNamespace(uid=uid, cached_len=lo, extend_len=hi-lo, input_ids=torch.zeros(hi),
                              mm_items=[SimpleNamespace(offsets=[(0, 300)])])
        batch = SimpleNamespace(is_prefill=True, padded_reqs=[req],
                                prompt_admissions=[(uid, 600, 0)] if admission else [])
        for layer in [0, 1]:
            collector.snapshot(layer, inputs[lo:hi], ids[lo:hi], batch)

    snapshot_chunk(collector, 0, 0, 300, True)
    snapshot_chunk(collector, 0, 300, 600, False)
    collector.flush(0)
    collector.snapshot(0, inputs[:1], ids[:1], SimpleNamespace(is_prefill=False))
    collector.close()
    progress = read_json(tmp_path / "progress.json")
    assert progress["sampling"]["0"] == progress["sampling"]["1"]
    assert progress["sample_state"]["counts"] == dict(vision=50, text=75)
    chosen = progress["sample_state"]["records"][0]["positions"]
    assert len(set(chosen["vision"])) == 50 and len(set(chosen["text"])) == 75
    collector = InputCollector(tmp_path, config, 4, 1, identity="samples")
    snapshot_chunk(collector, 0, 0, 600, True)
    collector.flush(0)
    collector.close()
    progress = read_json(tmp_path / "progress.json")
    assert progress["sample_state"]["counts"] == dict(vision=100, text=150)
    assert progress["tokens"] == dict(prefill=1200, decode=0)
    assert progress["sampling"]["0"] == progress["sampling"]["1"]
    second = progress["sample_state"]["records"][1]["positions"]
    assert second != chosen
    assert len(progress["sample_state"]["records"]) == 2
    finalize_collection(tmp_path, dict(sample_count=2), 2, 1)
    for mod in ['vision', 'text']:
        state = progress['buckets'][f'L0_E0_{mod}']
        actual = load_inputs(tmp_path, f'L0_E0_{mod}', state, 4)
        expected = torch.cat([inputs[torch.tensor(p[mod])] for p in [chosen, second]])
        assert torch.equal(actual, expected)


def test_per_sample_default_selection_matches_seeded_reference(tmp_path):
    from types import SimpleNamespace
    import numpy as np
    config = dict(sampling_policy='tablemoe_default', sample_count=1)
    collector = InputCollector(tmp_path, config, 4, 1)
    x = torch.randn(600, 4).half()
    req = SimpleNamespace(uid=1, cached_len=0, extend_len=600, input_ids=torch.zeros(600),
                          mm_items=[SimpleNamespace(offsets=[(0, 300)])])
    batch = SimpleNamespace(is_prefill=True, padded_reqs=[req], prompt_admissions=[(1, 600, 0)])
    collector.snapshot(0, x, torch.zeros(600, 1, dtype=torch.long), batch)
    collector.flush(1)
    collector.close()
    saved = read_json(tmp_path/'progress.json')
    record = saved['sample_state']['records'][0]
    rng = np.random.default_rng(42)
    assert record['positions']['vision'] == sorted(rng.choice(np.arange(300), 64, replace=False).tolist())
    assert record['positions']['text'] == sorted(rng.choice(np.arange(300, 600), 128, replace=False).tolist())
    assert saved['sampling']['0']['vision']['selected'] == 64
    assert saved['sampling']['0']['text']['selected'] == 128


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_fixed_decode_sampling_shares_text_quota_across_phases_layers_and_resume(tmp_path, device):
    from types import SimpleNamespace
    from freetoken.moe.tablemoe_builder import load_inputs

    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    config = dict(sampling_policy='tablemoe_default', sample_count=2, sample_decode_tokens=7,
                  min_vision_per_sample=3, max_vision_per_sample=3, max_vision_tokens=6,
                  min_text_per_sample=8, max_text_per_sample=8, max_text_tokens=16, export_inputs=False)
    x = torch.arange(68, device=device).reshape(17, 4).half()
    ids = torch.zeros(17, 1, dtype=torch.long, device=device)
    for sample in range(2):
        collector = InputCollector(tmp_path, config, 4, 1, identity='fixed')
        req = SimpleNamespace(uid=0, cached_len=0, extend_len=5, input_ids=torch.zeros(10),
                              mm_items=[SimpleNamespace(offsets=[(0, 5)])])
        batch = SimpleNamespace(is_prefill=True, padded_reqs=[req], prompt_admissions=[(0, 10, 0)])
        for lo in [0, 5]:
            req.cached_len = lo
            for layer in [0, 1]:
                collector.snapshot(layer, x[lo:lo+5], ids[lo:lo+5], batch)
        batch.is_prefill = False
        req.extend_len = 1
        for position in range(10, 17):
            req.cached_len = position
            for layer in [0, 1]:
                collector.snapshot(layer, x[position:position+1], ids[position:position+1], batch)
        collector.flush(0)
        collector.close()
    saved = read_json(tmp_path / 'progress.json')
    assert saved['tokens'] == dict(prefill=20, decode=14)
    assert saved['sample_state']['counts'] == dict(vision=6, text=16)
    assert saved['sampling']['0'] == saved['sampling']['1']
    for record in saved['sample_state']['records']:
        assert record['decode_seen'] == 7
        assert record['candidates'] == dict(vision=5, text=12)
        assert len(record['positions']['text']) == 8
    for mod in ['vision', 'text']:
        key = f'L0_E0_{mod}'
        actual = load_inputs(tmp_path, key, saved['buckets'][key], 4)
        expected = torch.cat([x[torch.tensor(record['positions'][mod])] for record in saved['sample_state']['records']])
        assert torch.equal(actual, expected.cpu())
    assert 0 < saved['sampling']['0']['text']['decode'] < 16
    finalize_collection(tmp_path, dict(sample_count=2), 2, 1)


def test_fixed_decode_sampling_rejects_incomplete_generation(tmp_path):
    from types import SimpleNamespace

    collector = InputCollector(tmp_path, dict(sampling_policy='tablemoe_default', sample_count=1,
                               sample_decode_tokens=7), 4, 1)
    req = SimpleNamespace(uid=0, cached_len=0, extend_len=10, input_ids=torch.zeros(10),
                          mm_items=[SimpleNamespace(offsets=[(0, 5)])])
    batch = SimpleNamespace(is_prefill=True, padded_reqs=[req], prompt_admissions=[(0, 10, 0)])
    collector.snapshot(0, torch.ones(10, 4), torch.zeros(10, 1, dtype=torch.long), batch)
    collector.flush(0)
    collector.close()
    with pytest.raises(ValueError, match='fixed-length collection coverage'):
        finalize_collection(tmp_path, dict(sample_count=1), 1, 1)
