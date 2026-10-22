import json
from types import SimpleNamespace
import hashlib

import pytest
import torch

from freetoken.moe.preverify_policy import PreverifyPolicy, merge_identity


def test_threshold_equality_and_ready_hits_do_not_double_identity():
    policy = PreverifyPolicy("modes", replacement="identity", importance=(.5, .5), tau=.0625)
    weights = torch.tensor([[.25, .125, .125, .125, .125, .0625, .0625, .125]])
    ready = torch.tensor([[False, False, False, False, False, True, False, False]])
    keep = policy.mask(weights, 0, ready)
    assert keep.tolist() == [[True, True, True, True, True, True, False, True]]
    hidden = torch.tensor([[2., 4.]])
    identity = policy.identity_term(hidden, weights, keep)
    hidden.zero_()
    assert identity.tolist() == [[.125, .25]]
    assert merge_identity(torch.zeros_like(identity), identity).tolist() == [[.125, .25]]


def test_stable_top_k_and_zero_all_skip():
    weights = torch.full((2, 8), .125)
    assert PreverifyPolicy(top_k=4).mask(weights, 0).tolist() == [[True] * 4 + [False] * 4] * 2
    policy = PreverifyPolicy("modes", replacement="identity", importance=(1.,), tau=1.)
    hidden = torch.tensor([[2., 3.], [4., 5.]])
    torch.testing.assert_close(policy.identity_term(hidden, weights, policy.mask(weights, 0)), hidden, rtol=0, atol=0)
    assert PreverifyPolicy().identity_term(hidden, weights, weights > 0) is None


def test_calibration_is_bound_to_model_replacement_and_layer_coverage(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    asset = dict(model_config_sha256=hashlib.sha256(b"{}").hexdigest(), dtype="bfloat16", top_k=8,
                 weight_index_sha256=hashlib.sha256(b"{}").hexdigest(),
                 score_dtype="float32", replacement="identity", status="validated", importance=[.5, .5], tau=.01)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(asset))
    config = SimpleNamespace(speculative_preverify_policy="modes", speculative_preverify_replacement="identity",
                             speculative_modes_calibration=str(path))
    assert PreverifyPolicy.from_config(config, layers=2, model_path=tmp_path).importance == (.5, .5)
    with pytest.raises(ValueError, match="coverage"):
        PreverifyPolicy.from_config(config, layers=3, model_path=tmp_path)
    config.speculative_preverify_replacement = "zero"
    with pytest.raises(ValueError, match="mismatch"):
        PreverifyPolicy.from_config(config, layers=2, model_path=tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("retained", [0, 4, 8])
def test_resident_invalid_routes_skip_poison_and_match_full_control(retained):
    from freetoken.moe.fused import fused_experts_impl
    torch.manual_seed(62)
    device = "cuda"
    hidden = torch.randn(2, 64, dtype=torch.bfloat16, device=device)
    w1 = torch.randn(8, 128, 64, dtype=torch.bfloat16, device=device) * .01
    w2 = torch.randn(8, 64, 64, dtype=torch.bfloat16, device=device) * .01
    ids = torch.arange(8, dtype=torch.int32, device=device).expand(2, -1).contiguous()
    weights = torch.full((2, 8), .125, device=device)
    reference = fused_experts_impl(hidden.clone(), w1, w2, weights, ids).clone() if retained == 8 else None
    w1[retained:] = float("nan")
    w2[retained:] = float("nan")
    ids[:, retained:] = -1
    weights[:, retained:] = 0
    output = fused_experts_impl(hidden.clone(), w1, w2, weights, ids, skip_invalid_routes=True)
    assert torch.isfinite(output).all()
    if retained == 0:
        assert torch.count_nonzero(output) == 0
    if reference is not None:
        torch.testing.assert_close(output, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_resident_reference_math_is_width_invariant_with_invalid_routes(monkeypatch):
    import freetoken.core as core
    from freetoken.core import Batch, Context
    from freetoken.moe.fused import fused_experts_impl
    torch.manual_seed(8)
    ctx = Context(1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    batch = Batch([], "decode")
    batch.speculative_reference_math = True
    hidden = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(8, 128, 64, dtype=torch.bfloat16, device="cuda")*.01
    w2 = torch.randn(8, 64, 64, dtype=torch.bfloat16, device="cuda")*.01
    ids = torch.arange(8, device="cuda", dtype=torch.int32).repeat(4, 1)
    ids[:, 4:] = -1
    weights = torch.full((4, 8), .125, device="cuda")
    weights[:, 4:] = 0
    with ctx.forward_batch(batch):
        one = fused_experts_impl(hidden[:1].clone(), w1, w2, weights[:1], ids[:1], skip_invalid_routes=True)
        many = fused_experts_impl(hidden.clone(), w1, w2, weights, ids, skip_invalid_routes=True)
    torch.testing.assert_close(many[:1], one, rtol=0, atol=0)
