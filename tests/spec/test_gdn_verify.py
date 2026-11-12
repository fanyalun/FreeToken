"""The GDN verify kernels: the fused recurrence's verify mode stashes each row's inputs, and gdn_state_replay over the first n rows reproduces decode mode bitwise for every n.

The layer-level verify (outputs, live state, commit) is checked against the reference in tests/layers/test_gdn.py.
"""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NK, NV, HD = 2, 4, 32
STEPS = 5  # verify width: k drafts + 1


@cuda
# the second geometry launches more layer x row x head programs than one grid axis holds (48 * 29 * 48 > 65535)
@pytest.mark.parametrize("layers,bs,h,hv,dim", [(2, 3, NK, NV, HD), (48, 29, 16, 48, 16)], ids=["small", "past-grid-axis-limit"])
def test_kernel_verify_stash_replay_matches_decode_bitwise(layers, bs, h, hv, dim):
    """The fused kernel's verify mode (disable_state_update + input stash) followed by
    gdn_state_replay over the first n tokens must reproduce decode mode BITWISE on
    identical per-token inputs, for every n, across layers and with verify rows that
    differ from the live slots; the verify itself must leave the live slab untouched."""
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update as fused
    from freetoken.kernel.fla import gdn_state_replay

    torch.manual_seed(0)
    dev = torch.device("cuda")
    steps, kd, vd = STEPS, dim, dim
    slots = [(i + 2) % bs for i in range(bs)]  # verify row i commits into live slot slots[i]

    def r(*s, dt=torch.bfloat16, sc=0.5):
        return (torch.randn(*s, device=dev) * sc).to(dt)

    q, k, v = r(layers, 1, bs * steps, h, kd), r(layers, 1, bs * steps, h, kd), r(layers, 1, bs * steps, hv, vd)
    a, b = r(layers, bs * steps, hv), r(layers, bs * steps, hv)
    a_log, dt_bias = r(layers, hv, dt=torch.float32, sc=0.1), r(layers, hv, dt=torch.float32, sc=0.1)
    state0 = torch.randn(layers, bs + 1, hv, kd, vd, device=dev) * 0.1
    slot_t = torch.tensor(slots, dtype=torch.int32, device=dev)
    common = dict(softplus_beta=1.0, softplus_threshold=20.0, scale=kd ** -0.5,
                  use_qk_l2norm_in_kernel=True, initial_state_indices=slot_t)

    per_step = []  # per_step[t][l] = live slab after tokens 0..t
    st_decode = state0.clone()
    for t in range(steps):
        rows = torch.arange(bs, device=dev) * steps + t
        for l in range(layers):
            fused(A_log=a_log[l], dt_bias=dt_bias[l], a=a[l][rows], q=q[l][:, rows], k=k[l][:, rows],
                  v=v[l][:, rows], b=b[l][rows], initial_state_source=st_decode[l],
                  cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=dev), **common)
        per_step.append(st_decode.clone())

    stash = tuple(torch.zeros(layers, bs, steps, *shape, device=dev)
                  for shape in ((h, kd), (hv, vd), (hv,), (hv,)))
    st_verify = state0.clone()
    for l in range(layers):
        fused(A_log=a_log[l], dt_bias=dt_bias[l], a=a[l], q=q[l], k=k[l], v=v[l], b=b[l],
              initial_state_source=st_verify[l],
              cu_seqlens=torch.arange(bs + 1, dtype=torch.int64, device=dev) * steps,
              disable_state_update=True, input_stash=tuple(t[l] for t in stash), **common)
    assert torch.equal(st_verify, state0)  # live slab untouched

    for n in range(1, steps + 1):
        st = state0.clone()
        meta = torch.tensor([list(range(bs)), slots, [n] * bs], dtype=torch.int32, device=dev)
        gdn_state_replay(st, stash, meta)
        assert torch.equal(st[:, slots], per_step[n - 1][:, slots]), f"n={n}"
        assert torch.equal(st[:, bs], state0[:, bs])  # the slot no row commits to


@cuda
@pytest.mark.parametrize("count", [0, 1, 3, 5])
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16])
def test_private_verify_commit_matches_sequential_decode(count, graph_mode, ssm_dtype):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kernel.causal_conv1d import causal_conv1d_decode
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update as fused
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.layers.linear_attn import GatedDeltaNet
    from freetoken.models.config import LinearGatedDeltaGroupConfig
    from freetoken.models.qwen3_5_moe.speculative_state import QwenGDNVerifyState
    from freetoken.utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    torch.manual_seed(83)
    device = torch.device("cuda")
    width, heads, value_heads, dim = 5, 2, 4, 128
    channels = (2 * heads + value_heads) * dim
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=heads, num_value_heads=value_heads,
        key_head_dim=dim, value_head_dim=dim, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group, 2, torch.bfloat16, device, tp_size=1)
    pool.recurrent_states = torch.randn_like(pool.recurrent_states).to(ssm_dtype) * 0.1
    pool.conv_states.normal_(std=0.1)
    with torch.device(device), torch_dtype(torch.bfloat16):
        layer = GatedDeltaNet(64, heads, value_heads, dim, dim, 4, 1e-6, 0)
    layer.conv1d.weight.normal_(std=0.1)
    layer.A_log.zero_()
    layer.dt_bias.zero_()
    raw = torch.randn(width, channels, device=device, dtype=torch.bfloat16)
    a, b = [torch.randn(width, value_heads, device=device, dtype=torch.bfloat16) for _ in range(2)]
    before_ssm, before_conv = pool.recurrent_states.clone(), pool.conv_states.clone()
    verify = QwenGDNVerifyState(pool, [layer], width, 1)
    verify.verify_layer(layer, raw, a, b)
    assert torch.equal(pool.recurrent_states, before_ssm)
    assert torch.equal(pool.conv_states, before_conv)

    expected_ssm, expected_conv = before_ssm[0].clone(), before_conv[0].clone()
    indices = torch.tensor([1], dtype=torch.int32, device=device)
    cu = torch.tensor([0, 1], dtype=torch.int32, device=device)
    for row in range(count):
        mixed = causal_conv1d_decode(raw[row:row + 1], expected_conv, layer._conv_weight(), indices)
        q, k, v = torch.split(mixed, [heads * dim, heads * dim, value_heads * dim], dim=-1)
        fused(A_log=layer.A_log, dt_bias=layer.dt_bias, a=a[row:row + 1], b=b[row:row + 1],
              q=q.reshape(1, 1, heads, dim), k=k.reshape(1, 1, heads, dim),
              v=v.reshape(1, 1, value_heads, dim), initial_state_source=expected_ssm,
              initial_state_indices=indices, cu_seqlens=cu, scale=dim ** -0.5,
              softplus_beta=1.0, softplus_threshold=20.0, use_qk_l2norm_in_kernel=True)
    for stash in verify.replay_inputs:
        stash[:, count:].fill_(float("nan"))
    accepted = torch.tensor(count, dtype=torch.int32, device=device)
    if graph_mode:
        verify.commit(torch.zeros_like(accepted))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            verify.commit(accepted)
        graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
    else:
        verify.commit(accepted)
    assert torch.equal(pool.recurrent_states[0], expected_ssm)
    assert torch.equal(pool.conv_states[0], expected_conv)
