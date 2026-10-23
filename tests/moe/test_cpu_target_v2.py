import pytest
import torch

from freetoken.kernel import _cpu_moe


def workspace(rows, qh, d, capacity, threads=46):
    return (torch.empty(rows, qh, d), torch.empty(threads, 128 * d),
            torch.empty((capacity + 255) // 256, rows, qh, d + 2))


def run(q, k, v, buffers, start):
    out = torch.empty_like(q)
    _cpu_moe.target_attention_v2(q, k, v, *buffers, out, start)
    return out


@pytest.mark.parametrize("context", [1, 63, 64, 65, 255, 256, 257, 512, 2048, 4096, 4191])
def test_v2_width_threads_and_poisoned_future_are_bitwise_exact(context):
    torch.manual_seed(context)
    previous = torch.get_num_threads()
    q = torch.randn(33, 6, 17).bfloat16()
    k = torch.randn(context + 33, 2, 17).bfloat16()
    v = torch.randn_like(k)
    buffers = workspace(33, 6, 17, k.shape[0])
    try:
        torch.set_num_threads(1)
        reference = torch.cat([run(q[r:r + 1], k, v, buffers, context + r) for r in range(33)])
        for threads in (1, 2, 4, 23, 46):
            torch.set_num_threads(threads)
            for rows in range(1, 34):
                actual = run(q[:rows], k, v, buffers, context)
                assert torch.equal(actual.view(torch.int16), reference[:rows].view(torch.int16))
        torch.set_num_threads(4)
        for rows in (1, 4, 17, 33):
            kk, vv = k.clone(), v.clone()
            kk[context + rows:] = float("nan")
            vv[context + rows:] = float("nan")
            actual = run(q[:rows], kk, vv, buffers, context)
            assert torch.equal(actual.view(torch.int16), reference[:rows].view(torch.int16))
            speculative = run(q, kk, vv, buffers, context)
            assert torch.equal(speculative[:rows].view(torch.int16), reference[:rows].view(torch.int16))
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("scale", [0, 1, 100])
def test_v2_real_head_geometry_fp64_reference(scale):
    torch.manual_seed(42)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        q = (torch.randn(5, 16, 256) * scale).bfloat16()
        k = torch.randn(270, 2, 256).bfloat16()
        v = torch.randn_like(k)
        out = run(q, k, v, workspace(5, 16, 256, 270), 255)
        for r in range(5):
            kk = k[:256 + r].double().repeat_interleave(8, 1)
            vv = v[:256 + r].double().repeat_interleave(8, 1)
            p = torch.einsum("hd,thd->ht", q[r].double(), kk).div(16).softmax(-1)
            expected = torch.einsum("ht,thd->hd", p, vv)
            torch.testing.assert_close(out[r].double(), expected, atol=0.008, rtol=0.006)
    finally:
        torch.set_num_threads(previous)


def test_v2_workspace_capacity_checked():
    q = torch.ones(1, 2, 8, dtype=torch.bfloat16)
    buffers = workspace(1, 2, 8, 1, threads=1)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with pytest.raises(RuntimeError, match="workspace capacity"):
            run(q, q, q, buffers, 0)
    finally:
        torch.set_num_threads(previous)


def test_attention_instrumentation_is_numerically_identical():
    torch.manual_seed(31)
    q = torch.randn(7, 4, 17).bfloat16()
    k = torch.randn(270, 2, 17).bfloat16()
    v = torch.randn_like(k)
    buffers = workspace(7, 4, 17, 270, max(46, torch.get_num_threads()))
    _cpu_moe.target_attention_workspace_v2(buffers[1], buffers[2], 2)
    expected = run(q, k, v, buffers, 255)
    actual, timing = torch.empty_like(q), torch.empty(18, dtype=torch.float64)
    _cpu_moe.target_attention_v2_profiled(q, k, v, *buffers, actual, 255, timing)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert timing.ge(0).all() and timing.sum() > 0


@pytest.mark.parametrize("rows", range(1, 34))
@pytest.mark.parametrize("k", [7, 16, 35, 129])
def test_packed_projection_width_threads_and_bf16_argmax(rows, k):
    from freetoken.engine.qwen_cpu_target import pack_projection
    torch.manual_seed(17)
    x = torch.randn(rows, k).bfloat16()
    w = pack_projection(torch.randn(67, k))
    tmp, maxima, indices = torch.empty(rows * k), torch.empty(rows * 3), torch.empty(rows * 3, dtype=torch.int64)
    def call(xx, argmax=False):
        return _cpu_moe.target_linear_v2(xx, w.data, w.outputs, tmp, maxima, indices, argmax)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        expected = torch.cat([call(row[None]) for row in x])
        for threads in (1, 2, 4):
            torch.set_num_threads(threads)
            assert torch.equal(call(x).view(torch.int16), expected.view(torch.int16))
            assert torch.equal(call(x, True), expected.argmax(-1))
        w.data.zero_()
        assert call(x, True).eq(0).all()
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("kd", [7, 17, 128])
def test_gdn_v2_replay_order_and_fp64_reference(kd):
    torch.manual_seed(17)
    state = torch.randn(2, 5, kd) * 0.1
    inputs = torch.randn(33, 2, kd + 7) * 0.1
    inputs[:, :, -2] = 0.9
    inputs[:, :, -1] = 0.2
    serial = state.clone()
    for item in inputs:
        _cpu_moe.target_replay_v2(serial, item[None], 1)
    for count in (0, 1, 7, 33):
        block = state.clone()
        _cpu_moe.target_replay_v2(block, inputs, count)
        _cpu_moe.target_replay_v2(block, inputs[count:].contiguous(), 33 - count)
        assert torch.equal(block.view(torch.int32), serial.view(torch.int32))
    reference = state.double()
    for item in inputs.double():
        key, value = item[:, :kd], item[:, kd:kd + 5]
        reference *= item[:, -2, None, None]
        delta = (value - (reference * key[:, None]).sum(-1)) * item[:, -1, None]
        reference += delta[:, :, None] * key[:, None]
    torch.testing.assert_close(serial.double(), reference, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("hidden,inter", [(17, 7), (64, 32)])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 7, 16, 33])
def test_packed_experts_route_heat_width_threads_and_invalid_routes(hidden, inter, rows):
    import os
    from freetoken.moe.cpu_target import CpuTargetExperts, pack_target_bank
    torch.manual_seed(19)
    gate = [pack_target_bank((torch.randn(12, 2 * inter, hidden) * 0.1).bfloat16(), gate_up=True)]
    down = [pack_target_bank((torch.randn(12, hidden, inter) * 0.1).bfloat16(), gate_up=False)]
    x = torch.randn(rows, hidden).bfloat16()
    ids = torch.stack([torch.randperm(12)[:8] for _ in range(rows)])
    weights = torch.rand(rows, 8)
    cores = sorted(os.sched_getaffinity(0))
    for hot in (False, True):
        if hot:
            ids[:] = torch.arange(8)
        ids[:, 3:5] = torch.tensor([-1, 12])
        weights[:, 2] = 0
        reference = None
        for threads in (1, 2, 4):
            ex = CpuTargetExperts(gate, down, 8, cores[:threads], rows, packed=True)
            try:
                serial = torch.cat([ex(0, x[r:r + 1], weights[r:r + 1], ids[r:r + 1]).clone()
                                    for r in range(rows)])
                block = ex(0, x, weights, ids).clone()
                assert torch.equal(serial.view(torch.int16), block.view(torch.int16))
                if reference is not None:
                    assert torch.equal(block.view(torch.int16), reference.view(torch.int16))
                reference = block
            finally:
                ex.close()


def test_packed_projection_and_experts_fp64_reference():
    import os
    from freetoken.engine.qwen_cpu_target import pack_projection
    from freetoken.moe.cpu_target import CpuTargetExperts, pack_target_bank
    torch.manual_seed(29)
    x = (torch.randn(3, 17) * 0.1).bfloat16()
    w = torch.randn(35, 17) * 0.2
    packed = pack_projection(w)
    actual = _cpu_moe.target_linear_v2(x, packed.data, 35, torch.empty(51),
        torch.empty(6), torch.empty(6, dtype=torch.int64), False)
    expected = x.double() @ w.double().T
    assert ((actual.double() - expected).abs() <= expected.abs() * 0.004 + 1e-6).all()
    gate = (torch.randn(8, 14, 17) * 0.1).bfloat16()
    down = (torch.randn(8, 17, 7) * 0.1).bfloat16()
    weights = torch.rand(3, 8)
    ids = torch.arange(8).repeat(3, 1)
    ex = CpuTargetExperts([pack_target_bank(gate, gate_up=True)],
        [pack_target_bank(down, gate_up=False)], 8, sorted(os.sched_getaffinity(0))[:2], 3, packed=True)
    try:
        actual = ex(0, x, weights, ids).double()
        reference = torch.zeros(3, 17, dtype=torch.float64)
        for r in range(3):
            for e in range(8):
                g, u = (gate[e].double() @ x[r].double()).chunk(2)
                activation = (torch.nn.functional.silu(g) * u).bfloat16().double()
                reference[r] += (down[e].double() @ activation) * weights[r, e].double()
        assert ((actual - reference).abs() <= reference.abs() * 0.005 + 1e-6).all()
    finally:
        ex.close()


def test_vector_norm_rope_and_gates_are_width_invariant_and_match_math():
    torch.manual_seed(37)
    x = torch.randn(33, 40).bfloat16()
    w = torch.randn(40) * 0.1
    angles = torch.randn(17)
    cosine, sine = angles.cos(), angles.sin()
    gate = torch.randn_like(x)
    operations = [lambda a, g: _cpu_moe.target_norm_v2(a, w, 1e-6, True),
                  lambda a, g: _cpu_moe.target_rope_v2(a, cosine, sine),
                  lambda a, g: _cpu_moe.target_gate_v2(a, g, False),
                  lambda a, g: _cpu_moe.target_gate_v2(a, g, True)]
    previous = torch.get_num_threads()
    try:
        for op in operations:
            torch.set_num_threads(1)
            expected = torch.cat([op(x[r:r + 1], gate[r:r + 1]) for r in range(33)])
            for threads in (1, 2, 4, 23):
                torch.set_num_threads(threads)
                for rows in (1, 2, 4, 7, 16, 33):
                    assert torch.equal(op(x[:rows], gate[:rows]).view(torch.int16), expected[:rows].view(torch.int16))
        reference = x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-6) * (w.double() + 1)
        actual = operations[0](x, gate).double()
        assert ((actual - reference).abs() <= reference.abs() * 0.004 + 1e-5).all()
        a, b = x[:, :17].double(), x[:, 17:34].double()
        reference = torch.cat((a * cosine.double() - b * sine.double(),
                               b * cosine.double() + a * sine.double(), x[:, 34:].double()), -1)
        actual = operations[1](x, gate).double()
        assert ((actual - reference).abs() <= reference.abs() * 0.004 + 1e-5).all()
        for silu in (False, True):
            reference = (x.double() * gate.double() * gate.double().sigmoid() if silu else
                         x.double() * gate.double().sigmoid().bfloat16().double())
            actual = _cpu_moe.target_gate_v2(x, gate, silu).double()
            assert ((actual - reference).abs() <= reference.abs() * 0.004 + 1e-5).all()
        broadcast = _cpu_moe.target_gate_v2(x, gate[:, :1].contiguous(), False)
        expanded = _cpu_moe.target_gate_v2(x, gate[:, :1].expand_as(x).contiguous(), False)
        assert torch.equal(broadcast.view(torch.int16), expanded.view(torch.int16))
    finally:
        torch.set_num_threads(previous)
