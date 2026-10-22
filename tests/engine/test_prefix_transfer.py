import multiprocessing as mp
import gc
import threading

import pytest
import torch

from freetoken.engine.prefix_transfer import PrefixAborted, PrefixReceiver, PrefixTicket, PrefixTransfer
from freetoken.engine.qwen_cpu_target import VerifyJob, bitwise_equal
from tests.engine.test_qwen_cpu_target import tiny_target, serial


@pytest.fixture(autouse=True)
def collect_shared_conditions():
    yield
    # Run semaphore finalizers outside the next test's resource-tracker registration.
    gc.collect()


def receiver_for(target, checkpoint, epoch=7):
    receiver = PrefixReceiver(target.config, 64)
    receiver.begin(epoch, checkpoint.length)
    for key, tensor in receiver.tensors.items():
        source = getattr(checkpoint, key)
        if key in ("keys", "values"):
            tensor = tensor[:, :checkpoint.length]
        tensor.copy_(source)
    return receiver


@pytest.mark.parametrize("backend", ["legacy", "optimized", "ggml_v2_all"])
@pytest.mark.parametrize("reject", [0, 2, None])
@pytest.mark.parametrize("eos", [False, True])
@pytest.mark.parametrize("wait_all", [False, True])
def test_streamed_matches_sync_bitwise(backend, reject, eos, wait_all):
    target, checkpoint = tiny_target(backend)
    try:
        candidates = [r.tokens[0] for r in serial(target, 6)[:4]]
        if reject is not None:
            candidates[reject] = (candidates[reject] + 1) % 64
        job = VerifyJob(7, 0, 2, 3, tuple(candidates), 100,
                        (candidates[0],) if eos else ())
        target.initialize(checkpoint, 7)
        expected = target.verify(job)
        expected_snapshot = target.snapshot()
        s = target.state
        following = target.verify(VerifyJob(7, 1, s.length, s.anchor, (), 100))
        receiver = receiver_for(target, checkpoint)
        target.initialize_streamed(PrefixTicket(7, 2, 3), receiver, wait_all=wait_all)
        for layer in range(2):
            receiver.publish(7, layer, 0, 2)
        actual = target.verify(job)
        snapshot = target.snapshot()
        assert actual.tokens == expected.tokens
        assert (actual.accepted, actual.input_count, actual.terminal) == (
            expected.accepted, expected.input_count, expected.terminal)
        for name in ("hidden", "predictions"):
            assert bitwise_equal(getattr(actual, name), getattr(expected, name))
        for name in ("keys", "values", "recurrent", "conv"):
            assert bitwise_equal(getattr(actual.checkpoint, name), getattr(expected.checkpoint, name))
            assert bitwise_equal(getattr(snapshot, name), getattr(expected_snapshot, name))
        s = target.state
        next_result = target.verify(VerifyJob(7, 1, s.length, s.anchor, (), 100))
        assert target.verify_timings["prefix_barrier_seconds"] == 0
        assert next_result.tokens == following.tokens
        assert bitwise_equal(next_result.hidden, following.hidden)
        for name in ("keys", "values", "recurrent", "conv"):
            assert bitwise_equal(getattr(next_result.checkpoint, name), getattr(following.checkpoint, name))
        assert len(target.prefix_timings) == 2
        assert receiver.consumers.value == 0
    finally:
        target.close()


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("wait_all", [False, True])
def test_prefix_policy_controls_forward_before_later_layer_arrives(cancel, wait_all):
    target, checkpoint = tiny_target()
    receiver = receiver_for(target, checkpoint)
    target.initialize_streamed(PrefixTicket(7, 2, 3), receiver, wait_all=wait_all)
    receiver.publish(7, 0, 0, 2)
    first_done = threading.Event()
    later_entered = threading.Event()
    errors = []
    original_moe, original_wait = target._moe, receiver.wait

    def moe(layer, x):
        result = original_moe(layer, x)
        if layer == 0:
            first_done.set()
        return result

    def wait(epoch, layer, timeout=60):
        if layer == 1:
            later_entered.set()
        original_wait(epoch, layer, timeout)

    target._moe, receiver.wait = moe, wait

    def verify():
        try:
            target.verify(VerifyJob(7, 0, 2, 3, (), 100))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=verify)
    thread.start()
    try:
        assert later_entered.wait(5)
        if wait_all:
            assert not first_done.is_set()
            assert not target.prefix_timings
        else:
            assert first_done.is_set()
        assert thread.is_alive()
        if cancel:
            receiver.abort()
        else:
            receiver.publish(7, 1, 0, 1)
            assert receiver.ready[1] == 1
            receiver.publish(7, 1, 1, 2)
        thread.join(5)
        assert not thread.is_alive()
        if cancel:
            assert len(errors) == 1 and isinstance(errors[0], PrefixAborted)
        else:
            assert not errors
            assert first_done.is_set()
            assert (target.verify_timings["prefix_barrier_seconds"] > 0) == wait_all
    finally:
        receiver.abort()
        thread.join(5)
        target.close()


@pytest.mark.parametrize("wait_all", [False, True])
def test_cancel_timeout_duplicates_and_reuse(wait_all):
    target, checkpoint = tiny_target()
    receiver = receiver_for(target, checkpoint)
    try:
        target.initialize_streamed(PrefixTicket(7, 2, 3), receiver, wait_all=wait_all)
        with pytest.raises(ValueError, match="consumers"):
            receiver.begin(8, 2)
        with pytest.raises(ValueError, match="gap"):
            receiver.publish(7, 1, 1, 2)
        receiver.publish(7, 1, 0, 1)
        with pytest.raises(ValueError, match="duplicate"):
            receiver.publish(7, 1, 0, 1)
        with pytest.raises(PrefixAborted, match="timed out"):
            receiver.wait(7, 0, timeout=0.01)
        with pytest.raises(PrefixAborted):
            target.verify(VerifyJob(7, 0, 2, 3, (), 100))
        target.release_prefix()
        receiver.begin(8, 2)
        assert not receiver.publish(7, 0, 0, 2)
        assert receiver.ready[0] == 0
        target.initialize_streamed(PrefixTicket(8, 2, 3), receiver, wait_all=wait_all)
        for layer in range(2):
            receiver.publish(8, layer, 0, 2)
        snapshot = target.snapshot()
        assert receiver.consumers.value == 0
        for name in ("keys", "values", "recurrent", "conv"):
            assert bitwise_equal(getattr(snapshot, name), getattr(checkpoint, name))
    finally:
        target.close()


def test_all_layer_barrier_has_one_timeout_budget(monkeypatch):
    from freetoken.engine import prefix_transfer

    target, checkpoint = tiny_target()
    receiver = receiver_for(target, checkpoint)
    now, deadlines = [0.0], []
    monkeypatch.setattr(prefix_transfer.time, "monotonic", lambda: now[0])

    def wait(epoch, layer, timeout):
        deadlines.append(timeout)
        now[0] += 4

    receiver.wait = wait
    try:
        receiver.wait_all(7, timeout=5)
        assert deadlines == [5, 1]
    finally:
        target.close()


def test_barrier_time_is_separate_from_forward_and_only_paid_once(monkeypatch):
    from freetoken.engine import qwen_cpu_target

    target, checkpoint = tiny_target()
    receiver = receiver_for(target, checkpoint)
    target.initialize_streamed(PrefixTicket(7, 2, 3), receiver, wait_all=True)
    for layer in range(2):
        receiver.publish(7, layer, 0, 2)
    clock = qwen_cpu_target.time.perf_counter
    offset, waits = [0.0], []
    original_wait = receiver.wait_all

    def wait_all(epoch):
        original_wait(epoch)
        offset[0] += 100
        waits.append(epoch)

    monkeypatch.setattr(qwen_cpu_target.time, "perf_counter", lambda: clock() + offset[0])
    receiver.wait_all = wait_all
    try:
        target.verify(VerifyJob(7, 0, 2, 3, (), 100))
        assert target.verify_timings["prefix_barrier_seconds"] >= 100
        assert 0 <= target.verify_timings["forward_seconds"] < 100
        state = target.state
        target.verify(VerifyJob(7, 1, state.length, state.anchor, (), 100))
        assert target.verify_timings["prefix_barrier_seconds"] == 0
        assert waits == [7]
    finally:
        target.close()


def _child_receive(receiver, pipe):
    try:
        receiver.wait(7, 0, timeout=10)
        pipe.send((receiver.tensors["recurrent"].sum().item(), torch.cuda.is_initialized()))
    finally:
        pipe.close()


def test_spawn_visibility_without_cuda_context():
    target, checkpoint = tiny_target()
    receiver = receiver_for(target, checkpoint)
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_child_receive, args=(receiver, child))
    process.start()
    child.close()
    try:
        receiver.publish(7, 0, 0, 2)
        assert parent.poll(20)
        total, cuda = parent.recv()
        assert total == checkpoint.recurrent.sum().item()
        assert not cuda
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()
        target.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_registered_shared_d2h_chunks_and_unregister():
    from types import SimpleNamespace
    from freetoken.kernel.pinned import _load_pinned_extension

    if not hasattr(_load_pinned_extension(), "host_unregister"):
        pytest.skip("requires rebuilt pinned extension")
    target, checkpoint = tiny_target()
    receiver = PrefixReceiver(target.config, 64, device=torch.device("cuda:0"))
    checkpoint.length = 5
    checkpoint.keys = checkpoint.keys.repeat(1, 3, 1, 1)[:, :5]
    checkpoint.values = checkpoint.values.repeat(1, 3, 1, 1)[:, :5]
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        keys = checkpoint.keys[0].to(device).unsqueeze(1)
        values = checkpoint.values[0].to(device).unsqueeze(1)
        rec = checkpoint.recurrent.to(device).unsqueeze(1).repeat(1, 2, 1, 1, 1)
        conv = checkpoint.conv.to(device).unsqueeze(1).repeat(1, 2, 1, 1)
    engine = SimpleNamespace(device=device, stream=stream,
        kv_cache=SimpleNamespace(k_cache=lambda _: keys, v_cache=lambda _: values),
        linear_state_pool=SimpleNamespace(recurrent_states=rec, conv_states=conv))
    transfer = PrefixTransfer(receiver, engine, chunk_size=2)
    try:
        transfer.begin(7, 5)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(500_000_000)
            for start, end in ((0, 2), (2, 4), (4, 5)):
                for layer in (0, 1):
                    transfer.enqueue_layer(layer, start, end, end == 5)
        assert not transfer.stream.query()
        transfer.finish()
        transfer.drain()
        assert list(receiver.ready) == [5, 5]
        assert len(transfer.records) == 4
        for name, tensor in receiver.tensors.items():
            if name in ("keys", "values"):
                tensor = tensor[:, :5]
            assert bitwise_equal(tensor, getattr(checkpoint, name))
        assert transfer.pinned_bytes > 0
        transfer.begin(8, 5)
        assert list(receiver.ready) == [0, 0]
        for layer in (0, 1):
            transfer.enqueue_layer(layer, 0, 5, True)
        transfer.finish()
        transfer.drain()
        assert list(receiver.ready) == [5, 5]
        assert not receiver.publish(7, 0, 0, 5)
    finally:
        transfer.close()
        target.close()
    assert transfer.closed and not transfer.registered


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_partial_registration_failure_unregisters_prior_ranges(monkeypatch):
    from types import SimpleNamespace
    from freetoken.kernel import pinned

    if not hasattr(pinned._load_pinned_extension(), "host_unregister"):
        pytest.skip("requires rebuilt pinned extension")
    target, _ = tiny_target()
    receiver = PrefixReceiver(target.config, 64, device=torch.device("cuda:0"))
    register, unregister = pinned.host_register, pinned.host_unregister
    registered, released = [], []

    def fail_second(addr, size):
        if registered:
            raise RuntimeError("injected registration failure")
        register(addr, size)
        registered.append(addr)

    def release(addr):
        unregister(addr)
        released.append(addr)

    monkeypatch.setattr(pinned, "host_register", fail_second)
    monkeypatch.setattr(pinned, "host_unregister", release)
    try:
        with pytest.raises(RuntimeError, match="injected registration failure"):
            PrefixTransfer(receiver, SimpleNamespace(device=torch.device("cuda:0")))
        assert released == registered and len(released) == 1
        tensor = receiver.tensors["keys"]
        register(tensor.data_ptr(), tensor.numel() * tensor.element_size())
        unregister(tensor.data_ptr())
    finally:
        target.close()
