from collections import deque
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.qwen_cpu_target import VerifyJob
from freetoken.engine.qwen_cpu_worker import CpuVerifierWorker


class Pipe:
    def __init__(self):
        self.messages, self.sent = deque(), []

    def poll(self):
        return bool(self.messages)

    def recv(self):
        return self.messages.popleft()

    def send(self, message):
        self.sent.append(message)


def worker():
    w = CpuVerifierWorker.__new__(CpuVerifierWorker)
    w.pipe = Pipe()
    w.closed, w.ready, w.initialized = False, True, True
    w.epoch = w.latest_epoch = 3
    w.inflight = None
    w.owners, w.readers, w.generations = [None, None], [None, None], [0, 0]
    w.process = SimpleNamespace(is_alive=lambda: True, exitcode=None)
    w.slots = [{"keys": torch.zeros(1, 16, 1, 2), "values": torch.zeros(1, 16, 1, 2),
        "recurrent": torch.zeros(1, 2, 2, 2), "conv": torch.zeros(1, 8, 3),
        "hidden": torch.zeros(8, 4), "predictions": torch.zeros(8, dtype=torch.int64)} for _ in range(2)]
    return w


def publish(w, slot=0, block=0, epoch=3):
    w.pipe.messages.append(dict(kind="result", slot=slot, generation=w.generations[slot],
        request_epoch=epoch, block_id=block, start=4, accepted=1, input_count=2,
        tokens=(5, 6), terminal=False, width=2))


def test_slots_wait_for_last_reader_and_do_not_reuse_published_result():
    w = worker()
    job = VerifyJob(3, 0, 4, 2, (5,), 10)
    w.submit(job)
    with pytest.raises(RuntimeError, match="busy"):
        w.submit(job)
    publish(w)
    result, _ = w.poll()
    done = SimpleNamespace(ready=False)
    event = SimpleNamespace(query=lambda: done.ready)
    w.release(result, event)
    w.submit(VerifyJob(3, 1, 6, 6, (), 10))
    assert w.inflight[0] == 1
    assert w.generations == [1, 1]
    with pytest.raises(RuntimeError, match="still have readers"):
        w._free_slot()
    done.ready = True
    assert w._free_slot() == 0
    assert w.owners[0] is None


def test_cancel_discards_late_result_and_waits_for_cancel_ack():
    w = worker()
    w.submit(VerifyJob(3, 0, 4, 2, (5,), 10))
    w.cancel(3)
    publish(w)
    assert w.poll() is None
    assert w.inflight == "cancel" and w.owners[0] is None
    w.pipe.messages.append(dict(kind="initialized", epoch=3))
    assert w.poll() is None and w.inflight == "cancel"
    w.pipe.messages.append(dict(kind="cancelled", epoch=3))
    assert w.poll()["kind"] == "cancelled"
    assert w.inflight is None and not w.initialized
    with pytest.raises(RuntimeError):
        w.submit(VerifyJob(3, 1, 4, 2, (), 10))


def test_generation_mismatch_and_worker_failure_fail_closed():
    w = worker()
    w.submit(VerifyJob(3, 0, 4, 2, (), 10))
    publish(w)
    w.generations[0] += 1
    w.owners[0] = (3, 0, 2)
    with pytest.raises(RuntimeError, match="generation mismatch"):
        w.poll()
    w.pipe.messages.append(dict(kind="error", traceback="test worker exception"))
    with pytest.raises(RuntimeError, match="test worker exception"):
        w.poll()
    w.process = SimpleNamespace(is_alive=lambda: False, exitcode=17)
    with pytest.raises(RuntimeError, match="code 17"):
        w.poll()


def test_slow_worker_poll_does_not_publish_provisional_output():
    w = worker()
    w.submit(VerifyJob(3, 0, 4, 2, (), 10))
    for _ in range(5):
        assert w.poll() is None
    assert w.inflight is not None
