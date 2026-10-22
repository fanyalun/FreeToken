"""Spawn-only CPU verifier with two bounded shared-memory result slots."""

from __future__ import annotations

from dataclasses import asdict
from contextlib import nullcontext
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import torch

from .qwen_cpu_target import CpuVerifyResult, PrefixCheckpoint, QwenCpuTarget, VerifyJob, bitwise_equal


def _cpu_main(pipe, slots, model, core_ids, max_context, max_rows, grouped, backend,
              kernel_profile="v1", components=None, bind_openmp=False):
    target = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.sched_setaffinity(0, core_ids)
        torch.set_num_threads(len(core_ids))
        torch.set_num_interop_threads(1)
        if bind_openmp:
            os.sched_setaffinity(0, [core_ids[0]])
        initialization_start = time.perf_counter()
        target = QwenCpuTarget.from_checkpoint(model, core_ids=core_ids, max_context=max_context,
                                               max_rows=max_rows, grouped=grouped, backend=backend,
                                               kernel_profile=kernel_profile, components=components)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU worker created a CUDA context")
        import hashlib
        from freetoken.kernel import _cpu_moe
        with open(_cpu_moe.__file__, "rb") as binary:
            extension_sha256 = hashlib.sha256(binary.read()).hexdigest()
        pipe.send({"kind": "ready", "pid": os.getpid(), "cuda_initialized": False,
                   "initialization_seconds": time.perf_counter() - initialization_start,
                   **target.arithmetic_metadata(), "extension_sha256": extension_sha256,
                   "core_ids": core_ids, "main_thread_affinity": sorted(os.sched_getaffinity(0)),
                   "thread_affinity": {p.name: sorted(os.sched_getaffinity(int(p.name)))
                       for p in Path("/proc/self/task").iterdir()},
                   "expert_isa": target.experts.ext.isa_name()})
        while True:
            command = pipe.recv()
            kind = command["kind"]
            if kind == "close":
                break
            if kind == "cancel":
                target.state = None
                pipe.send({"kind": "cancelled", "epoch": command["epoch"]})
                continue
            if kind == "audit":
                report = audit_cpu_blocks(target, command["tokens"], slots[0]["audit_logits"])
                pipe.send(dict(kind="audit", **report))
                continue
            slot = slots[command["slot"]]
            if kind == "initialize":
                length = command["length"]
                checkpoint = PrefixCheckpoint(length, command["anchor"], slot["keys"][:, :length],
                    slot["values"][:, :length], slot["recurrent"], slot["conv"])
                target.initialize(checkpoint, command["epoch"])
                pipe.send({"kind": "initialized", "epoch": command["epoch"]})
                continue
            if kind != "verify":
                raise ValueError("unknown CPU command")
            started = time.perf_counter()
            annotate = torch.cuda.nvtx.range if os.getenv("FREETOKEN_CPU_VERIFY_NVTX") == "1" else lambda _: nullcontext()
            with annotate("CPU Target complete forward and accepted state commit"):
                result = target.verify(VerifyJob(**command["job"]))
            verified_at = time.perf_counter()
            n = result.input_count
            with annotate("CPU accepted state publication to shared slot"):
                for key in ("keys", "values"):
                    slot[key][:, :n].copy_(getattr(result.checkpoint, key))
                slot["recurrent"].copy_(result.checkpoint.recurrent)
                slot["conv"].copy_(result.checkpoint.conv)
                slot["hidden"][:n].copy_(result.hidden)
                slot["predictions"][:result.predictions.numel()].copy_(result.predictions)
            if torch.cuda.is_initialized():
                raise RuntimeError("CPU verification created a CUDA context")
            pipe.send({"kind": "result", "slot": command["slot"], "generation": command["generation"],
                "request_epoch": result.request_epoch, "block_id": result.block_id, "start": result.start,
                "accepted": result.accepted, "input_count": n, "tokens": result.tokens,
                "terminal": result.terminal, "width": result.predictions.numel(),
                "compute_seconds": verified_at - started,
                "publish_seconds": time.perf_counter() - started,
                "rss_kib": _rss_kib()})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        pipe.send({"kind": "error", "traceback": traceback.format_exc()})
    finally:
        if target is not None:
            target.close()
        pipe.close()


def _rss_kib():
    with open("/proc/self/status") as file:
        for line in file:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return None


def audit_cpu_blocks(target, tokens, first_logits=None):
    import hashlib
    if not 2 <= tokens <= target.max_rows:
        raise ValueError("audit length must fit one reserved block and contain at least two rows")
    checkpoint, epoch = target.snapshot(), target.epoch
    reference = []
    for block in range(tokens + 1):
        s = target.state
        reference.append(target.verify(VerifyJob(epoch, block, s.length, s.anchor, (), tokens + 1)))
    if first_logits is not None:
        first_logits.copy_(target.logits(reference[0].hidden)[0].float())
    candidates = tuple(r.tokens[0] for r in reference[:tokens - 1])
    checks = []
    for reject in (0, (tokens - 1) // 2, None):
        target.initialize(checkpoint, epoch)
        trial = list(candidates)
        if reject is not None:
            trial[reject] = (trial[reject] + 1) % target.config.get("vocab_size", 248320)
        result = target.verify(VerifyJob(epoch, 0, checkpoint.length, checkpoint.anchor, tuple(trial), tokens + 1))
        count = tokens if reject is None else reject + 1
        expected = reference[count - 1]
        equal = (result.tokens == tuple(r.tokens[0] for r in reference[:count])
            and result.accepted == count - 1 and result.input_count == count
            and result.checkpoint.length == checkpoint.length + count
            and result.checkpoint.anchor == expected.checkpoint.anchor
            and bitwise_equal(result.checkpoint.recurrent, expected.checkpoint.recurrent)
            and bitwise_equal(result.checkpoint.conv, expected.checkpoint.conv)
            and bitwise_equal(result.hidden, torch.cat([r.hidden for r in reference[:count]]))
            and bitwise_equal(result.checkpoint.keys, torch.cat([r.checkpoint.keys for r in reference[:count]], 1))
            and bitwise_equal(result.checkpoint.values, torch.cat([r.checkpoint.values for r in reference[:count]], 1)))
        s = target.state
        following = target.verify(VerifyJob(epoch, 1, s.length, s.anchor, (), tokens + 1))
        next_ref = reference[count]
        next_equal = (following.tokens == next_ref.tokens and bitwise_equal(following.hidden, next_ref.hidden)
            and all(bitwise_equal(getattr(following.checkpoint, field), getattr(next_ref.checkpoint, field))
                    for field in ("keys", "values", "recurrent", "conv")))
        checks.append(dict(reject_at=reject, input_count=result.input_count,
                           exact_state_and_tokens=equal and next_equal, next_step_exact=next_equal))
    target.initialize(checkpoint, epoch)
    if not all(row["exact_state_and_tokens"] for row in checks):
        raise RuntimeError(f"CPU serial/block state mismatch: {checks}")
    hashes = []
    for result in reference:
        tensors = dict(hidden=result.hidden, predictions=result.predictions,
                       **{field: getattr(result.checkpoint, field)
                          for field in ("keys", "values", "recurrent", "conv")})
        if any(not bool(torch.isfinite(t).all()) for t in tensors.values() if t.is_floating_point()):
            raise RuntimeError("CPU serial audit produced a nonfinite valid hidden/KV/state tensor")
        hashes.append({key: hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                       for key, t in tensors.items()})
    return dict(tokens=[r.tokens[0] for r in reference[:tokens]], checks=checks,
                serial_tensor_sha256=hashes, valid_state_finite=True, cuda_initialized=torch.cuda.is_initialized())


class CpuVerifierWorker:
    def __init__(self, model, config, core_ids, *, max_context=4096, max_rows=33, grouped=True, backend="optimized",
                 kernel_profile="v1", components=None, bind_openmp=False):
        if backend not in ("legacy", "optimized") or (backend == "optimized" and not grouped):
            raise ValueError("optimized backend requires grouped experts; ungrouped is legacy-only")
        if not core_ids or max_context < 1 or max_rows < 1:
            raise ValueError("positive CPU worker capacity and affinity required")
        c = config
        linear = c["layer_types"].count("linear_attention")
        attention = c["layer_types"].count("full_attention")
        vh, kd, vd = c["linear_num_value_heads"], c["linear_key_head_dim"], c["linear_value_head_dim"]
        channels = 2 * c["linear_num_key_heads"] * kd + vh * vd
        shapes = {
            "keys": ((attention, max_context, c["num_key_value_heads"], c["head_dim"]), torch.bfloat16),
            "values": ((attention, max_context, c["num_key_value_heads"], c["head_dim"]), torch.bfloat16),
            "recurrent": ((linear, vh, vd, kd), torch.float32),
            "conv": ((linear, channels, c["linear_conv_kernel_dim"] - 1), torch.bfloat16),
            "hidden": ((max_rows, c["hidden_size"]), torch.bfloat16),
            "predictions": ((max_rows,), torch.int64),
            "audit_logits": ((c["vocab_size"],), torch.float32),
        }
        self.slots = [{key: torch.empty(shape, dtype=dtype).share_memory_()
                       for key, (shape, dtype) in shapes.items()} for _ in range(2)]
        self.generations = [0, 0]
        self.owners = [None, None]
        self.readers = [None, None]
        self.inflight = None
        self.epoch = None
        self.closed = False
        self.initialized = False
        self.latest_epoch = -1
        context = mp.get_context("spawn")
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=_cpu_main,
            args=(child, self.slots, str(model), list(core_ids), max_context, max_rows, grouped, backend,
                  kernel_profile, components, bind_openmp), daemon=True)
        saved_environment = {key: os.environ.get(key) for key in ("GOMP_CPU_AFFINITY", "OMP_PROC_BIND")}
        saved_affinity = os.sched_getaffinity(0) if bind_openmp else None
        try:
            if bind_openmp:
                os.environ["GOMP_CPU_AFFINITY"] = " ".join(map(str, core_ids))
                os.environ["OMP_PROC_BIND"] = "TRUE"
                # Spawn imports thread libraries before _cpu_main can set worker affinity.
                os.sched_setaffinity(0, core_ids)
            self.process.start()
        finally:
            for key, value in saved_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if saved_affinity is not None:
                os.sched_setaffinity(0, saved_affinity)
        child.close()
        self.ready = False
        self.metadata = {}

    def _receive(self):
        if self.closed:
            raise RuntimeError("CPU verifier is closed")
        if not self.pipe.poll():
            if not self.process.is_alive():
                raise RuntimeError(f"CPU verifier exited with code {self.process.exitcode}")
            return None
        try:
            message = self.pipe.recv()
        except EOFError as exc:
            raise RuntimeError("CPU verifier pipe closed") from exc
        if message["kind"] == "error":
            raise RuntimeError("CPU verifier failed:\n" + message["traceback"])
        return message

    def wait_ready(self, timeout=300):
        deadline = time.monotonic() + timeout
        while not self.ready:
            message = self._receive()
            if message is not None:
                if message["kind"] != "ready":
                    raise RuntimeError("unexpected worker startup response")
                self.metadata, self.ready = message, True
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("CPU verifier startup deadline exceeded")
            time.sleep(0.01)

    def _free_slot(self):
        self._reap_readers()
        for slot in range(2):
            if self.owners[slot] is None:
                return slot
        raise RuntimeError("both shared CPU slots still have readers")

    def _reap_readers(self):
        for slot in range(2):
            event = self.readers[slot]
            if event is not None and event.query():
                self.readers[slot], self.owners[slot] = None, None

    def initialize(self, checkpoint, request_epoch):
        self._reap_readers()
        if not self.ready or self.inflight is not None or any(owner is not None for owner in self.owners):
            raise RuntimeError("initialize requires a ready, drained CPU worker")
        if request_epoch <= self.latest_epoch:
            raise ValueError("request epochs must increase across reinitialization")
        self.latest_epoch, self.initialized = request_epoch, False
        self.epoch = request_epoch
        slot = self.slots[0]
        for key in ("keys", "values"):
            slot[key][:, :checkpoint.length].copy_(getattr(checkpoint, key))
        slot["recurrent"].copy_(checkpoint.recurrent)
        slot["conv"].copy_(checkpoint.conv)
        self.pipe.send(dict(kind="initialize", slot=0, length=checkpoint.length,
                            anchor=checkpoint.anchor, epoch=request_epoch))
        self.inflight = "initialize"

    def submit(self, job):
        if not self.ready or not self.initialized or self.inflight is not None or job.request_epoch != self.epoch:
            raise RuntimeError("CPU worker busy, uninitialized or stale epoch")
        slot = self._free_slot()
        self.generations[slot] += 1
        identity = (job.request_epoch, job.block_id, self.generations[slot])
        self.owners[slot] = identity
        self.inflight = (slot, identity)
        self.pipe.send(dict(kind="verify", slot=slot, generation=self.generations[slot], job=asdict(job)))

    def poll(self):
        message = self._receive()
        if message is None:
            return None
        kind = message["kind"]
        if kind == "initialized":
            if message["epoch"] != self.epoch:
                return None
            if self.inflight != "initialize":
                raise RuntimeError("unexpected CPU initialization acknowledgement")
            self.inflight, self.initialized = None, True
            return message
        if kind == "cancelled":
            if message["epoch"] != self.latest_epoch or self.inflight != "cancel":
                return None
            self.inflight, self.initialized = None, False
            return message
        if kind == "audit":
            self.inflight = None
            return message
        if kind != "result":
            raise RuntimeError("unexpected CPU worker response")
        slot_id = message["slot"]
        identity = (message["request_epoch"], message["block_id"], message["generation"])
        if self.owners[slot_id] != identity:
            raise RuntimeError("CPU result slot generation mismatch")
        if message["request_epoch"] != self.epoch:
            self.owners[slot_id] = None
            return None
        self.inflight = None
        n, slot = message["input_count"], self.slots[slot_id]
        checkpoint = PrefixCheckpoint(message["start"] + n, message["tokens"][-1],
            slot["keys"][:, :n], slot["values"][:, :n], slot["recurrent"], slot["conv"])
        result = CpuVerifyResult(message["request_epoch"], message["block_id"], message["start"],
            message["accepted"], n, message["tokens"], message["terminal"], checkpoint,
            slot["hidden"][:n], slot["predictions"][:message["width"]])
        return result, message

    def release(self, result, last_reader=None):
        slot = next((i for i, owner in enumerate(self.owners)
                     if owner is not None and owner[:2] == (result.request_epoch, result.block_id)), None)
        if slot is None or self.readers[slot] is not None:
            raise RuntimeError("stale or already released CPU result")
        if last_reader is None:
            self.owners[slot] = None
        else:
            self.readers[slot] = last_reader

    def cancel(self, request_epoch):
        if request_epoch != self.epoch:
            return
        self.epoch = None
        self.initialized = False
        self.pipe.send(dict(kind="cancel", epoch=request_epoch))
        self.inflight = "cancel"

    def audit(self, tokens):
        if not self.initialized or self.inflight is not None:
            raise RuntimeError("audit requires an initialized idle CPU worker")
        self.inflight = "audit"
        self.pipe.send(dict(kind="audit", tokens=tokens))

    def close(self):
        if self.closed:
            return
        self.closed = True
        for event in self.readers:
            if event is not None:
                event.synchronize()
        if self.process.is_alive():
            try:
                self.pipe.send(dict(kind="close"))
            except (BrokenPipeError, EOFError):
                pass
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        self.pipe.close()
