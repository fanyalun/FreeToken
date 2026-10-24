from __future__ import annotations

import mmap
import os

import torch


class SharedBankFd:
    def __init__(self, fd):
        self.fd = fd.detach() if hasattr(fd, "detach") else fd

    def __reduce__(self):
        from multiprocessing.reduction import DupFd
        return type(self), (DupFd(self.fd),)


def export_cpu_banks(bank_sources, layer_formats, num_experts):
    from .host_banks import _SHARED_BANKS

    buffers, views, seen = [], {}, {}
    for role, tensors in bank_sources.items():
        views[role] = []
        for tensor in tensors:
            size = tensor.element_size() * (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride())))
            bank = next((b for b in _SHARED_BANKS
                         if b.addr <= tensor.data_ptr() and tensor.data_ptr() + size <= b.addr + b.nbytes), None)
            if bank is None:
                raise ValueError(f"expert bank {role} has no shared backing; enable cpu_async before model load")
            if bank.addr not in seen:
                seen[bank.addr] = len(buffers)
                buffers.append(dict(fd=SharedBankFd(bank._share_fd), size=len(bank._buf)))
            views[role].append(dict(buffer=seen[bank.addr], offset=tensor.data_ptr() - bank.addr,
                                    shape=tuple(tensor.shape), stride=tuple(tensor.stride()), dtype=tensor.dtype))
    result = dict(shared_moe_banks_v1=True, buffers=buffers, views=views,
                  layer_formats=layer_formats, num_experts=num_experts)
    validate_shared_banks(result)
    return result


def validate_shared_banks(value):
    if value is None:
        return
    if not value.get("shared_moe_banks_v1"):
        if not all(t.device.type == "cpu" and t.is_shared()
                   for tensors in value["bank_sources"].values() for t in tensors):
            raise ValueError("CPU expert banks must use explicit shared backing")
        return
    if value["num_experts"] < 1 or not value["layer_formats"] or not value["views"]:
        raise ValueError("invalid shared expert bank geometry")
    for views in value["views"].values():
        if len(views) != len(value["layer_formats"]):
            raise ValueError("shared expert bank layer coverage mismatch")
        for view in views:
            buffer = value["buffers"][view["buffer"]]
            shape, stride, dtype = view["shape"], view["stride"], view["dtype"]
            if len(shape) != len(stride) or min(shape) < 1 or min(stride) < 0:
                raise ValueError("invalid shared bank shape or stride")
            extent = dtype.itemsize * (1 + sum((n - 1) * s for n, s in zip(shape, stride)))
            if view["offset"] < 0 or view["offset"] % dtype.itemsize or view["offset"] + extent > buffer["size"]:
                raise ValueError("shared bank view exceeds its mapping")


def materialize_cpu_banks(value):
    if value is None or not value.get("shared_moe_banks_v1"):
        return value
    validate_shared_banks(value)
    mappings = []
    try:
        for buffer in value["buffers"]:
            fd = buffer["fd"].fd
            try:
                mappings.append(mmap.mmap(fd, buffer["size"], flags=mmap.MAP_SHARED, prot=mmap.PROT_READ))
            finally:
                os.close(fd)
        sources = {}
        for role, views in value["views"].items():
            tensors = []
            for view in views:
                buffer = mappings[view["buffer"]]
                count = (len(buffer) - view["offset"]) // view["dtype"].itemsize
                base = torch.frombuffer(buffer, dtype=view["dtype"], count=count, offset=view["offset"])
                tensors.append(base.as_strided(view["shape"], view["stride"]))
            sources[role] = tensors
        return dict(bank_sources=sources, layer_formats=value["layer_formats"],
                    num_experts=value["num_experts"], _shared_mappings=mappings)
    except BaseException:
        for mapping in mappings:
            mapping.close()
        raise
