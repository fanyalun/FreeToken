"""CPU-only BF16 experts; no pinned allocation, stream or CUDA API calls."""

from __future__ import annotations

import torch


def pack_target_bank(weight, *, gate_up):
    experts, outputs, inputs = weight.shape
    if gate_up:
        inter = outputs // 2
        value = weight.view(experts, 2, inter, inputs)
        if inter % 32:
            value = torch.nn.functional.pad(value, (0, 0, 0, (-inter) % 32))
        return value.view(experts, 2, -1, 32, inputs).transpose(-1, -2).contiguous()
    value = torch.nn.functional.pad(weight, (0, 0, 0, (-outputs) % 32)) if outputs % 32 else weight
    return value.view(experts, -1, 32, inputs).transpose(-1, -2).contiguous()


class CpuTargetExperts:
    def __init__(self, gate_up, down, top_k, core_ids, max_rows, *, grouped=True, backend="optimized", packed=False):
        from freetoken.kernel import _cpu_moe

        if backend not in ("legacy", "optimized") or (backend == "optimized" and not grouped):
            raise ValueError("optimized CPU experts require grouped execution; ungrouped is legacy-only")
        if packed and (backend != "optimized" or not grouped):
            raise ValueError("packed experts require optimized grouped execution")
        self.backend = backend
        self.packed = packed
        self.banks = (gate_up, down)
        self.tables = [torch.tensor([t.data_ptr() for t in bank], dtype=torch.int64)
                       for bank in self.banks]
        if packed:
            experts, _, _, hidden, _ = gate_up[0].shape
            inter = down[0].shape[2]
        else:
            experts, twice_inter, hidden = gate_up[0].shape
            inter = twice_inter // 2
        if not core_ids or any(t.device.type != "cpu" or t.dtype != torch.bfloat16
                               or not t.is_contiguous() for bank in self.banks for t in bank):
            raise ValueError("CPU Target needs contiguous CPU BF16 banks and explicit affinity")
        gate_shape = (experts, 2, (inter + 31) // 32, hidden, 32) if packed else (experts, 2 * inter, hidden)
        down_shape = (experts, (hidden + 31) // 32, inter, 32) if packed else (experts, hidden, inter)
        if len(gate_up) != len(down) or any(t.shape != gate_shape for t in gate_up
                ) or any(t.shape != down_shape for t in down):
            raise ValueError("inconsistent CPU expert geometry")
        self.hidden, self.top_k, self.max_rows = hidden, top_k, max_rows
        self.ext = _cpu_moe.CpuMoeExecutor(
            len(core_ids), len(gate_up), experts, top_k, hidden, inter, max_rows,
            0, 0, 0, self.tables[0].data_ptr(), self.tables[1].data_ptr(),
            0, 0, 0, 0, 0, 0, 1.0, float("inf"), core_ids)
        if backend == "optimized":
            if not hasattr(self.ext, "create_grouped_compat_task"):
                raise RuntimeError("rebuild _cpu_moe for compatible multi-row CPU experts")
            self.ext.prepare_grouped(max_rows)
        elif grouped:
            if not hasattr(self.ext, "set_expert_grouped"):
                raise RuntimeError("rebuild _cpu_moe to enable CPU Target expert grouping")
            self.ext.set_expert_grouped(True)
        self.io = {}
        self.tasks = {}
        if backend == "optimized":
            buffers = (torch.empty(max_rows, hidden, dtype=torch.bfloat16),
                       torch.empty(max_rows, top_k, dtype=torch.int32),
                       torch.empty(max_rows, top_k),
                       torch.empty(max_rows, hidden, dtype=torch.bfloat16))
            for rows in range(1, max_rows + 1):
                self.io[rows] = tuple(t[:rows] for t in buffers)
                xx, ii, ww, yy = self.io[rows]
                for layer in range(len(gate_up)):
                    create = self.ext.create_grouped_packed_task if packed else self.ext.create_grouped_compat_task
                    self.tasks[layer, rows] = create(
                        layer, rows, xx.data_ptr(), ii.data_ptr(), ww.data_ptr(), yy.data_ptr())

    def __call__(self, layer, x, weights, ids):
        rows = x.shape[0]
        if self.ext is None:
            raise RuntimeError("CPU experts are closed")
        if not 1 <= rows <= self.max_rows or x.shape != (rows, self.hidden):
            raise ValueError("CPU expert input exceeds reserved geometry")
        if weights.shape != (rows, self.top_k) or ids.shape != weights.shape:
            raise ValueError("CPU routing shape mismatch")
        if any(t.device.type != "cpu" for t in (x, weights, ids)):
            raise ValueError("CPU experts cannot consume device tensors")
        if rows not in self.io:
            self.io[rows] = (torch.empty_like(x), torch.empty_like(ids, dtype=torch.int32),
                             torch.empty_like(weights, dtype=torch.float32), torch.empty_like(x))
        xx, ii, ww, yy = self.io[rows]
        xx.copy_(x)
        ii.copy_(ids)
        ww.copy_(weights)
        key = (layer, rows)
        if key not in self.tasks:
            self.tasks[key] = self.ext.create_task(layer, rows, xx.data_ptr(), ii.data_ptr(),
                                                  ww.data_ptr(), yy.data_ptr())
        self.ext.run_task(self.tasks[key])
        return yy if self.backend == "optimized" else yy.clone()

    def close(self):
        self.ext = None
        self.tasks.clear()
        self.io.clear()
