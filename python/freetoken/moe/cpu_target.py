"""CPU-only BF16 experts; no pinned allocation, stream or CUDA API calls."""

from __future__ import annotations

import torch


class CpuTargetExperts:
    def __init__(self, gate_up, down, top_k, core_ids, max_rows, *, grouped=True):
        from freetoken.kernel import _cpu_moe

        self.banks = (gate_up, down)
        self.tables = [torch.tensor([t.data_ptr() for t in bank], dtype=torch.int64)
                       for bank in self.banks]
        experts, twice_inter, hidden = gate_up[0].shape
        inter = twice_inter // 2
        if not core_ids or any(t.device.type != "cpu" or t.dtype != torch.bfloat16
                               or not t.is_contiguous() for bank in self.banks for t in bank):
            raise ValueError("CPU Target needs contiguous CPU BF16 banks and explicit affinity")
        if len(gate_up) != len(down) or any(t.shape != (experts, twice_inter, hidden) for t in gate_up
                ) or any(t.shape != (experts, hidden, inter) for t in down):
            raise ValueError("inconsistent CPU expert geometry")
        self.hidden, self.top_k, self.max_rows = hidden, top_k, max_rows
        self.ext = _cpu_moe.CpuMoeExecutor(
            len(core_ids), len(gate_up), experts, top_k, hidden, inter, max_rows,
            0, 0, 0, self.tables[0].data_ptr(), self.tables[1].data_ptr(),
            0, 0, 0, 0, 0, 0, 1.0, float("inf"), core_ids)
        if grouped:
            if not hasattr(self.ext, "set_expert_grouped"):
                raise RuntimeError("rebuild _cpu_moe to enable CPU Target expert grouping")
            self.ext.set_expert_grouped(True)
        self.io = {}
        self.tasks = {}

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
        return yy.clone()

    def close(self):
        self.ext = None
        self.tasks.clear()
        self.io.clear()
