"""Unweighted grouped expert execution for TableMoE Prefill."""

from __future__ import annotations

import torch
import triton

from freetoken.kernel.triton.moe_align import moe_align_block_size
from freetoken.kernel.triton.tablemoe_prefill import compact_group_blocks, grouped_expert_gemm, grouped_silu
from freetoken.moe.fused import try_get_optimal_moe_config


class GroupedPrefillExperts:
    def __init__(self, x, gate_up, down, route_slots, route_groups, workspace=None,
                 logical_ids=None, num_experts=None):
        if x.dtype != torch.float16 or gate_up.dtype != x.dtype or down.dtype != x.dtype:
            raise ValueError("grouped TableMoE Prefill requires FP16 inputs and experts")
        if route_slots.dtype != torch.int32 or route_groups.dtype != torch.int32:
            raise ValueError("grouped TableMoE Prefill requires int32 slots and groups")
        if route_slots.shape != route_groups.shape or route_slots.shape[0] != len(x):
            raise ValueError("TableMoE Prefill route shapes do not match inputs")
        if not all(t.is_contiguous() for t in (x, gate_up, down, route_slots, route_groups)):
            raise ValueError("grouped TableMoE Prefill tensors must be contiguous")
        if not x.is_cuda or any(t.device != x.device for t in (gate_up, down, route_slots, route_groups)):
            raise ValueError("grouped TableMoE Prefill tensors must share one CUDA device")
        self.x, self.gate_up, self.down = x, gate_up, down
        self.route_slots, self.route_groups = route_slots, route_groups
        self.tokens, self.top_k = route_slots.shape
        self.routes = route_slots.numel()
        self.hidden, self.inter = x.shape[1], down.shape[2]
        if gate_up.shape != (down.shape[0], 2 * self.inter, self.hidden) or down.shape[1] != self.hidden:
            raise ValueError("grouped TableMoE Prefill expert bank shapes do not match inputs")
        self.config = try_get_optimal_moe_config(gate_up.shape, down.shape, self.top_k, self.tokens)
        sort_ids, sort_experts = route_slots, gate_up.shape[0]
        if logical_ids is not None:
            if (logical_ids.shape != route_slots.shape or logical_ids.device != x.device or
                    logical_ids.dtype not in (torch.int32, torch.int64)):
                raise ValueError("grouped TableMoE Prefill logical expert ids do not match route slots")
            if num_experts is None or not 0 < num_experts <= gate_up.shape[0]:
                raise ValueError("grouped TableMoE Prefill requires the logical expert count")
            # Logical IDs avoid padding for every cache slot across all model layers.
            sort_ids = logical_ids.masked_fill(route_slots < 0, -1).to(torch.int32).contiguous()
            sort_experts = num_experts
        self.sorted_ids, self.expert_ids, self.num_padded = moe_align_block_size(
            sort_ids, self.config["BLOCK_SIZE_M"], sort_experts)
        self.max_groups = sort_experts
        self.max_blocks = triton.cdiv(self.sorted_ids.numel(), self.config["BLOCK_SIZE_M"])
        self.group_counts = torch.empty(self.max_groups + 1, device=x.device, dtype=torch.int32)
        self.group_offsets = torch.empty_like(self.group_counts)
        self.group_counters = torch.empty_like(self.group_counts)
        self.group_blocks = torch.empty(self.max_blocks, device=x.device, dtype=torch.int32)
        compact_group_blocks[(1,)](
            self.sorted_ids, self.num_padded, route_slots, route_groups, self.group_counts,
            self.group_offsets, self.group_counters, self.group_blocks, self.routes,
            self.max_blocks, self.config["BLOCK_SIZE_M"], self.max_groups + 1,
            triton.next_power_of_2(self.max_groups + 2), 256)
        self.buffers = (self.sorted_ids, self.expert_ids, self.num_padded, sort_ids,
                        self.group_counts, self.group_offsets, self.group_counters, self.group_blocks)
        self.grid_limit = 8 * torch.cuda.get_device_properties(x.device).multi_processor_count
        if workspace is None:
            workspace = (
                torch.empty(self.tokens, self.top_k, 2 * self.inter, device=x.device, dtype=x.dtype),
                torch.empty(self.routes, self.inter, device=x.device, dtype=x.dtype),
                torch.empty(self.tokens, self.top_k, self.hidden, device=x.device, dtype=x.dtype),
            )
        expected = [(self.tokens, self.top_k, 2 * self.inter), (self.routes, self.inter),
                    (self.tokens, self.top_k, self.hidden)]
        if len(workspace) != 3 or any(tuple(t.shape) != shape or t.dtype != x.dtype or t.device != x.device or
                                      not t.is_contiguous() for t, shape in zip(workspace, expected)):
            raise ValueError("grouped TableMoE Prefill workspace has incompatible shapes or dtype")
        self.gate_values = workspace[0].view(self.routes, 2 * self.inter)
        self.activated, self.outputs = workspace[1], workspace[2]
        tensors = (x, self.gate_values, self.activated, self.outputs)
        ranges = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in tensors]
        if any(max(a, c) < min(b, d) for i, (a, b) in enumerate(ranges) for c, d in ranges[i + 1:]):
            raise ValueError("grouped TableMoE Prefill workspaces and inputs must not overlap")
        self.outputs.zero_()

    def _gemm(self, inputs, weights, outputs, top_k, group_id):
        grid = (min(self.grid_limit, self.max_blocks *
                    triton.cdiv(weights.shape[1], self.config["BLOCK_SIZE_N"])),)
        grouped_expert_gemm[grid](
            inputs, weights, outputs, self.sorted_ids, self.route_slots,
            self.group_blocks, self.group_counts, self.group_offsets,
            group_id, weights.shape[1], weights.shape[2],
            self.routes, inputs.stride(0), inputs.stride(1), weights.stride(0), weights.stride(2),
            weights.stride(1), outputs.stride(0), outputs.stride(1),
            TOP_K=top_k, EVEN_K=weights.shape[2] % self.config["BLOCK_SIZE_K"] == 0, **self.config)

    def run(self, group_id):
        if group_id < -1 or group_id >= self.max_groups:
            return self.outputs
        self._gemm(self.x, self.gate_up, self.gate_values, self.top_k, group_id)
        grouped_silu[(min(self.grid_limit, self.max_blocks * triton.cdiv(self.inter, 128)),)](
            self.gate_values, self.activated, self.sorted_ids,
            self.group_blocks, self.group_counts, self.group_offsets,
            group_id, self.routes, self.inter, self.config["BLOCK_SIZE_M"], 128)
        self._gemm(self.activated, self.down, self.outputs.view(self.routes, self.hidden), 1, group_id)
        return self.outputs
