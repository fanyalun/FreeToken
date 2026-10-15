from __future__ import annotations

import torch

from freetoken.kernel.backend import is_sgl_kernel_installed


class QwenGDNVerifyState:
    def __init__(self, pool, layers, width: int, slot: int):
        if not 1 <= width <= 4 or not 0 <= slot < pool.num_slots:
            raise ValueError("GDN verification requires width 1..4 and a valid state slot")
        self.pool = pool
        self.layers = tuple(layers)
        self.width = width
        self.slot = slot
        self.conv_fp32_products = is_sgl_kernel_installed()
        device = pool.conv_states.device
        layers_count, _, channels, history = pool.conv_states.shape
        _, _, heads, key_dim, value_dim = pool.recurrent_states.shape
        key_heads = (channels - heads * value_dim) // (2 * key_dim)
        if key_heads <= 0 or heads % key_heads:
            raise ValueError("GDN conv and recurrent pool geometry disagree")
        self.indices = torch.tensor([slot], dtype=torch.int32, device=device)
        self.cu_seqlens = torch.tensor([0, width], dtype=torch.int32, device=device)
        self.conv_windows = torch.empty(
            layers_count, channels, history + width - 1, dtype=pool.conv_states.dtype, device=device,
        )
        self.replay_inputs = (
            torch.empty(layers_count, width, key_heads, key_dim, dtype=pool.conv_states.dtype, device=device),
            torch.empty(layers_count, width, heads, value_dim, dtype=pool.conv_states.dtype, device=device),
            torch.empty(layers_count, width, heads, dtype=torch.float32, device=device),
            torch.empty(layers_count, width, heads, dtype=torch.float32, device=device),
        )
        self.snapshots = None
        self.snapshot_index = None

    def verify_layer(self, layer, raw, a, b):
        # FLA imports probe CUDA, so defer them until the engine has bound its device.
        from freetoken.kernel.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
        from freetoken.kernel.fla.gdn_speculative import conv_verify

        if raw.shape[0] != self.width:
            raise ValueError("GDN window width differs from reserved state buffers")
        li = self.pool.local_index(layer.layer_id)
        mixed = conv_verify(
            raw, layer._conv_weight(), self.pool.conv_states[li], self.conv_windows[li], self.slot,
            fp32_products=self.conv_fp32_products,
        )
        q, k, v = torch.split(mixed, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
        reference = self.snapshots[layer.layer_id][0].unsqueeze(0) if self.snapshots is not None else None
        out = fused_sigmoid_gating_delta_rule_update(
            A_log=layer.A_log, a=a, b=b, dt_bias=layer.dt_bias,
            softplus_beta=1.0, softplus_threshold=20.0,
            q=q.reshape(1, self.width, layer.num_k_heads, layer.head_k_dim),
            k=k.reshape(1, self.width, layer.num_k_heads, layer.head_k_dim),
            v=v.reshape(1, self.width, layer.num_v_heads, layer.head_v_dim),
            initial_state_source=self.pool.recurrent_states[li], initial_state_indices=self.indices,
            cu_seqlens=self.cu_seqlens, scale=layer.head_k_dim ** -0.5,
            use_qk_l2norm_in_kernel=True, disable_state_update=True,
            replay_inputs=tuple(x[li] for x in self.replay_inputs),
            intermediate_states_buffer=reference, intermediate_state_indices=self.snapshot_index,
            round_state_each_step=self.pool.recurrent_states.dtype != torch.float32,
        )
        return out[0]

    def enable_snapshot_reference(self) -> None:
        self.snapshots = {}
        self.snapshot_index = torch.zeros(1, dtype=torch.int32, device=self.conv_windows.device)
        for layer in self.layers:
            li = self.pool.local_index(layer.layer_id)
            self.snapshots[layer.layer_id] = (
                torch.empty_like(self.pool.recurrent_states[li, :1]).expand(self.width, -1, -1, -1).clone(),
                self.conv_windows[li].unfold(-1, self.pool.conv_states.shape[-1], 1).permute(1, 0, 2),
            )

    def commit(self, input_count: torch.Tensor) -> None:
        from freetoken.kernel.fla.gdn_speculative import commit_gdn

        commit_gdn(
            self.pool.recurrent_states, self.pool.conv_states, self.replay_inputs,
            self.conv_windows, input_count, self.slot, self.width,
        )

    def bytes(self) -> int:
        tensors = [self.conv_windows, self.indices, self.cu_seqlens, *self.replay_inputs]
        if self.snapshots is not None:
            tensors.extend(group[0] for group in self.snapshots.values())
        if self.snapshot_index is not None:
            tensors.append(self.snapshot_index)
        return sum(t.numel() * t.element_size() for t in tensors)
