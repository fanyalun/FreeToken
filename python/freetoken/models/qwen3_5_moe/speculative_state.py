from __future__ import annotations

import os

import torch

from freetoken.kernel.backend import is_sgl_kernel_installed


class QwenGDNVerifyState:
    chunk_enabled = os.getenv("FREETOKEN_GDN_CHUNK_VERIFY", "0") == "1"
    chunk_min_width = 8
    chunk_max_width = 64

    def __init__(self, pool, layers, width: int, slot: int, *, workspace=None):
        if width < 1 or not 0 <= slot < pool.num_slots:
            raise ValueError("GDN verification requires positive width and a valid state slot")
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
        def allocate(shape, dtype, source=None):
            if source is not None:
                size = 1
                for dim in shape:
                    size *= dim
                if source.numel() < size or source.dtype != dtype or source.device != device:
                    raise ValueError("incompatible GDN scratch workspace")
                return source.view(-1)[:size].view(shape)
            return torch.empty(shape, dtype=dtype, device=device)

        self.conv_windows = allocate((layers_count, channels, history + width - 1), pool.conv_states.dtype,
                                     None if workspace is None else workspace.conv_windows)
        sources = (None,) * 4 if workspace is None else workspace.replay_inputs
        self.replay_inputs = (
            allocate((layers_count, width, key_heads, key_dim), pool.conv_states.dtype, sources[0]),
            allocate((layers_count, width, heads, value_dim), pool.conv_states.dtype, sources[1]),
            allocate((layers_count, width, heads), torch.float32, sources[2]),
            allocate((layers_count, width, heads), torch.float32, sources[3]),
        )
        self.snapshots = None
        self.snapshot_index = None

    def uses_chunk(self, dtype):
        from freetoken.core import get_active_batch

        return (self.chunk_enabled and self.chunk_min_width <= self.width <= self.chunk_max_width
                and dtype == torch.bfloat16
                and self.pool.recurrent_states.dtype == torch.float32
                and self.pool.recurrent_states.shape[-2] <= 256
                and self.snapshots is None
                and not getattr(get_active_batch(), "speculative_reference_math", False))

    def chunk_temporary_bytes(self):
        if not self.uses_chunk(self.pool.conv_states.dtype):
            return 0
        _, _, hv, k, v = self.pool.recurrent_states.shape
        channels = self.pool.conv_states.shape[2]
        h = (channels - hv * v) // (2 * k)
        return self.chunk_temporary_bytes_for_geometry(self.width, h, hv, k, v)

    @classmethod
    def chunk_temporary_bytes_for_geometry(cls, width, key_heads, value_heads, key_dim, value_dim,
                                         *, activation_dtype=torch.bfloat16, state_dtype=torch.float32):
        if (not cls.chunk_min_width <= width <= cls.chunk_max_width
                or activation_dtype != torch.bfloat16 or state_dtype != torch.float32 or key_dim > 256):
            return 0
        h, hv, k, v = key_heads, value_heads, key_dim, value_dim
        channels = 2 * h * k + hv * v
        # Bound simultaneous chunk intermediates; replay and conv windows are persistent elsewhere.
        return (2 * width * channels + 4 * width * h * k + 4 * width * hv
                + 2 * width * hv * 64 + 2 * width * hv * k + 6 * width * hv * v
                + 2 * ((width + 63) // 64) * hv * v * k)

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
        if self.uses_chunk(raw.dtype) and a.stride(-1) == b.stride(-1) == 1:
            from freetoken.kernel.fla.gdn_speculative import chunk_verify

            return chunk_verify(mixed, a, b, layer, self.pool.recurrent_states[li], self.indices,
                                tuple(x[li] for x in self.replay_inputs))
        q, k, v = torch.split(mixed, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
        from freetoken.core import get_active_batch

        small_value_tile = (self.width >= 2 and self.snapshots is None
                            and raw.dtype == torch.bfloat16
                            and self.pool.recurrent_states.dtype == torch.float32
                            and layer.head_k_dim == layer.head_v_dim == 128
                            and not getattr(get_active_batch(), "speculative_reference_math", False))
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
            small_value_tile=small_value_tile,
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
        return self.workspace_bytes((self,))

    @staticmethod
    def workspace_bytes(states) -> int:
        storages = {}
        for state in states:
            for tensor in state.tensors():
                storage = tensor.untyped_storage()
                storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
        return sum(storages.values())

    def tensors(self):
        tensors = [self.conv_windows, self.indices, self.cu_seqlens, *self.replay_inputs]
        if self.snapshots is not None:
            tensors.extend(group[0] for group in self.snapshots.values())
        if self.snapshot_index is not None:
            tensors.append(self.snapshot_index)
        return tensors
