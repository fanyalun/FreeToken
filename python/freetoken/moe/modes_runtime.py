"""MoDES and MoDES+TableMoE over the shared FP16 expert executor."""

import torch
import triton

from freetoken.kernel.triton.modes import _select
from freetoken.moe.modes import load_calibration
from freetoken.moe.tablemoe_optimized import TableMoeOptimizedRuntime
from freetoken.utils import load_tokenizer


class ModesRuntime(TableMoeOptimizedRuntime):
    def __init__(self, config):
        data = load_calibration(config.modes_config, config.model_path,
                                range(config.model_config.num_moe_layers))
        self.calibration = data
        super().__init__(config)
        self.importance = torch.tensor([[data["layers"][str(i)]["importance"][m]
                                        for m in ("visual", "text")] for i in range(self.num_layers)],
                                       device=self.device, dtype=torch.float32)
        self.decode_importance = self.importance
        if any("decode_importance" in layer for layer in data["layers"].values()):
            self.decode_importance = torch.tensor([
                [data["layers"][str(i)].get("decode_importance", data["layers"][str(i)]["importance"])[m]
                 for m in ("visual", "text")] for i in range(self.num_layers)],
                device=self.device, dtype=torch.float32)
        self.tau = torch.tensor([[data["layers"][str(i)]["tau"][m]
                                 for m in ("visual", "text")] for i in range(self.num_layers)],
                                device=self.device, dtype=torch.float32)
        self.decode_tau = self.tau
        if any("decode_tau" in layer for layer in data["layers"].values()):
            self.decode_tau = torch.tensor([
                [data["layers"][str(i)].get("decode_tau", data["layers"][str(i)]["tau"])[m]
                 for m in ("visual", "text")] for i in range(self.num_layers)],
                device=self.device, dtype=torch.float32)
        self.special_ids = torch.tensor(load_tokenizer(config.model_path).all_special_ids,
                                        device=self.device, dtype=torch.int32)

    def resident_gpu_bytes(self, target_cache_size=None):
        from freetoken.moe.tablemoe_optimized import gpu_storage_bytes

        return super().resident_gpu_bytes(target_cache_size) + gpu_storage_bytes(
            [self.importance, self.decode_importance, self.tau, self.decode_tau, self.special_ids])

    def _select(self, layer, weights, token_ids, ws, is_prefill=False):
        tau = self.tau if is_prefill else getattr(self, "decode_tau", self.tau)
        importance = self.importance if is_prefill else getattr(self, "decode_importance", self.importance)
        _select[(triton.cdiv(weights.numel(), 256),)](
            weights, token_ids, ws["request_slots"], ws["modality"], self.special_ids,
            importance[layer], tau[layer], ws["keep"], weights.shape[1],
            self.special_ids.numel(), self.dummy_slot, weights.numel(), 256)

    def stats(self):
        result = super().stats()
        if result.get("collected"):
            skipped = result["online_hits"] + result["offline_hits"] + result["dropped_routes"]
            valid = skipped + result["cpu_exact_routes"] + result["gpu_exact_routes"]
            recovered = result["online_hits"] + result["offline_hits"]
            result.update(moe_policy=self.config.moe_policy,
                          modes_prefill_policy=getattr(self.config, "modes_prefill_policy", "modes"),
                          valid_routes=valid, table_recovered_routes=recovered)
            if self.prefill_uses_tablemoe:
                result.update(skipped_routes=skipped, skip_fraction=skipped / valid if valid else 0.)
            else:
                result.update(modes_skipped_routes=skipped,
                              modes_skip_fraction=skipped / valid if valid else 0.,
                              table_recovery_fraction=recovered / skipped if skipped else 0.)
        return result

    def forward(self, layer_id, hidden_states, weights, ids, batch):
        if batch.tablemoe_slots is None:
            self.prepare_batch(batch)
        if batch.is_prefill and self.prefill_uses_tablemoe:
            return super()._prefill(layer_id, hidden_states, weights, ids, batch)
        weights = weights.to(torch.float16)
        if batch.is_prefill:
            return self._prefill(layer_id, hidden_states, weights, ids, batch)
        ws = self._workspace(hidden_states.shape[0])
        ws["request_slots"].copy_(batch.tablemoe_slots)
        ws["modality"].fill_(1)
        ws["update_rows"].zero_()
        ws["update_counts"].fill_(1)
        self._select(layer_id, weights, batch.input_ids, ws)
        return self._run(layer_id, hidden_states, weights, ids, ws, False)

    def _prefill(self, layer, x, weights, ids, batch):
        update = torch.full((x.shape[0],), -1, device=self.device, dtype=torch.int32)
        advance = torch.zeros_like(update)
        offset = 0
        for req in batch.padded_reqs:
            n = req.extend_len
            count = min(n, self.capacity)
            if count and req.uid != -1:
                update[offset + n - count:offset + n] = torch.arange(count, device=self.device)
                advance[offset + n - 1] = count
            offset += n
        output = torch.empty_like(x)
        for start in range(0, offset, 128):
            end = min(start + 128, offset)
            ws = self._workspace(end - start)
            ws["request_slots"].copy_(batch.tablemoe_slots[start:end])
            ws["update_rows"].copy_(update[start:end])
            ws["update_counts"].copy_(advance[start:end])
            ws["modality"].fill_(1)
            if batch.mm_rows is not None:
                rows = batch.mm_rows[(batch.mm_rows >= start) & (batch.mm_rows < end)] - start
                ws["modality"][rows.long()] = 0
            self._select(layer, weights[start:end], batch.input_ids[start:end], ws, is_prefill=True)
            output[start:end].copy_(self._run(layer, x[start:end], weights[start:end], ids[start:end], ws, True))
        return output
