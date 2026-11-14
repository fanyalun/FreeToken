"""Native DeepSeek-VL2 quality-evaluation policies with bounded expert offload."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Policy:
    name: str
    alpha: float = 1.
    top_h: int = 6
    switch_after: int | None = None
    late_top_h: int | None = None

    def decode_h(self, output_position):
        if self.switch_after is not None and output_position > self.switch_after:
            return self.late_top_h
        return self.top_h


POLICIES = {
    'baseline': Policy('baseline'),
    'tablemoe_06_top4': Policy('tablemoe_06_top4', .6, 4),
    'tablemoe_07_top6': Policy('tablemoe_07_top6', .7, 6),
    'tablemoe_06_top4_512_top6': Policy('tablemoe_06_top4_512_top6', .6, 4, 512, 6),
    'adapmoe_gating_top5': Policy('adapmoe_gating_top5', 1., 5),
}


def prefill_mask(ids, weights, attention, experts, alpha):
    if attention is None or attention.numel() != len(ids):
        raise ValueError('prefill importance requires the actual last-query attention scores')
    total = torch.zeros(experts, device=ids.device, dtype=torch.float32)
    total.scatter_add_(0, ids.reshape(-1), (weights * attention[:, None]).reshape(-1).float())
    active = torch.zeros(experts, device=ids.device, dtype=torch.bool)
    active.scatter_(0, ids.reshape(-1), True)
    count = (active.sum() * alpha).long().clamp_min(1)
    order = total.masked_fill(~active, -float('inf')).argsort(descending=True, stable=True)
    keep = torch.zeros_like(active)
    keep.scatter_(0, order, torch.arange(experts, device=ids.device) < count)
    return keep[ids]


def decode_mask(weights, h, resident):
    order = weights.argsort(dim=-1, descending=True, stable=True)
    keep = torch.zeros_like(weights, dtype=torch.bool)
    keep.scatter_(1, order[:, :h], True)
    return keep | resident


def merge_routes(outputs, weights):
    return (outputs.to(weights.dtype) * weights.unsqueeze(-1)).sum(1).to(outputs.dtype)


class Context:
    def __init__(self):
        self.reset(None)

    def reset(self, vision):
        self.vision = vision
        self.output_position = 1
        self.counters = Counter()


class Lookup:
    def __init__(self, layer, hidden, num_experts, cache_dir, pca_dir, device, capacity=128):
        self.layer, self.hidden, self.experts = layer, hidden, num_experts
        self.device, self.capacity = device, capacity
        self.pca, self.offline = {}, {}
        for modality, name in enumerate(('vision', 'text')):
            pca = torch.load(Path(pca_dir) / f'layer_{layer}/L{layer}_{name}_pca.pt', weights_only=True)
            mean, projection = pca['mean'].to(device), pca['proj'].to(device)
            if mean.dtype != torch.bfloat16 or projection.dtype != torch.bfloat16:
                raise ValueError('native BF16 PCA assets are required')
            self.pca[modality] = (projection, -mean @ projection)
            for expert in range(num_experts):
                path = Path(cache_dir) / f'layer_{layer}/L{layer}_E{expert}_{name}_cache.pt'
                if not path.is_file():
                    continue
                data = torch.load(path, weights_only=True, mmap=True, map_location='cpu')
                if (data['key'].dtype != torch.bfloat16 or data['value'].dtype != torch.bfloat16
                        or data['value'].shape != (len(data['key']), hidden)):
                    raise ValueError(f'incompatible table: {path}')
                keys = F.normalize(data['key'], dim=-1, eps=1e-6).to(device).float()
                self.offline[expert, modality] = (keys, data['value'])
        self.online_keys = torch.empty(capacity, hidden, dtype=torch.bfloat16, device=device)
        self.online_ids = torch.full((capacity, 6), -1, dtype=torch.long, device=device)
        self.online_values = torch.empty(capacity, 6, hidden, dtype=torch.bfloat16, device=device)
        self.reset()

    def reset(self):
        self.size, self.pointer = 0, 0
        self.online_ids.fill_(-1)

    def online(self, x, ids, reuse, outputs):
        hit = torch.zeros_like(reuse)
        if not self.size or not reuse.any():
            return hit
        query = F.normalize(x.float(), dim=-1, eps=1e-6).to(x.dtype)
        keys = F.normalize(self.online_keys[:self.size].float(), dim=-1, eps=1e-6).to(x.dtype)
        similarity, row = (query @ keys.T).max(-1)
        matches = ids[:, :, None] == self.online_ids[row, None, :]
        hit = matches.any(-1) & (similarity >= .7)[:, None] & reuse
        slot = matches.int().argmax(-1)
        values = self.online_values[row[:, None], slot]
        outputs[hit] = values[hit]
        return hit

    def offline_search(self, x, ids, modality, reuse, outputs, prefill):
        hit = torch.zeros_like(reuse)
        for mod in (0, 1):
            rows, slots = torch.where(reuse & (modality == mod)[:, None])
            if not len(rows):
                continue
            projection, bias = self.pca[mod]
            query = torch.addmm(bias, x, projection)
            query = (F.normalize(query, dim=-1, eps=1e-6).float() if prefill else
                     F.normalize(query.float(), dim=-1, eps=1e-6))
            selected = ids[rows, slots]
            for expert in selected.unique().tolist():
                if (expert, mod) not in self.offline:
                    continue
                positions = torch.where(selected == expert)[0]
                target_rows, target_slots = rows[positions], slots[positions]
                keys, values = self.offline[expert, mod]
                score, cluster = (query[target_rows] @ keys.T).max(-1)
                threshold = (.6 if mod == 0 else .7) if prefill else .7
                accepted = score >= threshold
                target_rows, target_slots, cluster = target_rows[accepted], target_slots[accepted], cluster[accepted]
                if len(target_rows):
                    outputs[target_rows, target_slots] = values[cluster.cpu()].to(x.device)
                    hit[target_rows, target_slots] = True
        return hit

    def update(self, x, ids, outputs, survived):
        n = min(len(x), self.capacity)
        rows = (torch.arange(n, device=x.device) + self.pointer) % self.capacity
        self.online_keys[rows] = x[-n:]
        self.online_ids[rows] = ids[-n:].masked_fill(~survived[-n:], -1)
        self.online_values[rows] = outputs[-n:].masked_fill(~survived[-n:, :, None], 0)
        self.pointer = (self.pointer + n) % self.capacity
        self.size = min(self.capacity, self.size + n)


class TableLayer(nn.Module):
    def __init__(self, original, policy, context, lookup):
        super().__init__()
        self.gate, self.shared_experts = original.gate, original.shared_experts
        self.experts = original.experts
        self.layer_id, self.top_k, self.num_routed_experts = original.layer_id, original.top_k, original.num_routed_experts
        self.policy, self.context, self.lookup = policy, context, lookup
        self.plan = original._plan_masked_slot_execution

    def forward(self, hidden_states):
        from table_moe.utils.modality import ModalityContext

        batch, length, hidden = hidden_states.shape
        if batch != 1:
            raise ValueError('the frozen writing evaluation uses batch size one')
        x = hidden_states.reshape(-1, hidden)
        ids, weights, _ = self.gate(hidden_states)
        ids = ids.reshape(len(x), -1)
        prefill = length > 1
        if prefill:
            exact = prefill_mask(ids, weights, ModalityContext.get_attn_weights(),
                                 self.num_routed_experts, self.policy.alpha)
            modality = torch.where(self.context.vision.reshape(-1), 0, 1)
        else:
            resident = torch.tensor([[not self.experts.registered_experts[self.layer_id, int(expert)].offloaded
                                      for expert in ids[0].tolist()]], device=x.device)
            h = self.policy.decode_h(self.context.output_position)
            exact = decode_mask(weights, h, resident)
            modality = torch.ones(len(x), device=x.device, dtype=torch.long)
            self.context.counters[f'decode_h_{h}_layer_calls'] += 1
            self.context.counters['decode_resident_routes'] += int(resident.sum())
        outputs = torch.zeros(len(x), ids.shape[1], hidden, device=x.device, dtype=x.dtype)
        online = (torch.zeros_like(exact) if prefill else self.lookup.online(x, ids, ~exact, outputs))
        offline = self.lookup.offline_search(x, ids, modality, ~exact & ~online, outputs, prefill)
        experts, positions = self.plan(ids, exact)
        for (_, expert), module in self.experts.load_experts(*((self.layer_id, expert) for expert in experts), unordered=True):
            rows, slots = positions[expert]
            outputs[rows, slots] = module(x[rows])
        survived = exact | online | offline
        self.lookup.update(x, ids, outputs, survived)
        phase = 'prefill' if prefill else 'decode'
        for name, mask in (('exact', exact), ('online_hit', online), ('offline_hit', offline), ('dropped', ~survived)):
            self.context.counters[f'{phase}_{name}_routes'] += int(mask.sum())
        self.context.counters[f'{phase}_total_routes'] += ids.numel()
        return (self.shared_experts(x) + merge_routes(outputs, weights)).reshape(batch, length, hidden)


def build_model(config, policy, context):
    import random
    from table_moe.models.deepseek_vl2.builder import build_offload_model_deepseek_baseline
    from table_moe.models.deepseek_vl2.deepseek_attention import HybridDeepSeekAttention
    from table_moe.models.deepseek_vl2.offload_config import update_deepseek_offload_config

    random.seed(42)
    update_deepseek_offload_config(model_dtype=torch.bfloat16, cache_size_per_layer=[0] + [36] * 29,
                                   buffer_size=72, prefetch=False, device=torch.device('cuda:0'))
    model, language, cache = build_offload_model_deepseek_baseline(config['model'], device_map='cuda:0')
    initial = {layer: sorted(expert for (owner, expert), info in cache.registered_experts.items()
                             if owner == layer and not info.offloaded) for layer in range(1, 30)}
    if any(len(experts) != 36 for experts in initial.values()) or len(cache.registered_experts) != 2088:
        raise ValueError('expert-cache capacity or checkpoint coverage mismatch')
    lookups = []
    for layer in range(1, language.num_hidden_layers):
        block = model.language.model.layers[layer]
        if policy.name.startswith('tablemoe'):
            lookup = Lookup(layer, language.hidden_size, language.n_routed_experts,
                            config['cache_dir'], config['pca_dir'], torch.device('cuda:0'))
            block.mlp = TableLayer(block.mlp, policy, context, lookup)
            block.self_attn = HybridDeepSeekAttention(block.self_attn)
            lookups.append(lookup)
        elif policy.name == 'adapmoe_gating_top5':
            block.mlp.gate.top_k = 5
            block.mlp.top_k = 5
    model.eval()
    return model, cache, initial, lookups


def reset_request(cache, initial, lookups):
    for layer, experts in initial.items():
        for _ in cache.load_experts(*((layer, expert) for expert in experts), unordered=False):
            pass
        resident = {expert for (owner, expert), info in cache.registered_experts.items()
                    if owner == layer and not info.offloaded}
        if resident != set(experts):
            raise ValueError('expert-cache reset did not restore the frozen resident set')
    for lookup in lookups:
        lookup.reset()
