"""Tree MTP for the scheduler's batched two-level verifier."""

from dataclasses import replace

import torch

from freetoken.core import Batch
from freetoken.engine.speculative import VerifyResult
from freetoken.engine.speculative_tree import build_mtp_trees, greedy_tree_commit, tree_attention_metadata
from freetoken.spec.proposer import SpecProposal

from .proposer import MTPProposer, with_resident_moe


class MTPTreeProposer(MTPProposer):
    @classmethod
    def adjust_config(cls, config, override):
        super().adjust_config(config, override)
        from freetoken.kvcache.linear_state_pool import ssm_state_dtype

        arch = config.hf_config.architectures[0]
        if not arch.startswith(("Qwen3_5", "Qwen3_6")) or config.model_config.slot_states:
            raise ValueError("tree MTP currently supports Qwen3.5/3.6 targets without extra slot states")
        if config.page_size != 1:
            raise ValueError("tree MTP currently requires page_size=1")
        if ssm_state_dtype() != torch.float32:
            raise ValueError("batched tree MTP requires fp32 recurrent states")
        if config.speculative_graph not in (None, "eager"):
            raise ValueError("tree MTP currently requires eager execution")
        override("attention_backend", "triton")
        override("cuda_graph_max_bs", 0)
        override("cuda_graph_bs", [])

    @classmethod
    def budget(cls, config):
        from freetoken.kvcache.base import spec_kv_bytes_per_token

        _, per_token = super().budget(config)
        draft_cls = cls._draft_class(config)
        rows, width = config.max_running_req + 1, config.decode_rows_per_req
        fixed = rows * (width + 2) * draft_cls.relay_width(config.model_config) * config.dtype.itemsize
        groups = draft_cls.draft_config(with_resident_moe(config.model_config)).kv_cache_group_specs()
        scratch = (config.speculative_num_steps - 1) * config.speculative_eagle_topk
        fixed += rows * scratch * sum(spec_kv_bytes_per_token(g, config) for g in groups)
        # Tree ancestry indices and the draft beam's fp32 logits/probabilities.
        fixed += rows * width * (config.max_seq_len + width) * 4
        fixed += rows * config.speculative_eagle_topk * config.model_config.vocab_size * 8
        group = config.model_config.linear_attention_group()
        if group is not None:
            channels = 2 * group.num_key_heads * group.key_head_dim + group.num_value_heads * group.value_head_dim
            fixed += rows * width * len(group.layer_ids) * (channels + group.num_value_heads * (group.value_head_dim + 2)) * config.dtype.itemsize
            fixed += rows * len(group.layer_ids) * channels * (group.conv_kernel_dim + width - 2) * config.dtype.itemsize
            fixed += rows * width * channels * config.dtype.itemsize * (config.speculative_num_steps + 1) * 4
            fixed += width * group.num_value_heads * group.key_head_dim * group.value_head_dim * 4
        return fixed, per_token

    def __init__(self, config, target_model):
        super().__init__(config, target_model)
        self.k = config.decode_rows_per_req - 1
        self.depth, self.topk = config.speculative_num_steps, config.speculative_eagle_topk
        # Draft beams have separate arenas beyond the target's physical pages.
        self.scratch_per_req = (self.depth - 1) * self.topk
        self.scratch_pages = (config.max_running_req + 1) * self.scratch_per_req

    def propose(self, reqs, rows, pending):
        bs = len(reqs)
        prefixes = [self.ctx.page_table[r.table_idx, :r.cached_len] for r in reqs]
        arenas = (torch.arange(bs * self.scratch_per_req, device=self.device,
                               dtype=self.ctx.page_table.dtype).view(bs, self.scratch_per_req)
                  + self.ctx.num_pages + 1)

        def forward(ids, hidden, paths, depths, slots):
            views = [self._view(r, r.cached_len, r.cached_len + self.topk) for r in reqs]
            batch = Batch(reqs=views, phase="verify", rows_per_req=self.topk)
            batch.padded_reqs = views
            batch.input_ids = ids
            batch.attn_metadata = tree_attention_metadata(prefixes, list(arenas), paths, depths)
            batch.positions = batch.attn_metadata.q_positions
            if self.mrope:
                delta = torch.tensor([r.mrope_delta for r in reqs], device=self.device, dtype=batch.positions.dtype)
                batch.mrope_positions = (batch.positions + delta.repeat_interleave(self.topk)).expand(3, -1).contiguous()
            batch.out_loc = arenas[:, slots].flatten().to(self.ctx.page_table.dtype)
            return self._forward(batch, self.chain_backend, self.draft.embed_tokens.forward(ids), hidden)

        trees = build_mtp_trees(
            pending, self.relay.hidden[rows], self.depth, self.topk, self.k + 1,
            lambda h: self.lm_head.logits(self.draft.head_input(h)),
            lambda t: t if self.hot_ids is None else self.hot_ids[t], forward,
        )
        return SpecProposal(torch.stack([t.tokens for t in trees]), trees=tuple(trees))


class _BatchedTreeState:
    def __init__(self, pool, reqs, trees):
        from freetoken.attention.linear import gdn_slot
        from freetoken.models.qwen3_5_moe.speculative_state import QwenGDNTreeVerifyState, QwenGDNVerifyState

        self.pool, self.states, self.layers = pool, [], []
        for i, (req, tree) in enumerate(zip(reqs, trees)):
            # The scheduler's replay uses fp32 k/v. Write straight into its stash
            # so stop-token truncation still selects the final committed count.
            chain = QwenGDNVerifyState(pool, (), tree.parents.numel(), gdn_slot(req),
                                      input_stash=tuple(t[:, i:i + 1] for t in pool.verify_scratch.stash))
            self.states.append(QwenGDNTreeVerifyState(chain, tree.paths, tree.depths,
                workspace=self.states[0].workspace if self.states else None))

    def verify_layer(self, layer, raw, a, b):
        self.layers.append(layer)
        width = self.states[0].paths.shape[0]
        return torch.cat([state.verify_layer(layer, raw[i * width:(i + 1) * width],
                                            a[i * width:(i + 1) * width], b[i * width:(i + 1) * width])
                          for i, state in enumerate(self.states)])

    def stage_commit(self, paths):
        for i, (state, path) in enumerate(zip(self.states, paths)):
            state.chain.layers = tuple(self.layers)
            state.stage_commit(path)
            history = self.pool.conv_states.shape[-1]
            # Both layouts store post-input windows; unfold the native compact
            # layout into the scheduler's count-indexed windows.
            for layer in self.layers:
                li = self.pool.local_index(layer.layer_id)
                raw = state.chain.conv_windows[li]
                self.pool.verify_scratch.conv_window[li, i, :state.chain.width].copy_(
                    raw.unfold(-1, history, 1).permute(1, 0, 2))


def verify_tree(engine, batch, args, proposer, trees):
    width, bs = batch.rows_per_req, batch.size
    original_positions, original_mrope = batch.positions, batch.mrope_positions
    locations = batch.out_loc.view(bs, width)
    prefixes = [engine.page_table[r.table_idx, :r.cached_len] for r in batch.reqs]
    batch.attn_metadata = tree_attention_metadata(prefixes, list(locations),
                                                  [t.paths for t in trees], [t.depths for t in trees])
    batch.positions = batch.attn_metadata.q_positions
    if original_mrope is not None:
        delta = original_mrope[:, ::width] - original_positions[::width]
        batch.mrope_positions = batch.positions.unsqueeze(0) + delta.repeat_interleave(width, dim=1)
    state = _BatchedTreeState(engine.linear_state_pool, batch.reqs, trees) if engine.linear_state_pool is not None else None
    batch.gdn_verify_state = state
    logits = engine.forward_logits(batch)[:bs * width]
    sampled = sample_tree(logits, args, trees, engine.sampler)
    outs, counts, paths = [], [], []
    for tree, tokens in zip(trees, sampled):
        commit, path = greedy_tree_commit(tree, VerifyResult(tokens, tokens[:, None]), remaining=width + 1)
        pad = width - path.numel()
        paths.append(torch.cat((path, path[-1:].expand(pad))))
        outs.append(torch.cat((commit.tokens, commit.tokens[-1:].expand(pad))))
        counts.append(commit.input_count - 1)
    paths = torch.stack(paths)
    if state is not None:
        state.stage_commit(paths)
    source = locations.gather(1, paths).flatten().long()
    target = locations.flatten().long()
    kv = engine.kv_cache._kv_buffer
    kv[:, :, target] = kv.index_select(2, source)
    hidden = proposer.verify_hidden[:bs * width].view(bs, width, -1)
    hidden.copy_(hidden.gather(1, paths[:, :, None].expand_as(hidden)))
    batch.positions, batch.mrope_positions = original_positions, original_mrope
    batch.gdn_verify_state = None
    return torch.stack(outs).long(), torch.stack(counts).long()


def sample_tree(logits, args, trees, sampler):
    width = trees[0].parents.numel()
    # Penalize only tokens on this node's ancestry; siblings have not been emitted.
    if args.penalties:
        logits = logits.float().clone()
        for i, counts, presence, frequency in args.penalties:
            block = logits[i * width:(i + 1) * width]
            block -= frequency * counts + presence * (counts > 0)
            tree = trees[i]
            ids = tree.verify_inputs().long()
            ancestors = tree.paths[:, 1:]
            tokens = ids[ancestors.clamp_min(0)]
            for col in range(ancestors.shape[1]):
                valid = ancestors[:, col] >= 0
                first = ~((tokens[:, :col] == tokens[:, col:col + 1]) & (ancestors[:, :col] >= 0)).any(1)
                extra = valid * (frequency + presence * ((counts[tokens[:, col]] == 0) & first))
                block.scatter_add_(1, tokens[:, col:col + 1], -extra[:, None].to(block.dtype))
        args = replace(args, penalties=[])
    return sampler.sample_verify(logits, args, torch.stack([t.tokens for t in trees])).view(len(trees), width)
