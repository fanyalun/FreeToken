# SPDX-License-Identifier: Apache-2.0
"""Bounded EAGLE-style candidate trees shared by both MTP runtimes."""

from dataclasses import dataclass

import torch

from .speculative import Commit, Proposal


@dataclass(frozen=True)
class TreeProposal(Proposal):
    parents: torch.Tensor
    paths: torch.Tensor
    depths: torch.Tensor


def tree_paths(parents, max_depth):
    """Root-inclusive paths, with -1 padding after each node."""
    parents = parents.long()
    single = parents.ndim == 1
    if single:
        parents = parents.unsqueeze(0)
    nodes = torch.arange(parents.shape[1], device=parents.device).expand_as(parents)
    ancestors = [nodes]
    for _ in range(max_depth):
        nodes = torch.where(nodes >= 0, parents.gather(1, nodes.clamp_min(0)), -1)
        ancestors.append(nodes)
    reverse = torch.stack(ancestors, dim=-1)
    depths = (reverse >= 0).sum(-1) - 1
    columns = depths[:, :, None] - torch.arange(max_depth + 1, device=parents.device)
    paths = torch.where(columns >= 0, reverse.gather(-1, columns.clamp_min(0)), -1)
    return (paths[0], depths[0]) if single else (paths, depths)


def select_tree(anchor, tokens, parents, scores, budget, max_depth):
    # Ancestors have at least their descendants' path probability. Stable ties keep
    # the earlier ancestor when FP32 multiplication rounds both scores equally.
    selected = torch.argsort(scores, descending=True, stable=True)[:budget - 1].sort().values
    remap = torch.full((scores.numel() + 1,), -1, dtype=torch.long, device=scores.device)
    remap[0] = 0
    remap[selected + 1] = torch.arange(1, selected.numel() + 1, device=scores.device)
    selected_parents = torch.cat((parents.new_full((1,), -1), remap[parents[selected]]))
    paths, depths = tree_paths(selected_parents, max_depth)
    return TreeProposal(anchor, tokens[selected], selected_parents, paths, depths)


def greedy_tree_commit(proposal, verified, *, remaining, eos_mask=None, eos_ids=()):
    """Follow target predictions through children and emit one correction/bonus."""
    device = verified.token_ids.device
    width = proposal.parents.numel()
    output_width = min(proposal.paths.shape[1], width)
    path = torch.zeros(output_width, dtype=torch.long, device=device)
    tokens = torch.empty(output_width, dtype=verified.token_ids.dtype, device=device)
    current = torch.zeros((), dtype=torch.long, device=device)
    count = torch.zeros((), dtype=torch.int32, device=device)
    accepted = torch.zeros_like(count)
    active = torch.ones((), dtype=torch.bool, device=device)
    terminal = torch.zeros_like(active)
    ids = proposal.verify_inputs()
    node_ids = torch.arange(width, device=device)
    for step in range(output_width):
        path[step] = current
        token = verified.token_ids[current]
        tokens[step] = token
        count += active.to(torch.int32)
        eos = torch.zeros_like(active) if eos_mask is None else eos_mask[token.long()]
        for eos_id in eos_ids:
            eos |= token == eos_id
        terminal |= active & (eos | (count >= remaining))
        matches = (proposal.parents == current) & (ids == token)
        child = torch.where(matches, node_ids, width).amin()
        found = child < width
        accepted += (active & found).to(torch.int32)
        active = active & found & ~terminal
        current = torch.where(active, child, current)
    return Commit(tokens, accepted, count, terminal), path


def build_mtp_tree(anchor, hidden, depth, topk, budget, logits_fn, map_tokens, forward):
    def single_forward(ids, previous, paths, depths, slots):
        return forward(ids, previous, paths[0], depths[0], slots)

    return build_mtp_trees(anchor.reshape(1), hidden, depth, topk, budget,
                           logits_fn, map_tokens, single_forward)[0]


def build_mtp_trees(anchors, hidden, depth, topk, budget, logits_fn, map_tokens, forward):
    """SGLang EAGLE/NEXTN: retain K beams per request, rank all K-squared children."""
    device, bs = anchors.device, anchors.numel()
    parents = torch.full((bs, 1), -1, dtype=torch.long, device=device)
    if depth == 0:
        paths, depths = tree_paths(parents[0], 0)
        return [TreeProposal(anchor, anchors.new_empty(0), parents[i], paths, depths)
                for i, anchor in enumerate(anchors)]
    scores = torch.ones(bs, 1, device=device)
    beam = torch.zeros(bs, 1, dtype=torch.long, device=device)
    slots = torch.zeros_like(beam)
    all_tokens, all_scores = [], []
    logits = logits_fn(hidden).view(bs, 1, -1)
    for step in range(depth):
        probs, tokens = logits.float().softmax(-1).topk(topk, dim=-1)
        tokens = map_tokens(tokens)
        expanded = (scores[:, :, None] * probs).flatten(1)
        offset = parents.shape[1]
        parents = torch.cat((parents, beam.repeat_interleave(topk, dim=1)), dim=1)
        slots = torch.cat((slots, torch.zeros_like(expanded, dtype=torch.long)), dim=1)
        all_tokens.append(tokens.flatten(1))
        all_scores.append(expanded)
        if step == depth - 1:
            break
        selected = torch.argsort(expanded, dim=1, descending=True, stable=True)[:, :topk]
        scores, beam = expanded.gather(1, selected), selected + offset
        slot_ids = torch.arange(topk, device=device) + step * topk
        slots.scatter_(1, beam, slot_ids.expand(bs, -1))
        paths, depths = tree_paths(parents, depth)
        selected_paths = paths.gather(1, beam[:, :, None].expand(-1, -1, depth + 1))[:, :, 1:]
        ancestry = slots.gather(1, selected_paths.clamp_min(0).flatten(1)).view(bs, topk, depth)
        positions = depths.gather(1, beam) - 1
        previous = hidden.view(bs, -1, hidden.shape[-1])
        previous = previous.gather(1, (selected // topk)[:, :, None].expand(-1, -1, hidden.shape[-1]))
        ids = tokens.flatten(1).gather(1, selected).flatten().to(torch.int32)
        # MTP consumed the anchor at C-1; candidate KV positions lag target by one.
        hidden = forward(ids, previous.flatten(0, 1), list(ancestry), list(positions), slot_ids)
        logits = logits_fn(hidden).view(bs, topk, -1)
    all_tokens, all_scores = torch.cat(all_tokens, dim=1), torch.cat(all_scores, dim=1)
    return [select_tree(anchors[i], all_tokens[i].to(torch.int32), parents[i, 1:], all_scores[i], budget, depth)
            for i in range(bs)]


def tree_attention_metadata(prefixes, locations, paths, depths):
    """One decode query per node, with a compact prefix-plus-ancestors KV list."""
    from freetoken.attention.triton import TritonMetadata

    indices, positions, offsets = [], [], [0]
    for prefix, loc, path, depth in zip(prefixes, locations, paths, depths):
        n, span = path.shape
        indices.append(torch.cat((prefix.expand(n, -1), loc[path.clamp_min(0)]), 1).flatten())
        positions.append(prefix.numel() + depth)
        base = offsets[-1]
        offsets.extend(base + (i + 1) * (prefix.numel() + span) for i in range(n))
    device = depths[0].device
    positions = torch.cat(positions).to(torch.int32)
    cu = torch.arange(positions.numel() + 1, dtype=torch.int32, device=device)
    indptr = torch.tensor(offsets, dtype=torch.int32, device="cpu", pin_memory=device.type == "cuda")
    indptr = indptr.to(device, non_blocking=True)
    return TritonMetadata(cu_seqlens_q_gpu=cu, indptr=indptr, indices=torch.cat(indices),
                          q_to_req=cu[:-1], q_positions=positions, is_decode=True,
                          prefix_lens=positions, max_q_len=1)


class MTPTreeRuntime:
    def __init__(self, runner):
        self.runner = runner
        self.topk = runner.engine.config.speculative_eagle_topk
        self.budget = runner.engine.config.speculative_num_draft_tokens
        self.context = None

    def propose(self, anchor, depth):
        runner = self.runner

        def forward(ids, hidden, paths, depths, slots):
            self.context = (paths, depths, slots)
            try:
                batch = runner._batch(ids, runner.frontier.target_valid_len)
                with runner.engine.ctx.forward_batch(batch):
                    return runner.engine.mtp.forward(batch.input_ids, hidden)
            finally:
                self.context = None

        return build_mtp_tree(anchor, runner.last_draft_hidden, depth, self.topk, self.budget,
                              runner.engine.mtp.logits, runner.engine.mtp.map_tokens, forward)

    def batch(self, ids, start, state):
        from copy import copy

        runner = self.runner
        paths, depths, slots = self.context
        # Keep chain metadata untouched: equal row counts can alternate between
        # tree verification and reconcile in the same scheduling turn.
        batch = copy(runner._decode_batch(ids, start, state))
        table = runner.engine.page_table[runner.table_idx]
        batch.attn_metadata = tree_attention_metadata([table[:start]], [table[start:]], [paths], [depths])
        batch.positions = batch.attn_metadata.q_positions
        batch.out_loc = table[start + slots]
        batch.gdn_verify_state = state
        return batch

    def verify(self, proposal, *, forward, state, remaining, eos_ids):
        from .speculative import VerifyResult
        from freetoken.models.qwen3_5_moe.speculative_state import QwenGDNTreeVerifyState

        runner = self.runner
        start = runner.frontier.target_valid_len
        n = proposal.parents.numel()
        tree_state = QwenGDNTreeVerifyState(state, proposal.paths, proposal.depths)
        self.context = (proposal.paths, proposal.depths, torch.arange(n, device=proposal.tokens.device))
        try:
            tokens, hidden = forward(proposal.verify_inputs(), start, tree_state)
        finally:
            self.context = None
        commit, path = greedy_tree_commit(proposal, VerifyResult(tokens, hidden), remaining=remaining,
                                          eos_mask=runner.eos_mask, eos_ids=eos_ids)
        # The callers retain their chain-width contract, including short output caps.
        path = path[:state.width]
        pad = state.width - path.numel()
        path = torch.cat((path, path[-1:].expand(pad)))
        output = torch.cat((commit.tokens[:state.width], commit.tokens[-1:].expand(pad)))
        commit = Commit(output, commit.accepted_drafts,
                        commit.input_count, commit.terminal)
        tree_state.stage_commit(path)
        table = runner.engine.page_table[runner.table_idx]
        source, target = table[start + path].long(), table[start:start + path.numel()].long()
        # The final slab belongs to MTP, whose beam slots use a different mapping.
        # Reconcile rebuilds that tail from the selected target hiddens.
        kv = runner.engine.kv_cache._kv_buffer[:, :-1]
        kv[:, :, target] = kv.index_select(2, source)
        experts = runner.three_level.experts
        experts.select_tree_path(path, commit.input_count, n)
        return commit.tokens, hidden.index_select(0, path), commit
