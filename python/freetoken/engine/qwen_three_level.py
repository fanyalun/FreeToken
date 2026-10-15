from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.models.qwen3_5_moe.speculative_state import QwenGDNVerifyState
from freetoken.moe.speculative_cache import SpeculativeExpertCache
from .speculative import Proposal, SpeculativeFrontier, VerifyResult
from .speculative_window import SpeculativeWindow, WindowBudget


def route_metrics(records, loops, experts, valid_rows=None):
    predicted, actual, predicted_union, actual_union = {}, {}, set(), set()
    for record in records:
        li = record["layer"]
        if record["phase"] == "preverify":
            loop = loops[record["loop"]]
            for index, ids in enumerate(record["ids"][:loop["returned"]]):
                predicted[li, loop["offset"] + index] = set(ids)
        else:
            for index, ids in enumerate(record["ids"]):
                actual[li, index] = set(ids)
    matched, denominator, unknown = 0, 0, 0
    valid_matches, valid_compared = 0, 0
    by_loop = [{"matches": 0, "compared": 0} for _ in loops]
    for key, ids in actual.items():
        li, _ = key
        actual_union.update(li * experts + expert for expert in ids)
        if key in predicted:
            matched += len(ids & predicted[key])
            denominator += len(ids)
            if valid_rows is None or key[1] < valid_rows:
                valid_matches += len(ids & predicted[key])
                valid_compared += len(ids)
            for number, loop in enumerate(loops):
                if loop["offset"] <= key[1] < loop["offset"] + loop["returned"]:
                    by_loop[number]["matches"] += len(ids & predicted[key])
                    by_loop[number]["compared"] += len(ids)
                    break
        else:
            unknown += len(ids)
    for (li, _), ids in predicted.items():
        predicted_union.update(li * experts + expert for expert in ids)
    overlap = len(predicted_union & actual_union)
    return {"route_matches": matched, "route_compared": denominator, "routes_without_prediction": unknown,
            "route_recall": matched / denominator if denominator else None,
            "accepted_input_route_matches": valid_matches, "accepted_input_route_compared": valid_compared,
            "by_loop": by_loop,
            "predicted_unique_experts": len(predicted_union), "target_unique_experts": len(actual_union),
            "expert_intersection": overlap,
            "expert_precision": overlap / len(predicted_union) if predicted_union else None,
            "expert_recall": overlap / len(actual_union) if actual_union else None}


class ThreeLevelCoordinator:
    def __init__(self, runner):
        self.runner, self.engine = runner, runner.engine
        engine = self.engine
        cache = engine.moe_offload_cache
        if cache is None or engine.mtp is None or runner.reference_ar:
            raise ValueError("three-level decoding requires native MTP and an offload expert cache")
        mc = engine.config.model_config
        self.budget = WindowBudget.resolve(cache.cache_size, mc.num_moe_layers, mc.num_experts_per_tok,
                                           cache.num_experts, engine.config.speculative_submit_tokens,
                                           engine.config.speculative_max_loops, engine.config.speculative_mtp_depth)
        self.experts = SpeculativeExpertCache(cache, engine.stream, prefetch=engine.config.speculative_prefetch,
                                               use_graph=runner.use_graph, audit=engine.config.speculative_audit,
                                               max_rows=max(4, self.budget.submit_rows), top_k=mc.num_experts_per_tok)
        pool = engine.linear_state_pool
        self.private_pool = SimpleNamespace(
            recurrent_states=torch.empty_like(pool.recurrent_states[:, 1:2]),
            conv_states=torch.empty_like(pool.conv_states[:, 1:2]),
            local_index=pool.local_index, num_slots=1,
        )
        layers = [layer.linear_attn for layer in engine.model.model.layers.op_list if layer._is_linear]
        self.private_states = {w: QwenGDNVerifyState(self.private_pool, layers, w, 0) for w in range(1, 5)}
        for width in range(5, self.budget.submit_rows + 1):
            runner.states[width] = QwenGDNVerifyState(pool, layers, width, 1)
        runner.max_verify_width = max(4, self.budget.submit_rows)
        self.epoch = 0

    def _forward(self, inputs, start, state):
        runner = self.runner
        batch = runner._batch(inputs, start, state=state)
        # Host route admission splits the target forward; only expert compute is graphed.
        runner.graph_fallbacks["target_host_schedule"] = runner.graph_fallbacks.get("target_host_schedule", 0) + 1
        with self.engine.model.forward_host_ctx(batch, False):
            return runner._target(batch)

    def step(self, anchor, remaining, eos_ids, cancelled):
        runner, engine = self.runner, self.engine
        exact = runner.frontier
        start = exact.target_valid_len
        self.epoch += 1
        window = SpeculativeWindow(self.epoch, start, min(self.budget.candidates, remaining - 1))
        pool = engine.linear_state_pool
        self.private_pool.recurrent_states.copy_(pool.recurrent_states[:, 1:2])
        self.private_pool.conv_states.copy_(pool.conv_states[:, 1:2])
        private = SpeculativeFrontier(exact.page_size, start, start, exact.allocated_len)
        first_anchor, current = anchor.clone(), anchor.clone()
        self.experts.begin(self.epoch)
        row = None
        graph_start = self.experts.graph_replays
        flush_reason = "output_or_submit_limit"
        try:
            while window.next_width(engine.config.speculative_mtp_depth, remaining) > 0 and not cancelled():
                width = window.next_width(engine.config.speculative_mtp_depth, remaining)
                ledger = self.experts.ledger
                previous_experts = len(ledger.refs)
                stop_reason = self.budget.stop_after_loop(len(window.loops) + 1, previous_experts, width)
                runner.frontier = private
                proposal = runner.coordinator.prepare(private, current, width - 1)
                inputs = proposal.verify_inputs()
                self.experts.set_phase("preverify", len(window.loops))
                tokens, hidden = self._forward(inputs, private.target_valid_len, self.private_states[width])
                verified = VerifyResult(tokens, hidden)
                commit = runner._accept(proposal, verified, width, eos_ids)
                self.private_states[width].commit(commit.input_count)
                drained = torch.cat((tokens, commit.input_count.reshape(1), commit.accepted_drafts.reshape(1),
                                     commit.terminal.reshape(1))).tolist()
                count, accepted = drained[-3:-1]
                self.experts.decide(len(window.loops), count)
                window.append(drained[:count], {"drafted": width - 1, "accepted": accepted,
                                                "accepted_before_stop": min(accepted, count - 1),
                                                "previous_unique_experts": previous_experts, "verify_rows": width,
                                                "worst_case_experts": previous_experts + width * self.budget.routes_per_row,
                                                "stop_after_loop": stop_reason})
                current = tokens[count - 1:count].clone()
                inner_eos = drained[count - 1] in eos_ids
                reconcile_inner = not inner_eos and not stop_reason and window.next_width(engine.config.speculative_mtp_depth, remaining) > 0 and not cancelled()
                if reconcile_inner:
                    runner.coordinator.proposer.reconcile(hidden, commit)
                private.commit_target(count)
                if reconcile_inner:
                    private.reconcile_done()
                else:
                    private.in_flight = False
                    private.check()
                if stop_reason:
                    flush_reason = stop_reason
                if inner_eos or stop_reason:
                    break
            runner.frontier = exact
            if cancelled():
                return [], anchor, {"cancelled": True}
            candidates = torch.tensor(window.candidates, dtype=anchor.dtype, device=anchor.device)
            proposal = Proposal(first_anchor, candidates)
            inputs = proposal.verify_inputs()
            end = runner._reserve(exact.reserve_end(len(inputs)))
            exact.begin(end, len(inputs))
            self.experts.set_phase("verify")
            state = runner.states[len(inputs)]
            tokens, hidden = self._forward(inputs, start, state)
            verified = VerifyResult(tokens, hidden)
            commit = runner._accept(proposal, verified, remaining, eos_ids)
            runner._commit_state(state, commit.input_count)
            drained = torch.cat((tokens, commit.input_count.reshape(1), commit.accepted_drafts.reshape(1),
                                 commit.terminal.reshape(1))).tolist()
            count, accepted, terminal = drained[-3:]
            reconcile = not terminal and not cancelled()
            if reconcile:
                runner.coordinator.proposer.reconcile(hidden, commit)
            runner.last_event = torch.cuda.Event()
            runner.last_event.record(engine.stream)
            from .qwen_mtp import _EARLY_COMMIT
            if not _EARLY_COMMIT:
                runner.last_event.synchronize()
            # Inner rounds can reserve beyond the outer accepted prefix.
            exact.allocated_len = runner._reserved_len
            runner.coordinator.drain(exact, count, reconciled=reconcile)
            row = {"drafted": len(candidates), "accepted": accepted, "returned": count,
                   "submit_rows": len(inputs), "submit_row_limit": self.budget.submit_rows,
                   "max_loops": self.budget.max_loops, "cache_slots": self.budget.cache_slots,
                   "loop_count": len(window.loops),
                   "loops": window.survival(min(accepted, count)), "epoch": self.epoch,
                   "flush_reason": "provisional_eos" if window.candidates and window.candidates[-1] in eos_ids else
                                   "single_row_submit" if not window.candidate_limit else flush_reason,
                   "route_metrics": route_metrics(self.experts.records, window.loops, self.experts.cache.num_experts, count)}
            if self.experts.audit:
                row["routes"] = self.experts.records
            return drained[:count], tokens[count - 1:count].clone(), row
        finally:
            runner.frontier = exact
            self.experts.close()
            if row is not None:
                row["cache_stats"] = dict(self.experts.stats)
                row["expert_graph_replays"] = self.experts.graph_replays - graph_start
                row["prefetched_unique_experts"] = len(self.experts.prefetched)
                row["unused_prefetched_unique_experts"] = len(self.experts.prefetched - self.experts.used_prefetch)
                if self.experts.audit:
                    row["transfer_audit"] = self.experts.transfer_audit()

    def invalidate(self):
        if self.experts.active:
            raise RuntimeError("cannot invalidate an active three-level window")
        self.experts.graphs.clear()
        self.private_states.clear()
        self.private_pool = None
