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
        if runner.active or getattr(runner, "coarse_graphs", None) is not None:
            raise RuntimeError("coordinator replacement requires an idle runner without captured coarse/full graphs")
        self.runner, self.engine = runner, runner.engine
        engine = self.engine
        cache = engine.moe_offload_cache
        if cache is None or engine.mtp is None or runner.reference_ar:
            raise ValueError("three-level decoding requires native MTP and an offload expert cache")
        mc = engine.config.model_config
        hybrid = getattr(engine.config, "moe_strategy", "offload") == "hybrid" and engine.config.speculative_expert_scheduler == "gpu"
        hybrid_phases = tuple(engine.config.speculative_hybrid_phases) if hybrid else ()
        self.draft_width = runner.draft_width
        coarse = getattr(engine.config, "speculative_graph_implementation", "reference") in ("coarse", "full")
        self.budget = WindowBudget.resolve(cache.cache_size, mc.num_moe_layers, mc.num_experts_per_tok,
                                           cache.num_experts, min(engine.config.speculative_submit_tokens or
                                               engine.config.speculative_max_output_tokens,
                                               engine.config.speculative_max_output_tokens),
                                           engine.config.speculative_max_loops, engine.config.speculative_mtp_depth,
                                           cpu_fallback="preverify" in hybrid_phases)
        self.inner_width, self.outer_width = self.budget.reachable_widths(engine.config.speculative_mtp_depth)
        if coarse and max(self.draft_width, self.budget.submit_rows) > engine.max_seq_len:
            raise ValueError("speculative graph width exceeds context capacity")
        cache_type = SpeculativeExpertCache
        extra = {}
        if getattr(engine.config, "speculative_expert_scheduler", "host") == "gpu":
            from freetoken.moe.speculative_gpu import GPUExpertCache
            cache_type = GPUExpertCache
            extra = dict(max_loops=self.budget.max_loops, max_rows=self.outer_width,
                         hybrid_phases=hybrid_phases,
                         hybrid_max_fetch=engine.config.moe_hybrid_max_fetch,
                         inner_rows=max(1, self.inner_width),
                         layer_quotas=getattr(engine.config, "speculative_prefetch_layer_quotas", None),
                         demand_fence=getattr(engine.config, "speculative_prefetch_demand_fence", False),
                         verify_layer_quota=getattr(engine.config, "speculative_verify_prefetch_layer_experts", 0),
                         verify_layer_quotas=getattr(engine.config, "speculative_verify_prefetch_layer_quotas", None),
                         top_k=mc.num_experts_per_tok,
                         compact_control=getattr(engine.config, "speculative_compact_control", True),
                         implementation=getattr(engine.config, "speculative_execution_implementation", "reference"))
        self.experts = cache_type(cache, engine.stream, prefetch=engine.config.speculative_prefetch,
                                               use_graph=runner.use_graph, audit=engine.config.speculative_audit,
                                               layer_quota=engine.config.speculative_prefetch_layer_experts,
                                               gap_capacity=engine.config.speculative_prefetch_gap_capacity, **extra)
        if hybrid_phases:
            self.experts.execution_mode = (engine.config.speculative_graph_implementation
                                           if runner.use_graph else "eager")
            profile_path = engine.config.speculative_hybrid_profile
            if engine.config.moe_cache_auto and (getattr(engine, "baseline_sd_memory_plan", None) is None
                                                 or getattr(engine, "_sd_sizing", False)):
                profile_path = None
            self.experts.prepare_hybrid(engine.speculative_hybrid_layer,
                profile_path)
        self.verify_prefetch_signature = self._verify_prefetch_signature()
        self.compact_control = getattr(self.experts, "compact_control", False)
        pool = engine.linear_state_pool
        self.private_pool = SimpleNamespace(
            recurrent_states=torch.empty_like(pool.recurrent_states[:, 1:2]),
            conv_states=torch.empty_like(pool.conv_states[:, 1:2]),
            local_index=pool.local_index, num_slots=1,
        )
        layers = [layer.linear_attn for layer in engine.model.model.layers.op_list if layer._is_linear]
        runner.max_verify_width = self.outer_width
        self.private_states = {}
        if not coarse:
            workspace = runner._gdn_workspace(runner.max_verify_width, layers)
            self.private_states = {w: QwenGDNVerifyState(self.private_pool, layers, w, 0, workspace=workspace)
                                   for w in range(1, self.inner_width + 1)}
        device = engine.device
        self.inputs = torch.empty(runner.max_verify_width, dtype=torch.int32, device=device)
        self.first_anchor = torch.empty(1, dtype=torch.int32, device=device)
        self.current = torch.empty_like(self.first_anchor)
        self.candidates = torch.empty(self.budget.candidates, dtype=torch.int32, device=device)
        self.candidate_host = torch.empty(self.budget.candidates, dtype=torch.int32,
                                          pin_memory=device.type == "cuda")
        self.epoch = 0

    def prepare_runtime(self):
        runner, engine = self.runner, self.engine
        with torch.cuda.stream(engine.stream):
            try:
                runner.eos_mask.zero_()
                capacity = engine.max_seq_len
                runner._reserve((capacity + engine.config.page_size - 1) // engine.config.page_size * engine.config.page_size)
                engine.kv_cache._kv_buffer.zero_()
                engine.linear_state_pool.recurrent_states.zero_()
                engine.linear_state_pool.conv_states.zero_()
                self.private_pool.recurrent_states.zero_()
                self.private_pool.conv_states.zero_()
                if runner.coarse_graphs is None:
                    for phase, states in (("preverify", self.private_states), ("verify", runner.states)):
                        for width, state in states.items():
                            if phase == "verify" and width > self.outer_width:
                                continue
                            self.experts.begin(0)
                            self.experts.set_phase(phase)
                            ids = self.inputs[:width].zero_()
                            tokens, hidden = self._forward(ids, capacity - width, state)
                            commit = runner._accept(Proposal(ids[:1], ids[1:]), VerifyResult(tokens, hidden), width, ())
                            if phase == "verify":
                                runner._commit_state(state, commit.input_count)
                            else:
                                state.commit(commit.input_count)
                            if phase == "preverify":
                                self.experts.finish_loop(True)
                            runner._execute("reconcile", runner._batch(ids, capacity - width), hidden,
                                            count=commit.input_count)
                            self.experts.close()
                    runner._execute("chain", runner._batch(self.inputs[:1], capacity - 1), runner.seed_hidden.zero_())
                runner._prefill(torch.zeros(engine.config.speculative_max_input_tokens, dtype=torch.int32), None)
            finally:
                self.experts.close()
                engine.stream.synchronize()
                engine.linear_state_pool.recurrent_states.zero_()
                engine.linear_state_pool.conv_states.zero_()
                engine.moe_offload_cache.reset()
                engine.stream.synchronize()
                runner._reserved_len = 0
                runner.next_draft = runner.last_draft_hidden = None
                runner.graph_replays, runner.graph_fallbacks = {}, {}

    def _verify_prefetch_signature(self):
        c = self.engine.config
        fixed_prefetch = (getattr(c, "speculative_graph_implementation", "reference") != "reference"
                          or getattr(c, "speculative_verify_prefetch_layer_experts", 0)
                          or any(getattr(c, "speculative_verify_prefetch_layer_quotas", None) or ())
                          or getattr(self.experts, "verify_prefetch_enabled", False))
        # Reference GPU forwards key their graphs by prefetch; host admission is eager.
        prefetch = (getattr(c, "speculative_prefetch", True), getattr(self.experts, "prefetch", True))
        return (getattr(c, "speculative_verify_prefetch_layer_experts", 0),
                tuple(getattr(c, "speculative_verify_prefetch_layer_quotas", None) or ()),
                prefetch if fixed_prefetch else None,
                getattr(self.experts, "verify_layer_quotas", None),
                getattr(self.experts, "verify_prefetch_enabled", False),
                tuple(getattr(c, "speculative_hybrid_phases", ())),
                getattr(c, "speculative_hybrid_profile", None), getattr(c, "moe_hybrid_max_fetch", -1),
                getattr(self.experts, "hybrid_phases", ()), getattr(self.experts, "hybrid_max_fetch", -1))

    def _forward(self, inputs, start, state):
        if self._verify_prefetch_signature() != self.verify_prefetch_signature:
            raise RuntimeError("verify prefetch configuration changed; rebuild the idle runner")
        runner = self.runner
        batch = runner._batch(inputs, start, state=state)
        if getattr(self.experts, "gpu_schedule", False):
            if self.experts.cross_layer_prefetch:
                self.experts.join_gap()
            else:
                self.engine.stream.wait_stream(self.experts.copy_stream)
            phase = f"three_level_{self.experts.phase}_{int(self.experts.prefetch)}"
            if self.compact_control:
                phase += "_compact"
            phase += f"_{self.experts.implementation}_{id(self)}"
            result = runner._execute(phase, batch)
            if not self.compact_control:
                self.experts.check_error()
            return result
        # Host route admission splits the target forward; only expert compute is graphed.
        runner.graph_fallbacks["target_host_schedule"] = runner.graph_fallbacks.get("target_host_schedule", 0) + 1
        profile = getattr(self.experts, "profiler", None)
        if profile is not None and self.experts.phase == "preverify":
            profile.verify_start(self.experts.loop, len(inputs), self.engine.stream)
        with self.engine.model.forward_host_ctx(batch, False):
            result = runner._target(batch)
        if profile is not None and self.experts.phase == "preverify":
            profile.verify_end(self.experts.loop, len(inputs), self.engine.stream)
        return result

    def _candidate_tokens(self, candidates, anchor):
        if getattr(self.experts, "stage_candidates", False):
            return self.experts.candidate_buffer[:len(candidates)]
        for row, token in enumerate(candidates):
            self.candidate_host[row] = token
        self.candidates[:len(candidates)].copy_(self.candidate_host[:len(candidates)], non_blocking=True)
        return self.candidates[:len(candidates)]

    def step(self, anchor, remaining, eos_ids, cancelled):
        runner, engine = self.runner, self.engine
        coarse = getattr(runner, "coarse_graphs", None)
        if coarse is not None:
            coarse.seed.copy_(runner.next_draft)
            coarse.seed_hidden.copy_(runner.last_draft_hidden)
        exact = runner.frontier
        start = exact.target_valid_len
        self.epoch += 1
        window = SpeculativeWindow(self.epoch, start, min(self.budget.candidates, remaining - 1))
        pool = engine.linear_state_pool
        private = SpeculativeFrontier(exact.page_size, start, start, exact.allocated_len)
        self.first_anchor.copy_(anchor)
        self.current.copy_(anchor)
        first_anchor, current = self.first_anchor, self.current
        self.experts.begin(self.epoch)
        row = None
        completed = False
        graph_start = self.experts.graph_replays
        flush_reason = "output_or_submit_limit"
        width = window.next_width(engine.config.speculative_mtp_depth, remaining)
        capacity = (dict(protected_experts=0, blocked_slots=0, available_slots=self.budget.cache_slots)
                    if self.compact_control else self.experts.ledger.capacity())
        stop_reason = self.budget.before_loop(0, capacity["protected_experts"], width, capacity["blocked_slots"])
        initial_admission = {**capacity, "verify_rows": width, "required_slots": width * self.budget.routes_per_row,
                             "stop_reason": stop_reason}
        try:
            while width > 0 and not stop_reason and not cancelled():
                if not window.loops:
                    self.private_pool.recurrent_states.copy_(pool.recurrent_states[:, 1:2])
                    self.private_pool.conv_states.copy_(pool.conv_states[:, 1:2])
                previous_experts = capacity["protected_experts"]
                runner.frontier = private
                if coarse is None:
                    proposal = runner.coordinator.prepare(private, current, width - 1)
                    inputs = self.inputs[:width]
                    inputs[:1].copy_(proposal.anchor)
                    inputs[1:].copy_(proposal.tokens)
                else:
                    end = runner._reserve(private.reserve_end(width))
                    private.begin(end, width)
                self.experts.input_offset = len(window.candidates)
                self.experts.set_phase("preverify", len(window.loops))
                if coarse is None:
                    tokens, hidden = self._forward(inputs, private.target_valid_len, self.private_states[width])
                    verified = VerifyResult(tokens, hidden)
                    commit = runner._accept(proposal, verified, width, eos_ids)
                else:
                    tokens, hidden, commit, control = coarse.body(current, inner=True,
                        start=private.target_valid_len, rows=width, loop=len(window.loops),
                        candidates=len(window.candidates), limit=window.candidate_limit, remaining=remaining)
                if self.compact_control:
                    if coarse is None:
                        control = self.experts.read_control(tokens, commit, loop=len(window.loops),
                            candidates=len(window.candidates), candidate_limit=window.candidate_limit,
                            remaining=remaining, depth=engine.config.speculative_mtp_depth)
                    count, accepted = control[:2]
                    drained = control[8:]
                    # Invalid forwards must never update private or canonical GDN state.
                    if coarse is None:
                        self.private_states[width].commit(commit.input_count)
                else:
                    self.private_states[width].commit(commit.input_count)
                    drained = torch.cat((tokens, commit.input_count.reshape(1), commit.accepted_drafts.reshape(1),
                                         commit.terminal.reshape(1))).tolist()
                    count, accepted = drained[-3:-1]
                    self.experts.decide(len(window.loops), count)
                loop = {"drafted": width - 1, "accepted": accepted,
                        "accepted_before_stop": min(accepted, count - 1),
                        "previous_unique_experts": previous_experts, "verify_rows": width,
                        "blocked_slots": capacity["blocked_slots"],
                        "worst_case_experts": previous_experts + width * self.budget.routes_per_row}
                window.append(drained[:count], loop)
                if self.compact_control:
                    width = control[5]
                    capacity = dict(protected_experts=control[6], blocked_slots=control[7],
                                    available_slots=self.budget.cache_slots - control[6] - control[7])
                    stop_reason = (None, "loop_limit", "cache_pressure")[control[4]]
                else:
                    width = window.next_width(engine.config.speculative_mtp_depth, remaining)
                    capacity = self.experts.ledger.capacity()
                    stop_reason = self.budget.before_loop(len(window.loops), capacity["protected_experts"], width,
                                                          capacity["blocked_slots"])
                window.loops[-1].update(next_admission={**capacity, "verify_rows": width,
                                                       "required_slots": width * self.budget.routes_per_row,
                                                       "stop_reason": stop_reason}, stop_after_loop=stop_reason)
                self.current.copy_(tokens[count - 1:count])
                current = self.current
                inner_eos = drained[count - 1] in eos_ids
                reconcile_inner = not inner_eos and not stop_reason and width > 0 and not cancelled()
                if coarse is not None:
                    coarse.tail(True, reconcile_inner)
                else:
                    self.experts.finish_loop(reconcile_inner)
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
            if stop_reason:
                flush_reason = stop_reason
            runner.frontier = exact
            if cancelled():
                return [], anchor, {"cancelled": True}
            verify_rows = len(window.candidates) + 1
            if coarse is None:
                candidates = self._candidate_tokens(window.candidates, anchor)
                proposal = Proposal(first_anchor, candidates)
                inputs = self.inputs[:verify_rows]
                inputs[:1].copy_(proposal.anchor)
                inputs[1:].copy_(proposal.tokens)
            end = runner._reserve(exact.reserve_end(verify_rows))
            exact.begin(end, verify_rows)
            self.experts.set_phase("verify")
            if getattr(self.experts, "gpu_schedule", False):
                self.experts.loop_count = len(window.loops)
                self.experts.verify_rows = verify_rows
            if coarse is None:
                state = runner.states[verify_rows]
                tokens, hidden = self._forward(inputs, start, state)
                verified = VerifyResult(tokens, hidden)
                commit = runner._accept(proposal, verified, remaining, eos_ids)
            else:
                tokens, hidden, commit, control = coarse.body(first_anchor, inner=False,
                    start=start, rows=verify_rows, remaining=remaining)
            if self.compact_control:
                if coarse is None:
                    control = self.experts.read_control(tokens, commit)
                count, accepted, terminal = control[:3]
                drained = control[8:]
                if coarse is None:
                    runner._commit_state(state, commit.input_count)
            else:
                runner._commit_state(state, commit.input_count)
                drained = torch.cat((tokens, commit.input_count.reshape(1), commit.accepted_drafts.reshape(1),
                                     commit.terminal.reshape(1))).tolist()
                count, accepted, terminal = drained[-3:]
            reconcile = not terminal and not cancelled()
            if coarse is not None:
                coarse.tail(False, reconcile)
            elif reconcile:
                runner.coordinator.proposer.reconcile(hidden, commit)
            runner.last_event = runner.reader_event
            runner.last_event.record(engine.stream)
            from .qwen_mtp import _EARLY_COMMIT
            if not _EARLY_COMMIT:
                runner.last_event.synchronize()
            # Inner rounds can reserve beyond the outer accepted prefix.
            exact.allocated_len = runner._reserved_len
            runner.coordinator.drain(exact, count, reconciled=reconcile, retain_reservation=True)
            collect_stats = not self.compact_control or self.experts.audit
            if getattr(self.experts, "gpu_schedule", False) and collect_stats:
                self.experts.export_records()
            row = {"drafted": len(window.candidates), "accepted": accepted, "returned": count,
                   "submit_rows": verify_rows, "submit_row_limit": self.budget.submit_rows,
                   "max_loops": self.budget.max_loops, "cache_slots": self.budget.cache_slots,
                   "target_reserve_slots": 0, "initial_admission": initial_admission,
                   "loop_count": len(window.loops),
                   "loops": window.survival(min(accepted, count)), "epoch": self.epoch,
                   "flush_reason": "provisional_eos" if window.candidates and window.candidates[-1] in eos_ids else
                                   "single_row_submit" if not window.candidate_limit else flush_reason}
            if collect_stats:
                row["route_metrics"] = route_metrics(self.experts.records, window.loops,
                                                     self.experts.cache.num_experts, count)
            if self.experts.audit:
                row["routes"] = self.experts.records
            completed = True
            runner.anchor_buffer.copy_(tokens[count - 1:count])
            return drained[:count], runner.anchor_buffer, row
        finally:
            runner.frontier = exact
            if self.compact_control:
                self.experts.close(abort=not completed)
            else:
                self.experts.close()
            if row is not None and (not self.compact_control or self.experts.audit):
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
