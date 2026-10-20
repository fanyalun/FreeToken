from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from bench_sparse_verify_motivation import CacheSnapshot, WindowExperiment, accepted_prefix
from qwen_mtp_config import prepare_experiment
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.speculative import Proposal, VerifyResult, SpeculativeFrontier, greedy_commit
from freetoken.mm.config import MultimodalConfig
from freetoken.scheduler.config import SchedulerConfig


def loop_candidates(inputs, predictions):
    accepted = accepted_prefix(inputs[1:], predictions)
    return inputs[1:accepted + 1] + [predictions[accepted]], accepted


def committed_route_rows(loop):
    return loop["record"]["layers"][0]["ids"][:loop["returned"]]


def widen_state(state, width):
    result = copy.copy(state)
    result.width = width
    result.cu_seqlens = torch.tensor([0, width], dtype=torch.int32, device=state.cu_seqlens.device)
    old = state.conv_windows
    result.conv_windows = old.new_empty(old.shape[0], old.shape[1], old.shape[2] - state.width + width)
    result.replay_inputs = tuple(x.new_empty(x.shape[0], width, *x.shape[2:]) for x in state.replay_inputs)
    result.snapshots, result.snapshot_index = None, None
    return result


class RouteRecord:
    def __init__(self, layers, width, device):
        self.ids = torch.empty((layers, width, 8), dtype=torch.int32, device=device)
        self.weights = torch.empty_like(self.ids, dtype=torch.float32)
        self.hits = torch.empty_like(self.ids, dtype=torch.bool)
        self.keep = torch.empty_like(self.hits)
        self.gpu_slots = torch.empty_like(self.ids)
        self.hidden = torch.empty((layers, width, 2048), dtype=torch.bfloat16, device=device)
        self.fetched = torch.empty((layers, max(32, width * 8)), dtype=torch.int32, device=device)
        self.count = torch.empty(layers, dtype=torch.int64, device=device)
        event = lambda: torch.cuda.Event(enable_timing=True, external=True)
        self.start, self.end = event(), event()
        self.ready = [event() for _ in range(layers)]
        self.copy_start = [event() for _ in range(layers)]
        self.copy_end = [event() for _ in range(layers)]

    def export(self):
        ids, weights, hits, keep, slots, fetched, count = [x.cpu().tolist() for x in (
            self.ids, self.weights, self.hits, self.keep, self.gpu_slots, self.fetched, self.count,
        )]
        rows = []
        for layer in range(len(ids)):
            assert 0 <= count[layer] <= len(fetched[layer])
            rows.append({
                "layer": layer, "ids": ids[layer], "weights": weights[layer], "hits": hits[layer],
                "keep": keep[layer], "gpu_slots": slots[layer], "fetched_ids": fetched[layer][:count[layer]],
                "ready_ms": self.start.elapsed_time(self.ready[layer]),
                "copy_start_ms": self.start.elapsed_time(self.copy_start[layer]),
                "copy_end_ms": self.start.elapsed_time(self.copy_end[layer]),
            })
        return {"stream_ms": self.start.elapsed_time(self.end), "layers": rows}


class ModelSnapshot:
    def __init__(self, engine, runner, start, end):
        self.engine, self.runner = engine, runner
        self.start, self.end = start, end
        pool = engine.linear_state_pool
        self.recurrent = pool.recurrent_states[:, 1].clone()
        self.conv = pool.conv_states[:, 1].clone()
        self.seed, self.hidden = runner.next_draft.clone(), runner.last_draft_hidden.clone()
        self.frontier = dataclasses.replace(runner.frontier)
        self.page_table = engine.page_table[0].clone()
        self.reserved = runner._reserved_len
        self.current, self.last_event = runner.model_state.current, runner.last_event
        ids = engine.config.model_config.kv_cache_group_specs()[0].layer_ids
        self.kv = []
        for layer in (*ids, engine.config.model_config.num_layers):
            for bank in (engine.kv_cache.k_cache(layer), engine.kv_cache.v_cache(layer)):
                view = bank.flatten(0, 1)[start:end]
                self.kv.append((view, view.clone()))

    def restore(self):
        pool, runner = self.engine.linear_state_pool, self.runner
        pool.recurrent_states[:, 1].copy_(self.recurrent)
        pool.conv_states[:, 1].copy_(self.conv)
        runner.next_draft, runner.last_draft_hidden = self.seed.clone(), self.hidden.clone()
        runner.frontier = dataclasses.replace(self.frontier)
        self.engine.page_table[0].copy_(self.page_table)
        runner._reserved_len = self.reserved
        runner.model_state.current, runner.last_event = self.current, self.last_event
        for view, value in self.kv:
            view.copy_(value)


class MultiloopExperiment(WindowExperiment):
    def __init__(self, engine, runner, output, repeats):
        super().__init__(engine, runner, output, [], repeats)
        self.aux_graphs = {}
        self.saved_probes = set()
        self.commit_warm_widths = set()
        self.quality_checks = []

        def copy_weights():
            record = self.record
            if record is None:
                return self.original_copy()
            layer = self.cache._pending_src_layer
            _, src, count = self.cache._pending_copy_plan or (
                self.cache.evict_slots, self.cache.src_indices, self.cache.num_indices,
            )
            record.count[layer].copy_(count.sum())
            size = min(record.fetched.shape[1], src.numel())
            record.fetched[layer, :size].copy_(src[:size])
            record.copy_start[layer].record()
            self.original_copy()
            record.copy_end[layer].record()

        self.cache.copy_missing = copy_weights

    def state(self, width):
        if width not in self.runner.states:
            self.runner.states[width] = widen_state(self.runner.states[4], width)
        return self.runner.states[width]

    def graph(self, policy, inputs, start, audit=False):
        width = inputs.numel()
        state = self.state(width)
        batch = self.runner._batch(inputs, start, state=state)
        key = policy, width, audit
        if key not in self.graphs:
            record = RouteRecord(40, width, self.engine.device) if audit else None
            self.policy, self.record = policy, record

            def run():
                if record is not None:
                    record.start.record()
                result = self.runner._target(batch)
                if record is not None:
                    record.end.record()
                return result

            with self.engine.model.forward_host_ctx(batch, True):
                run()
                self.engine.stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.engine.stream):
                    result = run()
            self.graphs[key] = graph, result, record
            self.policy, self.record = "full", None
        return self.graphs[key]

    def run_target(self, policy, inputs, start, audit=True):
        snapshot = CacheSnapshot(self.cache)
        predictions, times = self.measure(policy, inputs, start, snapshot)
        if audit:
            tokens, _, record = self.replay(policy, inputs, start, snapshot, audit=True)
            assert tokens.cpu().tolist() == predictions, "route logging changed model predictions"
        else:
            record = None
        hidden = self.graphs[(policy, len(inputs), audit)][1][1].clone()
        for layer in (record or {}).get("layers", []):
            keep = {expert for ids, selected in zip(layer["ids"], layer["keep"])
                    for expert, retained in zip(ids, selected) if retained}
            assert len(layer["fetched_ids"]) == len(set(layer["fetched_ids"]))
            assert set(layer["fetched_ids"]) <= keep
        return torch.tensor(predictions, dtype=torch.int32, device=self.engine.device), hidden, times, record

    def timed(self, function):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(self.engine.stream)
        result = function()
        end.record(self.engine.stream)
        end.synchronize()
        return result, begin.elapsed_time(end)

    def reconcile(self, hidden, commit):
        width = len(commit.tokens)
        if width <= 4:
            return self.runner.coordinator.proposer.reconcile(hidden, commit)
        runner = self.runner
        batch = runner._batch(commit.tokens, runner.frontier.target_valid_len)
        if width not in self.aux_graphs:
            static_hidden, count = hidden.clone(), commit.input_count.clone()
            with self.engine.model.forward_host_ctx(batch, True):
                runner._draft(batch, static_hidden, count)
                self.engine.stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.engine.stream):
                    result = runner._draft(batch, static_hidden, count)
            self.aux_graphs[width] = graph, static_hidden, count, result
        graph, static_hidden, count, result = self.aux_graphs[width]
        static_hidden.copy_(hidden)
        count.copy_(commit.input_count)
        with self.engine.model.forward_host_ctx(batch, True):
            graph.replay()
        runner.next_draft, runner.last_draft_hidden = result

    def commit(self, proposal, tokens, hidden, *, remaining=None):
        remaining = remaining or len(tokens)
        commit = self.runner._accept(proposal, VerifyResult(tokens, hidden), remaining, ())
        self.runner.model_state.current = self.state(len(tokens))
        self.runner.model_state.commit(commit.input_count)
        self.reconcile(hidden, commit)
        return commit

    def warm_commit(self, proposal, tokens, hidden, remaining):
        width = len(tokens)
        if width in self.commit_warm_widths:
            return
        pool, runner = self.engine.linear_state_pool, self.runner
        recurrent, conv = pool.recurrent_states[:, 1].clone(), pool.conv_states[:, 1].clone()
        seed, draft_hidden = runner.next_draft.clone(), runner.last_draft_hidden.clone()
        start = runner.frontier.target_valid_len
        layer = self.engine.config.model_config.num_layers
        banks = [bank.flatten(0, 1)[start:start + width] for bank in (
            self.engine.kv_cache.k_cache(layer), self.engine.kv_cache.v_cache(layer),
        )]
        copies = [bank.clone() for bank in banks]
        self.commit(proposal, tokens, hidden, remaining=remaining)
        pool.recurrent_states[:, 1].copy_(recurrent)
        pool.conv_states[:, 1].copy_(conv)
        for bank, saved in zip(banks, copies):
            bank.copy_(saved)
        runner.next_draft, runner.last_draft_hidden = seed, draft_hidden
        self.commit_warm_widths.add(width)

    def drain(self, commit):
        count = int(commit.input_count.item())
        self.runner.last_event = torch.cuda.Event()
        self.runner.last_event.record(self.engine.stream)
        self.runner.coordinator.drain(self.runner.frontier, count, retain_reservation=self.runner.fixed_mapping)
        return count

    def loop(self, anchor, *, policy="cheap", remaining=4, audit=True):
        runner = self.runner
        start = runner.frontier.target_valid_len
        width = min(4, remaining)
        proposal, draft_ms = self.timed(lambda: runner.coordinator.prepare(runner.frontier, anchor, width - 1))
        inputs = proposal.verify_inputs()
        tokens, hidden, times, record = self.run_target(policy, inputs, start, audit)
        # Capture setup is excluded from component timing for both the method and its control.
        self.runner._accept(proposal, VerifyResult(tokens, hidden), remaining, ())
        self.warm_commit(proposal, tokens, hidden, remaining)
        commit, commit_ms = self.timed(lambda: self.commit(proposal, tokens, hidden, remaining=remaining))
        count = self.drain(commit)
        candidate = tokens[:count].cpu().tolist()
        next_anchor = tokens[count - 1:count].clone()
        row = {"start": start, "inputs": inputs.cpu().tolist(), "predictions": tokens.cpu().tolist(),
               "tokens": candidate, "accepted": int(commit.accepted_drafts.item()), "returned": count,
               "draft_ms": draft_ms, "verify_ms": times, "commit_reconcile_ms": commit_ms,
               "component_ms": draft_ms + statistics.median(times) + commit_ms,
               "record": record}
        return next_anchor, row

    def save_probe(self, sample, prefix, loop_index, inputs):
        if sample != 0 or prefix != 96 or loop_index in self.saved_probes:
            return
        self.saved_probes.add(loop_index)
        record = self.graphs[("cheap", len(inputs), True)][2]
        layer = 24
        ids = record.ids[layer].cpu().long()
        selected, inverse = ids.unique(sorted=True, return_inverse=True)
        banks = [source[layer].index_select(0, selected).clone() for source, _ in self.cache.banks]
        torch.save({"hidden": record.hidden[layer].cpu(), "weights": record.weights[layer].cpu(),
                    "hidden_all": record.hidden.cpu(),
                    "keep": record.keep[layer].cpu(), "ids": inverse.reshape_as(ids).int(), "banks": banks,
                    "bank_schema": self.cache.bank_schema, "sample_id": sample, "prefix": prefix,
                    "loop": loop_index, "layer": layer}, self.output / f"probe_loop{loop_index}.pt")

    def save_workload(self, sample, prefix, loops):
        directory = self.output / "probe_workload"
        if sample != 0 or prefix != 96 or len(loops) != 4 or directory.exists():
            return
        directory.mkdir()
        metadata = []
        for layer in range(40):
            ids = sorted({expert for loop in loops for row in loop["record"]["layers"][layer]["ids"] for expert in row})
            selected = torch.tensor(ids, dtype=torch.int64)
            banks = [source[layer].index_select(0, selected).clone() for source, _ in self.cache.banks]
            torch.save({"ids": ids, "banks": banks}, directory / f"layer_{layer:02}.pt")
            metadata.append({"layer": layer, "experts": len(ids), "bytes": sum(x.numel() * x.element_size() for x in banks)})
        (directory / "config.json").write_text(json.dumps({"sample_id": sample, "prefix": prefix,
                                                           "loop_count": 4, "layers": metadata}, indent=2) + "\n")

    def outer(self, original, after_cache, anchor, candidates, start):
        original.restore()
        after_cache.restore(self.cache)
        inputs = torch.tensor([int(anchor.item()), *candidates], dtype=torch.int32, device=self.engine.device)
        page = self.engine.config.page_size
        self.runner._reserve((start + len(inputs) + page - 1) // page * page)
        self.runner.frontier.begin(self.runner._reserved_len, len(inputs))
        tokens, hidden, times, record = self.run_target("full", inputs, start)
        proposal = Proposal(anchor, inputs[1:])
        self.runner._accept(proposal, VerifyResult(tokens, hidden), len(inputs), ())
        self.warm_commit(proposal, tokens, hidden, len(inputs))
        commit, commit_ms = self.timed(lambda: self.commit(proposal, tokens, hidden))
        count = self.drain(commit)
        pool = self.engine.linear_state_pool
        states = pool.recurrent_states[:, 1].clone(), pool.conv_states[:, 1].clone()
        return {"inputs": inputs.cpu().tolist(), "predictions": tokens.cpu().tolist(),
                "accepted": int(commit.accepted_drafts.item()), "returned": count,
                "tokens": tokens[:count].cpu().tolist(), "verify_ms": times,
                "commit_reconcile_ms": commit_ms, "record": record}, states

    def reference(self, original, initial_cache, anchor, count, output_count):
        original.restore()
        initial_cache.restore(self.cache)
        tokens, states = [], None
        current = anchor.clone()
        start = original.frontier.target_valid_len
        page = self.engine.config.page_size
        self.runner._reserve((start + count + page - 1) // page * page)
        for index in range(count):
            batch = self.runner._batch(current, start + index)
            self.policy, self.record = "full", None
            current, _ = self.runner._target(batch)
            current = current.clone()
            tokens.append(int(current.item()))
            if index + 1 == output_count:
                pool = self.engine.linear_state_pool
                states = pool.recurrent_states[:, 1].clone(), pool.conv_states[:, 1].clone()
        return tokens, states

    def baseline(self, original, initial_cache, anchor, count):
        original.restore()
        initial_cache.restore(self.cache)
        current, outputs, rows = anchor.clone(), [], []
        while len(outputs) < count:
            current, row = self.loop(current, policy="full", remaining=min(4, count - len(outputs)), audit=False)
            outputs.extend(row["tokens"])
            rows.append({key: value for key, value in row.items() if key != "record"})
        return {"tokens": outputs, "loops": rows, "component_ms": sum(row["component_ms"] for row in rows)}

    def case(self, sample, prefix, anchor, loop_counts, policy):
        runner, engine = self.runner, self.engine
        start = runner.frontier.target_valid_len
        page = engine.config.page_size
        old_reserved, old_table = runner._reserved_len, engine.page_table[0].clone()
        end = min(start + max(loop_counts) * 4 + 8, engine.max_seq_len)
        runner._reserve((end + page - 1) // page * page)
        original = ModelSnapshot(engine, runner, start, end)
        initial_cache = CacheSnapshot(self.cache)
        occupied = initial_cache.values["id_of_slot"] >= 0
        bank_samples = [(bank.flatten(1)[:, :32], bank.flatten(1)[occupied, :32].clone())
                        for _, bank in self.cache.banks]
        prefix_kv = []
        ids = engine.config.model_config.kv_cache_group_specs()[0].layer_ids
        for layer in (*ids, engine.config.model_config.num_layers):
            for bank in (engine.kv_cache.k_cache(layer), engine.kv_cache.v_cache(layer)):
                view = bank.flatten(0, 1)[:start]
                prefix_kv.append((view, view.clone()))
        current, candidates, loops = anchor.clone(), [], []
        for index in range(1, max(loop_counts) + 1):
            current, row = self.loop(current, policy=policy)
            row["loop"] = index
            row["offset"] = len(candidates)
            candidates.extend(row["tokens"])
            loops.append(row)
            if policy == "cheap":
                self.save_probe(sample, prefix, index, row["inputs"])
            if index not in loop_counts:
                continue
            if policy == "cheap":
                self.save_workload(sample, prefix, loops)
            after = ModelSnapshot(engine, runner, start, end)
            after_cache = CacheSnapshot(self.cache)
            outer, states = self.outer(original, after_cache, anchor, candidates, start)
            next_input = torch.tensor(outer["tokens"][-1:], dtype=torch.int32, device=engine.device)
            runner._reserve((start + outer["returned"] + 1 + page - 1) // page * page)
            next_tokens, _ = runner._target(runner._batch(next_input, start + outer["returned"]))
            committed_next = int(next_tokens.item())
            reference, reference_states = self.reference(original, initial_cache, anchor, len(candidates) + 2, outer["returned"])
            output_parity = outer["tokens"] == reference[:outer["returned"]]
            next_parity = committed_next == reference[outer["returned"]]
            state_errors = []
            for actual, expected in zip(states, reference_states):
                error = actual.float() - expected.float()
                state_errors.append({"max_abs": float(error.abs().max().item()),
                                     "rms": float(error.square().mean().sqrt().item()),
                                     "reference_rms": float(expected.float().square().mean().sqrt().item())})
            baseline = self.baseline(original, initial_cache, anchor, outer["returned"])
            baseline_parity = baseline["tokens"] == outer["tokens"]
            original.restore()
            seed_bytes = sum(x.numel() * x.element_size() for x in (original.recurrent, original.conv))
            _, seed_ms = self.timed(lambda: (
                engine.linear_state_pool.recurrent_states[:, 1].copy_(original.recurrent),
                engine.linear_state_pool.conv_states[:, 1].copy_(original.conv),
            ))
            result = {"case_id": f"s{sample}_p{prefix}_{policy}_l{index}", "sample_id": sample,
                      "prefix_tokens": prefix, "policy": policy, "loop_count": index,
                      "start": start, "candidates": list(candidates), "loops": list(loops), "outer": outer,
                      "reference_tokens": reference, "outer_reference_parity": output_parity,
                      "post_commit_next_token_parity": next_parity,
                      "matched_baseline_token_parity": baseline_parity,
                      "ar_accepted": accepted_prefix(candidates, reference),
                      "gdn_commit_reference_errors": state_errors, "baseline": baseline,
                      "state_seed_bytes": seed_bytes, "state_seed_ms": seed_ms,
                      "initial_owner": initial_cache.values["id_of_slot"].cpu().tolist(),
                      "initial_usage": initial_cache.values["usage"].cpu().tolist(),
                      "after_owner": after_cache.values["id_of_slot"].cpu().tolist(),
                      "after_usage": after_cache.values["usage"].cpu().tolist()}
            with (self.output / "windows.jsonl").open("a") as file:
                file.write(json.dumps(result) + "\n")
            print(json.dumps({"case": result["case_id"], "candidates": len(candidates),
                              "accepted": outer["accepted"], "returned": outer["returned"],
                              "ar_parity": output_parity, "next_parity": next_parity,
                              "baseline_parity": baseline_parity,
                              "baseline_ms": baseline["component_ms"],
                              "method_ms": seed_ms + sum(row["component_ms"] for row in loops)
                              + statistics.median(outer["verify_ms"]) + outer["commit_reconcile_ms"]}), flush=True)
            self.cases.append(result["case_id"])
            self.quality_checks.append({"case_id": result["case_id"], "ar_output": output_parity,
                                        "ar_next": next_parity, "baseline_output": baseline_parity})
            after.restore()
            after_cache.restore(self.cache)
        original.restore()
        initial_cache.restore(self.cache)
        engine.page_table[0].copy_(old_table)
        runner._reserved_len = old_reserved
        runner.frontier.allocated_len = old_reserved
        assert all(torch.equal(view, saved) for view, saved in prefix_kv), "probe overwrote canonical prefix KV"
        for name, value in initial_cache.values.items():
            assert torch.equal(getattr(self.cache, name), value), f"cache restore changed {name}"
        assert all(torch.equal(view[occupied], saved) for view, saved in bank_samples), "cache restore changed expert weight samples"
        assert torch.equal(engine.linear_state_pool.recurrent_states[:, 1], original.recurrent)
        assert torch.equal(engine.linear_state_pool.conv_states[:, 1], original.conv)
        assert torch.equal(runner.next_draft, original.seed)
        assert torch.equal(runner.last_draft_hidden, original.hidden)


def build_engine(args):
    return Engine(SchedulerConfig(
        model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        max_running_req=1, attention_backend="triton", moe_strategy=args.backend,
        moe_cache_auto=True, kv_reserve_tokens=4096, memory_ratio=0.9,
        max_seq_len_override=4096, gpu_memory_budget_bytes=args.budget_gib << 30,
        speculative_mtp_depth=3, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 3, 4],
        speculative_max_input_tokens=4096 - max(args.tokens, 6), speculative_max_output_tokens=max(args.tokens, 6),
        cache_type="naive", max_extend_tokens=128,
        mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))),
    ))


def checkpointed_request(experiment, prompt, sample, args, *, probe):
    runner, engine = experiment.runner, experiment.engine
    anchor = runner._prefill(torch.tensor(prompt, dtype=torch.int32), 128)
    length, page = len(prompt), engine.config.page_size
    runner.frontier = SpeculativeFrontier(page, length, length, runner._reserved_len)
    output = [int(anchor.item())]
    for goal in (*args.prefixes, args.tokens):
        while len(output) < goal:
            anchor, row = experiment.loop(anchor, policy="full", remaining=min(4, goal - len(output)), audit=False)
            output.extend(row["tokens"])
        if probe and goal != args.tokens:
            for policy in args.policies:
                experiment.case(sample, goal, anchor, args.loops, policy)
    engine.stream.synchronize()
    runner._reset_request_mapping()
    engine.stream.synchronize()
    runner.frontier, runner.next_draft, runner.last_draft_hidden = None, None, None
    runner.model_state.current, runner.last_event = None, None
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data1/fanya/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--data", default="/home/fanya/replayssm_build_artifacts/gsm8k_test.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("offload", "hybrid"), required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--prefixes", nargs="+", type=int, default=[32, 96, 160])
    parser.add_argument("--loops", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--policies", nargs="+", choices=("cheap", "full"), default=["cheap", "full"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget-gib", type=int, default=40)
    parser.add_argument("--skip-probes", action="store_true")
    args = parser.parse_args()
    assert sorted(set(args.prefixes)) == args.prefixes and 0 < min(args.prefixes) < max(args.prefixes + [args.tokens])
    assert max(args.prefixes) < args.tokens and min(args.loops) > 0
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "complete.json").write_text('{"status":"incomplete"}\n')
    prompts, config = prepare_experiment(args.model, args.data, Path.cwd(), args.samples + args.sample_start)
    import freetoken
    from qwen_mtp_config import source_fingerprint
    runtime_root = Path(freetoken.__file__).resolve().parents[2]
    config["checkout_root"] = config["source_root"]
    config["source_root"] = str(runtime_root)
    config["source_sha256"] = source_fingerprint(runtime_root)
    prompts = prompts[args.sample_start:]
    config["prompt_sha256"] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    config.update({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    config.update({"sample_ids": list(range(args.sample_start, args.sample_start + args.samples)),
                   "policy": "all hits plus stable original top4 misses; original weights; no renormalization",
                   "scope": "real multiloop model trajectory and matched-output GPU component costs; not serving E2E",
                   "state_contract": "carry approximate KV/GDN/MTP across inner loops; full outer restarts canonical state",
                   "timing": "median uninstrumented graph target replay; draft/commit device events; setup/restores excluded",
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    engine, experiment = build_engine(args), None
    runner = engine.speculative_runner
    try:
        with torch.inference_mode(), torch.cuda.stream(engine.stream):
            for count in (2, 3, 4, 5, args.tokens):
                runner.generate(prompts[0], max_tokens=count)
            config.update({"gpu": torch.cuda.get_device_name(engine.device),
                           "moe_slots": engine.moe_offload_cache.cache_size, "kv_pages": engine.num_pages,
                           "hybrid_fetch_fraction": engine.moe_offload_cache.hybrid_fetch_fraction,
                           "expert_bytes": sum(bank[0].numel() * bank[0].element_size() for _, bank in engine.moe_offload_cache.banks)})
            (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            experiment = MultiloopExperiment(engine, runner, args.output, args.repeats)
            engine.moe_offload_cache.reset()
            reference = []
            for sample, prompt in enumerate(prompts, args.sample_start):
                tokens = checkpointed_request(experiment, prompt, sample, args, probe=False)
                row = {"sample_id": sample, "token_ids": tokens,
                       "scope": "uninstrumented-route checkpointed full-target control; not E2E timing"}
                reference.append(row)
                with (args.output / "baseline_requests.jsonl").open("a") as file:
                    file.write(json.dumps(row) + "\n")
                print(json.dumps({"baseline_sample": sample}), flush=True)
            engine.moe_offload_cache.reset()
            for sample, prompt in enumerate(prompts, args.sample_start):
                output = checkpointed_request(experiment, prompt, sample, args, probe=not args.skip_probes)
                expected = reference[sample - args.sample_start]["token_ids"]
                request_parity = output == expected
                if not request_parity:
                    (args.output / "request_parity_failure.json").write_text(json.dumps({
                        "sample_id": sample, "actual": output, "expected": expected,
                        "first_difference": next(i for i, (a, b) in enumerate(zip(output, expected)) if a != b),
                    }, indent=2) + "\n")
                assert request_parity, "snapshot restoration perturbed the identical checkpointed control"
                with (args.output / "audit_requests.jsonl").open("a") as file:
                    file.write(json.dumps({"sample_id": sample, "token_ids": output, "baseline_parity": request_parity,
                                           "protocol": "exact prefix checkpoints may shorten a standard MTP window"}) + "\n")
                print(json.dumps({"audit_sample": sample, "parity": request_parity}), flush=True)
            expected_cases = 0 if args.skip_probes else args.samples * len(args.prefixes) * len(args.loops) * len(args.policies)
            assert len(experiment.cases) == expected_cases
            engine.stream.synchronize()
            (args.output / "complete.json").write_text(json.dumps({
                "status": "completed", "windows": len(experiment.cases), "expected_windows": expected_cases,
                "requests": args.samples * 2, "returned_tokens": args.samples * args.tokens * 2,
                "outer_full_ar_parity": all(row["ar_output"] for row in experiment.quality_checks),
                "baseline_audit_parity": True,
                "post_commit_ar_next_parity": all(row["ar_next"] for row in experiment.quality_checks),
                "matched_baseline_output_parity": all(row["baseline_output"] for row in experiment.quality_checks),
                "quality_checks": experiment.quality_checks,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(engine.device),
                "hierarchical_serving_e2e_measured": False,
            }, indent=2) + "\n")
    finally:
        if experiment is not None:
            experiment.close()
        runner.invalidate()
        engine.shutdown()


if __name__ == "__main__":
    main()
