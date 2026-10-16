from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from qwen_mtp_config import prepare_experiment
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.mm.config import MultimodalConfig
from freetoken.moe.offload_cache import iter_offload_moe_layers
from freetoken.scheduler.config import SchedulerConfig


def retained_mask(weights, resident, top_h=4):
    ranks = weights.argsort(dim=-1, descending=True, stable=True).argsort(dim=-1)
    return resident | (ranks < top_h)


def candidate_tokens(inputs, predictions, max_drafts):
    accepted = 0
    for draft, prediction in zip(inputs[1:], predictions):
        if draft != prediction:
            break
        accepted += 1
    candidates = (inputs[1:1 + accepted] + [predictions[accepted]])[:max_drafts]
    return candidates, accepted


def accepted_prefix(drafts, predictions):
    count = 0
    for draft, prediction in zip(drafts, predictions):
        if draft != prediction:
            break
        count += 1
    return count


class CacheSnapshot:
    names = ("slot_for_id", "id_of_slot", "usage", "step", "expert_recency")

    def __init__(self, cache):
        self.values = {name: getattr(cache, name).clone() for name in self.names}

    def restore(self, cache):
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        old = self.values["id_of_slot"]
        changed = ((cache.id_of_slot != old) & (old >= 0)).nonzero().flatten()
        owners = old[changed].cpu().tolist()
        slots = changed.cpu().tolist()
        groups = {}
        for slot, owner in zip(slots, owners):
            layer, expert = divmod(owner, cache.num_experts)
            groups.setdefault(layer, []).append((slot, expert))
        for layer, rows in groups.items():
            dst = torch.tensor([row[0] for row in rows], dtype=torch.int32, device=cache.device)
            src = torch.tensor([row[1] for row in rows], dtype=torch.int32, device=cache.device)
            count = torch.tensor([len(rows)], dtype=torch.int64, device=cache.device)
            fast_index_copy_multi_jit(
                cache._copy_dst_ptrs, cache._copy_src_ptrs[layer], cache._copy_feat_bytes,
                dst, src, count,
            )
        for name, value in self.values.items():
            getattr(cache, name).copy_(value)


class RouteRecord:
    def __init__(self, layers, width, device):
        shape = (layers, width, 8)
        self.ids = torch.empty(shape, dtype=torch.int32, device=device)
        self.weights = torch.empty(shape, dtype=torch.float32, device=device)
        self.hits = torch.empty(shape, dtype=torch.bool, device=device)
        self.keep = torch.empty_like(self.hits)
        self.fetched = torch.empty((layers, 32), dtype=torch.int32, device=device)
        self.count = torch.empty(layers, dtype=torch.int64, device=device)
        self.gpu_slots = torch.empty_like(self.ids)
        self.hidden = torch.empty((layers, width, 2048), dtype=torch.bfloat16, device=device)
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        self.ready = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(layers)]
        self.copy_start = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(layers)]
        self.copy_end = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(layers)]

    def export(self):
        ids, weights, hits, keep, fetched, count, gpu_slots = [tensor.cpu().tolist() for tensor in (
            self.ids, self.weights, self.hits, self.keep, self.fetched, self.count, self.gpu_slots,
        )]
        layers = []
        for layer in range(len(ids)):
            layers.append({
                "layer": layer, "ids": ids[layer], "weights": weights[layer],
                "hits": hits[layer], "keep": keep[layer], "gpu_slots": gpu_slots[layer],
                "fetched_ids": fetched[layer][:count[layer]],
                "ready_ms": self.start.elapsed_time(self.ready[layer]),
                "copy_start_ms": self.start.elapsed_time(self.copy_start[layer]),
                "copy_end_ms": self.start.elapsed_time(self.copy_end[layer]),
            })
        return {"stream_ms": self.start.elapsed_time(self.end), "layers": layers}


class WindowExperiment:
    def __init__(self, engine, runner, output, rounds, repeats):
        self.engine, self.runner, self.cache = engine, runner, engine.moe_offload_cache
        self.output, self.rounds, self.repeats = output, set(rounds), repeats
        self.policy = "full"
        self.record = None
        self.graphs = {}
        self.sample = self.round = 0
        self.cases = []
        self.original_execute = runner._execute
        self.original_copy = self.cache.copy_missing
        self.original_layers = []
        self.layers = list(iter_offload_moe_layers(engine.model))
        assert len(self.layers) == 40 and [layer.layer_id for layer in self.layers] == list(range(40))
        for layer in self.layers:
            original = layer._decode_routed
            self.original_layers.append((layer, original))

            def routed(hidden, weights, ids, _original=original, _layer=layer.layer_id):
                record = self.record
                if self.policy == "full" and record is None:
                    return _original(hidden, weights, ids)
                raw = ids.clone()
                resident = self.cache.slot_for_id[_layer][raw.long()] >= 0
                keep = retained_mask(weights, resident) if self.policy == "cheap" else torch.ones_like(resident)
                if record is not None:
                    record.ready[_layer].record()
                    record.ids[_layer].copy_(raw)
                    record.weights[_layer].copy_(weights)
                    record.hits[_layer].copy_(resident)
                    record.keep[_layer].copy_(keep)
                    record.hidden[_layer].copy_(hidden)
                if self.policy == "cheap":
                    # A retained ID keeps skipped routes out of the miss planner without negative indexing.
                    first = weights.argmax(dim=-1, keepdim=True)
                    substitute = raw.gather(-1, first)
                    ids = torch.where(keep, raw, substitute).contiguous()
                    weights = torch.where(keep, weights, 0).contiguous()
                result = _original(hidden, weights, ids)
                if record is not None:
                    record.gpu_slots[_layer].copy_(ids)
                return result

            layer._decode_routed = routed

        def copy():
            record = self.record
            if record is None:
                return self.original_copy()
            layer = self.cache._pending_src_layer
            _, src, count = self.cache._pending_copy_plan or (
                self.cache.evict_slots, self.cache.src_indices, self.cache.num_indices,
            )
            record.count[layer].copy_(count.sum())
            size = min(32, src.numel())
            record.fetched[layer, :size].copy_(src[:size])
            record.copy_start[layer].record()
            self.original_copy()
            record.copy_end[layer].record()

        self.cache.copy_missing = copy

    def graph(self, policy, inputs, start, audit=False):
        width = inputs.numel()
        batch = self.runner._batch(inputs, start, state=self.runner.states[width])
        key = (policy, width, audit)
        if key not in self.graphs:
            record = RouteRecord(40, width, self.engine.device) if audit else None
            self.policy, self.record = policy, record

            def run():
                if record is not None:
                    record.start.record()
                output = self.runner._target(batch)
                if record is not None:
                    record.end.record()
                return output

            with self.engine.model.forward_host_ctx(batch, True):
                run()
                self.engine.stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.engine.stream):
                    output = run()
            self.graphs[key] = graph, output, record
            self.policy, self.record = "full", None
        return self.graphs[key]

    def replay(self, policy, inputs, start, snapshot, audit=False):
        graph, output, record = self.graph(policy, inputs, start, audit)
        snapshot.restore(self.cache)
        batch = self.runner._batch(inputs, start, state=self.runner.states[inputs.numel()])
        self.engine.stream.synchronize()
        before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before.record(self.engine.stream)
        with self.engine.model.forward_host_ctx(batch, True):
            graph.replay()
        after.record(self.engine.stream)
        after.synchronize()
        return output[0].clone(), before.elapsed_time(after), record.export() if record else None

    def measure(self, policy, inputs, start, snapshot):
        values, tokens = [], None
        for _ in range(self.repeats):
            result, ms, _ = self.replay(policy, inputs, start, snapshot)
            current = result.cpu().tolist()
            if tokens is not None:
                assert tokens == current, "same-snapshot graph replay changed tokens"
            tokens = current
            values.append(ms)
        return tokens, values

    def window(self, inputs, start, width, case_id):
        inputs = inputs[:width].clone()
        original = CacheSnapshot(self.cache)
        full_tokens, full_ms = self.measure("full", inputs, start, original)
        cheap_tokens, cheap_ms = self.measure("cheap", inputs, start, original)
        cheap_cache = CacheSnapshot(self.cache)
        input_list = inputs.cpu().tolist()
        candidates, inner_accepted = candidate_tokens(input_list, cheap_tokens, width - 1)
        candidate_input = torch.tensor([input_list[0], *candidates], device=self.engine.device, dtype=torch.int32)
        target_tokens, target_ms = self.measure("full", candidate_input, start, cheap_cache)
        full_audit, _, full_record = self.replay("full", inputs, start, original, audit=True)
        assert full_audit.cpu().tolist() == full_tokens
        sparse_tokens, _, cheap_record = self.replay("cheap", inputs, start, original, audit=True)
        assert sparse_tokens.cpu().tolist() == cheap_tokens
        cheap_cache_audit = CacheSnapshot(self.cache)
        for name in CacheSnapshot.names:
            assert torch.equal(cheap_cache.values[name], cheap_cache_audit.values[name]), name
        target_audit, _, target_record = self.replay("full", candidate_input, start, cheap_cache_audit, audit=True)
        assert target_audit.cpu().tolist() == target_tokens
        for record in (full_record, cheap_record, target_record):
            for layer in record["layers"]:
                fetched = layer["fetched_ids"]
                assert len(fetched) == len(set(fetched)), "duplicate expert transfer"
                retained = {expert for ids, keep in zip(layer["ids"], layer["keep"])
                            for expert, selected in zip(ids, keep) if selected}
                assert set(fetched) <= retained, "loaded an expert excluded from sparse computation"
        if width == 4 and not (self.output / "probe_inputs.pt").exists():
            layer = 24
            audit_record = self.graphs[("cheap", width, True)][2]
            raw_ids = audit_record.ids[layer].cpu().long()
            selected, inverse = raw_ids.unique(sorted=True, return_inverse=True)
            banks = [source[layer].index_select(0, selected).clone() for source, _ in self.cache.banks]
            torch.save({"hidden": audit_record.hidden[layer].cpu(), "weights": audit_record.weights[layer].cpu(),
                        "keep": audit_record.keep[layer].cpu(), "ids": inverse.reshape_as(raw_ids).int(),
                        "banks": banks, "bank_schema": self.cache.bank_schema,
                        "sample_id": self.sample, "round": self.round, "layer": layer}, self.output / "probe_inputs.pt")
        row = {
            "case_id": case_id, "sample_id": self.sample, "round": self.round, "start": start,
            "width": width, "inputs": input_list, "full_predictions": full_tokens,
            "cheap_predictions": cheap_tokens, "candidate_input": candidate_input.cpu().tolist(),
            "target_predictions": target_tokens, "inner_accepted": inner_accepted,
            "baseline_accepted": accepted_prefix(input_list[1:], full_tokens),
            "outer_accepted": accepted_prefix(candidates, target_tokens),
            "full_ms": full_ms, "cheap_ms": cheap_ms, "target_after_cheap_ms": target_ms,
            "initial_owner": original.values["id_of_slot"].cpu().tolist(),
            "initial_usage": original.values["usage"].cpu().tolist(),
            "after_cheap_owner": cheap_cache.values["id_of_slot"].cpu().tolist(),
            "after_cheap_usage": cheap_cache.values["usage"].cpu().tolist(),
            "records": {"full": full_record, "cheap": cheap_record, "target": target_record},
        }
        with (self.output / "windows.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")
        excluded = {"records", "initial_owner", "initial_usage", "after_cheap_owner", "after_cheap_usage"}
        self.cases.append({key: value for key, value in row.items() if key not in excluded})
        original.restore(self.cache)
        print(json.dumps({"case": case_id, "width": width, "baseline_returned": row["baseline_accepted"] + 1,
                          "outer_returned": row["outer_accepted"] + 1, "full_ms": statistics.median(full_ms),
                          "cheap_ms": statistics.median(cheap_ms), "target_ms": statistics.median(target_ms)}), flush=True)

    def execute(self, phase, batch, hidden=None, **kwargs):
        if phase != "verify":
            return self.original_execute(phase, batch, hidden, **kwargs)
        current = self.round
        self.round += 1
        if current not in self.rounds or batch.input_ids.numel() != 4:
            return self.original_execute(phase, batch, hidden, **kwargs)
        inputs, start, state = batch.input_ids.clone(), int(batch.positions[0].item()), batch.gdn_verify_state
        pool = self.engine.linear_state_pool
        recurrent, conv = pool.recurrent_states[:, 1].clone(), pool.conv_states[:, 1].clone()
        widths = (1, 2, 3, 4) if current == min(self.rounds) else (4,)
        for width in widths:
            self.window(inputs, start, width, f"s{self.sample}_r{current}_w{width}")
        assert torch.equal(recurrent, pool.recurrent_states[:, 1]), "probe advanced persistent SSM"
        assert torch.equal(conv, pool.conv_states[:, 1]), "probe advanced persistent conv"
        batch = self.runner._batch(inputs, start, state=state)
        return self.original_execute(phase, batch, hidden, **kwargs)

    def close(self):
        self.runner._execute = self.original_execute
        self.cache.copy_missing = self.original_copy
        for layer, original in self.original_layers:
            layer._decode_routed = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data1/fanya/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--data", default="/home/fanya/replayssm_build_artifacts/gsm8k_test.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("offload", "hybrid"), required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--rounds", type=int, nargs="+", default=[8, 24, 48])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget-gib", type=int, default=40)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "complete.json").write_text('{"status":"incomplete"}\n')
    prompts, config = prepare_experiment(args.model, args.data, Path(__file__).resolve().parents[1], args.samples)
    config.update(vars(args))
    config["experiment_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config.update({"policy": "all hit plus original top4 miss", "weight_mode": "original top8 weights, no renormalization",
                   "candidate_cap": 3, "warmups": "caps 2,3,4,5 then one full request",
                   "scope": "matched-prefix window diagnostic, not hierarchical end-to-end throughput",
                   "correctness_reference": "same backend and cache snapshot; original full top8 verifier"})
    engine = Engine(SchedulerConfig(
        model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        max_running_req=1, attention_backend="triton", moe_strategy=args.backend,
        moe_cache_auto=True, kv_reserve_tokens=4096, memory_ratio=0.9,
        max_seq_len_override=4096, gpu_memory_budget_bytes=args.budget_gib << 30,
        speculative_mtp_depth=3, cuda_graph_max_bs=4, cuda_graph_bs=[1, 2, 3, 4],
        cache_type="naive", max_extend_tokens=128,
        mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))),
    ))
    runner = engine.speculative_runner
    experiment = None
    try:
        with torch.inference_mode():
            for tokens in (2, 3, 4, 5, args.tokens):
                runner.generate(prompts[0], max_tokens=tokens)
            config.update({"moe_slots": engine.moe_offload_cache.cache_size, "kv_pages": engine.num_pages,
                           "gpu": torch.cuda.get_device_name(engine.device),
                           "hybrid_fetch_fraction": engine.moe_offload_cache.hybrid_fetch_fraction,
                           "expert_bytes": sum(bank[0].numel() * bank[0].element_size() for _, bank in engine.moe_offload_cache.banks)})
            (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            engine.moe_offload_cache.reset()
            baseline = []
            for sample, prompt in enumerate(prompts):
                started = time.perf_counter()
                result = runner.generate(prompt, max_tokens=args.tokens)
                engine.stream.synchronize()
                row = {"sample_id": sample, **dataclasses.asdict(result), "complete_wall_seconds": time.perf_counter() - started}
                assert len(result.token_ids) == args.tokens
                baseline.append(row)
                with (output / "baseline_requests.jsonl").open("a") as file:
                    file.write(json.dumps(row) + "\n")
                print(json.dumps({"baseline_sample": sample, "tokens": len(result.token_ids)}), flush=True)
            experiment = WindowExperiment(engine, runner, output, args.rounds, args.repeats)
            runner._execute = experiment.execute
            engine.moe_offload_cache.reset()
            audits = []
            for sample, prompt in enumerate(prompts):
                experiment.sample, experiment.round = sample, 0
                result = runner.generate(prompt, max_tokens=args.tokens)
                parity = result.token_ids == baseline[sample]["token_ids"] and result.rounds == baseline[sample]["rounds"]
                row = {"sample_id": sample, "token_ids": result.token_ids, "rounds": result.rounds,
                       "baseline_parity": parity, "diagnostic_wall_seconds": result.wall_seconds}
                audits.append(row)
                with (output / "audit_requests.jsonl").open("a") as file:
                    file.write(json.dumps(row) + "\n")
                assert parity, f"diagnostic perturbed baseline sample {sample}"
                print(json.dumps({"audit_sample": sample, "parity": parity}), flush=True)
            expected = args.samples * (len(args.rounds) + 3)
            assert len(experiment.cases) == expected, (len(experiment.cases), expected)
            (output / "case_index.json").write_text(json.dumps(experiment.cases, indent=2) + "\n")
            (output / "complete.json").write_text(json.dumps({
                "status": "completed", "requests": len(baseline), "returned_tokens": args.samples * args.tokens,
                "diagnostic_requests": len(audits), "windows": len(experiment.cases), "expected_windows": expected,
                "baseline_audit_parity": True, "persistent_gdn_unchanged": True,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(engine.device),
            }, indent=2) + "\n")
    finally:
        if experiment is not None:
            experiment.close()
        runner.invalidate()
        engine.shutdown()


if __name__ == "__main__":
    main()
