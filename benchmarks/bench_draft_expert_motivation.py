from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
from pathlib import Path
import statistics
import time

import torch

from qwen_mtp_config import digest, prepare_experiment


def retain_top_h(weights, ids, top_h):
    if not 1 <= top_h <= ids.shape[-1]:
        raise ValueError("top_h must be within the native routing width")
    if top_h == ids.shape[-1]:
        return weights, ids
    order = weights.argsort(dim=-1, descending=True, stable=True)[..., :top_h]
    # Preserve native accumulation order and weights after choosing the largest routes.
    order = order.sort(dim=-1).values
    return weights.gather(-1, order).contiguous(), ids.gather(-1, order).contiguous()


def route_counts(routes):
    counts = [len({expert for row in layer for expert in row}) for layer in routes]
    return counts, sum(counts)


def accepted_prefix(drafts, predictions):
    accepted = 0
    for draft, prediction in zip(drafts, predictions):
        if draft != prediction:
            break
        accepted += 1
    return accepted


def prediction_counts(draft_routes, target_routes, steps):
    predicted = {(layer, expert) for layer, rows in enumerate(draft_routes)
                 for row in rows[:steps] for expert in row}
    target = {(layer, expert) for layer, rows in enumerate(target_routes)
              for row in rows for expert in row}
    return {"predicted": len(predicted), "target": len(target),
            "intersection": len(predicted & target), "union": len(predicted | target)}


def summarize(windows, depth):
    full = [row for row in windows if row["drafted"] == depth]
    result = {"windows": len(windows), "full_length_windows": len(full),
              "tail_windows": len(windows) - len(full)}
    for name, rows in (("all", windows), ("full_length", full)):
        result[name] = {
            "windows": len(rows),
            "mean_unique_routed_experts": statistics.mean(row["unique_routed_experts"] for row in rows) if rows else None,
            "mean_accepted_drafts": statistics.mean(row["accepted_drafts"] for row in rows) if rows else None,
            "mean_returned_tokens": statistics.mean(row["returned_tokens"] for row in rows) if rows else None,
            "drafted_tokens": sum(row["drafted"] for row in rows),
            "accepted_drafts": sum(row["accepted_drafts"] for row in rows),
        }
        if rows and "expert_prediction" in rows[0]:
            result[name]["expert_prediction"] = {}
            for selection in rows[0]["expert_prediction"]:
                totals = {key: sum(row["expert_prediction"][selection][key] for row in rows)
                          for key in ("predicted", "target", "intersection", "union")}
                totals.update(recall=totals["intersection"] / totals["target"],
                              precision=totals["intersection"] / totals["predicted"] if totals["predicted"] else None,
                              iou=totals["intersection"] / totals["union"],
                              empty_prediction_windows=sum(row["expert_prediction"][selection]["predicted"] == 0 for row in rows))
                result[name]["expert_prediction"][selection] = totals
    return result


class SelfDraftProposer:
    def __init__(self, runner, top_h):
        self.runner, self.top_h = runner, top_h
        pool = runner.engine.linear_state_pool
        self.saved_recurrent = torch.empty_like(pool.recurrent_states[:, 1])
        self.saved_conv = torch.empty_like(pool.conv_states[:, 1])
        self.one = torch.ones((), dtype=torch.int32, device=pool.conv_states.device)

    def propose(self, anchor, depth):
        from freetoken.engine.speculative import Proposal

        runner, pool = self.runner, self.runner.engine.linear_state_pool
        self.saved_recurrent.copy_(pool.recurrent_states[:, 1])
        self.saved_conv.copy_(pool.conv_states[:, 1])
        token = anchor
        try:
            for row in range(depth):
                state = runner.states[1]
                batch = runner._batch(token, runner.frontier.target_valid_len + row, state=state)
                token, _ = runner._execute(f"self_draft_h{self.top_h}", batch)
                runner.proposal_buffer[row:row + 1].copy_(token)
                runner._commit_state(state, self.one)
        finally:
            # Private suffix KV is overwritten by final verify; canonical GDN must be restored first.
            pool.recurrent_states[:, 1].copy_(self.saved_recurrent)
            pool.conv_states[:, 1].copy_(self.saved_conv)
        return Proposal(anchor, runner.proposal_buffer[:depth])

    def reconcile(self, hidden, commit):
        pass


class RouteProbe:
    def __init__(self, engine, collect_draft_routes=False, draft_route_set="retained_top_h"):
        self.runner = engine.speculative_runner
        self.layers = [block.mlp.experts for block in engine.model.model.layers.op_list]
        self.layer_ids = [block._layer_id for block in engine.model.model.layers.op_list]
        if engine.config.moe_strategy != "fused" or not all(layer.top_k == 8 for layer in self.layers):
            raise ValueError("this diagnostic requires resident fused top-8 experts in every target layer")
        self.phase = None
        self.windows = []
        self.enabled = False
        self.draft_dispatch_widths = {}
        self.collect_draft_routes = collect_draft_routes
        self.draft_route_set = draft_route_set
        self.draft_step = 0
        self.draft_h = 0
        self.ids = torch.empty((len(self.layers), self.runner.draft_width, 8), dtype=torch.int32, device=engine.device)
        self.draft_ids = torch.empty_like(self.ids) if collect_draft_routes else None
        self.draft_scratch = torch.empty_like(self.ids[:, 0]) if collect_draft_routes else None
        self.original_execute = self.runner._execute
        self.original_layers = []
        self.runner.graphs.clear()
        for index, layer in enumerate(self.layers):
            original = layer._resident_gemm
            self.original_layers.append((layer, original))

            def routed(hidden, weights, ids, _original=original, _index=index):
                if self.phase == "verify":
                    self.ids[_index, :ids.shape[0]].copy_(ids)
                elif self.phase and self.phase.startswith("self_draft_h"):
                    if self.collect_draft_routes and self.draft_route_set == "native_top8":
                        self.draft_scratch[_index].copy_(ids[0])
                    weights, ids = retain_top_h(weights, ids, int(self.phase.removeprefix("self_draft_h")))
                    self.draft_dispatch_widths[self.phase] = ids.shape[-1]
                    if self.collect_draft_routes and self.draft_route_set == "retained_top_h":
                        self.draft_scratch[_index, :ids.shape[-1]].copy_(ids[0])
                return _original(hidden, weights, ids)

            layer._resident_gemm = routed
        self.runner._execute = self.execute

    def execute(self, phase, batch, hidden=None, **kwargs):
        self.phase = phase
        try:
            result = self.original_execute(phase, batch, hidden, **kwargs)
            if self.enabled and self.collect_draft_routes and phase.startswith("self_draft_h"):
                # Graph replay writes a fixed scratch address; archive each step outside capture.
                self.draft_h = 8 if self.draft_route_set == "native_top8" else int(phase.removeprefix("self_draft_h"))
                self.draft_ids[:, self.draft_step, :self.draft_h].copy_(self.draft_scratch[:, :self.draft_h])
                self.draft_step += 1
            if self.enabled and phase == "verify":
                width = batch.input_ids.numel()
                routes = self.ids[:, :width].cpu().tolist()
                inputs, predictions = batch.input_ids.cpu().tolist(), result[0].cpu().tolist()
                counts, total = route_counts(routes)
                self.windows.append({
                    "round": len(self.windows), "drafted": width - 1,
                    "accepted_drafts": accepted_prefix(inputs[1:], predictions),
                    "unique_routed_experts": total, "unique_routed_experts_per_layer": counts,
                    "verify_rows": width, "anchor": inputs[0], "draft_tokens": inputs[1:],
                    "target_predictions": predictions, "routes": routes,
                })
                if self.collect_draft_routes:
                    assert self.draft_step == width - 1
                    draft_routes = self.draft_ids[:, :self.draft_step, :self.draft_h].cpu().tolist()
                    window = self.windows[-1]
                    window["draft_routes"] = draft_routes
                    window["draft_route_set"] = self.draft_route_set
                    selections = (("all_generation_steps", self.draft_step),)
                    if self.draft_route_set == "retained_top_h":
                        selections += (("accepted_generation_steps", window["accepted_drafts"]),)
                    window["expert_prediction"] = {
                        selection: prediction_counts(draft_routes, routes, steps)
                        for selection, steps in selections}
                    self.draft_step = 0
            return result
        finally:
            self.phase = None

    def close(self):
        self.runner._execute = self.original_execute
        for layer, original in self.original_layers:
            layer._resident_gemm = original


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_report(output, summaries, expected):
    write_json(output / "metrics.json", {"expected_cells": expected, "completed_cells": len(summaries), "cells": summaries})
    lines = ["# 草稿长度、专家激活与接受长度动机测试", "",
             f"已完成 {len(summaries)}/{expected} 个配置。每组独立完整生成，原始验证路由见各组 windows.jsonl.gz。", "",
             "指标主表只纳入实际草稿数等于配置长度的完整窗口，输出末尾缩短窗口另计。",
             "专家数为各层内对所有验证输入行的路由专家去重后求和；不含 shared expert、草稿计算或缓存加载数。",
             "接受长度只计首个拒绝前匹配的草稿 token，不含纠正/奖励 token。全窗口统计及逐样本均值保存在 metrics.json。",
             "固定 top-h 只减少 routed experts，保留原始 top-8 权重，不重新归一化，不使用 ready-hit。",
             "这是带路由记录的动机实验，未进行严格 AR parity 验收，不提供性能或无损结论。", "",
             "| 草稿 | 长度 | 完整窗口数 | 平均激活专家总数 | 平均接受草稿长度 | 尾部窗口数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for cell in summaries:
        full = cell["full_length"]
        experts, accepted = full["mean_unique_routed_experts"], full["mean_accepted_drafts"]
        lines.append(f"| {cell['method']} | {cell['depth']} | {full['windows']} | "
                     f"{experts:.4f} | {accepted:.4f} | {cell['tail_windows']} |" if experts is not None else
                     f"| {cell['method']} | {cell['depth']} | 0 | unknown | unknown | {cell['tail_windows']} |")
    (output / "readme.md").write_text("\n".join(lines) + "\n")


def analyze(output):
    config = json.loads((output / "config.json").read_text())
    cells, request_count, tokens, window_count = [], 0, 0, 0
    for method in config["methods"]:
        for depth in config["depths"]:
            cell = output / f"{method}_d{depth:02d}"
            if not (cell / "complete.json").exists() or json.loads((cell / "complete.json").read_text())["status"] != "complete":
                continue
            requests = [json.loads(line) for line in (cell / "requests.jsonl").read_text().splitlines()]
            assert len(requests) == config["samples"]
            grouped = {sample: [] for sample in range(config["samples"])}
            with gzip.open(cell / "windows.jsonl.gz", "rt") as raw:
                for line in raw:
                    row = json.loads(line)
                    counts = [len(set(expert for route in layer for expert in route)) for layer in row["routes"]]
                    assert counts == row["unique_routed_experts_per_layer"]
                    assert sum(counts) == row["unique_routed_experts"]
                    assert len(row["routes"]) == len(config["layer_ids"])
                    assert all(len(layer) == row["verify_rows"] for layer in row["routes"])
                    assert all(len(route) == 8 and all(0 <= expert < 256 for expert in route)
                               for layer in row["routes"] for route in layer)
                    matches = [draft == predicted for draft, predicted in zip(row["draft_tokens"], row["target_predictions"])]
                    accepted = matches.index(False) if False in matches else len(matches)
                    assert accepted == row["accepted_drafts"]
                    if config.get("collect_draft_routes"):
                        draft_routes = row["draft_routes"]
                        assert len(draft_routes) == len(row["routes"])
                        assert all(len(layer) == row["drafted"] for layer in draft_routes)
                        route_width = 8 if config.get("draft_route_set") == "native_top8" else int(method.removeprefix("top_h"))
                        assert all(len(route) == route_width
                                   and all(0 <= expert < 256 for expert in route)
                                   for layer in draft_routes for route in layer)
                        for selection in row["expert_prediction"]:
                            steps = accepted if selection == "accepted_generation_steps" else row["drafted"]
                            assert row["expert_prediction"][selection] == prediction_counts(draft_routes, row["routes"], steps)
                    assert row["returned_tokens"] == accepted + 1
                    assert row["drafted"] == len(row["draft_tokens"]) == row["verify_rows"] - 1
                    grouped[row["sample_id"]].append(row)
                    window_count += 1
            all_windows = []
            for sample, request in enumerate(requests):
                assert request["sample_id"] == sample and len(request["token_ids"]) == config["tokens"]
                cursor = 1
                assert len(grouped[sample]) == len(request["rounds"])
                for number, row in enumerate(grouped[sample]):
                    assert row["round"] == number and row["anchor"] == request["token_ids"][cursor - 1]
                    count = row["returned_tokens"]
                    assert row["drafted"] == min(depth, config["tokens"] - cursor - 1)
                    assert request["token_ids"][cursor:cursor + count] == row["target_predictions"][:count]
                    cursor += count
                assert cursor == config["tokens"]
                all_windows.extend(grouped[sample])
                request_count += 1
                tokens += len(request["token_ids"])
            calculated = summarize(all_windows, depth)
            stored = json.loads((cell / "metrics.json").read_text())
            assert all(stored[key] == value for key, value in calculated.items())
            cells.append(stored)
    write_report(output, cells, config["expected_cells"])
    validation = {"valid": True, "matrix_complete": len(cells) == config["expected_cells"],
                  "completed_cells": len(cells), "expected_cells": config["expected_cells"],
                  "requests": request_count, "returned_tokens": tokens, "verification_windows": window_count,
                  "scope": "raw routing counts, accepted prefix, exact output length, request reconstruction and matrix coverage",
                  "strict_ar_parity": "not_tested"}
    write_json(output / "validation.json", validation)
    print(json.dumps(validation), flush=True)
    return validation


def main():
    parser = argparse.ArgumentParser(description="Standalone fused draft-length/top-h motivation sweep")
    parser.add_argument("--model", default="/data1/fanya/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--data", default="/home/fanya/replayssm_build_artifacts/gsm8k_test.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--collect-draft-routes", action="store_true")
    parser.add_argument("--draft-route-set", choices=("retained_top_h", "native_top8"), default="retained_top_h")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--depths", type=int, nargs="+", default=list(range(1, 33)))
    parser.add_argument("--methods", nargs="+", choices=("mtp", "top_h2", "top_h4", "top_h6", "top_h8"),
                        default=["mtp", "top_h2", "top_h4", "top_h6", "top_h8"])
    args = parser.parse_args()
    if args.analyze:
        analyze(args.output.resolve())
        return
    if args.collect_draft_routes and "mtp" in args.methods:
        parser.error("draft route prediction currently requires top-h self-draft methods")
    if (args.samples < 1 or args.tokens < 2 or not all(1 <= depth <= 32 for depth in args.depths)
            or len(set(args.depths)) != len(args.depths) or len(set(args.methods)) != len(args.methods)):
        parser.error("positive samples, at least 2 tokens, unique depths 1..32 and unique methods required")
    from freetoken.distributed import DistributedInfo
    from freetoken.engine import Engine
    from freetoken.mm.config import MultimodalConfig
    from freetoken.scheduler.config import SchedulerConfig
    from bench_qwen_mtp import ProcessMemoryMonitor

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "complete.json", {"status": "incomplete", "matrix_complete": False})
    root = Path(__file__).resolve().parents[1]
    prompts, config = prepare_experiment(args.model, args.data, root, args.samples)
    max_input = max(map(len, prompts))
    config.update({**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                   "driver_sha256": digest(__file__), "backend": "fused", "purpose": "diagnostic",
                   "scope": "complete fixed-length requests with instrumented final-verification routes",
                   "gpu_memory_budget_bytes": None,
                   "ignore_eos": True, "temperature": 0, "enable_thinking": False,
                   "weight_mode": "preserve native top8 weights without renormalization",
                   "ready_hit": False, "three_level": False, "strict_ar_parity": "not_tested",
                   "aggregation": "window pooled; also equal sample means; full-length and tail separate",
                   "expert_count": "sum over layers of unique routed expert IDs across anchor plus draft rows",
                   "shared_experts_in_count": False, "common_runtime_depth": max(args.depths),
                   "max_input_tokens": max_input, "expected_cells": len(args.methods) * len(args.depths),
                   "command": __import__("sys").argv})
    write_json(output / "config.json", config)
    write_json(output / "prompts.json", prompts)
    write_report(output, [], config["expected_cells"])
    monitor = ProcessMemoryMonitor()
    monitor.start()
    engine = probe = None
    try:
        engine = Engine(SchedulerConfig(
            model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
            max_running_req=1, attention_backend="triton", moe_strategy="fused", memory_ratio=0.95,
            speculative_mtp_depth=max(args.depths), speculative_max_input_tokens=max_input,
            speculative_max_output_tokens=args.tokens, kv_reserve_tokens=max_input + args.tokens,
            max_seq_len_override=max_input + args.tokens, cache_type="naive", max_extend_tokens=128,
            cuda_graph_max_bs=1, mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))),
        ))
        runner = engine.speculative_runner
        native_proposer = runner.coordinator.proposer
        with torch.inference_mode(), torch.cuda.stream(engine.stream):
            probe = RouteProbe(engine, args.collect_draft_routes, args.draft_route_set)
            config.update({"expert_slots": None, "kv_pages": engine.num_pages,
                           "gpu": torch.cuda.get_device_name(engine.device), "layer_ids": probe.layer_ids,
                           "memory_plan": engine.baseline_sd_memory_plan.as_dict(),
                           "gdn_private_state": "snapshot canonical slot1; commit each private draft step; restore before final verify",
                           "kv_private_state": "suffix only; final verifier overwrites from same start; prefix untouched"})
            write_json(output / "config.json", config)
            summaries = []
            for method in args.methods:
                runner.coordinator.proposer = (native_proposer if method == "mtp" else
                                               SelfDraftProposer(runner, int(method.removeprefix("top_h"))))
                for depth in args.depths:
                    runner.draft_width = depth + 1
                    cell = output / f"{method}_d{depth:02d}"
                    cell.mkdir()
                    write_json(cell / "complete.json", {"status": "incomplete"})
                    all_windows, requests = [], []
                    with gzip.open(cell / "windows.jsonl.gz", "wt", compresslevel=1) as raw:
                        for sample, prompt in enumerate(prompts):
                            probe.windows = []
                            probe.enabled = True
                            started = time.perf_counter()
                            result = runner.generate(prompt, max_tokens=args.tokens, eos_ids=(), prompt_chunk_size=128)
                            probe.enabled = False
                            assert len(result.token_ids) == args.tokens, "incomplete output"
                            assert len(result.rounds) == len(probe.windows), "missing/duplicated verify trace"
                            for window, accounting in zip(probe.windows, result.rounds):
                                assert window["accepted_drafts"] == accounting["matched_drafts"]
                                assert window["drafted"] == accounting["drafted"]
                                assert 0 <= window["accepted_drafts"] <= window["drafted"]
                                window.update({"sample_id": sample, "returned_tokens": accounting["returned_tokens"]})
                                raw.write(json.dumps(window, separators=(",", ":")) + "\n")
                            raw.flush()
                            row = {"sample_id": sample, **dataclasses.asdict(result),
                                   "instrumented_wall_seconds": time.perf_counter() - started,
                                   "metrics": summarize(probe.windows, depth)}
                            requests.append(row)
                            all_windows.extend({key: value for key, value in window.items() if key not in ("routes", "draft_routes")}
                                               for window in probe.windows)
                            with (cell / "requests.jsonl").open("a") as stream:
                                stream.write(json.dumps(row) + "\n")
                            print(json.dumps({"method": method, "depth": depth, "sample": sample,
                                              "tokens": len(result.token_ids), "seconds": row["instrumented_wall_seconds"],
                                              **row["metrics"]["full_length"]}), flush=True)
                    summary = {"method": method, "depth": depth, "samples": len(requests),
                               "actual_draft_dispatch_width": probe.draft_dispatch_widths.get(f"self_draft_h{method.removeprefix('top_h')}")
                               if method != "mtp" else None,
                               "returned_tokens": sum(len(row["token_ids"]) for row in requests),
                               **summarize(all_windows, depth)}
                    for metric in ("mean_unique_routed_experts", "mean_accepted_drafts"):
                        values = [row["metrics"]["full_length"][metric] for row in requests]
                        summary[f"equal_sample_{metric}"] = statistics.mean(values) if all(x is not None for x in values) else None
                    assert sum(row["returned_tokens"] for row in all_windows) + args.samples == args.samples * args.tokens
                    write_json(cell / "metrics.json", summary)
                    write_json(cell / "complete.json", {"status": "complete", "samples": len(requests),
                                                        "returned_tokens": summary["returned_tokens"]})
                    summaries.append(summary)
                    write_report(output, summaries, config["expected_cells"])
                    write_json(output / "progress.json", {"completed_cells": len(summaries),
                                                          "expected_cells": config["expected_cells"], "last_cell": cell.name,
                                                          "peak_process_vram_bytes": monitor.peak})
            write_json(output / "complete.json", {"status": "complete", "matrix_complete": True,
                                                  "completed_cells": len(summaries), "requests": args.samples * len(summaries),
                                                  "peak_process_vram_bytes": monitor.peak, "memory_samples": monitor.samples,
                                                  "strict_ar_parity": "not_tested", "purpose": "diagnostic"})
            analyze(output)
    except BaseException as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if probe is not None:
            probe.close()
        monitor.close()
        write_json(output / "memory.json", {"peak_process_vram_bytes": monitor.peak, "samples": monitor.samples,
                                            "budget_bytes": None})
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
