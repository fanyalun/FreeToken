from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
from freetoken.kernel.pinned import device_ptr
from freetoken.moe.fused import fused_experts_decode_impl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    raw = next(json.loads(line) for line in (args.inputs / "windows.jsonl").open()
               if (json.loads(line)["sample_id"], json.loads(line)["prefix_tokens"],
                   json.loads(line)["policy"], json.loads(line)["loop_count"]) == (0, 96, "cheap", 4))
    activation = [torch.load(args.inputs / f"probe_loop{i}.pt", map_location="cpu", weights_only=True)["hidden_all"] for i in range(1, 5)]
    compute_stream, copy_stream = torch.cuda.Stream(), torch.cuda.Stream()
    sources, compute_banks, destinations, ptrs, mapping = [], [], [], [], []
    for layer in range(40):
        data = torch.load(args.inputs / "probe_workload" / f"layer_{layer:02}.pt", map_location="cpu", weights_only=True)
        host = [bank.pin_memory() for bank in data["banks"]]
        sources.append(host)
        compute_banks.append([bank.cuda() for bank in host])
        destinations.append([torch.empty_like(bank, device="cuda") for bank in host])
        ptrs.append((torch.tensor([bank.data_ptr() for bank in destinations[-1]], dtype=torch.int64, device="cuda"),
                     torch.tensor([device_ptr(bank) for bank in host], dtype=torch.int64, device="cuda"),
                     torch.tensor([bank[0].numel() * bank.element_size() for bank in host], dtype=torch.int64, device="cuda")))
        mapping.append({expert: index for index, expert in enumerate(data["ids"])})
    operations, reference = [], []
    for number, loop in enumerate(raw["loops"]):
        for layer in loop["record"]["layers"]:
            li = layer["layer"]
            hidden = activation[number][li].cuda()
            weights = torch.tensor(layer["weights"], dtype=torch.float32, device="cuda")
            weights *= torch.tensor(layer["keep"], dtype=torch.bool, device="cuda")
            ids = torch.tensor([[mapping[li][expert] for expert in row] for row in layer["ids"]], dtype=torch.int32, device="cuda")
            missing = {expert for row, hits in zip(layer["ids"], layer["hits"]) for expert, hit in zip(row, hits) if not hit}
            missing.difference_update(layer["fetched_ids"])
            index = torch.tensor([mapping[li][expert] for expert in sorted(missing)], dtype=torch.int32, device="cuda")
            count = torch.tensor([len(index)], dtype=torch.int64, device="cuda")
            operations.append((li, hidden, weights, ids, index, count))
            reference.append(fused_experts_decode_impl(hidden, *compute_banks[li], weights, ids).clone())
    torch.cuda.synchronize()
    keepalive, results = [], []
    for mechanism in ("mapped_kernel", "cuda_memcpy"):
        graphs, outputs, components = {}, {}, {}
        for mode in ("compute", "copy", "serial", "concurrent"):
            release = [torch.cuda.Event() for _ in operations]
            joined = torch.cuda.Event()
            c0, c1, i0, i1 = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)]

            # CUDA memcpy capture uses host indices, prepared outside the captured region.
            host_indices = [operation[4].cpu().tolist() for operation in operations]

            def transfer_at(number):
                operation = operations[number]
                li, _, _, _, index, count = operation
                if not len(index):
                    return
                with torch.cuda.nvtx.range(f"loop{number // 40 + 1}/layer{li}/Expert weight prefetch/{mechanism}"):
                    if mechanism == "mapped_kernel":
                        fast_index_copy_multi_jit(*ptrs[li], index, index, count)
                    else:
                        for dst, src in zip(destinations[li], sources[li]):
                            for expert in host_indices[number]:
                                dst[expert].copy_(src[expert], non_blocking=True)

            with torch.cuda.stream(compute_stream):
                for number, operation in enumerate(operations):
                    li, hidden, weights, ids, _, _ = operation
                    fused_experts_decode_impl(hidden, *compute_banks[li], weights, ids)
                    transfer_at(number)
            compute_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=compute_stream):
                output = []
                c0.record(compute_stream)
                if mode == "copy":
                    i0.record(compute_stream)
                for number, operation in enumerate(operations):
                    li, hidden, weights, ids, _, _ = operation
                    if mode == "concurrent":
                        release[number].record(compute_stream)
                        copy_stream.wait_event(release[number])
                        with torch.cuda.stream(copy_stream):
                            if number == 0:
                                i0.record(copy_stream)
                            transfer_at(number)
                    if mode != "copy":
                        with torch.cuda.nvtx.range(f"loop{number // 40 + 1}/layer{li}/Expert compute"):
                            output.append(fused_experts_decode_impl(hidden, *compute_banks[li], weights, ids))
                    if mode in ("copy", "serial"):
                        if mode == "serial" and number == 0:
                            i0.record(compute_stream)
                        transfer_at(number)
                c1.record(compute_stream)
                if mode == "concurrent":
                    with torch.cuda.stream(copy_stream):
                        i1.record(copy_stream)
                        joined.record(copy_stream)
                    compute_stream.wait_event(joined)
                elif mode in ("copy", "serial"):
                    i1.record(compute_stream)
            graphs[mode], outputs[mode], components[mode] = graph, output, (c0, c1, i0, i1)
            keepalive.append((graph, output, release, joined, c0, c1, i0, i1))
        samples = {mode: [] for mode in graphs}
        for repeat in range(7):
            modes = list(graphs)
            modes = modes[repeat % 4:] + modes[:repeat % 4]
            for mode in modes:
                with torch.cuda.stream(compute_stream):
                    graphs[mode].replay()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(compute_stream)
                    for _ in range(3):
                        graphs[mode].replay()
                    end.record(compute_stream)
                end.synchronize()
                samples[mode].append(begin.elapsed_time(end) / 3)
        for mode in ("compute", "serial", "concurrent"):
            assert all(torch.equal(actual, expected) for actual, expected in zip(outputs[mode], reference)), mode
        copied = 0
        copied_by_layer = {layer: set() for layer in range(40)}
        for li, _, _, _, index, _ in operations:
            copied += len(index)
            copied_by_layer[li].update(index.cpu().tolist())
        for layer, indices in copied_by_layer.items():
            for dst, src in zip(destinations[layer], sources[layer]):
                assert torch.equal(dst[sorted(indices)].cpu(), src[sorted(indices)]), (mechanism, layer)
        medians = {mode: statistics.median(values) for mode, values in samples.items()}
        result = {"mechanism": mechanism, "expert_compute_operations": len(operations),
                  "copied_experts_including_repeats": copied,
                  "copied_bytes": copied * int(ptrs[0][2].sum().item()),
                  "median_ms": medians, "samples_ms": samples,
                  "serial_over_concurrent": medians["serial"] / medians["concurrent"],
                  "compute_outputs_bitwise_validated": True, "copied_weights_bitwise_validated": True}
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key != "samples_ms"}), flush=True)
        if args.profile and mechanism == "mapped_kernel":
            torch.cuda.cudart().cudaProfilerStart()
            with torch.cuda.stream(compute_stream):
                for mode in ("serial", "concurrent"):
                    with torch.cuda.nvtx.range(f"multiloop/{mode}/actual_4x40_moe"):
                        for _ in range(3):
                            graphs[mode].replay()
                        compute_stream.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        args.output.write_text(json.dumps({
            "scope": "isolated actual 4-loop x 40-layer MoE compute sequence and observed missing-weight copies; excludes dense/attention/MTP; fixed inputs and separate destination buffers; not serving E2E",
            "inputs": str(args.inputs), "case_id": raw["case_id"], "gpu": torch.cuda.get_device_name(),
            "torch": str(torch.__version__), "results": results,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
