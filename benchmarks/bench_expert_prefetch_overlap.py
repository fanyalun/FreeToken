from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

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
    data = torch.load(args.inputs, map_location="cpu", weights_only=True)
    torch.cuda.set_device(0)
    compute_stream, copy_stream = torch.cuda.Stream(), torch.cuda.Stream()
    hidden = data["hidden"].cuda()
    weights = torch.where(data["keep"], data["weights"], 0).cuda()
    ids = data["ids"].cuda()
    gate, down = [bank.cuda() for bank in data["banks"]]
    host_banks = []
    for bank in data["banks"]:
        repeats = (32 + bank.shape[0] - 1) // bank.shape[0]
        host_banks.append(bank.repeat((repeats,) + (1,) * (bank.ndim - 1))[:32].contiguous().pin_memory())
    destination = [torch.empty_like(bank, device="cuda") for bank in host_banks]
    dst_ptrs = torch.tensor([bank.data_ptr() for bank in destination], dtype=torch.int64, device="cuda")
    src_ptrs = torch.tensor([device_ptr(bank) for bank in host_banks], dtype=torch.int64, device="cuda")
    feature_bytes = torch.tensor([bank[0].numel() * bank.element_size() for bank in host_banks], dtype=torch.int64, device="cuda")
    indices = torch.arange(32, dtype=torch.int32, device="cuda")
    count = torch.zeros(1, dtype=torch.int64, device="cuda")
    keepalive, results = [], []

    def compute():
        return fused_experts_decode_impl(hidden, gate, down, weights, ids)

    with torch.cuda.stream(compute_stream):
        reference = compute().clone()
        fast_index_copy_multi_jit(dst_ptrs, src_ptrs, feature_bytes, indices, indices, count)
    compute_stream.synchronize()

    for mechanism in ("mapped_kernel", "cuda_memcpy"):
        for copied in (4, 16, 32):
            for repeats in (1, 32, 128):
                count.fill_(copied)
                torch.cuda.synchronize()

                def transfer():
                    if mechanism == "mapped_kernel":
                        fast_index_copy_multi_jit(dst_ptrs, src_ptrs, feature_bytes, indices, indices, count)
                    else:
                        for dst, src in zip(destination, host_banks):
                            dst[:copied].copy_(src[:copied], non_blocking=True)

                graphs, outputs, components = {}, {}, {}
                for mode in ("compute", "copy", "serial", "concurrent"):
                    fork, joined = torch.cuda.Event(), torch.cuda.Event()
                    c0, c1, i0, i1 = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)]
                    with torch.cuda.stream(compute_stream):
                        compute()
                        transfer()
                    compute_stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=compute_stream):
                        output = None
                        if mode == "concurrent":
                            fork.record(compute_stream)
                            copy_stream.wait_event(fork)
                            with torch.cuda.stream(copy_stream):
                                i0.record(copy_stream)
                                transfer()
                                i1.record(copy_stream)
                                joined.record(copy_stream)
                        if mode != "copy":
                            c0.record(compute_stream)
                            for _ in range(repeats):
                                output = compute()
                            c1.record(compute_stream)
                        if mode in ("copy", "serial"):
                            i0.record(compute_stream)
                            transfer()
                            i1.record(compute_stream)
                        if mode == "concurrent":
                            compute_stream.wait_event(joined)
                    graphs[mode], outputs[mode], components[mode] = graph, output, (c0, c1, i0, i1)
                    keepalive.append((graph, output, fork, joined, c0, c1, i0, i1))
                samples = {mode: [] for mode in graphs}
                for repeat in range(9):
                    modes = list(graphs)
                    modes = modes[repeat % 4:] + modes[:repeat % 4]
                    for mode in modes:
                        graph = graphs[mode]
                        with torch.cuda.stream(compute_stream):
                            graph.replay()
                            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            begin.record(compute_stream)
                            for _ in range(5):
                                graph.replay()
                            end.record(compute_stream)
                        end.synchronize()
                        samples[mode].append(begin.elapsed_time(end) / 5)
                for mode in ("compute", "serial", "concurrent"):
                    assert torch.equal(outputs[mode], reference), mode
                for dst, src in zip(destination, host_banks):
                    assert torch.equal(dst[:copied].cpu(), src[:copied]), mechanism
                medians = {mode: statistics.median(values) for mode, values in samples.items()}
                c0, c1, i0, i1 = components["concurrent"]
                row = {"mechanism": mechanism, "copied_experts": copied, "compute_repeats": repeats,
                       "bytes": copied * int(feature_bytes.sum().item()), "samples_ms": samples, "median_ms": medians,
                       "serial_over_concurrent": medians["serial"] / medians["concurrent"],
                       "concurrent_compute_ms": c0.elapsed_time(c1), "concurrent_copy_ms": i0.elapsed_time(i1),
                       "copy_and_compute_validated": True}
                results.append(row)
                print(json.dumps({key: value for key, value in row.items() if key != "samples_ms"}), flush=True)
                if args.profile and copied == 16 and repeats == 128:
                    torch.cuda.cudart().cudaProfilerStart()
                    with torch.cuda.stream(compute_stream):
                        for mode in ("serial", "concurrent"):
                            with torch.cuda.nvtx.range(f"{mechanism}/{mode}/experts16/compute128"):
                                for _ in range(3):
                                    graphs[mode].replay()
                                compute_stream.synchronize()
                    compute_stream.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                args.output.write_text(json.dumps({
                    "scope": "isolated real layer24 expert kernels/inputs plus pinned expert weight transfer; not full inference",
                    "inputs": str(args.inputs), "sample_id": data["sample_id"], "round": data["round"],
                    "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "results": results,
                }, indent=2) + "\n")


if __name__ == "__main__":
    main()
