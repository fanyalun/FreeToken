"""CPU projection A/B evidence against an explicitly selected native binary."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

_affinity = os.sched_getaffinity(0)
import torch
os.sched_setaffinity(0, _affinity)


def digest(value):
    return hashlib.sha256(value).hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--threads", type=int, default=23)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 17, 31, 33])
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1 or min(args.rows) < 1:
        parser.error("threads, repeats and rows must be positive")
    cores = sorted(_affinity)[:args.threads]
    if len(cores) != args.threads:
        parser.error("insufficient CPUs in process affinity")
    os.sched_setaffinity(0, cores)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.extension:
        spec = importlib.util.spec_from_file_location("_cpu_moe", args.extension.resolve())
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
    else:
        from freetoken.kernel import _cpu_moe as native
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(command=[sys.executable, *sys.argv], purpose="diagnostic",
        measurement_scope="synthetic CPU projection including allocation and dispatch",
        validity="incomplete", threads=args.threads, cores=cores, torch=torch.__version__,
        extension=str(Path(native.__file__).resolve()),
        extension_sha256=digest(Path(native.__file__).read_bytes()),
        script_sha256=digest(Path(__file__).read_bytes()),
        thread_environment={k: os.environ.get(k) for k in
            ("GOMP_CPU_AFFINITY", "OMP_PROC_BIND", "OMP_WAIT_POLICY")},
        torch_config=torch.__config__.show(), results=[])
    target = args.output / "results.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    for n, k in [(1, 2048), (32, 2048), (256, 2048), (512, 2048),
                 (2048, 512), (8192, 2048), (2048, 8192), (248320, 2048)]:
        torch.manual_seed(97)
        weight = torch.randn(n, k)
        x = torch.randn(max(args.rows), k).bfloat16()
        for rows in args.rows:
            for argmax in (False, True):
                for _ in range(2):
                    result = native.target_linear(x[:rows], weight, argmax)
                elapsed = []
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    result = native.target_linear(x[:rows], weight, argmax)
                    elapsed.append(time.perf_counter() - started)
                record = dict(outputs=n, inputs=k, rows=rows, argmax=argmax,
                    seconds=elapsed, median_seconds=statistics.median(elapsed),
                    result_sha256=digest(result.contiguous().view(torch.uint8).numpy().tobytes()))
                manifest["results"].append(record)
                print(json.dumps(record), flush=True)
        target.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["validity"] = "valid"
    manifest["cuda_initialized"] = torch.cuda.is_initialized()
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "complete.json").write_text(json.dumps(dict(validity="valid",
        measurements=len(manifest["results"]), pipeline_complete=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
