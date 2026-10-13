"""Real-model Graph, request lifecycle and live-cache rebuild gates."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request

from bench_tablemoe_matrix import REPO, stop


def call(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def payload():
    return dict(model="Qwen3-VL-30B-A3B-Instruct", messages=[dict(role="user", content="Count upwards from one.")],
                temperature=0, max_tokens=64, ignore_eos=True)


def generate(base):
    code, result = call(base, "/v1/chat/completions", payload())
    assert code == 200 and result["usage"]["completion_tokens"] == 64, (code, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19760)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    marker = dict(complete=False, modes=[])
    try:
        for mode in ("offload", "hybrid"):
            root = (args.output / mode).resolve()
            root.mkdir()
            command = json.loads((args.matrix / f"table_{mode}_graph/timing_0/command.json").read_text())["args"]
            command[command.index("--port") + 1] = str(args.port)
            env = os.environ.copy()
            env.update(CUDA_HOME="/usr/local/cuda-13.0", TABLEMOE_AUDIT_DIR=str(root), TABLEMOE_VERIFY="1",
                       PYTHONPATH=str(REPO / "benchmarks/tablemoe_audit") + ":" + str(REPO / "python"))
            env["PATH"] = str(REPO / ".venv/bin") + ":/usr/local/cuda-13.0/bin:" + env["PATH"]
            (root / "command.json").write_text(json.dumps(command, indent=2))
            base = f"http://127.0.0.1:{args.port}"
            with (root / "server.log").open("w") as log:
                proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try:
                    deadline = time.monotonic() + 900
                    while "API server is ready to serve" not in (root / "server.log").read_text(errors="replace"):
                        if proc.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError(f"server startup failed: {root}")
                        time.sleep(2)
                    baseline = generate(base)
                    before = call(base, "/v1/cache/status")[1]
                    code, rejected = call(base, "/v1/cache/rebuild", dict(num_pages=99_999_999))
                    assert (code, rejected["status"]) == (503, "rejected"), rejected
                    after_rejection = call(base, "/v1/cache/status")[1]
                    assert after_rejection["state"] == "serving"
                    assert after_rejection["geometry"] == before["geometry"]
                    generate(base)
                    code, rebuilt = call(base, "/v1/cache/rebuild", dict(num_pages=65536, moe_cache_size=3072))
                    assert (code, rebuilt["status"]) == (200, "ok"), rebuilt
                    after_rebuild = call(base, "/v1/cache/status")[1]
                    assert after_rebuild["state"] == "serving"
                    results = []
                    for bs in (1, 2, 3, 4):
                        barrier = threading.Barrier(bs)

                        def task(_):
                            barrier.wait()
                            return generate(base)

                        with ThreadPoolExecutor(max_workers=bs) as pool:
                            results.extend(pool.map(task, range(bs)))
                    body = payload()
                    body["stream"] = True
                    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                                     headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=900) as response:
                        for line in response:
                            if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                                event = json.loads(line[5:])
                                if any(c.get("delta", {}).get("content") for c in event.get("choices", [])):
                                    break
                    time.sleep(2)
                    generate(base)
                    observed = json.loads((root / "observed_batches.json").read_text())
                    assert {(r[0], r[1], r[2]) for r in observed} >= {(1, 1, True), (2, 2, True), (3, 4, True), (4, 4, True)}, observed
                    (root / "results.json").write_text(json.dumps(dict(before=before, rejected=rejected,
                        after_rejection=after_rejection, rebuilt=rebuilt, after_rebuild=after_rebuild,
                        baseline=baseline, concurrent=results, aborted_stream_recovered=True), indent=2))
                    marker["modes"].append(mode)
                finally:
                    stop(proc)
        marker["complete"] = True
    except Exception as exc:
        marker["error"] = repr(exc)
        raise
    finally:
        (args.output / "complete.json").write_text(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
