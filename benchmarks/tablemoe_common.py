"""Shared HTTP streaming and fresh-server lifecycle for TableMoE benchmarks."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


METRIC_VERSION = "tablemoe_complete_wall_v2"
PREDICTION_PROTOCOL = "full_response_v1"


def prediction_text(record):
    return record.get("prediction_text", record.get("reasoning_text", "") + record["text"])


class FreeTokenAPIAdapter:
    def __init__(self, port, model, max_tokens=128):
        self.port, self.model, self.max_tokens = port, model, max_tokens
        self.records = []

    def generate_inner(self, message, dataset=None):
        from tablemoe_data import api_messages

        row = dict(id=f"{dataset or 'api'}:{len(self.records)}", messages=api_messages(message))
        result = self.request_row(row, self.max_tokens, natural=True)
        self.records.append(result)
        return 0, prediction_text(result), result

    def generate(self, message, dataset=None):
        return self.generate_inner(message, dataset)[1]

    def request_row(self, row, tokens, natural):
        return stream_request(self.port, row, tokens, natural=natural, model=self.model)


def read_stream(lines, start, clock=time.perf_counter):
    content, reasoning, output, times, usage, response_id, finish = [], [], [], [], None, None, None
    done = False
    for line in lines:
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            done = True
            break
        if not payload:
            continue
        event = json.loads(payload)
        if event.get("error"):
            raise RuntimeError(event["error"])
        if event.get("id"):
            if response_id is not None and response_id != event["id"]:
                raise ValueError("stream response ID changed")
            response_id = event["id"]
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            text, reason = delta.get("content") or "", delta.get("reasoning_content") or ""
            if text or reason:
                times.append(clock() - start)
                content.append(text)
                reasoning.append(reason)
                output.append(reason + text)
    wall = clock() - start
    if not done or usage is None or finish is None or response_id is None:
        raise ValueError(f"incomplete SSE: done={done}, usage={usage}, finish={finish}, id={response_id}")
    tokens = usage.get("completion_tokens")
    prompt = usage.get("prompt_tokens")
    if not isinstance(tokens, int) or tokens < 0 or not isinstance(prompt, int) or prompt < 1:
        raise ValueError(f"invalid token usage: {usage}")
    if tokens and not times:
        raise ValueError("completion usage without a nonempty output delta")
    ttft = times[0] if times else None
    tpot = (wall - ttft) * 1000 / (tokens - 1) if tokens > 1 else None
    span = (times[-1] - ttft) * 1000 / (tokens - 1) if tokens > 1 else None
    return dict(text="".join(content), reasoning_text="".join(reasoning), prediction_text="".join(output), usage=usage,
                response_id=response_id, finish_reason=finish, ttft_s=ttft, tpot_ms=tpot,
                stream_span_tpot_ms=span, complete_wall_s=wall, event_times_s=times,
                metric_version=METRIC_VERSION)


def stream_request(port, row, tokens, *, natural=False, model="Qwen3-VL-30B-A3B-Instruct", timeout=1800):
    body = dict(model=model, messages=row["messages"], max_tokens=tokens, ignore_eos=not natural,
                temperature=0, stream=True, stream_options={"include_usage": True})
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = read_stream(response, start)
    if result["usage"]["completion_tokens"] > tokens or (
            not natural and result["usage"]["completion_tokens"] != tokens):
        raise ValueError("returned token count violates generation protocol")
    result.update(id=row["id"], index=row.get("index"), max_tokens=tokens,
                  request_sha256=hashlib.sha256(data).hexdigest(), natural_eos=natural)
    if "reference_prompt_tokens" in row and result["usage"]["prompt_tokens"] != row["reference_prompt_tokens"]:
        raise ValueError("expanded prompt token count mismatch")
    return result


def stop(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)


def wait_ready(proc, log_path, timeout=900):
    deadline = time.monotonic() + timeout
    while "API server is ready to serve" not in Path(log_path).read_text(errors="replace"):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode}: {log_path}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"server startup: {log_path}")
        time.sleep(1)


@contextmanager
def running_server(command, root, repo, env, timeout=900):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "server.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=repo, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_ready(proc, root / "server.log", timeout)
            yield proc
        finally:
            stop(proc)


def compare_runs(timing, audit):
    if len(timing) != len(audit) or [r["id"] for r in timing] != [r["id"] for r in audit]:
        raise ValueError("timing/audit request coverage or order differs")
    timing = [{**record, "prediction_text": prediction_text(record)} for record in timing]
    audit = [{**record, "prediction_text": prediction_text(record)} for record in audit]
    keys = ("request_sha256", "text", "reasoning_text", "prediction_text", "usage", "finish_reason", "max_tokens")
    return [dict(id=t["id"], differing=[key for key in keys if t.get(key) != a.get(key)])
            for t, a in zip(timing, audit) if any(t.get(key) != a.get(key) for key in keys)]
