"""Frozen five-method MMLongBench-Write quality campaign using the FreeToken engine."""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time


METHODS = ("baseline", "tablemoe_06_top4", "tablemoe_07_top6", "tablemoe_06_top4_512_top6", "adapmoe_gating_top5")


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_rows(path, values):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text("".join(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n" for value in values))
    temporary.replace(path)


def policy_options(method):
    options = dict(tablemoe_prefill_keep_rate=1., tablemoe_decode_keep_k=6)
    if method == "tablemoe_06_top4":
        options.update(tablemoe_prefill_keep_rate=.6, tablemoe_decode_keep_k=4)
    elif method == "tablemoe_07_top6":
        options.update(tablemoe_prefill_keep_rate=.7)
    elif method == "tablemoe_06_top4_512_top6":
        options.update(tablemoe_prefill_keep_rate=.6, tablemoe_decode_keep_k=4,
                       tablemoe_decode_switch_after=512, tablemoe_decode_late_top_h=6)
    elif method == "adapmoe_gating_top5":
        options.update(deepseek_vl2_router_top_k=5, tablemoe_decode_keep_k=5)
    elif method != "baseline":
        raise ValueError(f"unknown method: {method}")
    return options


def prepare(config):
    import torch
    from freetoken.mm.config import MultimodalConfig
    from freetoken.mm.processor import get_mm_processor
    from freetoken.utils import load_tokenizer

    root = Path(config["output"])
    source = Path(config["previous_campaign"]) / "prepare/MMLongBench-Write"
    evaluation = rows(source / "evaluation.jsonl")
    original = {item["id"]: item for item in rows(Path(config["previous_campaign"]) / "cells/baseline/requests.jsonl")}
    counts = Counter(row["metadata"]["category"] for row in evaluation)
    if len(evaluation) != 48 or len(counts) != 6 or set(counts.values()) != {8}:
        raise ValueError(f"unexpected frozen cohort: {counts}")
    tokenizer = load_tokenizer(config["model"])
    processor = get_mm_processor(config["model"], MultimodalConfig())
    prepared = []
    for row in evaluation:
        content = "".join("<image>\n" if item["type"] == "image" else item["value"] for item in row["vlmeval_message"])
        image_paths = [item["value"] for item in row["vlmeval_message"] if item["type"] == "image"]
        if image_paths != [item["path"] for item in row["images"]]:
            raise ValueError(f"image order differs: {row['id']}")
        prompt = tokenizer.apply_chat_template([dict(role="user", content=content)], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        images = [Path(path).read_bytes() for path in image_paths]
        mm = processor.apply(torch.tensor(prompt_ids, dtype=torch.int32), images)
        expected = original[row["id"]]["usage"]["prompt_tokens"]
        if len(mm.input_ids) != expected:
            raise ValueError(f"native prompt length mismatch: {row['id']}: {len(mm.input_ids)} != {expected}")
        prepared.append({**row, "actual_prompt": prompt, "prompt_token_ids": prompt_ids,
                         "expanded_prompt_tokens": len(mm.input_ids),
                         "image_crops": [item.model_specific_data["crop"] for item in mm.mm_items]})
    folder = root / "prepare/MMLongBench-Write"
    folder.mkdir(parents=True, exist_ok=True)
    write_rows(folder / "evaluation.jsonl", prepared)
    (folder / "split.json").write_bytes((source / "split.json").read_bytes())
    maximum = max(item["expanded_prompt_tokens"] for item in prepared)
    config.update(max_prompt_tokens=maximum, max_extend_tokens=maximum,
                  max_seq_len=((maximum + config["max_tokens"] + 127) // 128) * 128)
    save(root / "config.json", config)
    save(root / "prepare/complete.json", dict(complete=True, samples=48, category_samples=dict(Counter(
        item["metadata"]["category"] for item in prepared)), max_prompt_tokens=maximum,
        prompt_lengths_equal_native=True, official_prompts_preserved=True, image_order_preserved=True))
    print("PREPARED", maximum, config["max_seq_len"], flush=True)


def restore_initial_residency(llm, initial):
    import torch

    cache = llm.engine.tablemoe_runtime.expert_cache
    torch.cuda.synchronize()
    cache.reset()
    for bank in range(cache.num_layers):
        ids = torch.tensor([initial[str(bank + 1)]], dtype=torch.int32, device=llm.device)
        cache.ensure_experts(bank, ids)
        cache.copy_missing()
    torch.cuda.synchronize()
    observed = cache.slot_for_id.ge(0).cpu()
    expected = torch.zeros_like(observed)
    for bank in range(cache.num_layers):
        expected[bank, initial[str(bank + 1)]] = True
    if not torch.equal(observed, expected):
        raise ValueError("initial expert residency differs from the frozen 36-per-layer set")
    runtime = llm.engine.tablemoe_runtime
    if runtime._owners or runtime._uid_slots:
        raise ValueError("previous request did not release TableMoE online state")
    for cache_state in runtime.fixed_online.values():
        cache_state["size"].zero_()
        cache_state["pointer"].zero_()
    cache.reset_stats()


def worker(config, method, gpu):
    import torch
    from freetoken.core import SamplingParams
    from freetoken.llm import LLM
    from freetoken.message import DetokenizeMsg
    from freetoken.mm.config import MultimodalConfig

    root = Path(config["output"])
    folder = root / "cells" / method
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "generation_complete.json").exists():
        return
    torch.set_num_threads(8)
    torch.manual_seed(42)

    class RecordingLLM(LLM):
        def offline_send_result(self, replies):
            for reply in replies:
                if isinstance(reply, DetokenizeMsg):
                    self.returned.append(int(reply.next_token))
                    self.last_finished = bool(reply.finished)
                    if self.report and len(self.returned) % 128 == 0:
                        save(folder / "progress.json", dict(stage="generating", method=method, id=self.row_id,
                             samples=self.completed_samples, generated_tokens=len(self.returned), elapsed_s=time.monotonic()-self.generation_started))
            super().offline_send_result(replies)

    audit_path = folder / f"routes_attempt_{os.getpid()}.jsonl"
    llm = RecordingLLM(model_path=config["model"], dtype=torch.bfloat16, moe_strategy="tablemoe",
        tablemoe_expert_mode="offload", moe_cache_size=config["expert_slots"],
        tablemoe_pca_dir=config["pca_dir"], tablemoe_cache_dir=config["cache_dir"],
        tablemoe_implementation="optimized", tablemoe_renormalize_weights=False,
        tablemoe_online_cache_size=128, tablemoe_offline_cosine_threshold_vision=.6,
        tablemoe_offline_cosine_threshold_text=.7, tablemoe_decode_online_cosine_threshold=.7,
        tablemoe_decode_offline_cosine_threshold=.7, tablemoe_report_path=str(audit_path),
        tablemoe_stats=False, attention_backend="triton", max_running_req=1, cuda_graph_max_bs=1,
        max_extend_tokens=config["max_extend_tokens"], max_seq_len_override=config["max_seq_len"],
        num_page_override=config["max_seq_len"], cache_type="naive", memory_ratio=.9,
        mm=MultimodalConfig(encoder_weights="gpu"), **policy_options(method))
    runtime = llm.engine.tablemoe_runtime
    expected_top_k = 5 if method == "adapmoe_gating_top5" else 6
    if any(layer.mlp._config.num_experts_per_tok != expected_top_k for layer in llm.engine.model.model.layers.op_list[1:]):
        raise ValueError("native gate top-k differs from the requested method")
    if set(llm.engine.graph_runner.graph_map) != {1}:
        raise ValueError("decode CUDA Graph bs=1 was not captured")
    execution = dict(method=method, physical_gpu=gpu, pid=os.getpid(), options=policy_options(method),
                     model_module=type(llm.engine.model).__module__, model_source=__import__(type(llm.engine.model).__module__,fromlist=['x']).__file__,
                     engine_config=repr(llm.engine.config), audit_path=str(audit_path), native_gate_top_k=expected_top_k,
                     offline_files=len(runtime.offline_paths), graph_bs=[1])
    save(folder / "resolved_runtime.json", execution)
    evaluation = rows(root / "prepare/MMLongBench-Write/evaluation.jsonl")
    initial = read(root / "initial_resident_experts.json")

    def generate(row, maximum, report=True):
        restore_initial_residency(llm, initial)
        llm.returned, llm.last_finished, llm.report, llm.row_id = [], False, report, row["id"]
        llm.generation_started = time.monotonic()
        images = [Path(item["path"]).read_bytes() for item in row["images"]]
        result = llm.generate([row["prompt_token_ids"]], SamplingParams(max_tokens=maximum, temperature=0), images=[images])[0]
        elapsed = time.monotonic()-llm.generation_started
        content_ids = result["token_ids"]
        stopped = bool(llm.returned and llm.returned[-1] in llm.eos_token_ids)
        if not llm.last_finished or not llm.returned or (not stopped and len(llm.returned) != maximum):
            raise ValueError(f"incomplete output budget: {row['id']}: {len(llm.returned)} / {maximum}")
        text = llm.tokenizer.decode(content_ids, skip_special_tokens=True)
        return dict(id=row["id"], index=row["index"], text=text, prediction_text=text, reasoning_text="",
                    output_token_ids=content_ids, returned_token_ids=llm.returned,
                    raw_decoded_text=result["text"], usage=dict(prompt_tokens=row["expanded_prompt_tokens"],
                    completion_tokens=len(llm.returned), total_tokens=row["expanded_prompt_tokens"]+len(llm.returned)),
                    finish_reason="stop" if stopped else "length", max_tokens=maximum, natural_eos=True,
                    elapsed_s=elapsed, engine_backend="freetoken", physical_gpu=gpu,
                    source_manifest=str(root / "source_manifest.json"), per_request_cache_reset=True)

    reference = read(root / "native_reference.json")
    reference_row = next(row for row in evaluation if row["id"] == reference["id"])
    if not (folder / "policy_smoke.json").exists():
        probe = generate(reference_row, 32, False)
        save(folder / "policy_smoke.json", probe)
        if method == "baseline" and probe["output_token_ids"] != reference["tokens"]:
            raise ValueError("FreeToken baseline differs from saved 32-token native reference")
    previous = rows(folder / "requests.jsonl") if (folder / "requests.jsonl").exists() else []
    if [item["id"] for item in previous] != [row["id"] for row in evaluation[:len(previous)]]:
        raise ValueError("resume predictions do not match the frozen cohort order")
    for row in evaluation[len(previous):]:
        llm.completed_samples = len(previous)
        previous.append(generate(row, config["max_tokens"]))
        write_rows(folder / "requests.jsonl", previous)
        save(folder / "progress.json", dict(stage="request_complete", method=method, samples=len(previous),
             total=48, id=row["id"], usage=previous[-1]["usage"], elapsed_s=previous[-1]["elapsed_s"]))
        print("COMPLETED", method, len(previous), row["id"], previous[-1]["usage"], flush=True)
    save(folder / "generation_complete.json", dict(complete=True,samples=48,method=method,physical_gpu=gpu,
        native_gate_top_k=expected_top_k,per_request_cache_reset=True,graph_bs=1,
        initial_residency=str(root / "initial_resident_experts.json"),formal_performance_eligible=False))
    llm.engine.shutdown()


def environment(config, gpu):
    root = Path(config["output"])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TMPDIR": str(root / "tmp"),
           "CUDA_HOME": "/usr/local/cuda-13.0", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
           "PYTHONPATH": str(root / "runtime_snapshot/python") + ":" + str(root / "runtime_snapshot/benchmarks"),
           "PATH": str(Path(config["python"]).parent) + ":/usr/local/cuda-13.0/bin:" + os.environ["PATH"]}
    return env


def admitted(gpu):
    value = subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=memory.used,utilization.gpu",
                                     "--format=csv,noheader,nounits"],text=True)
    memory, utilization = map(int,value.strip().split(","))
    return memory < 512 and utilization == 0


def score_all(config):
    root = Path(config["output"])
    for gpu in config["gpus"]:
        if admitted(gpu):
            break
    else:
        raise RuntimeError("no idle GPU for the judge")
    env = environment(config,gpu)
    env["PYTHONPATH"] = str(root / "judge_runtime_snapshot/python") + ":" + str(root / "judge_runtime_snapshot/benchmarks")
    command = [config["python"],"-m","freetoken.cli","serve","--model",config["judge_model"],"--gpu","0",
        "--dtype","bfloat16","--moe-strategy","fused","--moe-cache-size","0","--cuda-graph-max-bs","4",
        "--max-running-requests","4","--num-pages","131072","--page-size","1","--cache-type","naive",
        "--mm-encoder-weights","gpu","--max-prefill-length","4096","--max-seq-len-override","65536",
        "--port",str(config["judge_port"]),"--mm-processor-kwargs",json.dumps(dict(size=dict(shortest_edge=602112,longest_edge=4014080))),
        "--reasoning-parser","off"]
    with (root / "logs/judge_server.log").open("a") as log:
        judge = subprocess.Popen(command,cwd=root / "judge_runtime_snapshot",env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root / "judge_child.json",dict(pid=judge.pid,physical_gpu=gpu,command=command))
        try:
            import urllib.request
            for _ in range(240):
                if judge.poll() is not None:
                    raise RuntimeError("judge exited before readiness")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{config['judge_port']}/health",timeout=3) as response:
                        if json.load(response).get("status") == "ok":
                            break
                except OSError:
                    pass
                time.sleep(3)
            else:
                raise TimeoutError("judge startup")
            for method in METHODS:
                folder = root / "cells" / method
                scoring = environment(config,gpu)
                scoring.update(LD_LIBRARY_PATH="/home/fanya/miniforge/envs/tablemoe/lib",LMUData="/home/fanya/tablemoe/LMUData")
                for name in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
                    scoring.pop(name,None)
                with (root / "logs" / f"{method}_judge.log").open("a") as output:
                    subprocess.run([config["kit_python"],str(root / "kit_worker.py"),"score","--dataset","MMLongBench-Write",
                                    "--folder",str(folder),"--port",str(config["judge_port"])],cwd=root,env=scoring,
                                   stdout=output,stderr=subprocess.STDOUT,check=True)
        finally:
            if judge.poll() is None:
                os.killpg(judge.pid,signal.SIGTERM)
                try:
                    judge.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(judge.pid,signal.SIGKILL)
                    judge.wait()
    comparison = {method:read(root / "cells" / method / "score/score.json") for method in METHODS}
    if any(not score["complete"] or score["scoring_denominator"] != 48 for score in comparison.values()):
        raise ValueError("incomplete quality scoring coverage")
    save(root / "comparison.json",comparison)
    save(root / "complete.json",dict(complete=True,samples_per_method=48,scores=comparison,formal_performance_eligible=False))


def run(config):
    root = Path(config["output"])
    manifest = read(root / "source_manifest.json")
    for name, expected in manifest["sources"].items():
        if hashlib.sha256((root / "runtime_snapshot" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen source changed: {name}")
    state = dict(stage="generation",methods={method:dict(stage="queued") for method in METHODS},workers={},failures=[])
    lock = threading.RLock()
    jobs = queue.Queue()
    for method in METHODS:
        jobs.put(method)

    def publish():
        with lock:
            state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
            save(root / "status.json",state)

    def lane(gpu):
        while True:
            try:
                method = jobs.get_nowait()
            except queue.Empty:
                with lock: state["workers"][str(gpu)] = dict(stage="lane_complete")
                publish()
                return
            folder = root / "cells" / method
            if (folder / "generation_complete.json").exists():
                with lock: state["methods"][method] = dict(stage="generation_complete",physical_gpu=gpu)
                jobs.task_done()
                continue
            try:
                with lock: state["workers"][str(gpu)] = dict(stage="waiting_for_gpu",method=method)
                while not admitted(gpu):
                    publish()
                    time.sleep(10)
                for attempt in range(2):
                    command = [config["python"],"-u",str(root / "runtime_snapshot/benchmarks" / Path(__file__).name),
                               "worker","--config",str(root / "config.json"),"--method",method,"--gpu",str(gpu)]
                    with (root / "logs" / f"{method}.log").open("a") as output:
                        child = subprocess.Popen(command,cwd=root / "runtime_snapshot",env=environment(config,gpu),
                                                 stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                        with lock:
                            state["methods"][method] = dict(stage="generating",physical_gpu=gpu,pid=child.pid,attempt=attempt+1)
                            state["workers"][str(gpu)] = dict(stage="generating",method=method,pid=child.pid)
                        save(root / f"child_gpu{gpu}.json",dict(pid=child.pid,method=method,command=command))
                        publish()
                        code = child.wait()
                    if code == 0:
                        break
                    time.sleep(3)
                else:
                    raise RuntimeError(f"worker failed twice: {method}")
                with lock: state["methods"][method].update(stage="generation_complete",samples=48)
            except BaseException as error:
                with lock:
                    state["methods"][method].update(stage="failed",error=str(error))
                    state["failures"].append(dict(method=method,error=str(error)))
            finally:
                jobs.task_done()
                publish()

    workers = [threading.Thread(target=lane,args=(gpu,)) for gpu in config["gpus"]]
    for thread in workers:
        thread.start()
    while any(thread.is_alive() for thread in workers):
        publish()
        time.sleep(10)
    if state["failures"]:
        raise RuntimeError(state["failures"])
    state["stage"] = "judging"
    publish()
    score_all(config)
    state["stage"] = "complete"
    publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","worker","run"))
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--method",choices=METHODS)
    parser.add_argument("--gpu",type=int)
    args = parser.parse_args()
    config = read(args.config)
    if args.action == "prepare":
        prepare(config)
    elif args.action == "worker":
        worker(config,args.method,args.gpu)
    else:
        with (Path(config["output"]) / ".campaign.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run(config)
            except BaseException:
                import traceback

                root = Path(config["output"])
                state = read(root / "status.json") if (root / "status.json").exists() else {}
                state.update(stage="failed",error=traceback.format_exc())
                save(root / "status.json",state)
                raise


if __name__ == "__main__":
    main()
