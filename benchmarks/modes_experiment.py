from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import urllib.request

import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.speculative import SpeculativeFrontier
from freetoken.mm.config import MultimodalConfig
from freetoken.scheduler.config import SchedulerConfig


REPO = Path(__file__).resolve().parents[1]
MODEL = "/data1/fanya/Qwen/Qwen3.6-35B-A3B"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def prepare(run, model=MODEL):
    run = Path(run)
    if (run / "selection.json").exists():
        selection = read(run / "selection.json")
        if selection["model_config_sha256"] != digest(Path(model) / "config.json"):
            raise ValueError("prepared model changed")
        if selection["train_sha256"] != digest(run / "gsm8k_train.jsonl"):
            raise ValueError("prepared training data changed")
        return selection
    run.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    raw = urllib.request.urlopen(url, timeout=90).read()
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 7473 or any(set(row) != {"question", "answer"} for row in rows):
        raise ValueError("unexpected GSM8K main train identity")
    order = list(range(len(rows)))
    random.Random(42).shuffle(order)
    test_path = Path("/home/fanya/replayssm_build_artifacts/gsm8k_test.jsonl")
    test = [json.loads(line) for line in test_path.read_text().splitlines()]
    test_order = list(range(2, len(test)))
    random.Random(42).shuffle(test_order)
    split_ids = dict(pilot=order[:16], calibration=order[16:80], development=order[80:96],
                     evaluation=[0, 1] + test_order[:30])
    index = Path(model) / "model.safetensors.index.json"
    shard_names = sorted(set(read(index)["weight_map"].values()))
    shard_stat = {name: dict(size=(Path(model)/name).stat().st_size,
                            mtime_ns=(Path(model)/name).stat().st_mtime_ns) for name in shard_names}
    train_questions = {row["question"] for row in rows}
    if any(row["question"] in train_questions for row in test):
        raise ValueError("GSM8K train/test question overlap")
    selection = dict(created_utc=datetime.now(timezone.utc).isoformat(), seed=42, model=str(Path(model).resolve()),
        model_config_sha256=digest(Path(model) / "config.json"), train_source=url,
        weight_index_sha256=digest(index), weight_shard_stat=shard_stat,
        train_sha256=hashlib.sha256(raw).hexdigest(), test_sha256=digest(test_path), ids=split_ids,
        splits={name: [dict(id=i, **(test if name == "evaluation" else rows)[i]) for i in ids]
                for name, ids in split_ids.items()})
    (run / "gsm8k_train.jsonl").write_bytes(raw)
    write(run / "selection.json", selection)
    geometry = {}
    for split in selection["splits"]:
        values, _ = prompts(selection, split)
        geometry[split] = dict(samples=len(values), min_input_tokens=min(map(len, values)),
                               max_input_tokens=max(map(len, values)))
    write(run / "input_validation.json", dict(status="validated", geometry=geometry, input_capacity=512))
    write(run / "complete.json", dict(status="incomplete", stage="prepared"))
    freeze(run / "runtime")
    return selection


def freeze(target):
    target = Path(target)
    suffixes = {".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".c", ".json"}
    files = [p for p in (REPO / "python/freetoken").rglob("*") if p.is_file() and p.suffix in suffixes]
    files += list((REPO / "benchmarks").glob("*.py"))
    files += [REPO / "AGENTS.md", REPO / "CONTRIBUTING.md", REPO / "docs/modes_preverify.md"]
    manifest = {}
    for source in files:
        relative = source.relative_to(REPO)
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        manifest[str(relative)] = digest(dest)
    for source in (REPO / "python/freetoken/kernel").glob("*.so"):
        dest = target / source.relative_to(REPO)
        shutil.copy2(source, dest)
        manifest[str(source.relative_to(REPO))] = digest(dest)
    write(target / "source_manifest.json", manifest)
    write(target / "source_identity.json", dict(commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        dirty=subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)))


def prompts(selection, split):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(selection["model"], local_files_only=True)
    values = [tokenizer.apply_chat_template([dict(role="user", content=row["question"])], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False) for row in selection["splits"][split]]
    if any(not isinstance(value, list) or not value or any(type(t) is not int for t in value) for value in values):
        raise ValueError("tokenizer did not return nonempty token ID lists")
    if max(map(len, values)) > 512:
        raise ValueError("GSM8K prompt exceeds explicit 512-token input capacity")
    return values, tokenizer


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig(SchedulerConfig):
    rendezvous_file: str = ""

    @property
    def distributed_addr(self):
        return "file://" + self.rendezvous_file


def fused_engine(run, model):
    return Engine(BenchmarkConfig(model_path=model, rendezvous_file=str(Path(run).resolve() / "rendezvous"),
        tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16, attention_backend="triton",
        max_running_req=1, moe_strategy="fused", moe_cache_auto=False, memory_ratio=.95,
        gpu_memory_budget_bytes=80 << 30, num_page_override=768, page_size=1, cache_type="naive",
        speculative_mtp_depth=3, speculative_max_input_tokens=512, speculative_max_output_tokens=256,
        speculative_three_level=True, speculative_max_loops=16, speculative_prefetch=False,
        speculative_reference_math=True,
        cuda_graph_max_bs=0, speculative_audit=False,
        mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio")))))


def initialize(runner, prompt):
    engine = runner.engine
    runner.active = False
    runner.three_level.experts.close()
    runner._reset_request_mapping()
    engine.linear_state_pool.recurrent_states.zero_()
    engine.linear_state_pool.conv_states.zero_()
    ids = torch.tensor(prompt, dtype=torch.int32)
    anchor = runner._prefill(ids, None)
    runner.frontier = SpeculativeFrontier(1, len(prompt), len(prompt), runner._reserved_len)
    runner.anchor_buffer.copy_(anchor)
    return runner.anchor_buffer.clone()


def checkpoint(runner):
    from bench_multiloop_sparse_verify import ModelSnapshot
    return ModelSnapshot(runner.engine, runner, 0, runner.engine.max_seq_len)


def verify_runtime(run):
    import freetoken
    runtime = Path(run).resolve() / "runtime"
    imported = Path(freetoken.__file__).resolve()
    if not imported.is_relative_to(runtime):
        raise ValueError("measurement must import the frozen runtime; run its benchmark with its PYTHONPATH")
    manifest = read(runtime / "source_manifest.json")
    changed = [name for name, expected in manifest.items() if digest(runtime / name) != expected]
    if changed:
        raise ValueError(f"frozen runtime fingerprint changed: {changed}")
    write(Path(run) / "runtime_binding.json", dict(imported_freetoken=str(imported),
        source_manifest_sha256=digest(runtime / "source_manifest.json"), verified_files=len(manifest),
        torch=str(torch.__version__), cuda=torch.version.cuda))
