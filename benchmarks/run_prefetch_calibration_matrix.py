from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


MODELS = {
    "qwen36": "/data1/fanya/Qwen/Qwen3.6-35B-A3B",
    "qwen235": "/data/fanya/models/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF/Q4_K_M",
}
DRAFT = "/data/fanya/models/RedHatAI/Qwen3-235B-A22B-Instruct-2507-speculator.eagle3"
SOURCE_DATA = "/home/fanya/freetoken_benchmark_results/speculative/offload_top4_loop4_32gib_20261023_092323/data"
SOURCE_DIRECTORIES = ("python", "benchmarks", "tests")
ROOT_FILES = ("pyproject.toml", "pytest.ini", "setup.cfg", "setup.py", "conftest.py")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def persistent(path):
    path = Path(path).expanduser().resolve()
    if path == Path("/tmp") or Path("/tmp") in path.parents:
        raise ValueError("all calibration outputs must use persistent paths outside /tmp")
    return path


def measured(path):
    return [row for row in (json.loads(line) for line in Path(path).read_text().splitlines()) if not row["warmup"]]


class Matrix:
    def __init__(self, args):
        self.args = args
        self.root = persistent(args.output)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / "manager.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.runtime = self.root / "runtime"
        self.state_path = self.root / "state.json"
        self.state = read(self.state_path) if self.state_path.exists() else dict(status="queued", completed=[])
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_HOME=args.cuda_home,
            TMPDIR=str(self.root / "tmp"), PYTHONPATH=str(self.runtime / "python") + os.pathsep + str(self.runtime / "benchmarks"),
            TORCH_EXTENSIONS_DIR=str(self.root / "cache/torch_extensions"),
            TRITON_CACHE_DIR=str(self.root / "cache/triton"), CUDA_CACHE_PATH=str(self.root / "cache/cuda"),
            TORCHINDUCTOR_CACHE_DIR=str(self.root / "cache/inductor"),
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", MAX_JOBS="4")
        self.env["PATH"] = str(Path(args.python).parent) + os.pathsep + args.cuda_home + "/bin:" + self.env.get("PATH", "")
        self.update(status="queued", manager_pid=os.getpid())

    def update(self, **fields):
        self.state.update(fields, updated_at=time.time())
        write(self.state_path, self.state)

    def wait_previous(self):
        if not self.args.wait_run:
            return
        path = Path(self.args.wait_run).resolve() / "state.json"
        while True:
            if not path.is_file():
                raise ValueError(f"previous run state is missing: {path}")
            previous = read(path)
            status = previous.get("status")
            if status == "complete":
                return
            if status in ("failed", "error", "cancelled", "stopped"):
                raise RuntimeError(f"previous run ended with {status}; inspect {path}")
            self.update(status="waiting_for_previous", previous_run=str(path.parent), previous_status=status)
            time.sleep(30)

    def freeze(self):
        protocol = self.root / "protocol.json"
        if protocol.exists():
            self.protocol = read(protocol)
            self.verify_source()
            return
        if self.runtime.exists():
            raise RuntimeError("partial source freeze is preserved; use a new output directory")
        repo = Path(__file__).resolve().parents[1]
        if self.args.expected_source_manifest:
            expected = read(self.args.expected_source_manifest)
            current = {str(path.relative_to(repo)) for directory in SOURCE_DIRECTORIES
                       for path in (repo / directory).rglob("*") if path.is_file()
                       and "__pycache__" not in path.parts and path.suffix != ".pyc"}
            current.update(name for name in ROOT_FILES if (repo / name).is_file())
            expected_scope = {name for name in expected if name in ROOT_FILES or
                              any(name.startswith(directory + "/") for directory in SOURCE_DIRECTORIES)}
            if current != expected_scope:
                differences = dict(added=sorted(current - expected_scope), removed=sorted(expected_scope - current))
                self.update(source_changed=True, source_set_changes=differences)
                raise RuntimeError("queued source file set changed before freeze; see state.json")
            for name, value in expected.items():
                path = repo / name
                if not path.is_file() or digest(path) != value:
                    self.update(source_changed=True, changed_path=name)
                    raise RuntimeError(f"queued source changed before freeze: {name}")
        self.runtime.mkdir()
        for name in SOURCE_DIRECTORIES:
            shutil.copytree(repo / name, self.runtime / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ROOT_FILES:
            if (repo / name).is_file():
                shutil.copy2(repo / name, self.runtime / name)
        catalog_tool = Path.home() / ".codex/skills/experiment-results-archive/scripts/results_catalog.py"
        (self.runtime / "tools").mkdir()
        shutil.copy2(catalog_tool, self.runtime / "tools/results_catalog.py")
        if self.args.expected_source_manifest:
            shutil.copy2(self.args.expected_source_manifest, self.root / "expected_source_manifest.json")
        revision = dict(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                        git_status=subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True))
        write(self.runtime / "source_revision.json", revision)
        records = {}
        for dataset in ("gsm8k", "humaneval"):
            source = Path(self.args.data_dir) / (dataset + ".jsonl")
            lines = source.read_text().splitlines()
            if len(lines) < 64:
                raise ValueError(f"source dataset needs the frozen 64 prompts: {source}")
            records[dataset] = dict(source=str(source.resolve()), sha256=digest(source), splits={})
            for phase, start, end in (("calibration", 16, 24), ("development", 24, 32), ("formal", 0, 16)):
                target = self.root / "data" / phase / (dataset + ".jsonl")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("\n".join(lines[start:end]) + "\n")
                records[dataset]["splits"][phase] = dict(path=str(target), sha256=digest(target),
                                                          source_rows=list(range(start + 1, end + 1)))
                if phase == "formal" and self.args.wait_run:
                    previous = Path(self.args.wait_run) / "data" / (dataset + ".jsonl")
                    if not previous.is_file() or previous.read_text().splitlines() != lines[:16]:
                        raise ValueError(f"formal inputs differ from previous first sixteen prompts: {dataset}")
        manifest = {str(path.relative_to(self.runtime)): digest(path)
                    for path in sorted(self.runtime.rglob("*")) if path.is_file()}
        write(self.root / "source_manifest.json", manifest)
        self.protocol = dict(schema_version=1, datasets=records, models=self.args.models, draft=self.args.draft,
            python=str(Path(self.args.python).absolute()), gpu=self.args.gpu, budget_gib=40, bootstrap_budget_gib=38,
            max_loops=4, preverify_top_h=4, depth=3, graph="full", backend="offload", cache_scope="unified",
            max_input_tokens=4096, max_output_tokens=1024, greedy=True, normal_eos=True,
            development=dict(samples_per_dataset=8, warmups=1, repeats=3),
            formal=dict(samples_per_dataset=16, warmups=2, repeats=3, requests_per_case=96),
            variants=["none", "legacy", "old", "calibrated", "half"], formal_variants=["sd", "none", "legacy", "selected"],
            resource_policy="python", nsys=self.args.nsys, source=revision,
            selection="per-dataset promotion gates; pooled throughput; within 1 percent choose lower logical bytes per token",
            fallback="zero quotas require valid correctness/accounting, coverage, complete formal requests and a no-gain receipt",
            strict_ar_parity="not_established")
        write(protocol, self.protocol)
        write(self.root / "environment.json", dict(python=self.args.python, environment={k: self.env[k] for k in
            ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "PYTHONPATH", "TMPDIR", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR")}))
        for name in ("tmp", "logs", "reports"):
            (self.root / name).mkdir(exist_ok=True)
        self.update(status="prepared")

    def verify_source(self):
        for name, expected in read(self.root / "source_manifest.json").items():
            path = self.runtime / name
            if not path.is_file() or digest(path) != expected:
                raise RuntimeError(f"frozen source changed: {name}")
        for dataset in self.protocol["datasets"].values():
            for split in dataset["splits"].values():
                if digest(split["path"]) != split["sha256"]:
                    raise RuntimeError("frozen dataset changed")

    def resource_gate(self):
        from bench_speculative_modes import resource_snapshot, resource_interference, resource_blocked

        while True:
            samples = []
            for _ in range(2):
                snapshot = resource_snapshot(1)
                samples.append(dict(resources=snapshot, interference=resource_interference(snapshot, {os.getpid()})))
            write(self.root / "resource_gate.json", dict(recorded_at=time.time(), samples=samples))
            if not any(resource_blocked(sample["interference"], "python") for sample in samples):
                return
            self.update(status="waiting_for_resources")
            time.sleep(30)

    def execute(self, name, command, *, output, formal=False, test=False, env_overrides=None):
        output = Path(output)
        marker = output / "complete.json"
        expected_command = dict(argv=command, cwd=str(self.runtime), output=str(output), env_overrides=env_overrides or {})
        command_path = self.root / "commands" / (name + ".json")
        if command_path.exists() and read(command_path) != expected_command:
            raise RuntimeError(f"resume command differs from frozen case: {name}")
        if marker.exists():
            if not command_path.is_file():
                raise RuntimeError(f"completed case lacks its frozen command receipt: {name}")
            receipt = read(marker)
            self.check_receipt(receipt, formal=formal, test=test)
            if name not in self.state["completed"]:
                self.state["completed"].append(name)
                self.update()
            return receipt
        if output.exists() or (self.root / "logs" / (name + ".log")).exists():
            raise RuntimeError(f"incomplete case preserved without overwrite: {name}")
        self.verify_source()
        self.resource_gate()
        write(command_path, expected_command)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.update(status="running", current_case=name)
        with (self.root / "logs" / (name + ".log")).open("x") as log:
            process = subprocess.Popen(command, cwd=self.runtime, env={**self.env, **(env_overrides or {})},
                                       stdout=log, stderr=subprocess.STDOUT)
            while True:
                self.update(case_pid=process.pid)
                try:
                    process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    continue
        self.state.pop("case_pid", None)
        if process.returncode:
            raise RuntimeError(f"case {name} exited with {process.returncode}; see its preserved log")
        if test:
            write(marker, dict(status="complete", valid=True, exit_code=0))
        if not marker.is_file():
            raise RuntimeError(f"case has no completion receipt: {name}")
        receipt = read(marker)
        self.check_receipt(receipt, formal=formal, test=test)
        self.state["completed"].append(name)
        self.update()
        return receipt

    @staticmethod
    def check_receipt(receipt, *, formal=False, test=False):
        if receipt.get("status") != "complete" or receipt.get("valid") is not True:
            raise RuntimeError("case completion or validity failed")
        if test:
            return
        if (receipt.get("source_changed") is not False or receipt.get("shutdown_complete") is not True
                or receipt.get("resource_eligible") is not True or (formal and receipt.get("formal_eligible") is not True)):
            raise RuntimeError("case source, shutdown or resource eligibility failed")

    def base_command(self, model, phase, output, *, mode="three_level_gpu"):
        command = [self.args.python, str(self.runtime / "benchmarks/bench_speculative_modes.py"),
            "--model", self.protocol["models"][model], "--mode", mode, "--backend", "offload", "--graph", "full",
            "--depth", "3", "--preverify-top-h", "4", "--max-loops", "4", "--budget-gib", "40",
            "--max-input-tokens", "4096", "--max-output-tokens", "1024", "--tokens", "1024",
            "--cpu-interference-policy", "python", "--output", str(output)]
        for dataset in ("gsm8k", "humaneval"):
            command += ["--data", self.protocol["datasets"][dataset]["splits"][phase]["path"]]
        samples, warmups = (16, 2) if phase == "formal" else (8, 1)
        command += ["--samples", str(samples), "--warmups", str(warmups), "--repeats", "3",
                    "--purpose", "diagnostic" if phase == "calibration" else "formal"]
        if model == "qwen235":
            command += ["--draft-model", self.protocol["draft"]]
        return command

    def create_variants(self, model):
        from freetoken.engine.speculative_prefetch_profile import validate_profile_data

        directory = self.root / model / "calibration"
        calibrated = read(directory / "profile_calibrated.json")
        half = read(directory / "profile_half.json")
        validate_profile_data(calibrated, allow_candidate=True)
        validate_profile_data(half, calibrated["identity"], allow_candidate=True)
        if half["geometry"] != calibrated["geometry"]:
            raise RuntimeError("calibration candidates have different cache geometry")
        count = len(calibrated["quotas"]["preverify"])
        variants = {"calibrated": calibrated, "half": half}
        for name, layer, gap, verify in (("none", 0, 0, 0), ("legacy", 1, 12, 2), ("old", 4, 0, 0)):
            data = copy.deepcopy(calibrated)
            data["quotas"] = dict(preverify=[layer] * count, gap_capacity=gap, verify=[0] + [verify] * (count - 1))
            data["evidence"].update(status="candidate", variant=name)
            variants[name] = data
        paths = {}
        for name, data in variants.items():
            path = self.root / model / "profiles" / (name + ".json")
            if path.exists() and read(path) != data:
                raise RuntimeError("resume candidate differs from preserved profile")
            if not path.exists():
                write(path, data)
            paths[name] = path
        return paths

    def promotion(self, baseline, candidate, coverage):
        from three_level_prefetch_calibration import evaluate_prefetch_promotion

        left, right = measured(baseline / "requests.jsonl"), measured(candidate / "requests.jsonl")
        a, b = read(baseline / "complete.json"), read(candidate / "complete.json")
        protocol_a, protocol_b = read(baseline / "protocol.json"), read(candidate / "protocol.json")
        source_a, source_b = protocol_a.get("source_commit"), protocol_b.get("source_commit")
        geometry_a, geometry_b = read(baseline / "memory_plan.json"), read(candidate / "memory_plan.json")
        fixed = ("model", "draft_model", "mode", "backend", "graph", "cache_scope", "depth", "preverify_top_h",
                 "max_loops", "budget_gib", "max_input_tokens", "max_output_tokens", "tokens", "samples",
                 "warmups", "repeats", "ignore_eos", "prompt_chunk_size", "cpu_interference_policy")
        arguments_a, arguments_b = protocol_a["arguments"], protocol_b["arguments"]
        pools_a = read(baseline / "prefetch_receipt.json")["actual_geometry"]
        pools_b = read(candidate / "prefetch_receipt.json")["actual_geometry"]
        audit = dict(baseline_complete=a.get("valid") is True, candidate_complete=b.get("valid") is True,
            baseline_formal_eligible=a.get("formal_eligible") is True, candidate_formal_eligible=b.get("formal_eligible") is True,
            same_source=source_a is not None and source_a == source_b and not a.get("source_changed") and not b.get("source_changed"),
            same_protocol=geometry_a["final_slots"] == geometry_b["final_slots"]
                and pools_a == pools_b and all(arguments_a.get(key) == arguments_b.get(key) for key in fixed)
                and protocol_a.get("dataset_sha256") == protocol_b.get("dataset_sha256"),
            coverage_complete=coverage, cleanup_valid=a.get("shutdown_complete") is True and b.get("shutdown_complete") is True,
            accounting_valid=a.get("accounting_valid") is True and b.get("accounting_valid") is True)
        by_dataset = {}
        for dataset in ("gsm8k.jsonl", "humaneval.jsonl"):
            by_dataset[dataset] = evaluate_prefetch_promotion(
                [row for row in left if Path(row["dataset"]).name == dataset],
                [row for row in right if Path(row["dataset"]).name == dataset], audit=audit)
        tokens = sum(row["returned_tokens"] for row in right)
        if any(row.get("metrics", {}).get("available") is not True
               or type(row["metrics"].get("logical_total_bytes")) is not int for row in right):
            raise RuntimeError("candidate selection requires measured expert-transfer byte accounting")
        payload = sum(row["metrics"]["logical_total_bytes"] for row in right)
        return dict(passed=all(row["eligible"] for row in by_dataset.values()), datasets=by_dataset,
            throughput=tokens / sum(row["wall_seconds"] for row in right), logical_bytes_per_token=payload / tokens,
            audit=audit)

    def run_model(self, model):
        calibration = self.root / model / "calibration"
        command = self.base_command(model, "calibration", calibration) + ["--calibrate-prefetch"]
        prefix = self.root / "reports" / (model + "_calibration")
        command = [self.args.nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
            "--cuda-graph-trace=node:nvtx-precapture", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
            "--output", str(prefix)] + command
        self.execute(model + "_calibration", command, output=calibration)
        coverage = read(calibration / "calibration_analysis.json").get("coverage_complete") is True
        if not coverage or not Path(str(prefix) + ".nsys-rep").is_file():
            raise RuntimeError("calibration requires complete shape coverage and a saved Nsight report")
        profiles = self.create_variants(model)
        slots = read(profiles["calibrated"])["geometry"]["total_slots"]
        for variant in self.protocol["variants"]:
            output = self.root / model / "development" / variant
            command = self.base_command(model, "development", output)
            if variant == "none":
                command += ["--no-prefetch", "--slots", str(slots)]
            else:
                command += ["--prefetch", "--prefetch-profile", str(profiles[variant]), "--allow-candidate-profile"]
            self.execute(model + "_development_" + variant, command, output=output, formal=True)
        development = self.root / model / "development"
        choices = {name: self.promotion(development / "none", development / name, coverage)
                   for name in self.protocol["variants"] if name != "none"}
        eligible = [name for name in choices if choices[name]["passed"]]
        selected = "none"
        if eligible:
            best = max(choices[name]["throughput"] for name in eligible)
            peers = [name for name in eligible if choices[name]["throughput"] >= best * .99]
            selected = min(peers, key=lambda name: (choices[name]["logical_bytes_per_token"], -choices[name]["throughput"], name))
        write(self.root / model / "selection.json", dict(selected=selected, choices=choices,
            reason="no candidate passes both dataset promotion gates" if selected == "none" else "passed development gates"))
        slots = read(profiles[selected])["geometry"]["total_slots"]
        quotas = read(profiles[selected])["quotas"]
        if any(quotas["preverify"]) or any(quotas["verify"]) or quotas["gap_capacity"] > 1:
            self.diagnose(model, profiles[selected])
        for variant in self.protocol["formal_variants"]:
            output = self.root / model / "formal" / variant
            command = self.base_command(model, "formal", output, mode="sd" if variant == "sd" else "three_level_gpu")
            if variant == "sd":
                command += ["--no-prefetch"]
            elif variant == "none" or (variant == "selected" and selected == "none"):
                command += ["--slots", str(slots), "--no-prefetch"]
            else:
                profile = profiles[selected if variant == "selected" else variant]
                command += ["--prefetch", "--prefetch-profile", str(profile), "--allow-candidate-profile"]
            receipt = self.execute(model + "_formal_" + variant, command, output=output, formal=True)
            if receipt.get("requests") != 96:
                raise RuntimeError("formal case must have exactly 96 measured requests")
        self.publish(model, selected, profiles, coverage)

    def diagnose(self, model, profile):
        output = self.root / model / "selected_diagnostic"
        command = self.base_command(model, "development", output)
        for name, value in (("--samples", "2"), ("--repeats", "1"), ("--tokens", "256"), ("--purpose", "diagnostic")):
            command[command.index(name) + 1] = value
        command += ["--prefetch", "--prefetch-profile", str(profile), "--allow-candidate-profile", "--profile-prefetch"]
        prefix = self.root / "reports" / (model + "_selected_diagnostic")
        command = [self.args.nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
            "--cuda-graph-trace=node:nvtx-precapture", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
            "--output", str(prefix)] + command
        receipt = self.execute(model + "_selected_diagnostic", command, output=output)
        diagnostics = read(output / "prefetch_diagnostics.json")
        if (receipt.get("requests") != 4 or not diagnostics.get("windows") or not diagnostics.get("stages")
                or not Path(str(prefix) + ".nsys-rep").is_file()):
            raise RuntimeError("selected-profile diagnosis is incomplete")
        for row in measured(output / "requests.jsonl"):
            if sum(row["graph_fallbacks"].values()) or row.get("profiled") is not True:
                raise RuntimeError("selected-profile diagnosis needs instrumented full Graph coverage")

    def publish(self, model, selected, profiles, coverage):
        from freetoken.engine.speculative_prefetch_profile import profile_directory, profile_key, validate_profile_data

        formal = self.root / model / "formal"
        evaluation = self.promotion(formal / "none", formal / "selected", coverage) if selected != "none" else None
        fallback = selected == "none" or not evaluation["passed"]
        final = "none" if fallback else selected
        evidence_cases = [formal / name for name in self.protocol["formal_variants"]]
        for case in evidence_cases:
            receipt = read(case / "complete.json")
            self.check_receipt(receipt, formal=True)
            if receipt.get("accounting_valid") is not True or receipt.get("requests") != 96:
                raise RuntimeError("profile promotion requires complete, accounted formal requests")
            for row in measured(case / "requests.jsonl"):
                fallback_count = row.get("graph_fallbacks")
                if not isinstance(fallback_count, dict) or sum(fallback_count.values()) != 0:
                    raise RuntimeError("profile promotion requires complete Graph coverage without fallback")
        data = copy.deepcopy(read(profiles[final]))
        artifacts = [self.root / "protocol.json", self.root / "source_manifest.json", self.root / "environment.json",
                     self.root / "validation/complete.json", self.root / "logs/validation.log",
                     self.root / model / "selection.json", self.root / model / "calibration/calibration_analysis.json",
                     self.root / model / "calibration/raw_calibration.json",
                     self.root / "reports" / (model + "_calibration.nsys-rep")]
        artifacts.extend(self.root / model / "calibration" / name for name in
                         ("protocol.json", "complete.json", "window_measurements.json", "copy_measurements.json", "calibration_result.json"))
        artifacts.extend(path for case in evidence_cases for path in (case / "complete.json", case / "requests.jsonl",
                         case / "prefetch_receipt.json", case / "memory_plan.json", case / "protocol.json",
                         case / "resource_validation.json"))
        diagnostic = self.root / model / "selected_diagnostic"
        if diagnostic.is_dir():
            artifacts.extend([diagnostic / name for name in ("complete.json", "prefetch_diagnostics.json", "requests.jsonl")])
            artifacts.append(self.root / "reports" / (model + "_selected_diagnostic.nsys-rep"))
        promotion = dict(passed=True, selected=final, development_selected=selected, final_evaluation=evaluation,
            coverage_complete=coverage, correctness="regression suite and complete-request accounting; strict AR parity not established",
            no_gain=fallback, reason="no reproducible positive gain; use zero prefetch quotas" if fallback else "passed both formal dataset gates")
        data["evidence"].update(status="validated", automatic_publishable=True, selection=final, promotion=promotion,
            artifact_sha256={str(path): digest(path) for path in artifacts})
        validate_profile_data(data)
        local = self.root / model / "validated_profile.json"
        if local.exists() and read(local) != data:
            raise RuntimeError("validated result differs from preserved publication")
        write(local, data)
        staged_cache = self.root / model / "reload_cache"
        staged_profile = staged_cache / "freetoken/speculative_prefetch" / (profile_key(data["identity"]) + ".json")
        if staged_profile.exists() and read(staged_profile) != data:
            raise RuntimeError("staged auto profile differs from preserved validation")
        if not staged_profile.exists():
            write(staged_profile, data)
        reload_output = self.root / model / "auto_reload"
        command = self.base_command(model, "formal", reload_output) + ["--prefetch-profile", "auto"]
        for name, value in (("--samples", "1"), ("--warmups", "0"), ("--repeats", "1"), ("--tokens", "32"), ("--purpose", "smoke")):
            command[command.index(name) + 1] = value
        receipt = self.execute(model + "_auto_reload", command, output=reload_output,
                               env_overrides={"XDG_CACHE_HOME": str(staged_cache)})
        reloaded = read(reload_output / "prefetch_receipt.json")
        if (receipt.get("requests") != 2 or reloaded.get("source") != "profile"
                or reloaded.get("profile_sha256") != digest(staged_profile)
                or reloaded.get("actual_geometry") != data["geometry"] or reloaded.get("validated") is not True):
            raise RuntimeError("auto-profile reload failed source, hash, geometry or completion validation")
        destination = persistent(profile_directory()) / (profile_key(data["identity"]) + ".json")
        if destination.exists() and read(destination) != data:
            raise RuntimeError(f"auto profile already exists with different evidence: {destination}")
        write(destination, data)
        write(self.root / model / "publication.json", dict(status="published", path=str(destination),
            sha256=digest(destination), source=str(local), selected=final, promotion=promotion,
            auto_reload_receipt=str(reload_output / "prefetch_receipt.json"),
            auto_reload_receipt_sha256=digest(reload_output / "prefetch_receipt.json")))

    def run(self):
        self.wait_previous()
        self.freeze()
        sys.path[:0] = [str(self.runtime / "python"), str(self.runtime / "benchmarks")]
        os.environ.update(self.env)
        tests = [self.args.python, "-m", "pytest", "-q", "tests/engine/test_speculative_modes.py", "tests/engine/test_speculative.py",
                 "tests/benchmarks/test_speculative_modes.py", "tests/benchmarks/test_analyze_three_level_demo.py",
                 "tests/moe/test_speculative_cache.py", "--basetemp", str(self.root / "tmp/pytest")]
        self.execute("validation", tests, output=self.root / "validation", test=True)
        for model in ("qwen36", "qwen235"):
            self.run_model(model)
        self.update(status="complete", formal_eligible=True, strict_ar_parity="not_established")
        write(self.root / "complete.json", dict(status="complete", valid=True, models=list(self.protocol["models"]),
            completed=self.state["completed"], source_unchanged=True, profiles_published=True))
        self.register()

    def register(self):
        from three_level_prefetch_calibration import summarize_prefetch_requests

        cases = {}
        for model in self.protocol["models"]:
            for directory in sorted((self.root / model / "formal").glob("*")):
                if not (directory / "complete.json").is_file() or not read(directory / "complete.json").get("formal_eligible"):
                    continue
                path = directory / "requests.jsonl"
                lines = path.read_text().splitlines()
                rows = [json.loads(line) for line in lines]
                result = summarize_prefetch_requests(rows)
                result.update(source=str(path), sha256=digest(path),
                    measured_physical_lines=[i for i, row in enumerate(rows, 1) if not row["warmup"]])
                cases[model + "/" + directory.name] = result
        write(self.root / "summary.json", dict(status=self.state["status"], cases=cases,
            scope="all measured complete requests; warmups excluded; no fastest-repeat selection",
            strict_ar_parity="not_established"))
        lines = ["# Prefetch calibration and complete-request validation", "",
                 f"Status: {self.state['status']}. Two models, one GPU, TP1, offload, 40 GiB, top-h 4 and loop4.", "",
                 "Calibration uses separate phase/layer/shape/context measurements and production copy curves.",
                 "Development selects a static profile; formal timing uses the original first sixteen prompts per dataset.",
                 "Strict AR parity and independent canonical-state audit are not established by these timing results.", "",
                 "| Case | Requests | Throughput (token/s) | Median TPOT (ms/token) |",
                 "| --- | ---: | ---: | ---: |"]
        lines += [f"| {name} | {row['requests']} | {row['throughput']:.3f} | {row['tpot_ms']:.3f} |"
                  for name, row in cases.items()]
        lines += ["", "See [protocol](protocol.json), [summary and source selectors](summary.json),",
                  "[source manifest](source_manifest.json), and each model's selection/publication receipts."]
        (self.root / "readme.md").write_text("\n".join(lines) + "\n")
        archive = persistent(self.args.archive_root)
        if not self.root.is_relative_to(archive):
            raise ValueError("matrix output must be inside --archive-root for catalog registration")
        run_id = "prefetch_calibration_" + self.root.name + "_" + hashlib.sha256(str(self.root).encode()).hexdigest()[:8]
        record = dict(id=run_id, category="speculative", series="speculative_prefetch_calibration",
            config=dict(models=self.protocol["models"], budget_gib=40, tp=1, concurrent_requests=1,
                max_loops=4, preverify_top_h=4, cpu_interference_policy="python", gpu=self.protocol["gpu"],
                source_commit=self.protocol["source"]["source_commit"]),
            measurement_scope="offline phase/copy calibration and separately unprofiled complete-request A/B",
            measured_date=datetime.now(timezone.utc).date().isoformat(), date_precision="day", purpose="formal",
            validity="valid" if self.state["status"] == "complete" else "failed", lifecycle="current",
            path=str(self.root.relative_to(archive)), entrypoint=str((self.root / "readme.md").relative_to(archive)),
            related_to=[], superseded_by=[], differences=dict(prefetch="calibrated per-layer quotas and fixed pool geometry"),
            limitations="Strict AR parity and independent canonical-state audit are not established")
        paths = [path for path in self.root.rglob("*") if path.is_file()
                 and not any(part in ("runtime", "cache", "reload_cache", "tmp", ".archive") for part in path.relative_to(self.root).parts)
                 and path.suffix in (".json", ".jsonl", ".log", ".md", ".nsys-rep")]
        artifacts = [dict(id=run_id + f"_artifact_{i:04}", run_id=run_id, path=str(path.relative_to(archive)),
            role="calibration_diagnostic" if "calibration" in path.parts or path.suffix == ".nsys-rep" else "performance_evidence",
            sha256=digest(path), bytes=path.stat().st_size, depends_on=[]) for i, path in enumerate(sorted(paths))]
        spec = importlib.util.spec_from_file_location("prefetch_results_catalog", self.runtime / "tools/results_catalog.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        validation = module.validate(archive, dict(runs=[record], artifacts=artifacts))
        if not validation["valid"]:
            raise RuntimeError(validation)
        with (archive / ".catalog.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            catalog = read(archive / "catalog.json")
            catalog["runs"] = [row for row in catalog["runs"] if row["id"] != run_id] + [record]
            catalog["artifacts"] = [row for row in catalog["artifacts"] if row["run_id"] != run_id] + artifacts
            catalog["updated_at"] = datetime.now(timezone.utc).isoformat()
            write(archive / "catalog.json", catalog)
            module.render(archive, catalog)
        write(self.root / ".archive/catalog_registration.json", validation)


def main():
    parser = argparse.ArgumentParser(description="Queue frozen single-request prefetch calibration and promotion")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-run", type=Path)
    parser.add_argument("--expected-source-manifest", type=Path)
    parser.add_argument("--archive-root", type=Path, default="/home/fanya/freetoken_benchmark_results")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-dir", type=Path, default=SOURCE_DATA)
    parser.add_argument("--qwen36", default=MODELS["qwen36"])
    parser.add_argument("--qwen235", default=MODELS["qwen235"])
    parser.add_argument("--draft", default=DRAFT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cuda-home", default="/usr/local/cuda-13.0")
    parser.add_argument("--nsys", default="/home/fanya/.local/bin/nsys")
    args = parser.parse_args()
    args.models = dict(qwen36=args.qwen36, qwen235=args.qwen235)
    matrix = Matrix(args)
    try:
        matrix.run()
    except Exception as exc:
        matrix.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        write(matrix.root / "complete.json", dict(status="failed", valid=False, error=str(exc),
            completed=matrix.state["completed"], profiles_published="see per-model publication receipts"))
        if hasattr(matrix, "protocol"):
            try:
                matrix.register()
            except Exception as archive_error:
                write(matrix.root / "archive_error.json", dict(error=str(archive_error)))
        raise


if __name__ == "__main__":
    main()
