"""Run Vision2Web Level 1 with the pinned official generation and evaluation engines."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlsplit
import uuid


OFFICIAL_REVISION = "d80a4a3d0ea7326541ad87314294269a633789e4"
PATH_EDITS = {
    "vision2web/core/constants.py": ("/tmp/workspace", "/v2w_runtime/staging"),
    "vision2web/core/sandbox.py": ("/tmp/workspace", "/v2w_runtime/staging"),
    "vision2web/inference/adapters/claude_code.py": ("/tmp/v2w_prompt_", "/v2w_runtime/v2w_prompt_"),
    "vision2web/inference/adapters/gemini_cli.py": ("/tmp/v2w_prompt_", "/v2w_runtime/v2w_prompt_"),
    "vision2web/evaluation/engine.py": ("/tmp/deploy.log", "/v2w_runtime/deploy.log"),
}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    partial.replace(path)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def persistent_path(value):
    path = Path(value).expanduser().resolve()
    if path == Path("/tmp") or Path("/tmp") in path.parents:
        raise ValueError(f"persistent output required, not /tmp: {path}")
    # The official copy-back command interpolates host paths into a shell pipeline.
    if not re.fullmatch(r"[a-zA-Z0-9_/.,-]+", str(path)):
        raise ValueError(f"official runner requires a shell-safe path: {path}")
    return path


def load_config(path):
    config = read(path)
    for key in ("official_root", "datasets_dir", "output"):
        config[key] = str(persistent_path(config[key]))
    if config.get("official_revision") != OFFICIAL_REVISION:
        raise ValueError(f"this adapter is tested against official revision {OFFICIAL_REVISION}")
    for key in ("generation", "evaluation"):
        endpoint = config[key]
        if "api_key" in endpoint:
            raise ValueError("use api_key_env; do not put credentials in saved configurations")
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", endpoint["model"]):
            raise ValueError("use a served model ID without slashes, colons or spaces")
        url = urlsplit(endpoint["base_url"])
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query:
            raise ValueError("base_url must be an HTTP endpoint without embedded credentials")
        if type(endpoint["max_workers"]) is not int or endpoint["max_workers"] < 1:
            raise ValueError("max_workers must be a positive integer")
    if urlsplit(config["generation"]["base_url"]).hostname in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("generation runs inside Docker; use host.docker.internal and bind the server to 0.0.0.0")
    if urlsplit(config["generation"]["base_url"]).path.rstrip("/") == "/v1":
        raise ValueError("Claude Code needs the Anthropic API root, without /v1")
    if config["generation"]["timeout"] <= 0:
        raise ValueError("generation timeout must be positive")
    output = Path(config["output"])
    for key in ("official_root", "datasets_dir"):
        source = Path(config[key])
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("output, dataset and official source must be separate directories")
    return config


def prepare(config):
    root, source = Path(config["output"]), Path(config["official_root"])
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if revision != OFFICIAL_REVISION or dirty.strip():
        raise ValueError("official source must be clean and checked out at the pinned revision")
    root.mkdir(parents=True, exist_ok=True)
    locked = root / "config.json"
    if locked.exists() and read(locked) != config:
        raise ValueError("configuration changed; use a new output directory")
    prior_manifest = root / "source_manifest.json"
    if prior_manifest.exists() and read(prior_manifest)["adapter_sha256"] != fingerprint(__file__):
        raise ValueError("adapter changed; use a new output directory")
    save(locked, config)
    for variable, name in (("TMPDIR", "work"), ("TMP", "work"), ("TEMP", "work"),
                           ("PYTHONPYCACHEPREFIX", "pycache")):
        target = root / name
        target.mkdir(exist_ok=True)
        os.environ[variable] = str(target)
    tempfile.tempdir = str(root / "work")
    sys.pycache_prefix = str(root / "pycache")
    for variable in ("NO_PROXY", "no_proxy"):
        os.environ[variable] = ",".join(filter(None, [os.environ.get(variable),
            "localhost", "127.0.0.1", "host.docker.internal"]))
    runtime = root / "official_runtime"
    files = subprocess.check_output(["git", "-C", str(source), "ls-files"], text=True).splitlines()
    files = [name for name in files if name.startswith(("vision2web/", "docker/")) or name == "README.md"]
    manifest = {}
    for name in files:
        original = (source / name).read_bytes()
        content = original
        if name in PATH_EDITS:
            old, new = PATH_EDITS[name]
            if old.encode() not in content:
                raise ValueError(f"official path changed: {name}")
            content = content.replace(old.encode(), new.encode())
        if name == "docker/Dockerfile":
            content = content.replace(b"/tmp/chrome.deb", b"/opt/vision2web_build/chrome.deb")
            content = content.replace(b"RUN wget -q", b"RUN mkdir -p /opt/vision2web_build && wget -q")
        destination = runtime / name
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f"frozen runtime changed: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(content)
        manifest[name] = dict(upstream_sha256=hashlib.sha256(original).hexdigest(),
                              runtime_sha256=hashlib.sha256(content).hexdigest(), modified=original != content)
    sys.path.insert(0, str(runtime))
    from vision2web.core.dataset import DatasetManager
    from vision2web.inference.prompts import get_prompt_for_task

    discovered = DatasetManager(Path(config["datasets_dir"])).discover_projects(task_type="webpage")
    projects = {project.name: project for project in discovered}
    if not config.get("projects") and len(projects) != 100:
        raise ValueError("full Level 1 requires all 100 official projects")
    selected = config.get("projects") or sorted(projects)
    if not selected or len(selected) != len(set(selected)) or set(selected) - projects.keys():
        raise ValueError("project selection is empty, duplicated or absent from the dataset")
    records = []
    for name in selected:
        project = projects[name]
        workflows = read(project.workflow_path)
        if any(item.get("content") for item in workflows):
            raise ValueError("this entry point supports screenshot-only Level 1 workflows")
        prototypes = {name for item in workflows for name in item.get("prototype", {})}
        if prototypes != {"desktop", "tablet", "mobile"}:
            raise ValueError(f"incomplete Level 1 prototype coverage: {name}")
        if any(not (project.prototypes_dir / f"{proto}.jpg").is_file() for proto in prototypes):
            raise ValueError(f"missing prototype image: {name}")
        records.append(dict(name=name, path=str(project.path), workflows=workflows,
            files={str(path.relative_to(project.path)): path.stat().st_size
                   for path in sorted(project.path.rglob("*")) if path.is_file()}))
    if (root / "projects.json").exists() and read(root / "projects.json") != records:
        raise ValueError("dataset metadata changed; use a new output directory")
    save(root / "projects.json", records)
    save(root / "source_manifest.json", dict(upstream="https://github.com/zai-org/Vision2Web",
        revision=revision, files=manifest, adapter_sha256=fingerprint(__file__)))
    prompt = get_prompt_for_task("webpage")
    (root / "generation_prompt.txt").write_text(prompt)
    save(root / "protocol.json", dict(dataset="Vision2Web", level=1, task="webpage",
        framework="claude_code", projects=selected, expected_projects=len(selected),
        expected_prototypes=3 * len(selected), official_revision=revision,
        prompt_source="vision2web/inference/prompts.py:WEBPAGE_PROMPT",
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        score_source="vision2web/evaluation/visual_scorer.py",
        aggregation_source="vision2web/analyze/analyzer.py",
        official_prompt_unchanged=True, metric="Visual Score",
        full_dataset=len(selected) == len(projects) == 100,
        formal_performance_eligible=False))
    return root


def doctor(config):
    root = Path(config["output"])
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ("click", "yaml", "openai", "playwright", "PIL")}
    image = subprocess.run(["docker", "image", "inspect", config["sandbox_image"],
                            "--format", "{{.Id}}"], capture_output=True, text=True) if shutil.which("docker") else None
    result = dict(ready=all(dependencies.values()) and image is not None and image.returncode == 0,
        dependencies=dependencies, sandbox_image=config["sandbox_image"],
        image_id=image.stdout.strip() if image and image.returncode == 0 else None,
        projects=len(read(root / "projects.json")), endpoint_inference_tested=False,
        build_command=["docker", "build", "-t", config["sandbox_image"],
                       "-f", str(root / "official_runtime/docker/Dockerfile"), str(root / "official_runtime/docker")])
    save(root / "doctor.json", result)
    return result


def api_key(endpoint):
    key = os.environ.get(endpoint["api_key_env"])
    if key:
        return key
    if urlsplit(endpoint["base_url"]).hostname in ("localhost", "127.0.0.1", "host.docker.internal"):
        return "EMPTY"
    raise ValueError(f"set {endpoint['api_key_env']} for the configured endpoint")


def official_config(config, stage):
    from vision2web.core.config import Config

    value = Config()
    value.datasets_dir = Path(config["datasets_dir"])
    value.results_dir = Path(config["output"]) / "results"
    value.sandbox.image = config["sandbox_image"]
    generation, evaluation = config["generation"], config["evaluation"]
    value.inference.framework = "claude_code"
    value.inference.model = generation["model"]
    value.inference.base_url = generation["base_url"]
    value.inference.api_key = api_key(generation) if stage == "inference" else None
    value.inference.max_workers = generation["max_workers"]
    value.inference.timeout = generation["timeout"]
    value.inference.task = "webpage"
    value.evaluation.visual_model = evaluation["model"]
    value.evaluation.base_url = evaluation["base_url"]
    value.evaluation.api_key = api_key(evaluation) if stage == "evaluate" else None
    value.evaluation.max_workers = evaluation["max_workers"]
    value.evaluation.task = "webpage"
    value.evaluation.framework = "claude_code"
    value.evaluation.model = generation["model"]
    value.ensure_dirs()
    return value


def persistent_sandbox(config, stage, logger):
    from vision2web.core.sandbox import SandboxManager

    root = Path(config["output"])

    class PersistentSandbox(SandboxManager):
        async def create_container(self, workspace, container_name=None):
            identity = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:12]
            backing = root / "container_work" / stage / identity
            for name in ("workspace", "runtime", "runtime/tmp"):
                (backing / name).mkdir(parents=True, exist_ok=True)
            self.backing = backing
            return await super().create_container(workspace, f"vision2web-{stage}-{identity}")

        async def _exec_docker_command(self, cmd, error_msg, timeout=None):
            if cmd[:2] == ["docker", "create"]:
                cmd = cmd[:2] + ["--add-host", "host.docker.internal:host-gateway",
                    "--mount", f"type=bind,src={self.backing / 'workspace'},dst=/workspace",
                    "--mount", f"type=bind,src={self.backing / 'runtime'},dst=/v2w_runtime",
                    "-e", "TMPDIR=/v2w_runtime/tmp", "-e", "TMP=/v2w_runtime/tmp",
                    "-e", "TEMP=/v2w_runtime/tmp"] + cmd[2:]
            return await super()._exec_docker_command(cmd, error_msg, timeout)

    sandbox = PersistentSandbox(image_name=config["sandbox_image"], logger=logger)
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        proxy = sandbox.proxy_env.get(variable)
        if proxy and urlsplit(proxy).hostname in ("localhost", "127.0.0.1", "::1"):
            del sandbox.proxy_env[variable]
            logger.warning("Ignoring host loopback %s inside Docker; external downloads use direct access", variable)
    bypass = ",".join(filter(None, [sandbox.proxy_env.get("NO_PROXY"),
        sandbox.proxy_env.get("no_proxy"), "localhost", "127.0.0.1", "host.docker.internal"]))
    sandbox.proxy_env.update(NO_PROXY=bypass, no_proxy=bypass)
    return sandbox


def workspaces(config):
    root = Path(config["output"])
    base = root / "results/webpage/claude_code" / config["generation"]["model"]
    return [(record, base / record["name"]) for record in read(root / "projects.json")]


def generation_coverage(config):
    missing = []
    for project, workspace in workspaces(config):
        result = workspace / "results" / f"{project['name']}.json"
        if not result.is_file() or read(result).get("status") != "success" or not (workspace / "start.sh").is_file():
            missing.append(project["name"])
    return missing


def evaluation_coverage(config):
    from PIL import Image

    missing = []
    for project, workspace in workspaces(config):
        result = workspace / "evaluation_result.json"
        if not result.is_file() or read(result).get("status") != "success":
            missing.append(f"{project['name']}:evaluation")
        for prototype in ("desktop", "tablet", "mobile"):
            folder = workspace / "test_results/prototypes"
            screenshot, scores = folder / f"{prototype}_actual.png", folder / f"{prototype}_scores.json"
            if not screenshot.is_file() or screenshot.stat().st_size == 0 or not scores.is_file():
                missing.append(f"{project['name']}:{prototype}")
                continue
            try:
                with Image.open(screenshot) as image:
                    if image.format != "PNG":
                        raise ValueError("expected a PNG screenshot")
                    image.verify()
                data = read(scores)
                valid = isinstance(data, list) and bool(data) and all(isinstance(item, dict)
                    and type(item.get("score")) in (int, float) and math.isfinite(item["score"])
                    and 0 <= item["score"] <= 1 for item in data)
            except (ValueError, OSError):
                valid = False
            if not valid:
                missing.append(f"{project['name']}:{prototype}:scores")
    return missing


async def run_stage(config, stage):
    from vision2web.core.logger import setup_logger
    from vision2web.inference.engine import InferenceEngine
    from vision2web.evaluation.engine import EvaluationEngine
    from vision2web.evaluation.visual_scorer import VisualScorer

    root = Path(config["output"])
    setup_logger("vision2web", log_file=str(root / f"{stage}.log"))
    if not doctor(config)["ready"]:
        raise RuntimeError(f"runtime dependencies or sandbox image missing; see {root / 'doctor.json'}")
    resolved = official_config(config, stage)
    if stage == "inference":
        missing_generation = generation_coverage(config)
        for project, workspace in workspaces(config):
            if (workspace / "start.sh").exists() and project["name"] in missing_generation:
                raise ValueError("partial generation would be silently skipped by upstream; use a fresh output")
        engine = InferenceEngine(resolved)
        engine.sandbox_manager = persistent_sandbox(config, stage, engine.logger)
        engine.adapter.sandbox_manager = engine.sandbox_manager
        try:
            await engine.run_all_projects(project_names=[r["name"] for r in read(root / "projects.json")], task_type="webpage")
        finally:
            await engine.sandbox_manager.cleanup_all()
        missing = generation_coverage(config)
    else:
        if generation_coverage(config):
            raise ValueError("generation coverage is incomplete")
        engine = EvaluationEngine(resolved)
        engine.sandbox_manager = persistent_sandbox(config, stage, engine.logger)

        class RecordedScorer(VisualScorer):
            async def _call_vlm(self, prompt, images, max_tokens=64000):
                path = root / "judge_records" / f"{uuid.uuid4().hex}.json"
                record = dict(model=self.visual_model, max_tokens=max_tokens,
                    messages=[dict(role="user", content=self._parse_prompt_with_images(prompt, images))])
                save(path, record)
                response, error = await super()._call_vlm(prompt, images, max_tokens)
                save(path, {**record, "response": response, "error": error})
                return response, error

        await engine.visual_scorer.client.close()
        engine.visual_scorer = RecordedScorer(api_key=resolved.evaluation.api_key,
            base_url=resolved.evaluation.base_url, visual_model=resolved.evaluation.visual_model)
        try:
            await engine.evaluate_all_projects(task_type="webpage", framework="claude_code", model=resolved.inference.model)
        finally:
            await engine.visual_scorer.client.close()
            await engine.sandbox_manager.cleanup_all()
        missing = evaluation_coverage(config)
    save(root / f"{stage}_complete.json", dict(complete=not missing, missing=missing,
        finished_utc=datetime.now(timezone.utc).isoformat()))
    if missing:
        raise RuntimeError(f"{stage} incomplete: {missing}")


def report(config):
    from vision2web.analyze.analyzer import ResultAnalyzer

    root = Path(config["output"])
    (root / "results").mkdir(exist_ok=True)
    analyzer = ResultAnalyzer(str(root / "results"))
    analysis = analyzer.analyze(datasets_dir=config["datasets_dir"])
    save(root / "official_analysis.json", analysis)
    missing_generation = generation_coverage(config)
    missing_evaluation = evaluation_coverage(config)
    expected = {f"webpage:claude_code:{config['generation']['model']}:{project['name']}"
                for project, _ in workspaces(config)}
    unexpected = sorted(set(analysis["results"]) - expected)
    result = dict(complete=not missing_generation and not missing_evaluation and not unexpected,
        expected_projects=len(read(root / "projects.json")),
        missing_generation=missing_generation, missing_evaluation=missing_evaluation,
        unexpected_results=unexpected,
        official_analysis=str(root / "official_analysis.json"), formal_performance_eligible=False)
    save(root / "complete.json", result)
    analyzer.print_summary(analysis)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "doctor", "inference", "evaluate", "report", "all"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = prepare(config)
    if args.stage == "prepare":
        print(json.dumps(read(root / "protocol.json"), indent=2))
        return
    if args.stage == "doctor":
        result = doctor(config)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["ready"] else 1)
    if args.stage in ("inference", "evaluate", "all"):
        for stage in (("inference", "evaluate") if args.stage == "all" else (args.stage,)):
            asyncio.run(run_stage(config, stage))
    if args.stage in ("report", "all"):
        raise SystemExit(0 if report(config)["complete"] else 1)


if __name__ == "__main__":
    main()
