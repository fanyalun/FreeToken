import asyncio
import logging
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from bench_vision2web import evaluation_coverage, generation_coverage, load_config, persistent_path
from bench_vision2web import persistent_sandbox, read, report, save


@pytest.fixture
def config(tmp_path):
    value = dict(output=str(tmp_path), generation=dict(model="test_model"))
    save(tmp_path / "projects.json", [dict(name="abc")])
    return value


def completed_project(config):
    from PIL import Image

    folder = Path(config["output"]) / "results/webpage/claude_code/test_model/abc"
    save(folder / "results/abc.json", dict(status="success"))
    (folder / "start.sh").write_text("python -m http.server 3000\n")
    save(folder / "evaluation_result.json", dict(status="success"))
    for name in ("desktop", "tablet", "mobile"):
        base = folder / "test_results/prototypes"
        save(base / f"{name}_scores.json", [dict(score=.75)])
        Image.new("RGB", (2, 2)).save(base / f"{name}_actual.png")
    return folder


@pytest.mark.parametrize("payload", [[], [{"score": float("nan")}], [{"score": 90}],
                                     [{"reason": "no score"}], {"score": 1}, [{"score": True}]])
def test_empty_or_invalid_judge_output_is_not_complete(config, payload):
    folder = completed_project(config)
    assert not evaluation_coverage(config)
    save(folder / "test_results/prototypes/mobile_scores.json", payload)
    assert evaluation_coverage(config) == ["abc:mobile:scores"]


def test_upstream_success_without_one_screenshot_is_incomplete(config):
    folder = completed_project(config)
    (folder / "test_results/prototypes/tablet_actual.png").unlink()
    assert evaluation_coverage(config) == ["abc:tablet"]


def test_invalid_screenshot_is_not_complete(config):
    folder = completed_project(config)
    (folder / "test_results/prototypes/tablet_actual.png").write_bytes(b"not an image")
    assert evaluation_coverage(config) == ["abc:tablet:scores"]


def test_start_script_alone_does_not_prove_generation_succeeded(config):
    folder = completed_project(config)
    save(folder / "results/abc.json", dict(status="timeout"))
    assert generation_coverage(config) == ["abc"]


def test_tmp_symlink_cannot_bypass_persistent_output_rule(tmp_path):
    link = tmp_path / "redirect"
    link.symlink_to("/tmp", target_is_directory=True)
    with pytest.raises(ValueError, match="persistent output"):
        persistent_path(link / "vision2web")


@pytest.fixture
def official_source():
    value = os.environ.get("VISION2WEB_OFFICIAL_ROOT")
    if not value:
        pytest.skip("set VISION2WEB_OFFICIAL_ROOT for pinned upstream integration checks")
    sys.path.insert(0, value)
    yield Path(value)
    sys.path.remove(value)


def test_persistent_mounts_keep_upstream_copy_source_separate(config, official_source, monkeypatch):
    from vision2web.core.sandbox import SandboxManager

    captured = []

    async def capture(self, command, error_msg, timeout=None):
        captured.append(command)
        return 0, "container-id", ""

    monkeypatch.setattr(SandboxManager, "_exec_docker_command", capture)
    monkeypatch.setenv("NO_PROXY", "upper.example")
    monkeypatch.setenv("no_proxy", "lower.example")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9876")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    config["sandbox_image"] = "test-image"
    workspace = Path(config["output"]) / "result"
    workspace.mkdir()
    sandbox = persistent_sandbox(config, "inference", logging.getLogger("test"))
    asyncio.run(sandbox.create_container(workspace, "upstream-name"))
    command = captured[0]
    mounts = [command[i + 1] for i, item in enumerate(command) if item == "--mount"]
    assert len(mounts) == 2
    assert any(mount.endswith("dst=/workspace") for mount in mounts)
    assert all(f"src={workspace}," not in mount for mount in mounts)
    assert "TMPDIR=/v2w_runtime/tmp" in command
    assert "host.docker.internal:host-gateway" in command
    assert "host.docker.internal" in sandbox.proxy_env["NO_PROXY"]
    assert "localhost" in sandbox.proxy_env["no_proxy"]
    assert "upper.example" in sandbox.proxy_env["no_proxy"]
    assert "lower.example" in sandbox.proxy_env["NO_PROXY"]
    assert "HTTP_PROXY" not in sandbox.proxy_env
    assert sandbox.proxy_env["HTTPS_PROXY"] == "http://proxy.example:8080"


def test_official_analyzer_scores_survive_but_incomplete_run_stays_incomplete(config, official_source):
    folder = completed_project(config)
    dataset = Path(config["output"]) / "dataset/webpage/abc"
    (dataset / "prototypes").mkdir(parents=True)
    for name in ("desktop", "tablet", "mobile"):
        (dataset / "prototypes" / f"{name}.jpg").write_bytes(b"prototype")
    save(dataset / "workflow.json", [])
    config["datasets_dir"] = str(dataset.parent.parent)
    assert report(config)["complete"]
    analysis = read(Path(config["output"]) / "official_analysis.json")
    assert analysis["results"]["webpage:claude_code:test_model:abc"]["prototypes"]["desktop"] == .75
    (folder / "test_results/prototypes/mobile_actual.png").unlink()
    assert not report(config)["complete"]


def test_generation_loopback_and_embedded_credentials_are_rejected(tmp_path):
    path = Path(__file__).resolve().parents[2] / "benchmarks/vision2web_example.json"
    value = read(path)
    value["generation"]["base_url"] = "http://127.0.0.1:23840"
    target = tmp_path / "config.json"
    save(target, value)
    with pytest.raises(ValueError, match="inside Docker"):
        load_config(target)
    value["generation"]["base_url"] = "https://user:secret@example.org"
    save(target, value)
    with pytest.raises(ValueError, match="embedded credentials"):
        load_config(target)
