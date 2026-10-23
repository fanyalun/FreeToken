from functools import partial
from pathlib import Path
import sys

import pytest


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    import frozen_cpu_runtime

    return frozen_cpu_runtime


def test_frozen_spawn_selection_is_scoped_to_constructor(runtime, monkeypatch, tmp_path):
    from freetoken.engine import qwen_cpu_worker

    original = qwen_cpu_worker._cpu_main
    captured = []

    def construct(*args, **kwargs):
        captured.append(qwen_cpu_worker._cpu_main)
        return "worker"

    monkeypatch.setattr(qwen_cpu_worker, "CpuVerifierWorker", construct)
    assert runtime.create_cpu_worker() == "worker"
    assert captured[-1] is original
    assert runtime.create_cpu_worker(frozen_root=tmp_path) == "worker"
    assert isinstance(captured[-1], partial)
    assert captured[-1].func is runtime._frozen_cpu_main
    assert captured[-1].args == (str(tmp_path.resolve()),)
    assert qwen_cpu_worker._cpu_main is original


def test_frozen_spawn_restores_constructor_after_error(runtime, monkeypatch, tmp_path):
    from freetoken.engine import qwen_cpu_worker

    original = qwen_cpu_worker._cpu_main

    def fail(*args, **kwargs):
        raise RuntimeError("constructor failed")

    monkeypatch.setattr(qwen_cpu_worker, "CpuVerifierWorker", fail)
    with pytest.raises(RuntimeError, match="constructor failed"):
        runtime.create_cpu_worker(frozen_root=tmp_path)
    assert qwen_cpu_worker._cpu_main is original


@pytest.mark.parametrize("options", [dict(backend="legacy"), dict(kernel_profile="ggml_v2"),
                                    dict(components=["attention"])])
def test_frozen_spawn_rejects_profile_mixtures(runtime, tmp_path, options):
    with pytest.raises(ValueError, match="optimized v1"):
        runtime.create_cpu_worker(frozen_root=tmp_path, **options)


def test_frozen_loader_rejects_preloaded_native_extension(runtime, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "freetoken.kernel._cpu_moe", object())
    with pytest.raises(RuntimeError, match="before native extension import"):
        runtime.load_frozen_target(tmp_path)
