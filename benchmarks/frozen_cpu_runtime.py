"""Load a frozen CPU Target in the benchmark's spawned verifier only."""

from functools import partial
import hashlib
import importlib
import json
from pathlib import Path
import sys


def load_frozen_target(root):
    root = Path(root).resolve()
    native_name = "freetoken.kernel._cpu_moe"
    if native_name in sys.modules:
        raise RuntimeError("frozen CPU runtime must be selected before native extension import")
    for package_name in ("engine", "moe", "kernel"):
        path = root / "python/freetoken" / package_name
        if not path.is_dir():
            raise ValueError(f"missing frozen CPU package: {path}")
        package = importlib.import_module("freetoken." + package_name)
        package.__path__ = [str(path)]
    for name in ("freetoken.engine.qwen_cpu_target", "freetoken.moe.cpu_target"):
        sys.modules.pop(name, None)
    module = importlib.import_module("freetoken.engine.qwen_cpu_target")
    experts = importlib.import_module("freetoken.moe.cpu_target")
    native = importlib.import_module(native_name)
    files = [Path(item.__file__).resolve() for item in (module, experts, native)]
    if not all(path.is_relative_to(root) for path in files):
        raise RuntimeError("CPU Target did not load entirely from the frozen runtime")
    identities = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in files}
    manifest_path = root.with_name(root.name + "_manifest.json")
    expected = json.loads(manifest_path.read_text())
    if any(expected.get(path) != fingerprint for path, fingerprint in identities.items()):
        raise RuntimeError("frozen CPU source or native extension differs from its manifest")

    class FrozenTarget(module.QwenCpuTarget):
        @classmethod
        def from_checkpoint(cls, path, *, kernel_profile="v1", components=None, **kwargs):
            if kernel_profile != "v1" or components or kwargs.get("backend", "optimized") != "optimized":
                raise ValueError("frozen CPU runtime requires optimized v1 without v2 components")
            return super().from_checkpoint(path, **kwargs)

        def arithmetic_metadata(self):
            return dict(super().arithmetic_metadata(), kernel_profile="v1", v2_components=[],
                        frozen_v1_runtime=True, runtime_source_root=str(root),
                        runtime_files=identities,
                        frozen_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())

    return FrozenTarget


def _frozen_cpu_main(root, *args):
    from freetoken.engine import qwen_cpu_worker

    try:
        qwen_cpu_worker.QwenCpuTarget = load_frozen_target(root)
    except BaseException:
        import traceback

        args[0].send({"kind": "error", "traceback": traceback.format_exc()})
        args[0].close()
        return
    qwen_cpu_worker._cpu_main(*args)


def create_cpu_worker(*args, frozen_root=None, **kwargs):
    from freetoken.engine import qwen_cpu_worker

    if frozen_root is None:
        return qwen_cpu_worker.CpuVerifierWorker(*args, **kwargs)
    if (kwargs.get("backend", "optimized") != "optimized"
            or kwargs.get("kernel_profile", "v1") != "v1" or kwargs.get("components")):
        raise ValueError("frozen CPU runtime requires optimized v1 without v2 components")
    original = qwen_cpu_worker._cpu_main
    try:
        # Spawn captures this callable; the parent GPU imports remain unchanged.
        qwen_cpu_worker._cpu_main = partial(_frozen_cpu_main, str(Path(frozen_root).resolve()))
        return qwen_cpu_worker.CpuVerifierWorker(*args, **kwargs)
    finally:
        qwen_cpu_worker._cpu_main = original
