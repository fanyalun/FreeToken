"""Run TableMoE and the affected CPU/cache regressions on the second GPU."""

import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
os.environ["PATH"] = str(Path(__file__).resolve().parents[1] / ".venv/bin") + ":/usr/local/cuda-13.0/bin:" + os.environ["PATH"]

import pytest

tests = [
    "tests/moe/test_tablemoe.py", "tests/moe/test_cpu_moe.py", "tests/moe/test_hybrid_fetch.py",
    "tests/kvcache/test_cache_unit_bytes.py", "tests/kvcache/test_pool_sizing_surface.py",
    "tests/engine/test_cache_budget.py", "tests/scheduler/test_cache_rebuild.py",
    "-q", "--tb=short", "-m", "not needs_weights and not slow",
]
if "--large-offset" in sys.argv:
    tests = ["tests/moe/test_tablemoe.py::test_mapped_value_offset_crosses_four_gib", "-q", "--tb=short"]
raise SystemExit(pytest.main(tests))
