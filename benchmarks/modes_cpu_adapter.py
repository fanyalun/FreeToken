from __future__ import annotations

import ast
from pathlib import Path
import shutil

from modes_experiment import REPO, digest, write


PARENT = Path("/home/fanya/freetoken_benchmark_results/speculative/cpu_async_12g_conditional_20261021/runtime")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"frozen CPU adapter source contract changed: {old[:100]!r}")
    return text.replace(old, new, 1)


def prepare_runtime(target, parent=PARENT):
    target, parent = Path(target), Path(parent)
    if target.exists():
        raise ValueError("CPU adapter output must be a new directory")
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    changed = {}
    path = target / "python/freetoken/moe/speculative_gpu.py"
    code = path.read_text()
    source_hash = digest(path)
    code = replace_once(code,
        "verify_prefetched=None, RETAIN_ROUTES: tl.constexpr = True):",
        "verify_prefetched=None, RETAIN_ROUTES: tl.constexpr = True, POLICY: tl.constexpr = False, "
        "RETAIN_TOPK: tl.constexpr = 4, importance=None, TAU: tl.constexpr = 0.0):")
    code = replace_once(code, '"hybrid_preverify_cpu_unique_experts", "hybrid_verify_cpu_unique_experts",',
        '"hybrid_preverify_cpu_unique_experts", "hybrid_verify_cpu_unique_experts",\n'
        '    "preverify_threshold_retained_routes", "preverify_extra_hit_routes",')
    code = replace_once(code, "        keep = valid & ((ready & KEEP_HITS) | (rank < 4))",
        "        if POLICY:\n"
        "            threshold_keep = valid & (w.to(tl.float32) * tl.load(importance + LAYER) >= TAU)\n"
        "        else:\n"
        "            threshold_keep = valid & (rank < RETAIN_TOPK)\n"
        "        keep = valid & ((ready & KEEP_HITS) | threshold_keep)\n"
        "        tl.atomic_add(stats + 24, tl.sum(threshold_keep.to(tl.int32), 0))\n"
        "        tl.atomic_add(stats + 25, tl.sum((keep & ~threshold_keep).to(tl.int32), 0))")
    code = replace_once(code, "    if CHECK_PENDING:\n", "    if CHECK_PENDING and not POLICY:\n")
    code = replace_once(code, "        keep = valid\n    count =",
        "        keep = valid\n"
        "    if CHECK_PENDING and POLICY:\n"
        "        if tl.sum((keep & (slot >= 0) & ~ready).to(tl.int32), 0) > 0:\n"
        "            tl.store(error, 3)\n"
        "    count =")
    code = replace_once(code, "hybrid_phases=(), hybrid_max_fetch=-1):",
        "hybrid_phases=(), hybrid_max_fetch=-1, preverify_policy=None):\n"
        "        from .preverify_policy import PreverifyPolicy\n"
        "        self.preverify_policy = preverify_policy or PreverifyPolicy()\n"
        "        if renormalize and self.preverify_policy.replacement == 'identity':\n"
        "            raise ValueError('identity requires original weights')\n"
        "        self.importance = torch.tensor(self.preverify_policy.importance or (1.,)*cache.num_layers,\n"
        "                                       dtype=torch.float32, device=cache.device)")
    code = replace_once(code,
        'self.verify_prefetched if self.phase == "verify" else None, self.retain_routes)',
        'self.verify_prefetched if self.phase == "verify" else None, self.retain_routes,\n'
        '            self.preverify_policy.kind == "modes", self.preverify_policy.top_k,\n'
        '            self.importance, self.preverify_policy.tau)\n'
        '        identity = None\n'
        '        if self.phase == "preverify" and self.preverify_policy.replacement == "identity":\n'
        '            identity = hidden.float() * (weights.float()-selected.float()).sum(-1, keepdim=True)')
    code = replace_once(code, "        return result\n\n    def decide",
        "        from .preverify_policy import merge_identity\n"
        "        return merge_identity(result, identity)\n\n    def decide")
    ast.parse(code)
    path.write_text(code)
    changed[str(path.relative_to(target))] = dict(before=source_hash, after=digest(path))
    source = REPO / "python/freetoken/moe/preverify_policy.py"
    shutil.copy2(source, target / "python/freetoken/moe/preverify_policy.py")
    path = target / "python/freetoken/engine/qwen_async_three_level.py"
    source_hash = digest(path)
    code = replace_once(path.read_text(), 'compact_control=True, implementation="optimized")',
        'compact_control=True, implementation="optimized", preverify_policy=type(self).preverify_policy)')
    ast.parse(code)
    path.write_text(code)
    changed[str(path.relative_to(target))] = dict(before=source_hash, after=digest(path))
    path = target / "benchmarks/bench_cpu_verify_feasibility.py"
    source_hash = digest(path)
    code = replace_once(path.read_text(), "import argparse", "import argparse\nfrom types import SimpleNamespace\n"
        "from freetoken.moe.preverify_policy import PreverifyPolicy, add_policy_arguments, policy_config")
    code = replace_once(code, "    args = parser.parse_args()",
        "    add_policy_arguments(parser)\n"
        "    args = parser.parse_args()\n"
        "    QwenAsyncGpuWindow.preverify_policy = PreverifyPolicy.from_config(\n"
        "        SimpleNamespace(**policy_config(args)), layers=40, model_path=args.model)")
    code = replace_once(code, 'config["preverify_topk"] = 4',
        'config["preverify_topk"] = args.preverify_top_k if args.preverify_policy == "fixed_topk" else None\n'
        '    config["preverify_policy"] = policy_config(args)\n'
        '    config["cpu_overlap_criterion"] = "CPU publish <= GPU generation; replay follows publication"')
    code = replace_once(code, 'max(r["cpu_publish_seconds"],\n'
        '                r["gpu_generation_seconds"] + r["install_replay_seconds"])',
        'max(r["cpu_publish_seconds"], r["gpu_generation_seconds"]) + r["install_replay_seconds"]')
    code = replace_once(code, 'sum(r["cpu_publish_seconds"] > r["gpu_generation_seconds"]\n'
        '                                     + r["install_replay_seconds"] for r in cells)',
        'sum(r["cpu_publish_seconds"] > r["gpu_generation_seconds"] for r in cells)')
    code = replace_once(code, 'all(r["cpu_publish_seconds"] <= r["gpu_generation_seconds"]\n'
        '                                       + r["install_replay_seconds"] for r in cells)',
        'all(r["cpu_publish_seconds"] <= r["gpu_generation_seconds"] and r["original_b_retained"] '
        'and not r["deadline_miss"] for r in cells)')
    code = replace_once(code, 'row["cycle_margin_seconds"] = gpu_seconds + restore_seconds - timing["publish_seconds"]',
        'row["cycle_margin_seconds"] = gpu_seconds - timing["publish_seconds"]\n'
        '                        row["cycle_seconds"] = max(gpu_seconds, timing["publish_seconds"]) + restore_seconds\n'
        '                        row["estimated_cpu_wait_seconds"] = max(0., timing["publish_seconds"]-gpu_seconds)')
    code = replace_once(code, 'row["cycle_feasible"] = row["cycle_margin_seconds"] >= 0',
        'row["cycle_feasible"] = row["cycle_margin_seconds"] >= 0 and row["original_b_retained"] and not row["deadline_miss"]')
    code = replace_once(code, 'row.update(preverify_topk=config["preverify_topk"],',
        'row.update(preverify_topk=config["preverify_topk"], preverify_policy=config["preverify_policy"],\n'
        '                            gpu_a_threshold_retained_routes=a_counters[24],\n'
        '                            gpu_a_extra_hit_routes=a_counters[25],\n'
        '                            gpu_b_threshold_retained_routes=b_counters[24]-a_counters[24],\n'
        '                            gpu_b_extra_hit_routes=b_counters[25]-a_counters[25],')
    code = replace_once(code, 'complete_curve = (set(args.loops) == {1, 2, 3, 4, 6, 8} and set(args.threads) == {23, 46}\n'
        '                          and args.samples == 4 and args.repeats == 3 and not args.profile)',
        'complete_curve = (set(args.loops).issubset({3, 4, 6, 8}) and set(args.threads) == {23}\n'
        '                          and args.repeats == 1 and not args.profile)')
    ast.parse(code)
    path.write_text(code)
    changed[str(path.relative_to(target))] = dict(before=source_hash, after=digest(path))
    manifest = {str(p.relative_to(target)): digest(p) for p in target.rglob("*") if p.is_file() and p.suffix in (".py", ".so")}
    write(target / "source_manifest.json", manifest)
    write(target / "adaptation.json", dict(parent=str(parent), changed=changed,
        semantics="threshold OR ready, original weights, selected weighted identity replacement",
        measurement_scope="fixed-window prototype; no complete-request async throughput"))
    return target
