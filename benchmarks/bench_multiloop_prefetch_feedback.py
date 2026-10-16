from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import torch

import bench_multiloop_sparse_verify as benchmark


class FeedbackExperiment(benchmark.MultiloopExperiment):
    def save_probe(self, sample, prefix, loop_index, inputs):
        pass

    def save_workload(self, sample, prefix, loops):
        pass

    def case(self, sample, prefix, anchor, loop_counts, policy):
        directory = self.output
        control = directory / "no_prefetch"
        control.mkdir(exist_ok=True)
        try:
            self.output, self.feedback_enabled = control, False
            count, checks = len(self.cases), len(self.quality_checks)
            super().case(sample, prefix, anchor, loop_counts, policy)
            del self.cases[count:]
            del self.quality_checks[checks:]
            self.output = directory
            self.protected_experts = set()
            self.feedback_enabled = policy == "cheap"
            return super().case(sample, prefix, anchor, loop_counts, policy)
        finally:
            self.output = directory
            self.feedback_enabled = False

    def loop(self, anchor, *, policy="cheap", remaining=4, audit=True):
        anchor, row = super().loop(anchor, policy=policy, remaining=remaining, audit=audit)
        if policy != "cheap" or not getattr(self, "feedback_enabled", False):
            return anchor, row
        loaded, predicted = [], set()
        for layer in row["record"]["layers"]:
            for ids in layer["ids"][:row["returned"]]:
                self.protected_experts.update(layer["layer"] * 256 + expert for expert in ids)
            predicted.update(layer["layer"] * 256 + expert for ids in layer["ids"] for expert in ids)

        def prefetch():
            for layer in row["record"]["layers"]:
                ids = torch.tensor(layer["ids"], dtype=torch.int32, device=self.engine.device)
                self.cache.ensure_experts(layer["layer"], ids)
                count = int(self.cache.num_indices.sum().item())
                loaded.extend(layer["layer"] * 256 + expert for expert in self.cache.src_indices[:count].cpu().tolist())
                self.original_copy()
            owners = self.cache.id_of_slot
            protected = torch.tensor(sorted(self.protected_experts), dtype=torch.int32, device=self.engine.device)
            protected_slots = torch.isin(owners, protected)
            self.cache.usage[~protected_slots] = 0
            self.cache.usage[protected_slots] = self.cache.step + 1

        _, elapsed = self.timed(prefetch)
        resident = set(self.cache.id_of_slot.cpu().tolist())
        row["feedback_prefetch"] = {
            "scope": "real cache boundary prefetch; all current-window top8 ready before next loop; availability upper control, not asynchronous performance",
            "loaded": loaded, "predicted": len(predicted), "protected": len(self.protected_experts),
            "protected_not_resident": len(self.protected_experts - resident), "serial_prefetch_ms": elapsed,
            "prefetch_time_in_component_ms": False,
        }
        print(json.dumps({"feedback_loop_start": row["start"], "prefetched": len(loaded),
                          "protected": len(self.protected_experts), "lost_protected": len(self.protected_experts - resident)}), flush=True)
        return anchor, row


if __name__ == "__main__":
    benchmark.MultiloopExperiment = FeedbackExperiment
    benchmark.main()
    directory = Path(sys.argv[sys.argv.index("--output") + 1])
    config = json.loads((directory / "config.json").read_text())
    config.update({"experiment_variant": "real-cache loop-boundary all-window prefetch with rejection release",
                   "availability": "previous window's predictions all loaded before next loop; upper availability control",
                   "paired_control": "no_prefetch/windows.jsonl; both branches restore the same model and cache snapshot within one process",
                   "timing_valid_for_async_performance": False,
                   "feedback_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
