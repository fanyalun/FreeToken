"""Run paired startup windows after the independent prefix parity gates pass."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("audit", "performance"), required=True)
    parser.add_argument("--audit-root", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    from qwen_mtp_config import source_fingerprint
    source_hashes = source_fingerprint(root)
    old = json.loads((args.baseline / "retained_matrix_complete.json").read_text())["jobs"]
    manifest = dict(phase=args.phase, status="incomplete", expected_windows=72 if args.phase == "performance" else 0,
                    jobs=[], scope="startup windows; no full-request throughput")
    for budget in (24, 16):
        source = args.baseline / ("budget_24g_retained" if budget == 24 else "budget_16g_retained_retry")
        config = json.loads((source / "config.json").read_text())
        template = next(j["command"] for j in old if j["command"][j["command"].index("--budget-gib") + 1] == str(budget))
        for chunk in (None, 512):
            pair = f"budget_{budget}_chunk_{chunk or 'full'}"
            if args.phase == "performance":
                if args.audit_root is None:
                    parser.error("performance requires --audit-root")
                audit = json.loads((args.audit_root / pair / "complete.json").read_text())
                if audit.get("status") != "prefix_parity_validated" or not audit.get("all_prefixes_exact"):
                    raise RuntimeError(f"prefix parity gate missing: {pair}")
                audited_config = json.loads((args.audit_root / pair / "config.json").read_text())
                if audited_config["source_sha256"] != source_hashes:
                    raise RuntimeError("runtime changed after the bitwise audit")
            modes = ("layerwise",) if args.phase == "audit" else ("sync", "layerwise_barrier", "layerwise")
            for mode in modes:
                name = pair if args.phase == "audit" else f"{pair}_{mode}"
                output = args.output / name
                command = list(template)
                command[0] = sys.executable
                command.remove("--auto-slots")
                command += ["--gpu-slots", str(config["actual_slots"]), "--startup-timing",
                            "--prefix-transfer", mode]
                command[command.index("--output") + 1] = str(output)
                if chunk:
                    command += ["--prompt-chunk-size", str(chunk)]
                if args.phase == "audit":
                    command.remove("--formal-timing")
                    command += ["--prefix-audit-only"]
                environment = dict(os.environ, PYTHONPATH=str(root / "python"), CUDA_VISIBLE_DEVICES="0",
                    CUDA_HOME="/usr/local/cuda-13.0",
                    PATH=f"{Path(sys.executable).parent}:/usr/local/cuda-13.0/bin:" + os.environ["PATH"])
                job = dict(name=name, command=command, actual_slots=config["actual_slots"],
                           status="running", paired_chunk=chunk, prefix_transfer=mode)
                manifest["jobs"].append(job)
                (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                with (args.output / f"{name}.log").open("w") as log:
                    result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
                job["returncode"] = result.returncode
                complete = json.loads((output / "complete.json").read_text()) if (output / "complete.json").exists() else {}
                valid = (complete.get("all_prefixes_exact") if args.phase == "audit" else
                         complete.get("coverage_formal_timing_complete") and complete.get("measured_windows") == 6)
                job["status"] = "valid" if not result.returncode and valid else "incomplete_or_invalid"
                if not result.returncode and (output / "config.json").exists():
                    current = json.loads((output / "config.json").read_text())
                    if (current.get("actual_slots") != config["actual_slots"]
                            or current.get("actual_kv_pages") != config["actual_kv_pages"]):
                        job["status"] = "incomplete_or_invalid"
                        job["error"] = "paired capacities changed or were not recorded"
                    binary = root / "python/freetoken/kernel/_cpu_moe.cpython-312-x86_64-linux-gnu.so"
                    job["cpu_binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
                (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                if job["status"] != "valid":
                    raise RuntimeError(f"gate failed: {name}; inspect its log and completion marker")
    manifest["status"] = "complete"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
