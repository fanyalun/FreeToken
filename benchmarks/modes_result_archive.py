from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import importlib.util
from pathlib import Path

from modes_experiment import digest, read, write


ROOT = Path("/home/fanya/freetoken_benchmark_results")


def register(run, *, validity=None):
    preserve_existing = validity is None
    if preserve_existing:
        validity = "incomplete"
    run = Path(run).resolve()
    root = ROOT
    if not run.is_relative_to(root):
        raise ValueError("MoDES results must be inside the local result archive")
    selection = read(run / "selection.json")
    run_id = "modes_text_preverify_" + run.parent.name.removeprefix("modes_text_preverify_") + "_" + run.name
    relative = run.relative_to(root)
    row = dict(id=run_id, category="speculative", series="modes_text_preverify",
        config=dict(model=selection["model"], model_config_sha256=selection["model_config_sha256"],
            dtype="bfloat16", score_dtype="float32", tp=1, batch=1, mtp_depth=3,
            splits={name: len(ids) for name, ids in selection["ids"].items()},
            budgets_gib=[40, 24, 16, 12], calibration_arithmetic="resident_rowwise_reference_fixed_kv_tiles_and_gated_norm",
            runtime_arithmetic="existing fast path", gpu=0),
        measurement_scope="staged fused calibration, complete GPU requests, and CPU window feasibility",
        measured_date=selection["created_utc"][:10], date_precision="day",
        purpose="formal", validity=validity, lifecycle="current", path=str(relative),
        entrypoint=str(relative / "readme.md"), differences=["text-only Qwen3.6", "nominal skip25/50",
            "runtime threshold OR ready", "CPU replay excluded from overlap deadline"], related_to=[])
    artifacts = []
    for name, role in [("selection.json", "experiment_configuration"),
                       ("input_validation.json", "input_validation"),
                       ("runtime/source_manifest.json", "source_fingerprint")]:
        path = run / name
        artifacts.append(dict(id=run_id+"_"+name.replace("/", "_"), run_id=run_id,
            path=str(path.relative_to(root)), role=role, sha256=digest(path), bytes=path.stat().st_size,
            depends_on=[]))
    with (root / ".catalog.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        catalog = read(root / "catalog.json")
        # Preparation must preserve the existing validation decision and receipt.
        if preserve_existing and any(r["id"] == run_id for r in catalog["runs"]):
            return
        catalog["runs"] = [r for r in catalog["runs"] if r["id"] != run_id] + [row]
        ids = {a["id"] for a in artifacts}
        catalog["artifacts"] = [a for a in catalog["artifacts"] if a["id"] not in ids] + artifacts
        catalog["updated_at"] = datetime.now(timezone.utc).isoformat()
        write(root / "catalog.json", catalog)
        renderer = Path.home() / ".codex/skills/experiment-results-archive/scripts/results_catalog.py"
        if renderer.exists():
            spec = importlib.util.spec_from_file_location("modes_catalog_renderer", renderer)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.render(root, catalog)
    write(run / ".archive/validation.json", dict(status="validated", validity=validity,
        run_id=run_id, artifacts=artifacts, historical_results_modified=False))
