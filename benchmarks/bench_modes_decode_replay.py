"""Calibrate decode text scores on existing complete, frozen reference answers."""

import argparse
from collections import Counter
from pathlib import Path
import types

import torch

from bench_modes_decode_calibration import export_config, extend_samples, histogram_threshold
from modes_calibration import CalibrationModel
from freetoken.moe.tablemoe_artifacts import checkpoint_identity, file_hash, finish_stage, read_json, verify_stage, write_json
from tablemoe_data import read_rows, verify_disjoint, write_rows


def replay_mlp_forward(module, hidden_states):
    state = module._modes_calibration
    shape = hidden_states.shape
    flat = hidden_states.reshape(-1, shape[-1])
    modality = state.modality.reshape(-1)
    if (state.is_decode and state.ablation is not None
            and state.ablation == (module._modes_layer, "text")):
        kept = modality != 1
        result = flat.clone()
        # Match the upstream identity ablation, restricted to decode text rows.
        if kept.any():
            _, weights, ids = module.gate(flat[kept])
            result[kept] = module.experts(flat[kept], ids, weights.half())
        return result.reshape(shape)
    _, weights, ids = module.gate(flat)
    weights = weights.half()
    layer = state.calibration["layers"][str(module._modes_layer)]
    thresholds = dict(layer["tau"])
    if state.is_decode:
        thresholds["text"] = state.decode_text_tau
    exact = torch.ones_like(weights, dtype=torch.bool)
    for name, mod in (("visual", 0), ("text", 1)):
        importance = layer.get("decode_importance", layer["importance"]) if state.is_decode else layer["importance"]
        score = (weights.float() * importance[name]).half()
        threshold = torch.tensor(thresholds[name], device=weights.device, dtype=torch.float16)
        rows = modality == mod
        exact &= ~(rows[:, None] & (score < threshold))
        if state.is_decode and name == "text" and getattr(state, "collect_scores", True):
            bits = score[rows].contiguous().view(torch.int16).long().flatten()
            state.histogram.scatter_add_(0, bits, torch.ones_like(bits))
            state.routes[module._modes_layer] += rows.sum() * weights.shape[1]
            state.skipped[module._modes_layer] += ((~exact) & rows[:, None]).sum()
    return module.experts(flat, ids, weights.masked_fill(~exact, 0)).reshape(shape)


class DecodeReplayModel(CalibrationModel):
    def __init__(self, model, tokenizer, calibration, sequence_chunk=1024):
        super().__init__(model, tokenizer, sequence_chunk=sequence_chunk)
        self.calibration = calibration
        self.histogram = torch.zeros(65536, device=model.device, dtype=torch.int64)
        self.routes = torch.zeros(len(self.layers), device=model.device, dtype=torch.int64)
        self.skipped = torch.zeros_like(self.routes)
        self.is_decode = False
        for module in self.layers.values():
            module.forward = types.MethodType(replay_mlp_forward, module)
        model.register_forward_pre_hook(self._phase, with_kwargs=True)

    def _phase(self, module, args, kwargs):
        self.is_decode = kwargs.get("past_key_values") is not None

    @torch.inference_mode()
    def replay(self, inputs, mask, tau):
        self.decode_text_tau = tau
        self.histogram.zero_()
        self.routes.zero_()
        self.skipped.zero_()
        if self.sequence_chunk is None:
            raise ValueError("decode replay requires a cached prompt and answer suffix")
        self.hidden(inputs, mask)
        routes, skipped = self.routes.cpu(), self.skipped.cpu()
        if not torch.all(routes == routes[0]) or routes.sum() <= 0:
            raise ValueError("decode replay layer coverage differs")
        return dict(histogram=self.histogram.cpu().clone(), routes=routes, skipped=skipped)


def prepare(config):
    root = Path(config["output"])
    source = Path(config["prepared_source"])
    stage = verify_stage(source / "prepare")
    source_experiment = read_json(source / "experiment.json")
    data = Path(config["dataset_root"])
    original = read_rows(data / "prepare/calibration.jsonl")
    old = read_json(Path(config["previous_calibration"]) / "prepare/samples.json")
    rows, missing, additions = extend_samples(original, [r["id"] for r in old])
    verify_disjoint(rows, read_rows(data / "prepare/evaluation.jsonl"))
    calibration_path = Path(config["source_config"])
    verify_stage(calibration_path.parent)
    calibration = read_json(calibration_path)
    identity = checkpoint_identity(calibration["checkpoint"]["path"], expected=calibration["checkpoint"])
    if identity != calibration["checkpoint"] or identity != source_experiment["checkpoint"]:
        raise ValueError("replay checkpoint binding differs")
    records = {r["id"]: r for r in read_json(source / "prepare/samples.json")}
    selected = [records[r["id"]] for r in rows]
    if stage["truncated_samples"] != 0:
        raise ValueError("existing prepared answers were truncated")
    baseline = data / "results/calibration_72/predictions/baseline/completed_predictions.json"
    if source_experiment["reference_source"]["predictions_sha256"] != file_hash(baseline):
        raise ValueError("existing baseline answer binding differs")
    root.mkdir(parents=True, exist_ok=True)
    write_rows(root / "calibration.jsonl", rows)
    write_json(root / "samples.json", selected)
    write_json(root / "config.json", config)
    info = dict(dataset="MMLongBench-Write", scope="decode_text_only", target_skip_fraction=.5,
        tolerance=config.get("tolerance", .01), samples=len(rows), previous_samples=len(old),
        missing_categories=missing, added_ids=[r["id"] for r in additions], added_per_missing_category=5,
        categories=dict(Counter(r["metadata"]["category"] for r in rows)), sample_ids=[r["id"] for r in rows],
        source_config_sha256=file_hash(calibration_path), prepared_manifest_sha256=file_hash(source / "prepare/manifest.json"),
        reference_predictions_sha256=file_hash(baseline), checkpoint=identity,
        original_row_equality=read_rows(root / "calibration.jsonl") == rows,
        heldout_samples_used=0, new_reference_generation=False, answers_truncated=False,
        preserved_importance=True, preserved_prefill_tau=True,
        execution="HF FP16 cached prompt followed by teacher-forced answer chunks; no free-running generation",
        runtime_decode_graph_verified=False, sequence_chunk=config.get("sequence_chunk", 1024),
        denominator="ordinary answer text token top-k routes summed over all physical MoE layers",
        calibration_source=str(source), runtime_performance_measured=False)
    write_json(root / "selection.json", info)
    write_json(root / "complete.json", dict(complete=False))
    return calibration, selected, info


def collect(adapter, config, records, folder, tau):
    folder.mkdir(parents=True, exist_ok=True)
    inputs = dict(tau=tau, sample_ids=[r["id"] for r in records],
                  prepared_manifest_sha256=file_hash(Path(config["prepared_source"]) / "prepare/manifest.json"),
                  sequence_chunk=adapter.sequence_chunk)
    if (folder / "complete.json").exists():
        verify_stage(folder, inputs)
        return read_json(folder / "result.json")
    histogram, values = torch.zeros(65536, dtype=torch.int64), []
    outputs = []
    for record in records:
        path = Path(config["prepared_source"]) / "prepare" / record["path"]
        if file_hash(path) != record["sha256"]:
            raise ValueError("existing prepared tensor changed")
        saved = torch.load(path, weights_only=True)
        batch = {key: value.to(adapter.model.device) for key, value in saved["inputs"].items()}
        value = adapter.replay(batch, saved["mask"].to(adapter.model.device), tau)
        output = folder / f"sample_{len(values):03d}.pt"
        torch.save(value, output)
        outputs.append(output)
        histogram += value["histogram"]
        total, skipped = int(value["routes"].sum()), int(value["skipped"].sum())
        values.append(dict(id=record["id"], category=record["category"],
            total_text_routes=total, skipped_text_routes=skipped, skip_fraction=skipped / total,
            original_tokens=record["tokens"], original_answer_tokens=record["answer_tokens"]))
        write_json(folder / "progress.json", dict(samples=values, tau=tau))
        write_json(folder.parent / "progress.json", dict(stage=folder.name, completed_samples=len(values),
            total_samples=len(records), last_id=record["id"], complete=False))
        print(folder.name, len(values), len(records), values[-1], flush=True)
    total = sum(r["total_text_routes"] for r in values)
    skipped = sum(r["skipped_text_routes"] for r in values)
    categories = {}
    for category in {r["category"] for r in values}:
        route_count = sum(r["total_text_routes"] for r in values if r["category"] == category)
        skipped_count = sum(r["skipped_text_routes"] for r in values if r["category"] == category)
        categories[category] = dict(total_text_routes=route_count, skipped_text_routes=skipped_count,
                                    skip_fraction=skipped_count / route_count)
    result = dict(tau=tau, samples=len(values), records=values, total_text_routes=total,
        skipped_text_routes=skipped, observed_skip_fraction=skipped / total, categories=categories,
        histogram_threshold=histogram_threshold(histogram), new_reference_generation=False)
    torch.save(histogram, folder / "histogram.pt")
    write_json(folder / "result.json", result)
    finish_stage(folder, inputs, [*outputs, folder / "histogram.pt", folder / "result.json"])
    return result


def main():
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    root = Path(config["output"])
    calibration, records, info = prepare(config)
    model_path = calibration["checkpoint"]["path"]
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_path, dtype=torch.float16,
        attn_implementation="sdpa", local_files_only=True).to("cuda")
    tokenizer = AutoProcessor.from_pretrained(model_path, local_files_only=True).tokenizer
    adapter = DecodeReplayModel(model, tokenizer, calibration, config.get("sequence_chunk", 1024))
    try:
        smoke = collect(adapter, config, records[:1], root / "smoke", 0.)
        if smoke["skipped_text_routes"]:
            raise ValueError("zero text decode tau skipped routes")
        initial = collect(adapter, config, records, root / "full_decode", 0.)
        tau = initial["histogram_threshold"]["tau"]
        for index in range(config.get("max_iterations", 3)):
            result = collect(adapter, config, records, root / f"candidate_{index}", tau)
            rate = result["observed_skip_fraction"]
            if abs(rate - .5) <= info["tolerance"]:
                path = export_config(Path(config["source_config"]), root / "modes_config", tau,
                    dict(info, observed_skip_fraction=rate, validation_result_sha256=file_hash(root / f"candidate_{index}/result.json")))
                write_json(root / "complete.json", dict(complete=True, scope="fixed_answer_decode_replay_calibration",
                    samples=len(records), tau=tau, target_skip_fraction=.5, observed_skip_fraction=rate,
                    tolerance=info["tolerance"], modes_config=str(path), new_reference_generation=False,
                    heldout_samples_used=0, free_running_skip_verified=False, quality_evaluated=False))
                write_json(root / "progress.json", dict(stage="complete", complete=True))
                return
            tau = result["histogram_threshold"]["tau"]
        raise ValueError("fixed-answer decode skip calibration did not converge")
    except BaseException as error:
        write_json(root / "error.json", dict(error=repr(error), complete=False))
        raise


if __name__ == "__main__":
    main()
