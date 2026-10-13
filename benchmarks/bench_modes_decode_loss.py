"""Decode-only layer KL importance and loss-aware text threshold calibration."""

import argparse
from collections import Counter
import copy
import gc
import math
from pathlib import Path
import shutil
import statistics
import time

import torch

from bench_modes_decode_calibration import collect as collect_free, extend_samples, histogram_threshold
from bench_modes_decode_replay import DecodeReplayModel
from freetoken.moe.modes import validate_calibration
from freetoken.moe.tablemoe_artifacts import checkpoint_identity, digest, file_hash, finish_stage, read_json, verify_stage, write_json
from mmlongbench_write import CATEGORIES
from tablemoe_data import read_rows, verify_disjoint, write_rows


REPO = Path(__file__).resolve().parents[1]


def balanced_samples(rows, per_category=5):
    if per_category < 1 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("invalid balanced calibration selection")
    selected = []
    for category in CATEGORIES:
        pool = sorted((r for r in rows if r["metadata"]["category"] == category), key=lambda r: r["index"])
        if len(pool) < per_category:
            raise ValueError(f"not enough calibration samples in {category}")
        selected.extend(pool[:per_category])
    return selected


class DecodeLossModel(DecodeReplayModel):
    @torch.inference_mode()
    def prompt(self, inputs, mask):
        if inputs["input_ids"].shape[0] != 1 or not mask.any() or self.sequence_chunk is None:
            raise ValueError("decode loss calibration requires one cached, complete answer")
        prefix = int(mask[0].nonzero()[0])
        length = inputs["input_ids"].shape[1]
        if "image_grid_thw" in inputs or "video_grid_thw" in inputs:
            positions, _ = self.model.model.get_rope_index(
                inputs["input_ids"], mm_token_type_ids=inputs["mm_token_type_ids"],
                image_grid_thw=inputs.get("image_grid_thw"), video_grid_thw=inputs.get("video_grid_thw"),
                attention_mask=inputs.get("attention_mask"))
        else:
            positions = torch.arange(length, device=self.model.device)[None, None, :].expand(4, 1, -1)
        piece = dict(inputs)
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if key in piece:
                piece[key] = piece[key][:, :prefix]
        piece["position_ids"] = positions[:, :, :prefix]
        self.ablation = None
        self.decode_text_tau = 0.
        self.collect_scores = False
        output = self.forward_piece(piece, use_cache=True)
        ordinary = mask & ~torch.isin(inputs["input_ids"], torch.tensor(self.tokenizer.all_special_ids, device=self.model.device))
        if not ordinary.any():
            raise ValueError("no ordinary decode text positions")
        return dict(inputs=inputs, positions=positions, mask=ordinary, prefix=prefix,
                    cache=output.past_key_values, answer_tokens=int(ordinary.sum()))

    def forward_piece(self, piece, **kwargs):
        ids = piece["input_ids"]
        visual = (ids == self.model.config.image_token_id) | (ids == self.model.config.video_token_id)
        text = ~torch.isin(ids, torch.tensor(self.tokenizer.all_special_ids, device=ids.device))
        self.modality = torch.where(visual, 0, torch.where(text, 1, 2))
        return self.model(**piece, return_dict=True, **kwargs)

    @torch.inference_mode()
    def answer(self, context, tau=0., ablation=None, collect_scores=False):
        self.ablation, self.decode_text_tau, self.collect_scores = ablation, tau, collect_scores
        self.histogram.zero_()
        self.routes.zero_()
        self.skipped.zero_()
        inputs = context["inputs"]
        # DynamicCache mutates on append; every candidate starts from the same prompt.
        cache = copy.deepcopy(context["cache"])
        states = []
        for start in range(context["prefix"], inputs["input_ids"].shape[1], self.sequence_chunk):
            end = min(start + self.sequence_chunk, inputs["input_ids"].shape[1])
            piece = dict(input_ids=inputs["input_ids"][:, start:end], position_ids=context["positions"][:, :, start:end])
            if "attention_mask" in inputs:
                piece["attention_mask"] = inputs["attention_mask"][:, :end]
            if "mm_token_type_ids" in inputs:
                piece["mm_token_type_ids"] = inputs["mm_token_type_ids"][:, start:end]
            output = self.forward_piece(piece, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            states.append(output.logits[context["mask"][:, start:end]])
        hidden = torch.cat(states)
        counts = dict(histogram=self.histogram.cpu().clone(), routes=self.routes.cpu().clone(), skipped=self.skipped.cpu().clone()) if collect_scores else None
        if counts is not None:
            expected = context["answer_tokens"] * self.model.config.text_config.num_experts_per_tok
            if not torch.all(counts["routes"] == expected) or int(counts["histogram"].sum()) != expected * len(self.layers):
                raise ValueError("decode histogram layer or token coverage differs")
        return hidden, counts


def source_hashes():
    names = ("benchmarks/bench_modes_decode_loss.py", "benchmarks/bench_modes_decode_replay.py",
             "benchmarks/bench_modes_decode_calibration.py", "benchmarks/modes_calibration.py", "benchmarks/modes_decode_audit.py")
    paths = [REPO / name for name in names]
    paths.extend(p for p in (REPO / "python/freetoken").rglob("*")
                 if p.is_file() and p.suffix in (".py", ".cpp", ".cu", ".h", ".cuh", ".so"))
    return {str(p.relative_to(REPO)): file_hash(p) for p in sorted(paths)}


def prepare(config):
    root, source, dataset = (Path(config[key]) for key in ("output", "prepared_source", "dataset_root"))
    prepared = verify_stage(source / "prepare")
    verify_stage(dataset / "prepare")
    calibration_path = Path(config["source_config"])
    verify_stage(calibration_path.parent)
    calibration = read_json(calibration_path)
    identity = checkpoint_identity(calibration["checkpoint"]["path"], expected=calibration["checkpoint"])
    experiment = read_json(source / "experiment.json")
    if identity != calibration["checkpoint"] or identity != experiment["checkpoint"] or prepared["truncated_samples"]:
        raise ValueError("checkpoint mismatch or truncated prepared answers")
    baseline = dataset / "results/calibration_72/predictions/baseline/completed_predictions.json"
    if file_hash(baseline) != experiment["reference_source"]["predictions_sha256"]:
        raise ValueError("baseline reference answer binding differs")
    original = read_rows(dataset / "prepare/calibration.jsonl")
    old = read_json(Path(config["previous_calibration"]) / "prepare/samples.json")
    pool, _, _ = extend_samples(original, [r["id"] for r in old])
    rows = balanced_samples(pool, config.get("per_category", 5))
    verify_disjoint(rows, read_rows(dataset / "prepare/evaluation.jsonl"))
    by_id = {r["id"]: r for r in read_json(source / "prepare/samples.json")}
    records = [by_id[r["id"]] for r in rows]
    info = dict(scope="decode_text_layer_kl_importance_and_threshold", dataset="MMLongBench-Write",
        sample_ids=[r["id"] for r in rows], samples=len(rows), categories=dict(Counter(r["metadata"]["category"] for r in rows)),
        target_skip_fraction=.5, tolerance=config.get("tolerance", .01), heldout_samples_used=0,
        original_row_equality=True, new_reference_generation=False, answers_truncated=False,
        preserved_prefill_importance=True, preserved_prefill_tau=True, vision_recalibrated=False,
        ablation="upstream identity MoE-output ablation on one layer's ordinary decode text inputs only",
        loss="D_KL(teacher || candidate) over vocabulary at ordinary decode input positions predicting the next token; terminal special tokens excluded",
        teacher="full text expert execution during decode, conditioned on the frozen original MoDES prefill",
        normalization="sum KL over samples per layer, then normalize across the 48 decode text layers",
        sequence_chunk=config.get("sequence_chunk", 1024), logits_chunk=config.get("logits_chunk", 128),
        source_config_sha256=file_hash(calibration_path), prepared_manifest_sha256=file_hash(source / "prepare/manifest.json"),
        reference_predictions_sha256=file_hash(baseline), checkpoint=identity, source=source_hashes())
    root.mkdir(parents=True, exist_ok=True)
    if (root / "selection.json").exists() and read_json(root / "selection.json") != info:
        raise ValueError("calibration selection or sources changed on resume")
    write_rows(root / "calibration.jsonl", rows)
    write_json(root / "samples.json", records)
    write_json(root / "config.json", config)
    write_json(root / "selection.json", info)
    for name in info["source"]:
        if name.endswith(".py") and (name.startswith("benchmarks/") or "/modes" in name):
            target = root / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, target)
    hooks = root / "audit_hooks"
    hooks.mkdir(exist_ok=True)
    (hooks / "sitecustomize.py").write_text("import os\nif os.environ.get('MODES_DECODE_AUDIT_DIR'):\n    from modes_decode_audit import install\n    install()\n")
    return calibration, rows, records, info


def load_sample(adapter, config, record):
    path = Path(config["prepared_source"]) / "prepare" / record["path"]
    if file_hash(path) != record["sha256"]:
        raise ValueError("prepared tensor changed")
    saved = torch.load(path, weights_only=True)
    inputs = {k: v.to(adapter.model.device) for k, v in saved["inputs"].items()}
    return adapter.prompt(inputs, saved["mask"].to(adapter.model.device))


def update_progress(root, stage, **kwargs):
    value = dict(stage=stage, complete=False, **kwargs)
    write_json(root / "progress.json", value)
    print(stage, kwargs, flush=True)


def importance(adapter, config, records, info):
    root, folder = Path(config["output"]), Path(config["output"]) / "importance"
    folder.mkdir(exist_ok=True)
    stage_inputs = dict(selection_sha256=digest(info), sample_ids=[r["id"] for r in records])
    if (folder / "complete.json").exists():
        verify_stage(folder, stage_inputs)
        return read_json(folder / "result.json")
    started = time.perf_counter()
    outputs, pages = [], []
    for index, record in enumerate(records):
        path = folder / f"sample_{index:03d}.json"
        if path.exists():
            page = read_json(path)
            if page["id"] != record["id"] or page["selection_sha256"] != digest(info):
                raise ValueError("partial importance sample binding differs")
        else:
            context = load_sample(adapter, config, record)
            teacher, _ = adapter.answer(context)
            teacher = teacher.cpu()
            repeated, _ = adapter.answer(context)
            repeat_kl = adapter.kl(teacher, repeated)
            if repeat_kl / context["answer_tokens"] > 1e-5:
                raise ValueError("cached prompt teacher replay is not stable")
            del repeated
            losses = {}
            for layer in adapter.layers:
                tick = time.perf_counter()
                candidate, _ = adapter.answer(context, ablation=(layer, "text"))
                losses[str(layer)] = adapter.kl(teacher, candidate)
                del candidate
                update_progress(root, "importance", completed_samples=index, total_samples=len(records),
                    id=record["id"], layer=layer, layer_kl=losses[str(layer)], forward_and_kl_s=time.perf_counter()-tick,
                    elapsed_s=time.perf_counter()-started)
            page = dict(id=record["id"], category=record["category"], answer_tokens=context["answer_tokens"],
                        losses=losses, teacher_repeat_kl=repeat_kl, selection_sha256=digest(info))
            write_json(path, page)
            del context, teacher
        pages.append(page)
        outputs.append(path)
    losses = {str(layer): sum(page["losses"][str(layer)] for page in pages) for layer in adapter.layers}
    total = sum(losses.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("decode text layer losses do not define importance")
    result = dict(importance={layer: loss / total for layer, loss in losses.items()}, raw_losses=losses,
        samples=len(records), answer_tokens=sum(p["answer_tokens"] for p in pages), records=pages,
        elapsed_s=time.perf_counter()-started, ablation=info["ablation"], loss=info["loss"])
    write_json(folder / "result.json", result)
    finish_stage(folder, stage_inputs, [*outputs, folder / "result.json"])
    return result


def set_importance(adapter, result):
    for layer_id, layer in adapter.calibration["layers"].items():
        layer["decode_importance"] = dict(layer["importance"], text=result["importance"][layer_id])


def evaluate(adapter, config, records, folder, tau, info):
    folder.mkdir(parents=True, exist_ok=True)
    inputs = dict(tau=tau, sample_ids=[r["id"] for r in records], selection_sha256=digest(info),
                  decode_importance={i: v["decode_importance"] for i, v in adapter.calibration["layers"].items()})
    if (folder / "complete.json").exists():
        verify_stage(folder, inputs)
        return read_json(folder / "result.json")
    started = time.perf_counter()
    histogram, values, outputs = torch.zeros(65536, dtype=torch.int64), [], []
    for index, record in enumerate(records):
        context = load_sample(adapter, config, record)
        teacher, _ = adapter.answer(context)
        if tau == 0.:
            hidden, counts = adapter.answer(context, collect_scores=True)
        else:
            hidden, counts = adapter.answer(context, tau=tau, collect_scores=True)
        kl = adapter.kl(teacher, hidden)
        total, skipped = int(counts["routes"].sum()), int(counts["skipped"].sum())
        values.append(dict(id=record["id"], category=record["category"], answer_tokens=context["answer_tokens"],
                           total_text_routes=total, skipped_text_routes=skipped, skip_fraction=skipped/total,
                           kl_sum=kl, kl_per_token=kl/context["answer_tokens"]))
        histogram += counts["histogram"]
        output = folder / f"sample_{index:03d}.pt"
        torch.save(counts, output)
        outputs.append(output)
        update_progress(Path(config["output"]), folder.name, completed_samples=index+1, total_samples=len(records),
                        tau=tau, last=values[-1], elapsed_s=time.perf_counter()-started)
        del context, teacher, hidden
    total = sum(v["total_text_routes"] for v in values)
    skipped = sum(v["skipped_text_routes"] for v in values)
    tokens = sum(v["answer_tokens"] for v in values)
    kl_sum = sum(v["kl_sum"] for v in values)
    result = dict(tau=tau, samples=len(values), records=values, total_text_routes=total,
        skipped_text_routes=skipped, observed_skip_fraction=skipped/total, answer_tokens=tokens,
        kl_sum=kl_sum, kl_per_token=kl_sum/tokens, elapsed_s=time.perf_counter()-started,
        histogram_threshold=histogram_threshold(histogram))
    torch.save(histogram, folder / "histogram.pt")
    write_json(folder / "result.json", result)
    finish_stage(folder, inputs, [*outputs, folder / "histogram.pt", folder / "result.json"])
    return result


def select_candidate(candidates, tolerance=.01):
    feasible = [v for v in candidates if abs(v["observed_skip_fraction"]-.5) <= tolerance]
    if not feasible:
        raise ValueError("no loss-evaluated candidate meets the decode skip target")
    return min(feasible, key=lambda v: (v["kl_per_token"], abs(v["observed_skip_fraction"]-.5)))


def search(adapter, config, records, info):
    root = Path(config["output"])
    full = evaluate(adapter, config, records, root / "full_decode", 0., info)
    if full["skipped_text_routes"] or full["kl_per_token"] > 1e-5:
        raise ValueError("full-decode teacher control failed")
    histogram = torch.load(root / "full_decode/histogram.pt", weights_only=True)
    targets = (.5, .5-info["tolerance"]*.75, .5+info["tolerance"]*.75)
    queue = [histogram_threshold(histogram, target)["tau"] for target in targets]
    candidates, seen = [], set()
    for index in range(config.get("max_candidates", 8)):
        if not queue:
            break
        tau = queue.pop(0)
        if tau in seen:
            continue
        seen.add(tau)
        result = evaluate(adapter, config, records, root / f"candidate_{index}", tau, info)
        candidates.append(dict(result, folder=f"candidate_{index}"))
        if abs(result["observed_skip_fraction"]-.5) > info["tolerance"]:
            queue.append(result["histogram_threshold"]["tau"])
    selected = select_candidate(candidates, info["tolerance"])
    report = dict(selected=selected, candidates=candidates,
                  objective="minimum token-weighted KL among candidates within target skip tolerance")
    write_json(root / "search.json", report)
    return selected


def export(calibration, folder, tau, info):
    data = copy.deepcopy(calibration)
    for layer in data["layers"].values():
        layer["decode_tau"] = dict(layer["tau"], text=tau)
    data["decode_calibration_info"] = info
    data["control"] = False
    validate_calibration(data, [int(i) for i in data["layers"]])
    path = folder / "modes_config.json"
    write_json(path, data)
    finish_stage(folder, dict(calibration=info, tau=tau), [path],
        prefill_importance_preserved=True, prefill_thresholds_preserved=True)
    return path


def load_adapter(calibration, config):
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

    path = calibration["checkpoint"]["path"]
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(path, dtype=torch.float16,
        attn_implementation="sdpa", local_files_only=True).to("cuda")
    tokenizer = AutoProcessor.from_pretrained(path, local_files_only=True).tokenizer
    adapter = DecodeLossModel(model, tokenizer, calibration, config.get("sequence_chunk", 1024))
    adapter.logits_chunk = config.get("logits_chunk", 128)
    return adapter


def free_validation(config, calibration, rows, info, selected):
    root = Path(config["output"])
    baseline = read_json(Path(config["dataset_root"]) / "results/calibration_72/predictions/baseline/completed_predictions.json")
    samples = []
    for category in CATEGORIES:
        pool = [r for r in rows if r["metadata"]["category"] == category]
        lengths = {r["id"]: baseline[r["id"]]["usage"]["completion_tokens"] for r in pool}
        median = statistics.median(lengths.values())
        samples.append(min(pool, key=lambda r: (abs(lengths[r["id"]]-median), r["index"])))
    write_rows(root / "free_samples.jsonl", samples)
    tau, rounds = selected["tau"], []
    free_config = dict(gpu=config["gpu"], port=config["port"], max_tokens=config.get("max_tokens", 4096),
                       server_command=config["server_command"])
    for index in range(config.get("max_free_iterations", 3)):
        path = export(calibration, root / f"free_candidate_{index}_config", tau,
                      dict(info, free_round=index, fixed_answer_selected_tau=selected["tau"]))
        if index == 0:
            smoke = collect_free(dict(free_config, max_tokens=32), root / "free_smoke", samples[:1], path)
            if not smoke["graph_decode_verified"]:
                raise ValueError("free-generation graph smoke failed")
        update_progress(root, "free_validation", iteration=index, tau=tau, sample_ids=[r["id"] for r in samples])
        result = collect_free(free_config, root / f"free_{index}", samples, path)
        tau_bits = int(torch.tensor(tau, dtype=torch.float16).view(torch.int16))
        for record in result["request_records"]:
            counts = torch.load(root / f"free_{index}" / record["histogram"], weights_only=True)
            predicted = counts["histogram"][:, :tau_bits].sum(1)
            if not torch.equal(predicted, counts["skipped"]):
                raise ValueError("free decode score histogram and actual route mask differ")
        rounds.append(dict(result, tau=tau, folder=f"free_{index}"))
        write_json(root / "free_validation.json", dict(rounds=rounds, selected=rounds[-1],
            threshold_refitted=index>0, heldout_samples_used=0, selection="one nearest median reference length per category within the calibration 30"))
        if abs(result["observed_skip_fraction"]-.5) <= info["tolerance"]:
            return rounds[-1]
        tau = result["histogram_threshold"]["tau"]
    raise ValueError("free-generation decode skip target did not converge")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("prepare", "smoke", "all"), default="all")
    args = parser.parse_args()
    config = read_json(args.config)
    root = Path(config["output"])
    started = time.perf_counter()
    calibration, rows, records, info = prepare(config)
    if args.stage == "prepare":
        print("PREPARED", info["sample_ids"], info["categories"], flush=True)
        return
    try:
        adapter = load_adapter(calibration, config)
        if args.stage == "smoke":
            context = load_sample(adapter, config, records[0])
            teacher, _ = adapter.answer(context)
            repeated, counts = adapter.answer(context, collect_scores=True)
            zero_kl = adapter.kl(teacher, repeated)
            ablated, _ = adapter.answer(context, ablation=(0, "text"))
            ablation_kl = adapter.kl(teacher, ablated)
            all_skip, skipped = adapter.answer(context, tau=1., collect_scores=True)
            assert zero_kl/context["answer_tokens"] <= 1e-5 and not counts["skipped"].sum()
            assert torch.equal(skipped["routes"], skipped["skipped"])
            value = dict(complete=True, samples=1, id=records[0]["id"], answer_tokens=context["answer_tokens"],
                zero_kl=zero_kl, ablation_kl=ablation_kl, all_skipped_kl=adapter.kl(teacher, all_skip),
                prompt_cache_reused=True, prompt_cache_length=context["cache"].get_seq_length(), elapsed_s=time.perf_counter()-started)
            write_json(root / "smoke_result.json", value)
            print("SMOKE", value, flush=True)
            return
        importance_result = importance(adapter, config, records, info)
        set_importance(adapter, importance_result)
        selected = search(adapter, config, records, info)
        export(calibration, root / "fixed_answer_config", selected["tau"], dict(info, kl_per_token=selected["kl_per_token"]))
        write_json(root / "fixed_calibration.json", dict(complete=True, samples=len(records), selected=selected,
            importance_recomputed=True, loss_evaluated=True, free_running_skip_verified=False))
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
        free = free_validation(config, calibration, rows, info, selected)
        final_evaluation = selected
        if free["tau"] != selected["tau"]:
            adapter = load_adapter(calibration, config)
            final_evaluation = evaluate(adapter, config, records, root / "final_loss", free["tau"], info)
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
        if source_hashes() != info["source"]:
            raise ValueError("runtime or calibration sources changed during the run")
        original = read_json(config["source_config"])
        if any(layer["importance"] != original["layers"][i]["importance"] or layer["tau"] != original["layers"][i]["tau"]
               for i, layer in calibration["layers"].items()):
            raise ValueError("prefill importance or thresholds changed")
        final = export(calibration, root / "modes_config", free["tau"], dict(info,
            fixed_answer_skip_fraction=final_evaluation["observed_skip_fraction"], kl_per_token=final_evaluation["kl_per_token"],
            free_running_skip_fraction=free["observed_skip_fraction"], free_threshold_refitted=free["tau"] != selected["tau"]))
        summary = dict(complete=True, samples=len(records), categories=info["categories"], tau=free["tau"],
            fixed_answer_skip_fraction=final_evaluation["observed_skip_fraction"], fixed_answer_kl_per_token=final_evaluation["kl_per_token"],
            fixed_answer_kl_sum=final_evaluation["kl_sum"], free_running_skip_fraction=free["observed_skip_fraction"],
            target_skip_fraction=.5, tolerance=info["tolerance"], free_samples=6, importance_recomputed=True,
            loss_evaluated=True, task_quality_evaluated=False, heldout_samples_used=0,
            free_threshold_refitted=free["tau"] != selected["tau"], graph_decode_verified=free["graph_decode_verified"],
            modes_config=str(final), elapsed_s=time.perf_counter()-started)
        predictions = read_json(root / free["folder"] / "predictions.json")
        summary["free_records"] = [dict(record, completion_tokens=prediction["usage"]["completion_tokens"],
            finish_reason=prediction["finish_reason"])
            for record, prediction in zip(free["request_records"], predictions)]
        summary["free_categories"] = free["categories"]
        write_json(root / "summary.json", summary)
        dependencies = [root / "importance", root / "full_decode", root / selected["folder"], root / free["folder"], final.parent]
        if free["tau"] != selected["tau"]:
            dependencies.append(root / "final_loss")
        output_files = [root / name for name in ("selection.json", "samples.json", "calibration.jsonl", "search.json", "free_validation.json", "summary.json")]
        output_files.extend(root / free["folder"] / r["histogram"] for r in free["request_records"])
        finish_stage(root, dict(config=config, selection=info), output_files,
            scope=info["scope"], dependencies={str(p): file_hash(p/"manifest.json") for p in dependencies})
        update_progress(root, "complete", result=summary)
        write_json(root / "progress.json", dict(stage="complete", complete=True, result=summary))
        print("COMPLETE", summary, flush=True)
    except BaseException as error:
        write_json(root / "error.json", dict(complete=False, error=repr(error)))
        raise


if __name__ == "__main__":
    main()
