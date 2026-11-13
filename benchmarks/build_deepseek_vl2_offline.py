"""Build prompt-only DeepSeek-VL2 tables from a frozen multimodal split."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

from freetoken.moe.tablemoe_artifacts import digest, file_hash, finish_stage, read_json, verify_stage, write_json
from freetoken.moe.tablemoe_builder import InputCollector, algorithm_config, batched_kmeans, expert_values, load_inputs, train_pca


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def conversation(row):
    content, images = [], []
    for item in row["vlmeval_message"]:
        if item["type"] == "image":
            content.append("<image>\n")
            images.append(item["value"])
        elif item["type"] == "text":
            content.append(item["value"])
        else:
            raise ValueError("unsupported prompt item")
    if images != [image["path"] for image in row["images"]]:
        raise ValueError("frozen image order differs from the prompt")
    return [dict(role="<|User|>", content="".join(content), images=images),
            dict(role="<|Assistant|>", content="")]


def image_offsets(mask, counts):
    positions = torch.where(mask.cpu().reshape(-1))[0].tolist()
    if sum(counts) != len(positions):
        raise ValueError("processor image-token coverage mismatch")
    result, offset = [], 0
    for count in counts:
        selected = positions[offset:offset + count]
        if not selected or selected != list(range(selected[0], selected[-1] + 1)):
            raise ValueError("an image must occupy a contiguous token span")
        result.append((selected[0], selected[-1] + 1))
        offset += count
    return result


def collection_batch(sample, uid):
    ids = sample.input_ids.tolist()
    spans = image_offsets(sample.images_seq_mask, sample.num_image_tokens)
    request = SimpleNamespace(uid=uid, input_ids=ids, cached_len=0, extend_len=len(ids),
                              mm_items=[SimpleNamespace(offsets=[span]) for span in spans])
    return SimpleNamespace(is_prefill=True, padded_reqs=[request], prompt_admissions=[(uid, len(ids), None)])


def collection_image_ids(row):
    mode = "cropped" if len(row["images"]) <= 2 else "uncropped"
    return [image["sha256"] + ":" + mode for image in row["images"]]


class NativeCollector(InputCollector):
    def __init__(self, *args, first_moe_layer, **kwargs):
        self.first_moe_layer = first_moe_layer
        super().__init__(*args, **kwargs)
        if not hasattr(self, "batch_sizes"):
            self.batch_sizes = {"prefill": {}, "decode": {}}

    def snapshot(self, layer, x, ids, batch):
        if layer == self.first_moe_layer and layer != 0:
            sizes = self.batch_sizes["prefill"]
            sizes["1"] = sizes.get("1", 0) + 1
        super().snapshot(layer, x, ids, batch)

    def add(self, layer, x, ids, modality, phase, sampled_counts=None):
        super().add(layer, x, ids, modality, phase, sampled_counts)
        # The generic collector counts tokens at layer zero; DeepSeek starts with a dense FFN.
        if layer == self.first_moe_layer and layer != 0:
            self.tokens[phase] += sum(value["candidates"] for value in sampled_counts.values())


def native_classes(config):
    sys.path.insert(0, config["native_repo"])
    from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
    from deepseek_vl2.utils.io import load_pil_images

    return DeepseekVLV2ForCausalLM, DeepseekVLV2Processor, load_pil_images


def status(config, stage, **details):
    value = dict(stage=stage, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **details)
    write_json(Path(config["output"]) / "status.json", value)
    print(json.dumps(value), flush=True)


def prepare(config):
    root, frozen = Path(config["output"]), Path(config["frozen_prepare"])
    target = root / "prepare"
    bindings = dict(frozen=str(frozen), files={name: file_hash(frozen / name)
                    for name in ("calibration.jsonl", "evaluation.jsonl", "split.json")},
                    model=str(Path(config["model"]).resolve()), seed=42, ratio=[.6, .4])
    if (target / "complete.json").exists():
        verify_stage(target, bindings)
        return
    calibration, evaluation = [read_rows(frozen / f"{name}.jsonl") for name in ("calibration", "evaluation")]
    if len(calibration) != 72 or len(evaluation) != 48:
        raise ValueError("expected the frozen 72/48 Write split")
    for key in ("id", "group_id"):
        if {row[key] for row in calibration} & {row[key] for row in evaluation}:
            raise ValueError(f"calibration/evaluation overlap: {key}")
    image_sets = [{image["sha256"] for row in rows for image in row["images"]} for rows in (calibration, evaluation)]
    if image_sets[0] & image_sets[1]:
        raise ValueError("calibration/evaluation images overlap")
    categories = [Counter(row["metadata"]["category"] for row in rows) for rows in (calibration, evaluation)]
    if len(categories[0]) != 6 or set(categories[0].values()) != {12} or set(categories[1].values()) != {8}:
        raise ValueError("expected 12/8 per category")
    target.mkdir(parents=True, exist_ok=True)
    sharing, prompts = {}, []
    for rows in (calibration, evaluation):
        for row in rows:
            for image in row["images"]:
                if not Path(image["path"]).is_file():
                    raise ValueError(f"missing image: {image['path']}")
                sharing.setdefault(image["sha256"], []).append(row["id"])
            prompts.append(dict(id=row["id"], messages=conversation(row),
                                prompt_source="official Write question and all images; frozen benchmark extension"))
    outputs = []
    for name, rows in (("calibration", calibration), ("evaluation", evaluation), ("messages", prompts)):
        path = target / f"{name}.jsonl"
        write_rows(path, rows)
        outputs.append(path)
    write_json(target / "image_sharing.json", sharing)
    write_json(target / "split.json", dict(source=read_json(frozen / "split.json"), calibration_categories=categories[0],
                                           evaluation_categories=categories[1], disjoint=True))
    outputs.extend((target / "image_sharing.json", target / "split.json"))
    finish_stage(target, bindings, outputs, sample_count=72, test_count=48, unique_images=len(sharing))


def preflight(config):
    root = Path(config["output"])
    verify_stage(root / "prepare")
    model_class, processor_class, load_pil_images = native_classes(config)
    language = model_class.config_class.from_pretrained(config["model"]).language_config
    processor = processor_class.from_pretrained(config["model"])
    records = []
    for row in read_rows(root / "prepare/calibration.jsonl"):
        messages = conversation(row)
        images = load_pil_images(messages)
        try:
            sample = processor(conversations=messages, images=images, force_batchify=False, system_prompt="")
        finally:
            for image in images:
                image.close()
        batch = collection_batch(sample, len(records))
        records.append(dict(id=row["id"], prompt_tokens=len(sample.input_ids), image_count=len(row["images"]),
                            image_offsets=[item.offsets[0] for item in batch.padded_reqs[0].mm_items],
                            collection_image_ids=collection_image_ids(row)))
    write_json(root / "preflight.json", dict(complete=True, sample_count=len(records), records=records,
               nominal_context=language.max_position_embeddings,
               min_prompt_tokens=min(row["prompt_tokens"] for row in records),
               max_prompt_tokens=max(row["prompt_tokens"] for row in records),
               samples_exceeding_nominal_context=sum(row["prompt_tokens"] > language.max_position_embeddings for row in records),
               image_preprocessing="native processor defaults; no truncation; crop mode included in dedup identity"))


def collect(config):
    root = Path(config["output"])
    prepared = verify_stage(root / "prepare")
    rows = read_rows(root / "prepare/calibration.jsonl")
    model_class, processor_class, load_pil_images = native_classes(config)
    model_config = model_class.config_class.from_pretrained(config["model"])
    language = model_config.language_config
    layers = [i for i in range(language.num_hidden_layers)
              if i >= language.first_k_dense_replace and i % language.moe_layer_freq == 0]
    algorithm = algorithm_config({**config["builder"], "sample_count": len(rows),
                                  "collection_image_ids": [collection_image_ids(row) for row in rows]})
    inputs = dict(checkpoint=config["checkpoint"], processor=dict(native_defaults=True), convert_bf16=False,
                  algorithm=algorithm, sample_count=len(rows), calibration_ids=[row["id"] for row in rows],
                  prepare=prepared["input_sha256"], sources=config["sources"], layer_ids=layers)
    target = root / "assets/collect"
    if (target / "complete.json").exists() and read_json(target / "complete.json").get("complete"):
        verify_stage(target, inputs, hash_outputs=False)
        return
    dtype = getattr(torch, config["dtype"])
    collector = NativeCollector(target, algorithm, language.hidden_size, language.n_routed_experts,
                                identity=digest(inputs), dtype=dtype, first_moe_layer=layers[0])
    previous = read_rows(target / "requests.jsonl") if (target / "requests.jsonl").exists() else []
    if (collector.requests != list(range(len(previous))) or len(collector.sample_state["records"]) != len(previous)
            or [row["id"] for row in previous] != inputs["calibration_ids"][:len(previous)]
            or collector.tokens["prefill"] != sum(row["prompt_tokens"] for row in previous)):
        collector.close()
        raise ValueError("interrupted collection is not at a completed request boundary; use a fresh output")
    processor = processor_class.from_pretrained(config["model"])
    if len(previous) < len(rows):
        model_config._attn_implementation = config["attention"]
        status(config, "loading_model", samples=len(previous), total=len(rows))
        model = model_class.from_pretrained(config["model"], config=model_config, torch_dtype=dtype,
                                            device_map={"": config["device"]}).eval()
        active, handles = {}, []
        for layer in layers:
            def hook(module, args, result, layer=layer):
                x = args[0].reshape(-1, language.hidden_size)
                ids = result[0].reshape(len(x), -1)
                collector.snapshot(layer, x, ids, active["batch"])
            handles.append(model.language.model.layers[layer].mlp.gate.register_forward_hook(hook))
        try:
            for ordinal in range(len(previous), len(rows)):
                row = rows[ordinal]
                started = time.perf_counter()
                messages = conversation(row)
                images = load_pil_images(messages)
                try:
                    sample = processor(conversations=messages, images=images, force_batchify=False, system_prompt="")
                finally:
                    for image in images:
                        image.close()
                active["batch"] = collection_batch(sample, ordinal)
                batch = processor.batchify([sample]).to(config["device"], dtype=dtype)
                with torch.inference_mode():
                    embeddings = model.prepare_inputs_embeds(**batch)
                    model.language.model(inputs_embeds=embeddings, attention_mask=batch.attention_mask,
                                         use_cache=False, return_dict=False)
                torch.cuda.synchronize()
                collector.flush(ordinal)
                previous.append(dict(id=row["id"], prompt_tokens=len(sample.input_ids), decode_tokens=0,
                                     nominal_context=language.max_position_embeddings,
                                     exceeds_nominal_context=len(sample.input_ids) > language.max_position_embeddings,
                                     sampling=collector.sample_state["records"][-1], elapsed_s=time.perf_counter()-started))
                write_rows(target / "requests.jsonl", previous)
                status(config, "collection", samples=len(previous), total=len(rows), last=previous[-1])
                del embeddings, batch, sample
        finally:
            for handle in handles:
                handle.remove()
            collector.close()
    else:
        collector.close()
    progress = read_json(target / "progress.json")
    if len(progress["requests"]) != len(rows) or progress["tokens"]["decode"]:
        raise ValueError("prompt-only request coverage mismatch")
    expected = progress["sample_state"]["counts"]
    for layer in layers:
        for mod in ("vision", "text"):
            if progress["sampling"][str(layer)][mod]["selected"] != expected[mod]:
                raise ValueError("incomplete layer sampling coverage")
    outputs = [target / "progress.json", target / "requests.jsonl"]
    for key, state in progress["buckets"].items():
        for suffix, length in (("bin", state["kept"] * language.hidden_size * 2), ("phase", state["kept"])):
            path = target / f"layer_{state['layer']}" / f"{key}.{suffix}"
            if not path.is_file() or path.stat().st_size != length:
                raise ValueError(f"missing or truncated collection: {path}")
            outputs.append(path)
    finish_stage(target, inputs, outputs, hash_outputs=False, buckets=progress["buckets"], tokens=progress["tokens"],
                 hidden=language.hidden_size, layer_ids=layers, num_layers=language.num_hidden_layers,
                 num_experts=language.n_routed_experts, dtype=config["dtype"], sampling=progress["sampling"])


def cluster(config):
    root, algorithm = Path(config["output"]) / "assets", algorithm_config(config["builder"])
    source = verify_stage(root / "collect", hash_outputs=False)
    inputs = dict(collect=source["input_sha256"], algorithm=algorithm, sources=config["sources"])
    target = root / "cluster"
    if (target / "complete.json").exists() and read_json(target / "complete.json").get("complete"):
        verify_stage(target, inputs, hash_outputs=False)
        return
    if (target / "stage_inputs.json").exists() and read_json(target / "stage_inputs.json") != inputs:
        raise ValueError("cluster source changed")
    write_json(target / "stage_inputs.json", inputs)
    outputs, missing = [target / "stage_inputs.json"], []
    dtype, device = getattr(torch, config["dtype"]), config["device"]
    for layer in source["layer_ids"]:
        folder = target / f"layer_{layer}"
        folder.mkdir(parents=True, exist_ok=True)
        for mod in ("vision", "text"):
            buckets = [(key, state) for key, state in sorted(source["buckets"].items())
                       if state["layer"] == layer and state["modality"] == mod]
            samples = []
            for key, state in buckets:
                x = load_inputs(root / "collect", key, state, source["hidden"], dtype=dtype)
                generator = torch.Generator().manual_seed(algorithm["seed"] ^ int(digest(key)[:8], 16))
                samples.append(x[torch.randperm(len(x), generator=generator)[:algorithm["pca_per_expert"]]])
            if not samples:
                raise ValueError(f"PCA coverage gap at {layer}/{mod}")
            pca = train_pca(torch.cat(samples).to(device), algorithm["pca_dim"])
            path = folder / f"L{layer}_{mod}_pca.pt"
            torch.save(pca, path)
            outputs.append(path)
            pending = []
            for expert in range(source["num_experts"]):
                key = f"L{layer}_E{expert}_{mod}"
                if key not in source["buckets"]:
                    missing.append(key)
                else:
                    high = load_inputs(root / "collect", key, source["buckets"][key], source["hidden"], dtype=dtype).to(device)
                    low = (high.float() - pca["mean"].to(device).float()) @ pca["proj"].to(device).float()
                    pending.append((low, high, key))
                if pending and (len(pending) == algorithm.get("cluster_batch_size", 8) or expert == source["num_experts"] - 1):
                    for name, data in batched_kmeans(pending, algorithm).items():
                        path = folder / f"{name}_clusters.pt"
                        torch.save(data, path)
                        outputs.append(path)
                    pending.clear()
        status(config, "pca_and_cluster", layer=layer, files=len(outputs), missing=len(missing))
    finish_stage(target, inputs, outputs, hash_outputs=False, missing_tables=missing,
                 source_inputs=source["inputs"], layer_ids=source["layer_ids"], num_experts=source["num_experts"],
                 hidden=source["hidden"], dtype=config["dtype"])


def checkpoint_experts(model, layers, num_experts):
    from safetensors import safe_open

    model = Path(model)
    index = read_json(model / "model.safetensors.index.json")["weight_map"]
    with ExitStack() as stack:
        handles = {}
        for layer in layers:
            for expert in range(num_experts):
                weights = {}
                for role in ("gate_proj", "up_proj", "down_proj"):
                    key = f"language.model.layers.{layer}.mlp.experts.{expert}.{role}.weight"
                    shard = index[key]
                    if shard not in handles:
                        handles[shard] = stack.enter_context(safe_open(model / shard, framework="pt", device="cpu"))
                    weights[role] = handles[shard].get_tensor(key)
                yield layer, expert, weights


def materialize(config):
    root = Path(config["output"]) / "assets"
    source = verify_stage(root / "cluster", hash_outputs=False)
    target = root / "cache"
    inputs = dict(cluster=source["input_sha256"], checkpoint=config["checkpoint"], sources=config["sources"],
                  algorithm=config["builder"], activation="silu", convert_bf16=False)
    if (target / "complete.json").exists() and read_json(target / "complete.json").get("complete"):
        verify_stage(target, inputs, hash_outputs=False)
        return
    if (target / "stage_inputs.json").exists() and read_json(target / "stage_inputs.json") != inputs:
        raise ValueError("materialize source changed")
    write_json(target / "stage_inputs.json", inputs)
    outputs, examples = [target / "stage_inputs.json"], []
    dtype, device = getattr(torch, config["dtype"]), config["device"]
    for layer, expert, weights in checkpoint_experts(config["model"], source["layer_ids"], source["num_experts"]):
        if any(value.dtype != dtype for value in weights.values()):
            raise ValueError("checkpoint native precision differs from the requested table dtype")
        gate_up = torch.cat((weights["gate_proj"], weights["up_proj"])).to(device)
        down = weights["down_proj"].to(device)
        for mod in ("vision", "text"):
            key = f"L{layer}_E{expert}_{mod}"
            path = root / "cluster" / f"layer_{layer}/{key}_clusters.pt"
            if str(path.relative_to(root / "cluster")) not in source["outputs"]:
                continue
            data = torch.load(path, weights_only=True)
            value = expert_values(data["centroid"].to(device), gate_up, down, config["builder"]["value_chunk"])
            if data["key"].dtype != dtype or value.dtype != dtype or value.shape != (len(data["key"]), source["hidden"]):
                raise ValueError("materialized table shape/dtype mismatch")
            out = target / f"layer_{layer}/{key}_cache.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(id=torch.arange(len(value)), key=data["key"], value=value), out)
            outputs.append(out)
            if len(examples) < 3:
                examples.append(dict(path=str(out), key_shape=list(data["key"].shape), value_shape=list(value.shape)))
        if expert == source["num_experts"] - 1:
            status(config, "materialize", layer=layer, tables=len(outputs)-1)
    pca = {name: None for name in source["outputs"] if name.endswith("_pca.pt")}
    finish_stage(target, inputs, outputs, hash_outputs=False, pca_root=str((root / "cluster").resolve()),
                 pca_outputs=pca, missing_tables=source["missing_tables"], dtype=config["dtype"],
                 layer_ids=source["layer_ids"], examples=examples, provenance=source["source_inputs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "preflight", "collect", "cluster", "materialize", "all"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if config["dtype"] != "bfloat16" or config["builder"]["sampling_policy"] != "tablemoe_default" or config["builder"]["sample_decode_tokens"]:
        raise ValueError("this runner requires native BF16, prompt-only TableMoE sampling")
    torch.set_num_threads(config.get("cpu_threads", 8))
    torch.manual_seed(config["builder"]["seed"])
    timings = {}
    for name, function in (("prepare", prepare), ("preflight", preflight), ("collect", collect),
                           ("cluster", cluster), ("materialize", materialize)):
        if args.stage in (name, "all"):
            started = time.perf_counter()
            status(config, name)
            function(config)
            timings[name] = time.perf_counter() - started
            write_json(Path(config["output"]) / f"{name}_timing.json", dict(elapsed_s=timings[name]))
    if args.stage in ("all", "materialize"):
        root = Path(config["output"])
        cache = verify_stage(root / "assets/cache", hash_outputs=False)
        collection = verify_stage(root / "assets/collect", hash_outputs=False)
        requests = read_rows(root / "assets/collect/requests.jsonl")
        result = dict(complete=True, calibration_samples=len(requests), dtype=config["dtype"],
                      tables=sum(name.endswith("_cache.pt") for name in cache["outputs"]), pca_files=len(cache["pca_outputs"]),
                      missing_tables=len(cache["missing_tables"]), tokens=collection["tokens"],
                      samples_exceeding_nominal_context=sum(row["exceeds_nominal_context"] for row in requests),
                      cache_dir=str(root / "assets/cache"), pca_dir=str(root / "assets/cluster"),
                      runtime_validated=False, accuracy_evaluated=False, formal_eligible=False)
        write_json(root / "complete.json", result)
        status(config, "complete", **result)


if __name__ == "__main__":
    main()
