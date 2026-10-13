"""Collect routed inputs, train layer PCA, and materialize FP16 expert tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import queue
import threading

import numpy as np
import torch
import torch.nn.functional as F

from freetoken.moe.tablemoe_artifacts import digest, file_hash, finish_stage, read_json, verify_stage, write_json


DEFAULTS = dict(seed=42, reservoir_size=30000, pca_dim=64, pca_per_expert=1000,
                clusters=256, iterations=30, distance_chunk=2048, value_chunk=64, device="cuda",
                export_inputs=True, vision_budget=0, vision_population=0)

SAMPLE_DEFAULTS = dict(sampling_policy="all_tokens", sample_count=0,
                       sample_decode_tokens=0,
                       min_vision_per_sample=1, max_vision_per_sample=64,
                       min_text_per_sample=8, max_text_per_sample=128,
                       max_vision_tokens=100000, max_text_tokens=40000)


def algorithm_config(config):
    result = {**DEFAULTS, **SAMPLE_DEFAULTS, **config}
    for key in ("reservoir_size", "pca_dim", "pca_per_expert", "clusters", "iterations",
                "distance_chunk", "value_chunk"):
        if result[key] < 1:
            raise ValueError(f"{key} must be positive")
    if result.get("cluster_batch_size", 1) < 1:
        raise ValueError("cluster_batch_size must be positive")
    if not 0 <= result["vision_budget"] <= result["vision_population"]:
        raise ValueError("vision sampling budget exceeds its verified population")
    if result["sampling_policy"] not in ("all_tokens", "tablemoe_default"):
        raise ValueError("unknown collection sampling policy")
    if type(result["sample_decode_tokens"]) is not int or result["sample_decode_tokens"] < 0:
        raise ValueError("sample_decode_tokens must be a nonnegative integer")
    if result["sampling_policy"] == "tablemoe_default":
        if result["sample_count"] < 1 or result["vision_budget"]:
            raise ValueError("per-sample sampling requires sample_count and no global exact budget")
        for mod in ("vision", "text"):
            if not 0 <= result[f"min_{mod}_per_sample"] <= result[f"max_{mod}_per_sample"]:
                raise ValueError("invalid per-sample sampling bounds")
            if result[f"max_{mod}_tokens"] < 0:
                raise ValueError("negative collection budget")
    return result


def adaptive_quota(avail, remaining_budget, remaining_samples, history_lengths, min_per_sample, max_per_sample):
    if avail <= 0 or remaining_budget <= 0:
        return 0
    median_len = max(float(np.median(history_lengths)) if history_lengths else float(avail), 1.0)
    weight = np.sqrt(max(float(avail), 1.0) / median_len)
    quota = int(round(remaining_budget / max(1, remaining_samples) * weight))
    return max(0, min(avail, remaining_budget, max_per_sample, max(min_per_sample, quota)))


def _draw(seed, count):
    # A counter keyed PRNG makes resumed reservoirs independent of worker scheduling.
    mask = (1 << 64) - 1
    z = (seed + count * 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) % count


def _file_hasher(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            result.update(block)
    return result


class InputCollector:
    """Bounded writer queue with disk reservoirs; never retains mutable engine views."""

    def __init__(self, root, config, hidden, num_experts, identity=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = algorithm_config(config)
        self.hidden, self.num_experts = hidden, num_experts
        self.buckets, self.tokens, self.requests = {}, {"prefill": 0, "decode": 0}, []
        self.sampling = {}
        self.sample_state = dict(records=[], counts=dict(vision=0, text=0), history=dict(vision=[], text=[]))
        self.sample_rng = np.random.default_rng(self.config["seed"])
        self.active_samples = {}
        self.sample_batch = None
        self.sample_batch_key = None
        self.hashers = {}
        self.identity = identity
        if (self.root / "progress.json").exists():
            saved = read_json(self.root / "progress.json")
            if saved["identity"] != identity or saved["config"] != self.config or saved["hidden"] != hidden:
                raise ValueError("collection resume configuration mismatch")
            for key, state in saved["buckets"].items():
                folder = self.root / f"layer_{state['layer']}"
                hashes = []
                for suffix in ("bin", "phase"):
                    h = _file_hasher(folder / f"{key}.{suffix}")
                    if h.hexdigest() != state[f"{suffix}_sha256"]:
                        raise ValueError("interrupted or changed collection reservoir")
                    hashes.append(h)
                self.hashers[key] = hashes
            self.buckets, self.tokens, self.requests = saved["buckets"], saved["tokens"], saved["requests"]
            self.sampling = saved.get("sampling", {})
            self.sample_state = saved.get("sample_state", self.sample_state)
            if "rng_state" in self.sample_state:
                self.sample_rng.bit_generator.state = self.sample_state["rng_state"]
        self.error = None
        self.dirty = set()
        self.handles = {}
        self.pending = queue.Queue(maxsize=4)
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def snapshot(self, layer, x, ids, batch):
        phase = "prefill" if batch.is_prefill else "decode"
        sampled_counts = None
        if (self.config["sampling_policy"] == "tablemoe_default" and not batch.is_prefill
                and not self.config["sample_decode_tokens"]):
            return
        n = sum(r.extend_len if batch.is_prefill else 1 for r in batch.padded_reqs if r.uid != -1)
        if self.config["sampling_policy"] == "tablemoe_default":
            key = (batch.is_prefill, tuple((r.uid, r.cached_len) for r in batch.padded_reqs if r.uid != -1))
            if self.sample_batch is not batch or self.sample_batch_key != key:
                sampler = self._sample_prefill if batch.is_prefill else self._sample_decode
                self.sample_batch_rows = sampler(batch, n)
                self.sample_batch, self.sample_batch_key = batch, key
            indices, modality, sampled_counts = self.sample_batch_rows
            device_indices = indices.to(x.device)
            source_inputs, source_routes = x.index_select(0, device_indices), ids.index_select(0, device_indices)
            n = len(indices)
        else:
            source_inputs, source_routes = x[:n], ids[:n]
            modality = torch.ones(n, dtype=torch.int8)
            if batch.is_prefill and batch.mm_rows is not None:
                rows = batch.mm_rows.detach().cpu()
                modality[rows[rows < n].long()] = 0
        # D2H owns its storage before expert kernels can mutate x or rewrite ids.
        ready = None
        if x.is_cuda:
            inputs = torch.empty((n, self.hidden), dtype=torch.float16, pin_memory=True)
            routes = torch.empty(source_routes.shape, dtype=ids.dtype, pin_memory=True)
            inputs.copy_(source_inputs.detach(), non_blocking=True)
            routes.copy_(source_routes.detach(), non_blocking=True)
        else:
            inputs = source_inputs.detach().to("cpu", dtype=torch.float16, copy=True)
            routes = source_routes.detach().to("cpu", copy=True)
        if x.is_cuda:
            ready = torch.cuda.Event()
            ready.record()
        self._raise()
        self.pending.put((layer, inputs, routes, modality, phase, sampled_counts, ready))

    def _sample_prefill(self, batch, n):
        admissions = {uid: length for uid, length, _ in getattr(batch, "prompt_admissions", [])}
        selected, modalities = [], []
        candidates = dict(vision=0, text=0)
        offset = 0
        for req in batch.padded_reqs:
            if req.uid == -1:
                continue
            if req.uid not in self.active_samples:
                ordinal = len(self.sample_state["records"])
                if ordinal >= self.config["sample_count"] or req.cached_len:
                    raise ValueError("per-sample collection requires an uncached first prompt chunk")
                length = admissions.get(req.uid, len(req.input_ids))
                vision = np.zeros(length, dtype=bool)
                for item in getattr(req, "mm_items", None) or []:
                    for lo, hi in item.offsets:
                        vision[lo:hi] = True
                record = dict(uid=req.uid, ordinal=ordinal, prompt_tokens=length,
                              candidates={}, selected={}, positions={}, decode_seen=0)
                for mod, mask in (("vision", vision), ("text", ~vision)):
                    positions = np.flatnonzero(mask)
                    if mod == "text" and self.config["sample_decode_tokens"]:
                        # Fixed-length generation makes the future decode-input positions known.
                        positions = np.concatenate((positions, np.arange(length, length + self.config["sample_decode_tokens"])))
                    count = adaptive_quota(len(positions),
                        self.config[f"max_{mod}_tokens"] - self.sample_state["counts"][mod],
                        self.config["sample_count"] - ordinal, self.sample_state["history"][mod],
                        self.config[f"min_{mod}_per_sample"], self.config[f"max_{mod}_per_sample"])
                    chosen = positions if count == len(positions) else self.sample_rng.choice(positions, count, replace=False)
                    record["positions"][mod] = sorted(chosen.tolist())
                    record["candidates"][mod], record["selected"][mod] = len(positions), count
                    self.sample_state["counts"][mod] += count
                    self.sample_state["history"][mod].append(len(positions))
                self.sample_state["records"].append(record)
                self.active_samples[req.uid] = record
            record = self.active_samples[req.uid]
            lo, hi = req.cached_len, req.cached_len + req.extend_len
            for modality, mod in enumerate(("vision", "text")):
                positions = np.asarray(record["positions"][mod], dtype=np.int64)
                chosen = positions[(positions >= lo) & (positions < hi)]
                selected.extend((chosen - lo + offset).tolist())
                modalities.extend([modality] * len(chosen))
            vision_count = sum(max(0, min(hi, b) - max(lo, a))
                               for item in getattr(req, "mm_items", None) or [] for a, b in item.offsets)
            candidates["vision"] += vision_count
            candidates["text"] += req.extend_len - vision_count
            offset += req.extend_len
        if offset != n:
            raise ValueError("per-sample collection batch row mismatch")
        indices = torch.tensor(selected, dtype=torch.long)
        modality = torch.tensor(modalities, dtype=torch.int8)
        counts = {mod:dict(candidates=candidates[mod], selected=int((modality == i).sum()))
                  for i, mod in enumerate(("vision", "text"))}
        self.sample_state["rng_state"] = self.sample_rng.bit_generator.state
        return indices, modality, counts

    def _sample_decode(self, batch, n):
        selected = []
        requests = [r for r in batch.padded_reqs if r.uid != -1]
        for index, req in enumerate(requests):
            record = self.active_samples[req.uid]
            if (req.cached_len != record["prompt_tokens"] + record["decode_seen"] or
                    record["decode_seen"] >= self.config["sample_decode_tokens"]):
                raise ValueError("decode collection position differs from the fixed-length plan")
            if req.cached_len in record["positions"]["text"]:
                selected.append(index)
            record["decode_seen"] += 1
        counts = dict(vision=dict(candidates=0, selected=0), text=dict(candidates=n, selected=len(selected)))
        return torch.tensor(selected, dtype=torch.long), torch.ones(len(selected), dtype=torch.int8), counts

    def _raise(self):
        if self.error is not None:
            raise RuntimeError("TableMoE collection writer failed") from self.error

    def _worker(self):
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                if self.error is None:
                    *values, ready = item
                    if ready is not None:
                        ready.synchronize()
                    self.add(*values)
            except BaseException as exc:
                self.error = exc
            finally:
                self.pending.task_done()

    def add(self, layer, x, ids, modality, phase, sampled_counts=None):
        inputs, routes = x.numpy(), ids.numpy()
        if not np.isfinite(inputs).all() or (routes < 0).any() or (routes >= self.num_experts).any():
            raise ValueError("invalid raw collection input/routes")
        if layer == 0:
            self.tokens[phase] += (sum(s["candidates"] for s in sampled_counts.values())
                                   if sampled_counts is not None else len(x))
        counters = self.sampling.setdefault(str(layer), {
            mod: dict(candidates=0, selected=0, prefill=0, decode=0) for mod in ("vision", "text")})
        selected = torch.ones(len(x), dtype=torch.bool)
        for mod, name in enumerate(("vision", "text")):
            if sampled_counts is not None:
                count = sampled_counts[name]["candidates"]
                take = sampled_counts[name]["selected"]
                counters[name]["candidates"] += count
                counters[name]["selected"] += take
                counters[name][phase] += take
                continue
            positions = torch.where(modality == mod)[0]
            state = counters[name]
            count = len(positions)
            take = count
            if name == "vision" and self.config["vision_budget"]:
                remaining = self.config["vision_population"] - state["candidates"]
                needed = self.config["vision_budget"] - state["selected"]
                if count > remaining:
                    raise ValueError("vision token population changed since verified calibration")
                # Hypergeometric draws sample uniformly across the entire known population.
                seed = self.config["seed"] + state["candidates"]
                take = int(np.random.default_rng(seed).hypergeometric(needed, remaining - needed, count)) if count else 0
                selected[positions] = False
                generator = torch.Generator().manual_seed(seed)
                selected[positions[torch.randperm(count, generator=generator)[:take]]] = True
            state["candidates"] += count
            state["selected"] += take
            state[phase] += take
        # Group routes once; tiny decode batches otherwise launch hundreds of CPU tensor operations.
        row_indices = np.repeat(np.arange(len(inputs)), routes.shape[-1])
        unique = np.unique(routes.reshape(-1) * len(inputs) + row_indices)
        if not len(inputs):
            return
        experts, row_indices = unique // len(inputs), unique % len(inputs)
        valid = selected.numpy()[row_indices]
        experts, row_indices = experts[valid], row_indices[valid]
        groups = experts * 2 + modality.numpy()[row_indices]
        order = np.argsort(groups, kind="stable")
        groups, row_indices = groups[order], row_indices[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(groups)) + 1, len(groups)]
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            if begin == end:
                continue
            group = int(groups[begin])
            expert, name = group // 2, ("vision", "text")[group % 2]
            rows = np.ascontiguousarray(inputs[row_indices[begin:end]])
            key = f"L{layer}_E{expert}_{name}"
            state = self.buckets.setdefault(key, dict(layer=layer, expert=expert, modality=name,
                                                      seen=0, prefill=0, decode=0, kept=0))
            if key not in self.hashers:
                self.hashers[key] = [hashlib.sha256(), hashlib.sha256()]
            folder = self.root / f"layer_{layer}"
            folder.mkdir(exist_ok=True)
            data = folder / f"{key}.bin"
            labels = folder / f"{key}.phase"
            seed = self.config["seed"] ^ int(digest(key)[:16], 16)
            if key not in self.handles:
                self.handles[key] = (data.open("r+b" if data.exists() else "w+b", buffering=0),
                                     labels.open("r+b" if labels.exists() else "w+b", buffering=0))
            vectors, phases = self.handles[key]
            raw = memoryview(rows).cast("B")
            initial = min(len(rows), max(0, self.config["reservoir_size"] - state["seen"]))
            if initial:
                self.dirty.add(key)
                vectors.seek(state["seen"] * self.hidden * 2)
                vectors.write(raw[:initial * self.hidden * 2])
                phases.seek(state["seen"])
                phases.write(bytes([phase == "decode"]) * initial)
                if self.hashers[key] is not None:
                    self.hashers[key][0].update(raw[:initial * self.hidden * 2])
                    self.hashers[key][1].update(bytes([phase == "decode"]) * initial)
                state["seen"] += initial
                state[phase] += initial
            for index in range(initial, len(rows)):
                state["seen"] += 1
                state[phase] += 1
                count, capacity = state["seen"], self.config["reservoir_size"]
                slot = count - 1 if count <= capacity else _draw(seed, count)
                if slot < capacity:
                    self.dirty.add(key)
                    self.hashers[key] = None
                    vectors.seek(slot * self.hidden * 2)
                    vectors.write(raw[index * self.hidden * 2:(index + 1) * self.hidden * 2])
                    phases.seek(slot)
                    phases.write(bytes([phase == "decode"]))
            state["kept"] = min(state["seen"], self.config["reservoir_size"])

    def flush(self, uid=None):
        self.pending.join()
        self._raise()
        if uid is not None:
            self.requests.append(uid)
            self.active_samples.pop(uid, None)
        for key, state in self.buckets.items():
            if key not in self.dirty and "bin_sha256" in state:
                continue
            folder = self.root / f"layer_{state['layer']}"
            for index, suffix in enumerate(("bin", "phase")):
                hashes = self.hashers.get(key)
                state[f"{suffix}_sha256"] = (hashes[index].hexdigest() if hashes is not None else
                                              file_hash(folder / f"{key}.{suffix}"))
        self.dirty.clear()
        write_json(self.root / "progress.json", dict(buckets=self.buckets, tokens=self.tokens,
                                                      requests=self.requests, hidden=self.hidden,
                                                      identity=self.identity, config=self.config, sampling=self.sampling,
                                                      sample_state=self.sample_state))

    def close(self):
        self.flush()
        self.pending.put(None)
        self.worker.join()
        for handles in self.handles.values():
            for handle in handles:
                handle.close()
        self.handles.clear()


def finalize_collection(root, inputs, num_layers, num_experts):
    root = Path(root)
    progress = read_json(root / "progress.json")
    expected = inputs["sample_count"]
    if len(progress["requests"]) != expected:
        raise ValueError("collection request coverage mismatch")
    if progress["config"]["sampling_policy"] == "tablemoe_default":
        state = progress["sample_state"]
        decode = progress["config"].get("sample_decode_tokens", 0)
        if (len(state["records"]) != expected or progress["tokens"]["decode"] != expected * decode or
                any(r.get("decode_seen", 0) != decode for r in state["records"])):
            raise ValueError("per-sample fixed-length collection coverage mismatch")
        for layer in range(num_layers):
            for mod in ("vision", "text"):
                if (progress["sampling"][str(layer)][mod]["selected"] != state["counts"][mod] or
                        state["counts"][mod] > progress["config"][f"max_{mod}_tokens"]):
                    raise ValueError("per-sample selections are incomplete or exceed the budget")
    if progress["config"].get("vision_budget"):
        for layer in range(num_layers):
            state = progress["sampling"][str(layer)]["vision"]
            if (state["candidates"] != progress["config"]["vision_population"] or
                    state["selected"] != progress["config"]["vision_budget"]):
                raise ValueError("vision sampling population or budget incomplete")
    outputs = [root / "progress.json"]
    if (root / "requests.jsonl").exists():
        outputs.append(root / "requests.jsonl")
    for key, state in progress["buckets"].items():
        folder = root / f"layer_{state['layer']}"
        for suffix in ("bin", "phase"):
            if file_hash(folder / f"{key}.{suffix}") != state[f"{suffix}_sha256"]:
                raise ValueError("interrupted collection reservoir hash mismatch")
        if not progress["config"].get("export_inputs", True):
            outputs.extend(folder / f"{key}.{suffix}" for suffix in ("bin", "phase"))
            continue
        x = torch.from_file(str(folder / f"{key}.bin"), size=state["kept"] * progress["hidden"],
                            dtype=torch.float16).reshape(-1, progress["hidden"]).clone()
        phases = torch.from_file(str(folder / f"{key}.phase"), size=state["kept"], dtype=torch.uint8).clone()
        path = folder / f"{key}_inputs.pt"
        temp = path.with_suffix(".partial")
        torch.save(dict(inputs=x, phase=phases), temp)
        temp.replace(path)
        outputs.append(path)
    return finish_stage(root, inputs, outputs, buckets=progress["buckets"], tokens=progress["tokens"],
                        hidden=progress["hidden"], num_layers=num_layers, num_experts=num_experts,
                        sampling=progress.get("sampling", {}))


def load_inputs(root, key, state, hidden):
    path = Path(root) / f"layer_{state['layer']}"
    exported = path / f"{key}_inputs.pt"
    if exported.exists():
        return torch.load(exported, weights_only=True)["inputs"]
    return torch.from_file(str(path / f"{key}.bin"), size=state["kept"] * hidden,
                           dtype=torch.float16).reshape(-1, hidden)


def train_pca(x, dim):
    x = x.float()
    if len(x) <= dim or x.shape[1] < dim or not torch.isfinite(x).all():
        raise ValueError(f"PCA coverage gap: {tuple(x.shape)} cannot train {dim} dimensions")
    mean = x.mean(0)
    covariance = (x - mean).T @ (x - mean) / (len(x) - 1)
    values, vectors = torch.linalg.eigh(covariance)
    if (values[-dim] <= values[-1].clamp_min(1e-20) * 1e-7):
        raise ValueError(f"PCA rank coverage gap for {dim} dimensions")
    proj = vectors[:, -dim:].flip(1)
    largest = proj.abs().argmax(0)
    proj *= proj[largest, torch.arange(dim)].sign()
    return dict(mean=mean.cpu().half(), proj=proj.cpu().half())


def _assign(x, centers, chunk):
    labels, distances = [], []
    for part in x.split(chunk):
        distance = (part.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None]
                    - 2 * part @ centers.T).clamp_min_(0)
        d, idx = distance.min(-1)
        labels.append(idx)
        distances.append(d)
    return torch.cat(labels), torch.cat(distances)


def _cluster_means(values, labels, counts):
    # Stable segments avoid CUDA atomics changing the centroids between rebuilds.
    order = torch.argsort(labels, stable=True)
    return torch.segment_reduce(values[order], "mean", lengths=counts[counts > 0])


def kmeans(x, high, config):
    x, high = x.float(), high.float()
    n, k = len(x), min(config["clusters"], len(x))
    if not n:
        raise ValueError("cannot cluster an empty expert")
    generator = torch.Generator(device=x.device).manual_seed(config["seed"])
    indices = [int(torch.randint(n, (1,), generator=generator, device=x.device))]
    nearest = torch.full((n,), float("inf"), device=x.device)
    for _ in range(1, k):
        nearest = torch.minimum(nearest, (x - x[indices[-1]]).square().sum(-1))
        if nearest.sum() == 0:
            break
        indices.append(int(torch.multinomial(nearest, 1, generator=generator)))
    centers = x[indices].clone()
    previous = None
    for _ in range(config["iterations"]):
        labels, _ = _assign(x, centers, config["distance_chunk"])
        counts = torch.bincount(labels, minlength=len(centers))
        centers = _cluster_means(x, labels, counts)
        if previous is not None and torch.equal(labels, previous) and (counts > 0).all():
            break
        previous = labels
    labels, _ = _assign(x, centers, config["distance_chunk"])
    counts = torch.bincount(labels, minlength=len(centers))
    active = counts > 0
    return dict(key=_cluster_means(x, labels, counts).cpu().half(),
                centroid=_cluster_means(high, labels, counts).cpu().half(), counts=counts[active].cpu())


def batched_kmeans(samples, config):
    device = samples[0][1].device
    batch, maximum = len(samples), max(len(high) for _, high, _ in samples)
    k = min(config["clusters"], maximum)
    low = torch.zeros(batch, maximum, samples[0][0].shape[1], device=device)
    high = torch.zeros(batch, maximum, samples[0][1].shape[1], device=device)
    sizes = torch.tensor([len(x) for x, _, _ in samples], device=device)
    valid = torch.arange(maximum, device=device)[None] < sizes[:, None]
    first, uniforms = [], []
    for i, (x, h, key) in enumerate(samples):
        low[i, :len(x)] = x.float()
        high[i, :len(h)] = h.float()
        generator = torch.Generator().manual_seed(config["seed"] ^ int(digest(key)[:8], 16))
        first.append(int(torch.randint(len(x), (1,), generator=generator)))
        uniforms.append(torch.rand(k, generator=generator))
    uniforms = torch.stack(uniforms).to(device)
    centers = torch.zeros(batch, k, low.shape[-1], device=device)
    active = torch.zeros(batch, k, dtype=torch.bool, device=device)
    centers[:, 0] = low[torch.arange(batch, device=device), torch.tensor(first, device=device)]
    active[:, 0] = True
    nearest = torch.full((batch, maximum), float("inf"), device=device)
    for j in range(1, k):
        nearest = torch.minimum(nearest, (low - centers[:, j - 1, None]).square().sum(-1))
        nearest.masked_fill_(~valid, 0)
        mass = nearest.sum(-1)
        indices = (nearest.cumsum(-1) < (uniforms[:, j] * mass)[:, None]).sum(-1).clamp_max(maximum - 1)
        centers[:, j] = low[torch.arange(batch, device=device), indices]
        active[:, j] = (mass > 0) & (j < sizes) & active[:, j - 1]
    low_flat, high_flat = low[valid], high[valid]
    offsets = (torch.arange(batch, device=device) * k)[:, None]
    square = low.square().sum(-1, keepdim=True)
    previous = None
    for _ in range(config["iterations"] + 1):
        distance = square + centers.square().sum(-1)[:, None] - 2 * torch.bmm(low, centers.transpose(1, 2))
        labels = distance.masked_fill(~active[:, None], float("inf")).argmin(-1)
        ids = (labels + offsets)[valid]
        counts = torch.bincount(ids, minlength=batch * k)
        means = _cluster_means(low_flat, ids, counts)
        updated = torch.zeros_like(centers).reshape(batch * k, -1)
        updated[counts > 0] = means
        centers = updated.reshape_as(centers)
        active = counts.reshape(batch, k) > 0
        if previous is not None and torch.equal(labels, previous):
            break
        previous = labels
    centroids = torch.zeros(batch * k, high.shape[-1], device=device)
    centroids[counts > 0] = _cluster_means(high_flat, ids, counts)
    centroids = centroids.reshape(batch, k, -1)
    return {key: dict(key=centers[i, active[i]].cpu().half(), centroid=centroids[i, active[i]].cpu().half(),
                      counts=counts.reshape(batch, k)[i, active[i]].cpu())
            for i, (_, _, key) in enumerate(samples)}


def cluster(root, config):
    root, config = Path(root), algorithm_config(config)
    source = verify_stage(root / "collect")
    target = root / "cluster"
    inputs = dict(collect=source["input_sha256"], collect_outputs=source["outputs"], algorithm=config,
                  builder_sha256=file_hash(__file__), torch_version=torch.__version__, cuda_version=torch.version.cuda)
    if (target / "complete.json").exists() and read_json(target / "complete.json").get("complete"):
        return verify_stage(target, inputs)
    stage_input = target / "stage_inputs.json"
    if not stage_input.exists() and any(target.glob("layer_*/*.pt")):
        raise ValueError("incomplete cluster has no verified resume input manifest")
    if stage_input.exists() and read_json(stage_input) != inputs:
        raise ValueError("incomplete cluster resume input mismatch")
    write_json(stage_input, inputs)
    write_json(target / "complete.json", dict(complete=False))
    outputs, missing = [stage_input], []
    device = config["device"]
    for layer in range(source["num_layers"]):
        folder = target / f"layer_{layer}"
        folder.mkdir(parents=True, exist_ok=True)
        for mod in ("vision", "text"):
            buckets = [(key, v) for key, v in source["buckets"].items()
                       if v["layer"] == layer and v["modality"] == mod]
            samples = []
            for key, state in sorted(buckets):
                x = load_inputs(root / "collect", key, state, source["hidden"])
                generator = torch.Generator().manual_seed(config["seed"] ^ int(digest(key)[:8], 16))
                samples.append(x[torch.randperm(len(x), generator=generator)[:config["pca_per_expert"]]])
            if not samples:
                raise ValueError(f"PCA coverage gap: layer {layer} modality {mod} has no inputs")
            pca = train_pca(torch.cat(samples).to(device), config["pca_dim"])
            path = folder / f"L{layer}_{mod}_pca.pt"
            torch.save(pca, path)
            outputs.append(path)
            pending = []
            for expert in range(source["num_experts"]):
                key = f"L{layer}_E{expert}_{mod}"
                if key not in source["buckets"]:
                    missing.append(key)
                    continue
                high = load_inputs(root / "collect", key, source["buckets"][key], source["hidden"]).to(device)
                # Fit the clustering against the FP16 assets that inference will load.
                low = (high.float() - pca["mean"].to(device).float()) @ pca["proj"].to(device).float()
                pending.append((low, high, key))
                if len(pending) == config.get("cluster_batch_size", 1) or expert == source["num_experts"] - 1:
                    results = (batched_kmeans(pending, config) if config.get("cluster_batch_size", 1) > 1 else
                               {name: kmeans(x, h, {**config, "seed": config["seed"] ^ int(digest(name)[:8], 16)})
                                for x, h, name in pending})
                    for name, data in results.items():
                        path = folder / f"{name}_clusters.pt"
                        torch.save(data, path)
                        outputs.append(path)
                    pending.clear()
            if pending:
                for name, data in batched_kmeans(pending, config).items():
                    path = folder / f"{name}_clusters.pt"
                    torch.save(data, path)
                    outputs.append(path)
    return finish_stage(target, inputs, outputs, missing_tables=missing, source_inputs=source["inputs"],
                        num_layers=source["num_layers"], num_experts=source["num_experts"])


def expert_values(x, gate_up, down, chunk=64):
    result = []
    for part in x.split(chunk):
        gate, up = F.linear(part, gate_up).chunk(2, -1)
        result.append(F.linear(F.silu(gate) * up, down).cpu())
    return torch.cat(result).half()


def materialize(root, config, pieces=None, model_config=None):
    root, config = Path(root), algorithm_config(config)
    source = verify_stage(root / "cluster")
    provenance = source["source_inputs"]
    from freetoken.moe.tablemoe_artifacts import checkpoint_identity

    if pieces is None:
        identity = checkpoint_identity(provenance["checkpoint"]["path"], expected=provenance["checkpoint"])
        if identity != provenance["checkpoint"]:
            raise ValueError("checkpoint changed since collection")
        pieces = iter_checkpoint_experts(identity["path"])
    target = root / "cache"
    inputs = dict(cluster=source["input_sha256"], cluster_outputs=source["outputs"],
                  checkpoint=provenance["checkpoint"], convert_bf16=provenance["convert_bf16"],
                  processor=provenance["processor"], algorithm=config, builder_sha256=file_hash(__file__),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda)
    if (target / "complete.json").exists() and read_json(target / "complete.json").get("complete"):
        return verify_stage(target, inputs)
    stage_input = target / "stage_inputs.json"
    if not stage_input.exists() and any(target.glob("layer_*/*.pt")):
        raise ValueError("incomplete materialize has no verified resume input manifest")
    if stage_input.exists() and read_json(stage_input) != inputs:
        raise ValueError("incomplete materialize resume input mismatch")
    write_json(stage_input, inputs)
    write_json(target / "complete.json", dict(complete=False))
    outputs, visited = [stage_input], set()
    for layer, e0, e1, piece in pieces:
        allowed = {torch.float16, torch.bfloat16} if provenance["convert_bf16"] else {torch.float16}
        if not {piece["gate_up"].dtype, piece["down"].dtype} <= allowed:
            raise ValueError("checkpoint conversion policy mismatch")
        folder = target / f"layer_{layer}"
        folder.mkdir(exist_ok=True)
        for expert in range(e0, e1):
            if (layer, expert) in visited:
                raise ValueError("duplicate checkpoint expert")
            visited.add((layer, expert))
            gate = piece["gate_up"][expert - e0].to(config["device"], dtype=torch.float16)
            down = piece["down"][expert - e0].to(config["device"], dtype=torch.float16)
            for mod in ("vision", "text"):
                key = f"L{layer}_E{expert}_{mod}"
                path = root / "cluster" / f"layer_{layer}" / f"{key}_clusters.pt"
                if str(path.relative_to(root / "cluster")) not in source["outputs"]:
                    continue
                data = torch.load(path, weights_only=True)
                value = expert_values(data["centroid"].to(config["device"]), gate, down, config["value_chunk"])
                out = folder / f"{key}_cache.pt"
                torch.save(dict(id=torch.arange(len(value)), key=data["key"], value=value), out)
                outputs.append(out)
    if len(visited) != source["num_layers"] * source["num_experts"]:
        raise ValueError("checkpoint expert coverage gap")
    return finish_stage(target, inputs, outputs, pca_root=str((root / "cluster").resolve()),
                        missing_tables=source["missing_tables"], provenance=provenance,
                        pca_outputs={name: h for name, h in source["outputs"].items() if name.endswith("_pca.pt")})


def iter_checkpoint_experts(model):
    from safetensors import safe_open
    from freetoken.models.qwen3_vl.weight import _linear_layout, rename_vl_prefix
    from freetoken.moe.expert_pieces import packed_expert_source_info

    geometry = read_json(Path(model) / "config.json")
    if geometry["architectures"] != ["Qwen3VLMoeForConditionalGeneration"]:
        raise ValueError("builder requires Qwen3-VL-MoE")
    sources = {}
    for path in sorted(Path(model).glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                info = packed_expert_source_info(rename_vl_prefix(key))
                if info is not None:
                    if info in sources:
                        raise ValueError("duplicate checkpoint expert tensor")
                    sources[info] = (path, key)
    from contextlib import ExitStack
    text = geometry["text_config"]
    for layer in range(text["num_hidden_layers"]):
        with ExitStack() as stack:
            handles = {}
            tensors = {}
            for leaf, role in (("gate_up_proj", "gate_up"), ("down_proj", "down")):
                path, key = sources[layer, leaf]
                if path not in handles:
                    handles[path] = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                tensors[role] = handles[path].get_slice(key)
            for expert in range(text["num_experts"]):
                yield layer, expert, expert + 1, {
                    role: _linear_layout(view[expert:expert + 1]).contiguous() for role, view in tensors.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "cluster", "materialize", "all"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, help="one dataset's asset root for cluster/materialize")
    args = parser.parse_args()
    config = read_json(args.config)
    root = args.root or Path(config.get("asset_root", config["output"]))
    if args.stage in ("collect", "all"):
        # Collection drives the real server through the same suite lifecycle.
        import subprocess
        repo = Path(__file__).resolve().parents[3]
        command = [config["python"], str(repo / "benchmarks/bench_tablemoe_suite.py"), "build", "--config", str(args.config)]
        if args.stage == "collect":
            command.append("--collect-only")
        subprocess.run(command, check=True)
        return
    if args.stage == "cluster":
        cluster(root, config.get("builder", {}))
    if args.stage == "materialize":
        materialize(root, config.get("builder", {}))


if __name__ == "__main__":
    main()
