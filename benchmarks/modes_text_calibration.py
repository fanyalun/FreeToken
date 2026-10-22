from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import statistics

import torch

from modes_experiment import checkpoint, digest, initialize, prompts, read, write
from freetoken.moe.preverify_policy import PreverifyPolicy


@torch.inference_mode()
def replay(runner, snapshot, tokens, policy, *, ablation=None):
    control = runner.three_level.experts
    snapshot.restore()
    control.policy, control.ablation_layer = policy, ablation
    control.begin(0)
    control.set_phase("preverify")
    count = len(tokens)
    control.signals = (torch.empty((40, count, 8), device=runner.engine.device),
                       torch.empty((40, count, 8), dtype=torch.bool, device=runner.engine.device))
    hidden = []
    start = runner.frontier.target_valid_len
    try:
        for offset in range(0, count, 4):
            ids = torch.tensor(tokens[offset:offset + 4], dtype=torch.int32, device=runner.engine.device)
            runner._reserve(start + offset + ids.numel())
            control.signal_offset = offset
            state = runner.states[ids.numel()]
            batch = runner._batch(ids, start + offset, state=state)
            _, output = runner._target(batch, need_logits=False)
            hidden.append(output.clone())
            state.commit(torch.tensor([ids.numel()], dtype=torch.int32, device=runner.engine.device))
        weights, keep = control.signals
        signals = dict(weights=weights.cpu(), keep=keep.cpu(), native=keep.numel(), retained=int(keep.sum()))
        return torch.cat(hidden), signals
    finally:
        control.signals = None
        control.ablation_layer = None
        control.close()


@torch.inference_mode()
def generate_teacher(runner, prompt, eos_ids, max_tokens=256):
    anchor = initialize(runner, prompt)
    tokens = []
    count = torch.ones(1, dtype=torch.int32, device=runner.engine.device)
    for offset in range(max_tokens):
        token = int(anchor.item())
        if token in eos_ids:
            break
        tokens.append(token)
        runner._reserve(len(prompt) + offset + 1)
        state = runner.states[1]
        batch = runner._batch(anchor, len(prompt) + offset, state=state)
        anchor, _ = runner._target(batch)
        anchor = anchor.clone()
        state.commit(count)
    if not tokens:
        raise ValueError("teacher answer has no decode positions")
    return tokens


@torch.inference_mode()
def kl(runner, teacher, candidate):
    total = torch.zeros((), dtype=torch.float64, device=runner.engine.device)
    teacher = teacher.to(runner.engine.device)
    head = runner.engine.model.lm_head
    def project(hidden):
        if runner.engine.config.speculative_reference_math:
            return torch.cat([head.quant_method.apply(head, row) for row in hidden.split(1)])
        return head.quant_method.apply(head, hidden)
    for start in range(0, candidate.shape[0], 4):
        p = project(teacher[start:start + 4]).float().log_softmax(-1)
        q = project(candidate[start:start + 4]).float().log_softmax(-1)
        total += (p.exp() * (p - q)).sum().double()
    value = float(total)
    if not math.isfinite(value) or value < -1e-4:
        raise ValueError("non-finite or materially negative full-vocabulary KL")
    return max(0., value)


def teacher_record(runner, selection, split, index, prompt, tokenizer, run, replacement):
    folder = Path(run) / "teachers" / split
    path = folder / f"sample_{index:03d}.pt"
    identity = dict(sample_id=selection["ids"][split][index], model_config_sha256=selection["model_config_sha256"],
                    replacement=replacement, decode_chunk=4, max_tokens=256)
    anchor = initialize(runner, prompt)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("teacher identity changed")
        return checkpoint(runner), saved
    tokens = generate_teacher(runner, prompt, {tokenizer.eos_token_id})
    initialize(runner, prompt)
    snapshot = checkpoint(runner)
    policy = PreverifyPolicy(top_k=8, replacement=replacement)
    teacher, signals = replay(runner, snapshot, tokens, policy)
    repeated, _ = replay(runner, snapshot, tokens, policy)
    if not torch.equal(teacher, repeated) or signals["retained"] != signals["native"]:
        raise RuntimeError("full teacher replay or native-route control failed")
    saved = dict(identity=identity, tokens=tokens, hidden=teacher.cpu(), weights=signals["weights"],
                 length_limited=len(tokens) == 256, teacher_repeat_equal=True)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(saved, path)
    write(path.with_suffix(".json"), dict(**identity, tokens=len(tokens), length_limited=saved["length_limited"],
                                         sha256=digest(path), teacher_repeat_equal=True))
    return snapshot, saved


@torch.inference_mode()
def calibration_smoke(engine, selection, run):
    runner = engine.speculative_runner
    values, tokenizer = prompts(selection, "pilot")
    tokens = generate_teacher(runner, values[0], {tokenizer.eos_token_id}, max_tokens=8)
    initialize(runner, values[0])
    snapshot = checkpoint(runner)
    full = PreverifyPolicy(top_k=8)
    teacher, signals = replay(runner, snapshot, tokens, full)
    repeated, _ = replay(runner, snapshot, tokens, full)
    if not torch.equal(teacher, repeated) or signals["retained"] != signals["native"]:
        raise RuntimeError("calibration smoke full-repeat or route-count failure")
    losses = {}
    for replacement in ("zero", "identity"):
        hidden, _ = replay(runner, snapshot, tokens, PreverifyPolicy(top_k=8, replacement=replacement), ablation=0)
        losses[replacement] = kl(runner, teacher, hidden)
    write(Path(run) / "calibration_smoke.json", dict(status="complete", tokens=len(tokens),
        teacher_repeat_equal=True, native_routes=signals["native"], ablation_kl=losses))


@torch.inference_mode()
def calibrate(engine, selection, run):
    run = Path(run)
    pilot = read(run / "pilot/result.json")
    if pilot["measured_rows"] != 240:
        raise ValueError("calibration requires the complete 16-sample replacement pilot")
    replacement = pilot["replacement"]
    runner = engine.speculative_runner
    values, tokenizer = prompts(selection, "calibration")
    raw_losses = torch.zeros(40, dtype=torch.float64)
    teacher_weights, total_tokens = [], 0
    full_policy = PreverifyPolicy(top_k=8, replacement=replacement)
    for index, prompt in enumerate(values):
        snapshot, saved = teacher_record(runner, selection, "calibration", index, prompt, tokenizer, run, replacement)
        path = run / "importance" / f"sample_{index:03d}.json"
        record = read(path) if path.exists() else dict(sample_id=selection["ids"]["calibration"][index], losses={})
        if record["sample_id"] != saved["identity"]["sample_id"]:
            raise ValueError("importance sample mismatch")
        for layer in range(40):
            if str(layer) not in record["losses"]:
                hidden, _ = replay(runner, snapshot, saved["tokens"], full_policy, ablation=layer)
                record["losses"][str(layer)] = kl(runner, saved["hidden"], hidden)
                write(path, record)
                print("importance", index, layer, record["losses"][str(layer)], flush=True)
        raw_losses += torch.tensor([record["losses"][str(i)] for i in range(40)], dtype=torch.float64)
        teacher_weights.append(saved["weights"])
        total_tokens += len(saved["tokens"])
        del snapshot
    if not torch.isfinite(raw_losses).all() or raw_losses.sum() <= 0:
        raise ValueError("layer ablations did not define finite positive importance")
    importance = (raw_losses / raw_losses.sum()).float()
    write(run / "importance/result.json", dict(status="complete", replacement=replacement,
        importance=importance.tolist(), raw_kl=raw_losses.tolist(), decode_tokens=total_tokens,
        samples=64, ablation="replace all routed experts using selected replacement; shared branch remains exact"))
    scores = torch.cat([(w * importance[:, None, None]).flatten() for w in teacher_weights])
    assets = {}
    for target in (.25, .5):
        assets[str(target)] = search(engine, selection, run, replacement, importance, scores, target)
    write(run / "calibration/result.json", dict(status="complete", replacement=replacement,
        assets=assets, samples=64, scope="fused decode text; cache-hit retention disabled"))


@torch.inference_mode()
def evaluate(engine, selection, run, replacement, importance, tau, split, target):
    runner = engine.speculative_runner
    values, tokenizer = prompts(selection, split)
    policy = PreverifyPolicy("modes", replacement=replacement, importance=tuple(importance.tolist()), tau=tau)
    loss, native, retained, tokens = 0., 0, 0, 0
    scores, records = [], []
    for index, prompt in enumerate(values):
        snapshot, saved = teacher_record(runner, selection, split, index, prompt, tokenizer, run, replacement)
        hidden, signals = replay(runner, snapshot, saved["tokens"], policy)
        sample_loss = kl(runner, saved["hidden"], hidden)
        loss += sample_loss
        native += signals["native"]
        retained += signals["retained"]
        tokens += len(saved["tokens"])
        scores.append((signals["weights"] * importance[:, None, None]).flatten())
        records.append(dict(id=saved["identity"]["sample_id"], kl=sample_loss,
                            native=signals["native"], retained=signals["retained"], tokens=len(saved["tokens"])))
        del snapshot
    quantiles = sorted({max(0., min(1., target + delta)) for delta in (-.1, -.025, -.005, 0, .005, .025, .1)})
    thresholds = torch.quantile(torch.cat(scores), torch.tensor(quantiles))
    return dict(tau=tau, skip_fraction=1-retained/native, kl_per_token=loss/tokens, total_kl=loss,
        native=native, retained=retained, tokens=tokens, records=records,
        quantiles={str(q): float(t) for q, t in zip(quantiles, thresholds)})


def search(engine, selection, run, replacement, importance, scores, target):
    folder = Path(run) / "search" / str(target)
    folder.mkdir(parents=True, exist_ok=True)
    trials = read(folder / "trials.json") if (folder / "trials.json").exists() else []
    proposed = float(torch.quantile(scores, target))
    for iteration in range(len(trials), 24):
        seen = {r["tau"] for r in trials}
        if proposed in seen:
            ordered = sorted([(0., 0.)] + [(r["tau"], r["skip_fraction"]) for r in trials]
                             + [(float(importance.max()) + 1e-6, 1.)])
            brackets = [(a, b) for a, b in zip(ordered, ordered[1:]) if (a[1]-target)*(b[1]-target) <= 0]
            if not brackets:
                raise ValueError("no measured threshold bracket")
            a, b = min(brackets, key=lambda pair: abs(pair[0][1]-target) + abs(pair[1][1]-target))
            proposed = float(torch.tensor((a[0]+b[0])/2, dtype=torch.float32))
        result = evaluate(engine, selection, run, replacement, importance, proposed, "calibration", target)
        trials.append(result)
        write(folder / "trials.json", trials)
        print("threshold", target, len(trials), proposed, result["skip_fraction"], result["kl_per_token"], flush=True)
        feasible = [r for r in trials if abs(r["skip_fraction"]-target) <= .01]
        if feasible and len(feasible) >= 3:
            break
        if feasible:
            fraction = target + (-.005 if len(feasible) == 1 else .005)
            proposed = result["quantiles"][str(fraction)]
        else:
            proposed = result["quantiles"][str(target)]
    feasible = [r for r in trials if abs(r["skip_fraction"]-target) <= .01]
    if not feasible:
        raise ValueError(f"MoDES skip target {target} did not converge in 24 measured trials")
    best = min(feasible, key=lambda r: (r["kl_per_token"], abs(r["skip_fraction"]-target), r["tau"]))
    path = Path(run) / "calibration" / f"modes_skip{int(target*100)}.json"
    asset = dict(status="validated", model_config_sha256=selection["model_config_sha256"],
        weight_index_sha256=selection["weight_index_sha256"], weight_shard_stat=selection["weight_shard_stat"],
        dtype="bfloat16", score_dtype="float32", top_k=8, replacement=replacement,
        calibration_arithmetic="resident_rowwise_reference_fixed_kv_tiles_and_gated_norm",
        importance=importance.tolist(), tau=best["tau"], target_skip_fraction=target,
        measured_skip_fraction=best["skip_fraction"], kl_per_token=best["kl_per_token"],
        calibration_samples=64, runtime_keep_ready_hits=True, calibration_keep_ready_hits=False,
        heldout_samples_used=0)
    write(path, asset)
    return str(path.resolve())


@torch.inference_mode()
def development(engine, selection, run):
    run = Path(run)
    calibration = read(run / "calibration/result.json")
    runner, coordinator = engine.speculative_runner, engine.speculative_runner.three_level
    budget = coordinator.budget
    values, _ = prompts(selection, "development")
    results = []
    for target, path in calibration["assets"].items():
        asset = read(path)
        importance = torch.tensor(asset["importance"])
        fixed = evaluate(engine, selection, run, asset["replacement"], importance, asset["tau"],
                         "development", float(target))
        policy = PreverifyPolicy("modes", replacement=asset["replacement"],
                                  importance=tuple(asset["importance"]), tau=asset["tau"])
        rows = []
        for index, prompt in enumerate(values):
            anchor = initialize(runner, prompt)
            snapshot = checkpoint(runner)
            for loops in (8, 12, 16):
                snapshot.restore()
                coordinator.budget = replace(budget, max_loops=loops, submit_rows=4 * loops + 1)
                coordinator.experts.policy = policy
                _, _, row = coordinator.step(anchor, 128, (), lambda: False)
                if row["loop_count"] != loops:
                    raise RuntimeError("development window was truncated")
                rows.append(dict(sample=index, loops=loops, accepted=row["accepted"], drafted=row["drafted"]))
            del snapshot
        results.append(dict(target=float(target), fixed_trajectory=fixed, windows=rows,
                            mean_accepted=statistics.mean(r["accepted"] for r in rows)))
        write(run / "development/result.json", dict(status="incomplete", results=results))
    coordinator.budget = budget
    write(run / "development/result.json", dict(status="complete", samples=16, results=results,
          heldout_samples_used=0, thresholds_refitted=False))
