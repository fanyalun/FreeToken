from __future__ import annotations

import argparse
from dataclasses import replace
import itertools
from pathlib import Path
import statistics
import time
import traceback

import torch

from modes_experiment import MODEL, checkpoint, fused_engine, initialize, prepare, prompts, read, write, verify_runtime
from freetoken.moe.preverify_policy import PreverifyPolicy


def choose_replacement(rows):
    means = {}
    for replacement in ("zero", "identity"):
        groups = []
        for top_k, loops in itertools.product((4, 6), (8, 12, 16)):
            group = [r["accepted"] for r in rows if (r["top_k"], r["requested_loops"], r["replacement"])
                     == (top_k, loops, replacement)]
            if not group:
                raise ValueError("incomplete replacement matrix")
            groups.append(statistics.mean(group))
        means[replacement] = statistics.mean(groups)
    winner = "identity" if means["identity"] > means["zero"] * 1.01 else "zero"
    return dict(replacement=winner, group_equal_weight_mean_accepted=means, minimum_relative_gain=.01,
                metric="full Target accepted candidate prefix, excluding bonus")


@torch.inference_mode()
def pilot(engine, selection, run, samples=16):
    run = Path(run)
    runner, control = engine.speculative_runner, engine.speculative_runner.three_level
    expert = control.experts
    values, _ = prompts(selection, "pilot")
    expected = samples * 15
    rows = read(run / "pilot/rows.json") if (run / "pilot/rows.json").exists() else []
    keys = {(r["sample"], r["top_k"], r["replacement"], r["requested_loops"]) for r in rows}
    budget = control.budget
    expert.audit = False
    for sample, prompt in enumerate(values[:samples]):
        anchor = initialize(runner, prompt)
        snapshot = checkpoint(runner)
        cases = [(8, "zero", loops) for loops in (8, 12, 16)]
        cases += list(itertools.product((4, 6), ("zero", "identity"), (8, 12, 16)))
        for top_k, replacement, loops in cases:
            if (sample, top_k, replacement, loops) in keys:
                continue
            snapshot.restore()
            control.budget = replace(budget, max_loops=loops, submit_rows=4 * loops + 1)
            expert.policy = PreverifyPolicy(top_k=top_k, replacement=replacement)
            original_forward = control._forward
            trace = []
            def traced_forward(inputs, start, state):
                inputs_copy = inputs.clone()
                tokens, hidden = original_forward(inputs, start, state)
                trace.append(dict(phase=expert.phase, start=start, inputs=inputs_copy.cpu(),
                                  predictions=tokens.cpu(), hidden=hidden.cpu()))
                return tokens, hidden
            if top_k == 8:
                control._forward = traced_forward
            engine.stream.synchronize()
            started = time.perf_counter()
            try:
                _, _, row = control.step(anchor, 128, (), lambda: False)
            finally:
                control._forward = original_forward
            engine.stream.synchronize()
            if row["loop_count"] != loops or row["flush_reason"] != "loop_limit":
                raise RuntimeError("pilot window did not execute the requested loop count")
            if top_k == 8 and row["accepted"] != row["drafted"]:
                write(run / "pilot/full_control_failure.json", dict(sample=sample, loops=loops, row=row))
                torch.save(trace, run / "pilot/full_control_trace.pt")
                probes = []
                probe_hidden = {}
                for name, inputs, state, phase in (
                    ("canonical_one", trace[0]["inputs"][:1], runner.states[1], "verify"),
                    ("canonical_four", trace[0]["inputs"], runner.states[len(trace[0]["inputs"])], "verify"),
                    ("private_four", trace[0]["inputs"], control.private_states[len(trace[0]["inputs"])], "preverify"),
                    ("canonical_outer", trace[-1]["inputs"], runner.states[len(trace[-1]["inputs"])], "verify")):
                    snapshot.restore()
                    control.private_pool.recurrent_states.copy_(engine.linear_state_pool.recurrent_states[:, 1:2])
                    control.private_pool.conv_states.copy_(engine.linear_state_pool.conv_states[:, 1:2])
                    expert.begin(control.epoch + 1)
                    expert.set_phase(phase)
                    prediction, hidden = original_forward(inputs.to(engine.device), trace[0]["start"], state)
                    probe_hidden[name] = hidden.cpu()
                    probes.append(dict(name=name, predictions=prediction.cpu().tolist()))
                    expert.close()
                probes.append(dict(private_vs_canonical_four_max_abs=float((
                    probe_hidden["canonical_four"].float()-probe_hidden["private_four"].float()).abs().max())))
                write(run / "pilot/full_control_probes.json", probes)
                torch.save(probe_hidden, run / "pilot/full_control_probe_hidden.pt")
                raise RuntimeError(f"full-top8 resident control accepted {row['accepted']}/{row['drafted']}; stop replacement selection")
            row.update(sample=sample, sample_id=selection["ids"]["pilot"][sample], top_k=top_k,
                replacement=replacement, requested_loops=loops, wall_seconds=time.perf_counter()-started,
                acceptance_rate=row["accepted"] / row["drafted"] if row["drafted"] else None,
                purpose="diagnostic", measurement_scope="one resident window including full Target")
            rows.append(row)
            write(run / "pilot/rows.json", rows)
            print("pilot", sample, top_k, replacement, loops, row["drafted"], row["accepted"], flush=True)
        del snapshot
    control.budget = budget
    expected_keys = {(sample, top_k, replacement, loops) for sample in range(samples)
                     for top_k, replacement, loops in [(8, "zero", n) for n in (8, 12, 16)]
                     + list(itertools.product((4, 6), ("zero", "identity"), (8, 12, 16)))}
    actual_keys = {(r["sample"], r["top_k"], r["replacement"], r["requested_loops"]) for r in rows}
    if len(rows) != expected or actual_keys != expected_keys:
        raise ValueError("pilot matrix coverage mismatch")
    result = dict(status="complete", purpose="diagnostic" if samples == 16 else "smoke",
                  expected_rows=expected, measured_rows=len(rows), **choose_replacement(rows))
    write(run / "pilot/result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--stage", choices=("prepare", "pilot", "calibrate", "development", "all"), default="all")
    parser.add_argument("--pilot-samples", type=int, choices=range(1, 17), default=16)
    args = parser.parse_args()
    selection = prepare(args.run, args.model)
    if args.stage == "prepare":
        return
    verify_runtime(args.run)
    write(args.run / "complete.json", dict(status="incomplete", stage=args.stage))
    try:
        engine = fused_engine(args.run, args.model)
        with torch.cuda.stream(engine.stream):
            if args.stage in ("pilot", "all"):
                pilot(engine, selection, args.run, args.pilot_samples)
                from modes_text_calibration import calibration_smoke
                calibration_smoke(engine, selection, args.run)
            if args.stage in ("calibrate", "all"):
                from modes_text_calibration import calibrate
                calibrate(engine, selection, args.run)
            if args.stage in ("development", "all"):
                from modes_text_calibration import development
                development(engine, selection, args.run)
        write(args.run / "fused_complete.json", dict(status="complete", stage=args.stage,
              calibration_complete=(args.run / "calibration/result.json").exists(),
              development_complete=(args.run / "development/result.json").exists()))
    except BaseException:
        write(args.run / "failure.json", dict(stage=args.stage, traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
