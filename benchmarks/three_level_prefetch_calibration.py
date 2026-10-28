from __future__ import annotations

import math
import statistics


def gap_capacity_for_experts(experts):
    return experts + 1 if experts else 0


def gap_experts_for_capacity(capacity):
    return max(0, capacity - 1)


def select_quotas(windows, stages, copies, *, layer_layouts, num_experts):
    """Fit static quotas from arithmetic means of one request and one-expert copies."""
    layers = len(layer_layouts)
    if not layers or num_experts < 1:
        raise ValueError("prefetch calibration requires MoE layers and experts")
    io = {}
    for layout in set(layer_layouts):
        values = [value for row in copies if row["layout"] == layout for value in row["samples_ms"]]
        if not values or any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"missing positive expert IO measurements for layout {layout}")
        io[layout] = statistics.mean(values)
    means, counts = {}, {}
    for phase in ("preverify", "verify"):
        rows = [row for row in windows if row["phase"] == phase]
        if set(row["layer"] for row in rows) != set(range(layers)):
            raise ValueError(f"single sample did not exercise every {phase} layer")
        values = [row["compute_ms"] for row in rows]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("compute durations must be finite and nonnegative")
        means[phase + "_layer_ms"] = statistics.mean(values)
        means[phase + "_ms"] = statistics.mean(row["duration_ms"] for row in stages if row["phase"] == phase)
        counts[phase] = len(rows) // layers
    draft = [row["duration_ms"] for row in stages if row["phase"] == "draft"]
    gaps = [row["duration_ms"] for row in stages if row["phase"] == "draft_gap"]
    if not draft or any(not math.isfinite(value) or value < 0 for value in draft + gaps):
        raise ValueError("single sample did not provide valid Draft timings")
    means["draft_ms"] = statistics.mean(draft)
    means["draft_gap_ms"] = statistics.mean(gaps) if gaps else 0.
    counts.update(draft=len(draft), draft_gap=len(gaps))
    preverify = [min(num_experts, max(1, math.floor(means["preverify_layer_ms"] / io[layout])))
                 for layout in layer_layouts]
    verify = [min(num_experts, math.floor(means["verify_layer_ms"] / io[layout]))
              for layout in layer_layouts]
    preverify[-1] = verify[0] = 0
    # The gap has one shared quota; charge each copy at the slowest layout's mean.
    gap = min(num_experts * layers, math.floor(means["draft_gap_ms"] / max(io.values())))
    return dict(quotas=dict(preverify=preverify, verify=verify, gap_capacity=gap_capacity_for_experts(gap)),
                means=means, calls=counts, expert_io_mean_ms={str(k): v for k, v in io.items()},
                formula="floor(mean compute window / mean one-expert IO); minimum one for preverify; one shared gap quota")
