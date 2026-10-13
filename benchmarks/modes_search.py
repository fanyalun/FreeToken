"""Coarse-to-fine threshold search with explicit time and validation budgets."""

import math


def length_stratified_subset(records, count):
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(records):
        raise ValueError("search subset must select between one and all calibration pages")
    ordered = sorted(records, key=lambda record: (record["tokens"], record["id"]))
    selected = {ordered[min(len(ordered) - 1, int((i + .5) * len(ordered) / count))]["id"]
                for i in range(count)}
    return [record for record in records if record["id"] in selected]


def spaced_grid(grid, points):
    if isinstance(points, bool) or not isinstance(points, int) or not 2 <= points <= len(grid):
        raise ValueError("grid points must select between two and all grid entries")
    return sorted({grid[round(i * (len(grid) - 1) / (points - 1))] for i in range(points)})


def local_grid(grid, coarse, center, points):
    index = coarse.index(center)
    lower, upper = coarse[max(0, index - 1)], coarse[min(len(coarse) - 1, index + 1)]
    nearby = [value for value in grid if lower <= value <= upper]
    return sorted(set(spaced_grid(nearby, min(points, len(nearby)))) | {center})


def coarse_to_fine(text_grid, visual_grid, evaluate, target, *, coarse_points=9, fine_points=5,
                   continue_phase=lambda phase: True, on_phase=lambda phase: None):
    if not 0 < target < 1:
        raise ValueError("target must lie strictly between zero and one")
    cache, phase_complete = {}, {}

    def measured(text, visual):
        key = (text, visual)
        if key not in cache:
            loss, rate = evaluate(text, visual)
            if not math.isfinite(loss) or loss < 0 or not math.isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError("invalid candidate loss or skip proportion")
            cache[key] = loss, rate
        return cache[key]

    maximum = (text_grid[-1], visual_grid[-1])
    if measured(*maximum)[1] < target:
        raise ValueError("maximum tau does not reach target on the search subset")
    best = maximum

    def frontier(texts, visuals, phase):
        nonlocal best
        on_phase(phase)
        j = len(visuals) - 1
        for text in texts:
            while j >= 0:
                key = (text, visuals[j])
                if key not in cache and not continue_phase(phase):
                    phase_complete[phase] = False
                    return
                loss, rate = measured(*key)
                if rate >= target:
                    if loss < cache[best][0]:
                        best = key
                    j -= 1
                else:
                    break
            if j + 1 < len(visuals):
                key = (text, visuals[j + 1])
                if key not in cache and not continue_phase(phase):
                    phase_complete[phase] = False
                    return
                loss, rate = measured(*key)
                if rate >= target and loss < cache[best][0]:
                    best = key
        phase_complete[phase] = True

    coarse_text, coarse_visual = [spaced_grid(grid, coarse_points) for grid in (text_grid, visual_grid)]
    frontier(coarse_text, coarse_visual, "coarse")
    coarse_best = best
    fine_text = local_grid(text_grid, coarse_text, best[0], fine_points)
    fine_visual = local_grid(visual_grid, coarse_visual, best[1], fine_points)
    frontier(fine_text, fine_visual, "fine")
    return dict(tau=dict(zip(("text", "visual"), best)), best_val=cache[best][0],
                skip_proportion=cache[best][1], coarse_best=list(coarse_best),
                coarse_grids=dict(text=coarse_text, visual=coarse_visual),
                fine_grids=dict(text=fine_text, visual=fine_visual),
                phase_complete=phase_complete, evaluated_tau_pairs=len(cache))


def validate_full_target(anchor, maximum, evaluate, target, *, max_evaluations=4, tolerance=.02,
                         continue_evaluation=lambda: True):
    if max_evaluations < 1 or tolerance < 0:
        raise ValueError("invalid full validation budget")
    measured, feasible = [], []
    # Both thresholds follow the same scale so their modality balance survives full-set correction.
    lower = (0., 0.)
    upper = None
    scale = 1.
    for _ in range(max_evaluations):
        if not continue_evaluation():
            break
        tau = tuple(min(anchor[name] * scale, maximum[name]) for name in ("text", "visual"))
        loss, rate = evaluate(*tau)
        if not math.isfinite(loss) or loss < 0 or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("invalid full-set result")
        measured.append(dict(tau=dict(zip(("text", "visual"), tau)), loss=loss, skip_proportion=rate,
                             scale=scale))
        if rate >= target:
            feasible.append(measured[-1])
            upper = (scale, rate)
            if rate <= target + tolerance:
                break
        else:
            lower = (scale, rate)
        if upper is None:
            scale *= min(1.5, max(1.05, target / max(rate, 1e-6)))
        else:
            fraction = (target - lower[1]) / (upper[1] - lower[1])
            scale = lower[0] + min(.9, max(.1, fraction)) * (upper[0] - lower[0])
        if measured[-1]["tau"] == maximum:
            break
    if not feasible:
        raise ValueError("full validation did not find a feasible threshold within the budget")
    best = min(feasible, key=lambda result: result["loss"])
    return dict(tau=best["tau"], best_val=best["loss"], skip_proportion=best["skip_proportion"],
                full_validation=measured, target_tolerance_met=best["skip_proportion"] <= target + tolerance)
