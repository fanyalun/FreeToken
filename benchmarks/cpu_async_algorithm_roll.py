"""Serially simulate fixed-token replay and rolling CPU verification."""

from dataclasses import asdict

from freetoken.engine.qwen_async_three_level import GpuCandidateBlock, retained_tail
from cpu_async_algorithm_pairs import _quality


def slide_cells(handoffs, output_tokens):
    episodes, chain, start, returned = [], 0, 0, 1
    for row in handoffs:
        returned += row["input_count"]
        if row["terminal"]:
            if chain:
                episodes.append(dict(start_handoff=start, end_handoff=row["handoff"],
                    cells=chain, exact=False, reason="output_cap"))
            break
        if row["retained"]:
            chain += 1
            continue
        censored = row["reason"] in ("no_ahead_capacity", "no_retained_candidates")
        if censored and (not row["full_accept"] or output_tokens - returned != 1):
            raise RuntimeError("an empty ahead block was not explained by the output cap")
        if chain or not censored:
            episodes.append(dict(start_handoff=start, end_handoff=row["handoff"], cells=chain,
                exact=not censored, reason="output_cap" if censored else row["reason"]))
        chain, start = 0, row["handoff"] + 1
    if sum(row["cells"] for row in episodes) != sum(row["retained"] for row in handoffs):
        raise RuntimeError("slide-cell episodes disagree with retained handoffs")
    histogram = {str(k): sum(row["cells"] == k for row in episodes)
                 for k in sorted({row["cells"] for row in episodes})}
    return dict(episodes=episodes, histogram_observed=histogram,
        attempts=len(episodes), zero_slide_attempts=sum(row["cells"] == 0 for row in episodes),
        at_least_two=sum(row["cells"] >= 2 for row in episodes),
        output_censored=sum(not row["exact"] for row in episodes))


def run_rolling(session, *, recovery="fixed_replay"):
    if recovery not in ("fixed_replay", "recheck"):
        raise ValueError("rolling recovery must be fixed_replay or recheck")
    initial = session.reset()
    window = session.window
    output_budget = session.output_tokens
    output = [initial.anchor]
    input_end = initial.length + output_budget - 1
    loops = 4
    block_cap = loops * (session.engine.config.speculative_mtp_depth + 1)
    block, initial_inner = session.generate(initial.anchor, loops=loops,
        candidate_cap=min(block_cap, max(0, output_budget - 2)))
    traces, chains = [], []
    chain = longest = 0
    source = "initial"
    old_ahead_mask = (False,) * len(block.candidates)

    while len(output) < output_budget:
        if len(traces) >= output_budget - 1:
            raise RuntimeError("rolling verification failed to advance the confirmed prefix")
        remaining = output_budget - len(output)
        block = GpuCandidateBlock(block.start, block.anchor, block.candidates[:remaining - 1])
        old_ahead_mask = old_ahead_mask[:len(block.candidates)]
        before = window.runner.frontier.target_valid_len
        if before != block.start + len(block.candidates):
            raise RuntimeError("next ahead generation must continue after the complete proposal")
        ahead_cap = min(block_cap, max(0, input_end - 1 - before)) if remaining > 1 else 0
        ahead, ahead_inner = session.ahead(block, loops=loops, candidate_cap=ahead_cap)
        after = window.runner.frontier.target_valid_len
        if ahead.start != before or after != ahead.start + len(ahead.candidates):
            raise RuntimeError("ahead generation did not advance from the proposal's private frontier")
        result = session.verify(block, remaining=remaining)
        if (result.start != block.start or result.input_count < 1
                or result.input_count != len(result.tokens)
                or result.checkpoint.length != result.start + result.input_count):
            raise RuntimeError("CPU result has inconsistent accepted-prefix accounting")
        if not 0 <= result.accepted <= len(block.candidates):
            raise RuntimeError("CPU accepted length exceeds the submitted proposal")
        output.extend(result.tokens)
        if len(output) > output_budget:
            raise RuntimeError("CPU verification exceeded the fixed output budget")
        if tuple(output) != tuple(session.reference_tokens[:len(output)]):
            raise RuntimeError("rolling CPU output differs from the serial CPU Target")
        session.install(result)
        if (window.canonical_length != result.checkpoint.length
                or window.runner.frontier.target_valid_len != result.checkpoint.length):
            raise RuntimeError("GPU installation did not restore the confirmed CPU frontier")

        full = result.accepted == len(block.candidates)
        offset = result.checkpoint.length - ahead.start - 1
        bonus_match = (0 <= offset < len(ahead.candidates)
                       and ahead.candidates[offset] == result.tokens[-1])
        tail = retained_tail(result, ahead)
        usable = tail is not None and bool(tail.candidates)
        record = dict(handoff=len(traces), proposal_source=source,
            submitted=asdict(block), ahead=asdict(ahead), ahead_inner=ahead_inner,
            fresh_ahead_start=before, fresh_ahead_end=after,
            cpu_block_id=result.block_id, cpu_tokens=list(result.tokens),
            cpu_predictions=result.predictions.tolist(), proposed=len(block.candidates),
            accepted=result.accepted, input_count=result.input_count,
            confirmed_end=result.checkpoint.length, bonus=result.tokens[-1],
            full_accept=full, bonus_match=bonus_match, terminal=result.terminal,
            retained=False, retained_candidates=0, replay=None,
            recheck=None, cpu_gate_retained=usable,
            submitted_old_ahead_mask=old_ahead_mask,
            cpu_accepted_old_ahead_tokens=sum(old_ahead_mask[:result.accepted]),
            next_proposal=None, discarded_ahead_candidates=0, success_chain=chain)

        if result.terminal:
            if len(output) != output_budget:
                raise RuntimeError("CPU terminated before the fixed output budget")
            record["reason"] = "cpu_terminal"
            traces.append(record)
            break

        if usable:
            before_replay = window.runner.frontier.target_valid_len
            old_tail = tail
            old_quality = _quality(session, old_tail)
            if recovery == "recheck":
                checked = session.recheck(old_tail)
                tail = checked.block
                matched = checked.matched_old_tokens
                record["recheck"] = dict(**asdict(checked), old_block=asdict(old_tail),
                    old_quality=old_quality, checked_quality=_quality(session, tail))
            else:
                session.replay(tail)
                matched = len(tail.candidates)
            after_replay = window.runner.frontier.target_valid_len
            replay_inputs = (tail.anchor, *tail.candidates[:-1])
            if (before_replay != tail.start
                    or after_replay != tail.start + len(replay_inputs)
                    or (recovery == "fixed_replay" and after_replay != after)):
                raise RuntimeError("fixed replay did not reconstruct the old ahead block's frontier")
            full_reuse = matched == len(old_tail.candidates)
            reason = ("retained_and_replayed" if recovery == "fixed_replay" else
                      "retained_and_rechecked" if full_reuse else
                      "recheck_modified" if matched else "recheck_zero_match")
            record.update(reason=reason, retained=full_reuse,
                retained_candidates=matched, replay=dict(
                    start=before_replay, end=after_replay, rows=len(replay_inputs),
                    inputs=list(replay_inputs), pending_anchor=tail.candidates[-1])
                    if recovery == "fixed_replay" else None,
                discarded_ahead_candidates=len(old_tail.candidates) - matched,
                next_proposal=asdict(tail), old_tail_quality=old_quality)
            block = tail
            source = ("replayed_ahead" if recovery == "fixed_replay" else
                      "rechecked_ahead" if full_reuse else "recheck_replacement")
            old_ahead_mask = (True,) * matched + (False,) * (len(tail.candidates) - matched)
            if full_reuse:
                chain += 1
                longest = max(longest, chain)
            else:
                if chain:
                    chains.append(chain)
                chain = 0
        else:
            if chain:
                chains.append(chain)
                chain = 0
            reason = ("A_rejected" if not full else
                      "no_ahead_capacity" if not ahead.candidates else
                      "no_retained_candidates" if tail is not None else
                      "bonus_or_overlap_mismatch")
            cap = min(block_cap, max(0, output_budget - len(output) - 1))
            block, inner = session.generate(result.tokens[-1], loops=loops, candidate_cap=cap)
            record.update(reason=reason, discarded_ahead_candidates=len(ahead.candidates),
                regeneration_inner=inner, next_proposal=asdict(block))
            source = "regenerated"
            old_ahead_mask = (False,) * len(block.candidates)
        record["success_chain"] = chain
        traces.append(record)

    if chain:
        chains.append(chain)
    exact = tuple(output) == tuple(session.reference_tokens[:output_budget])
    consumed = sum(row["input_count"] for row in traces)
    if not exact or consumed != output_budget - 1:
        raise RuntimeError("rolling request failed the serial CPU reference or input-count audit")
    return dict(algorithm="fixed_token_replay_rolling" if recovery == "fixed_replay" else "recheck_rolling",
        recovery=recovery, request_complete=True,
        output_tokens=output, returned_tokens=len(output), committed_input_rows=consumed,
        cpu_serial_output_exact=exact, initial_inner=initial_inner, handoffs=traces,
        retained_handoffs=sum(row["retained"] for row in traces),
        regenerated_handoffs=sum(not row["retained"] and not row["terminal"]
                                  for row in traces),
        cpu_accepted_old_ahead_tokens=sum(row["cpu_accepted_old_ahead_tokens"] for row in traces),
        partially_reused_handoffs=sum(not row["retained"] and row["retained_candidates"] > 0 for row in traces),
        slide_cells=slide_cells(traces, output_budget),
        discarded_ahead_candidates=sum(row["discarded_ahead_candidates"] for row in traces),
        success_chain_lengths=chains, longest_success_chain=longest)
