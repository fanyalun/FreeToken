"""Paired acceptance probes for the CPU-confirmed sliding-window algorithm."""

from __future__ import annotations

from dataclasses import asdict

from freetoken.engine.qwen_async_three_level import GpuCandidateBlock, retained_tail


def _full(result, block):
    return (result.accepted == len(block.candidates)
            and result.input_count == len(block.candidates) + 1)


def _quality(session, block, *, valid_context=True, count=None):
    candidates = tuple(block.candidates if count is None else block.candidates[:count])
    offset = block.start - session.initial.length
    reference = session.reference_tokens
    if offset < 0 or offset + len(candidates) >= len(reference):
        raise ValueError("candidate block exceeds the serial CPU reference")
    row = dict(proposed=len(candidates), valid_context=valid_context,
               reference_start=offset + 1, accepted=None, all_accepted=None,
               first_rejection_offset=None, first_rejection_position=None)
    if not valid_context:
        return row
    if block.anchor != reference[offset]:
        raise RuntimeError("a valid candidate block has the wrong serial CPU anchor")
    expected = reference[offset + 1:offset + 1 + len(candidates)]
    accepted = 0
    for candidate, token in zip(candidates, expected):
        if candidate != token:
            break
        accepted += 1
    row.update(accepted=accepted, all_accepted=accepted == len(candidates))
    if accepted < len(candidates):
        row.update(first_rejection_offset=accepted,
                   first_rejection_position=block.start + 1 + accepted)
    return row


def _block_row(session, block, inner, *, valid_context=True):
    return dict(block=asdict(block), inner=inner,
        consumed_start=block.start, consumed_end=block.start + len(block.candidates),
        pending_token=block.candidates[-1] if block.candidates else block.anchor,
        quality=_quality(session, block, valid_context=valid_context))


def _cpu_row(result, block):
    return dict(start=result.start, candidates=len(block.candidates), accepted=result.accepted,
        full_proposal_accepted=_full(result, block), input_count=result.input_count,
        confirmed_end=result.checkpoint.length, bonus=result.tokens[-1],
        tokens=list(result.tokens), predictions=result.predictions.tolist(), terminal=result.terminal)


def _retention(result, submitted, ahead):
    tail = retained_tail(result, ahead)
    offset = result.checkpoint.length - ahead.start - 1
    boundary_token = ahead.candidates[offset] if 0 <= offset < len(ahead.candidates) else None
    if result.terminal:
        reason = "cpu_terminal"
    elif not _full(result, submitted):
        reason = "proposal_rejected"
    elif not ahead.candidates:
        reason = "empty_ahead"
    elif boundary_token is not None and boundary_token != result.tokens[-1]:
        reason = "bonus_mismatch"
    elif tail is None:
        reason = "boundary_or_overlap_mismatch"
    elif not tail.candidates:
        reason = "empty_retained_tail"
    else:
        reason = "retained"
    usable = tail is not None and bool(tail.candidates)
    return (tail if usable else None), dict(retained=usable, reason=reason,
        full_proposal_accepted=_full(result, submitted),
        bonus_matches=boundary_token == result.tokens[-1] if boundary_token is not None else None,
        original_candidates=len(ahead.candidates), retained_candidates=len(tail.candidates) if usable else 0,
        tail=asdict(tail) if usable else None)


def _raw_ab(session):
    checkpoint = session.reset()
    a, inner_a = session.generate(checkpoint.anchor, loops=4)
    b, inner_b = session.ahead(a, loops=4)
    pending = a.candidates[-1] if a.candidates else a.anchor
    if b.start != a.start + len(a.candidates) or b.anchor != pending:
        raise RuntimeError("raw B does not extend the uncorrected A trajectory")
    return a, inner_a, b, inner_b


def _assert_same_ab(expected_a, expected_b, a, b):
    if a != expected_a or b != expected_b:
        raise RuntimeError("paired branches regenerated different A/B proposals from the same cold prefix")


def _aligned_quality(session, first, second):
    if first.start != second.start or first.anchor != second.anchor:
        raise RuntimeError("paired candidate windows have different anchors or positions")
    count = min(len(first.candidates), len(second.candidates))
    if count == 0:
        return dict(status="skipped", reason="empty_common_candidate_range", candidates=0)
    return dict(status="complete", start=first.start, anchor=first.anchor, candidates=count,
        first=_quality(session, first, count=count), second=_quality(session, second, count=count))


def run_pairs(session):
    """Run reset4/raw8 and replayed-B/exact-B C probes for one prompt."""
    checkpoint = session.reset()
    clean_a, inner_a = session.generate(checkpoint.anchor, loops=4)
    clean_a_result = session.verify(clean_a)
    clean_a_row = _block_row(session, clean_a, inner_a)
    session.install(clean_a_result)
    clean_b, inner_b = session.generate(clean_a_result.tokens[-1], loops=4)
    clean_b_row = _block_row(session, clean_b, inner_b)
    clean_b_result = session.verify(clean_b)
    reset4 = dict(a=clean_a_row, b=clean_b_row,
        a_cpu=_cpu_row(clean_a_result, clean_a), b_cpu=_cpu_row(clean_b_result, clean_b))

    a, inner_a, b, inner_b = _raw_ab(session)
    if a != clean_a:
        raise RuntimeError("reset4 and raw8 branches have different initial A proposals")
    result_a = session.verify(a)
    full_a = _full(result_a, a)
    tail_b, retention_b = _retention(result_a, a, b)
    ab = GpuCandidateBlock(a.start, a.anchor, a.candidates + b.candidates)
    raw8 = dict(a=_block_row(session, a, inner_a),
        b=_block_row(session, b, inner_b, valid_context=full_a),
        ab_quality=_quality(session, ab), a_cpu=_cpu_row(result_a, a),
        retention=retention_b, b_tail=asdict(tail_b) if tail_b is not None else None,
        b_cpu=None)
    aligned_b = dict(status="skipped", reason=retention_b["reason"])
    if tail_b is not None:
        result_b = session.verify(tail_b)
        raw8["b_cpu"] = _cpu_row(result_b, tail_b)
        aligned_b = _aligned_quality(session, tail_b, clean_b)
        aligned_b.update(first_name="raw8_retained_b", second_name="reset4_fresh_b")
    result = dict(experiment1=dict(reset4=reset4, continuous8=raw8, aligned_b=aligned_b))
    if tail_b is None:
        result["experiment2"] = dict(status="skipped", eligible=False,
            skip_reason=retention_b["reason"], raw_a_candidates=len(a.candidates),
            discarded_b_candidates=len(b.candidates))
        return result

    replay_a, _, replay_b, _ = _raw_ab(session)
    _assert_same_ab(a, b, replay_a, replay_b)
    result_a = session.verify(replay_a)
    retained_b, repeated_retention = _retention(result_a, replay_a, replay_b)
    if retained_b != tail_b:
        raise RuntimeError("the repeated replay branch has a different retained B")
    session.install(result_a)
    session.replay(retained_b)
    replay_end = session.window.runner.frontier.target_valid_len
    if replay_end != retained_b.start + len(retained_b.candidates):
        raise RuntimeError("fixed B replay did not end at B's pending anchor")
    c, inner_c = session.ahead(retained_b, loops=4)
    if c.start != replay_end or c.anchor != retained_b.candidates[-1]:
        raise RuntimeError("C was not generated beyond replayed B")
    result_b = session.verify(retained_b)
    full_b = _full(result_b, retained_b)
    _, retention_c = _retention(result_b, retained_b, c)
    replay_c = dict(a_cpu=_cpu_row(result_a, replay_a), b_retention=repeated_retention,
        b_replay=dict(block=asdict(retained_b), input_tokens=[retained_b.anchor, *retained_b.candidates[:-1]],
                      consumed_start=retained_b.start, consumed_end=replay_end),
        b_cpu=_cpu_row(result_b, retained_b), c=_block_row(session, c, inner_c, valid_context=full_b),
        c_quality=_quality(session, c, valid_context=full_b), c_retention=retention_c,
        wasted_c_candidates=0 if full_b else len(c.candidates),
        discarded_c_candidates=len(c.candidates) if not retention_c["retained"] else
            len(c.candidates) - retention_c["retained_candidates"])
    experiment2 = dict(status="complete", eligible=True, replay_old_b=replay_c, exact_b=None,
                       paired_c=dict(status="skipped", reason="b_rejected"))
    result["experiment2"] = experiment2
    if not full_b:
        return result
    if len(retained_b.candidates) <= 1:
        experiment2["paired_c"] = dict(status="skipped", reason="insufficient_b_candidates_for_aligned_control")
        return result

    exact_a, _, exact_b, _ = _raw_ab(session)
    _assert_same_ab(a, b, exact_a, exact_b)
    exact_a_result = session.verify(exact_a)
    exact_tail, _ = _retention(exact_a_result, exact_a, exact_b)
    if exact_tail != retained_b:
        raise RuntimeError("the exact-B control has a different retained B")
    session.install(exact_a_result)
    # Keep the same pending B token as replay(), rather than consuming it twice.
    prefix_b = GpuCandidateBlock(exact_tail.start, exact_tail.anchor, exact_tail.candidates[:-1])
    exact_b_result = session.verify(prefix_b)
    if not _full(exact_b_result, prefix_b):
        raise RuntimeError("a prefix of the independently accepted B was rejected")
    session.install(exact_b_result)
    if exact_b_result.checkpoint.length != c.start or exact_b_result.tokens[-1] != c.anchor:
        raise RuntimeError("the exact-B control is not aligned with replayed C")
    exact_c, exact_inner = session.generate(exact_b_result.tokens[-1], loops=4)
    experiment2["exact_b"] = dict(aligned_cpu_prefix=_cpu_row(exact_b_result, prefix_b),
        full_b_acceptance_evidence="replay_old_b.b_cpu",
        c=_block_row(session, exact_c, exact_inner), c_quality=_quality(session, exact_c))
    experiment2["paired_c"] = _aligned_quality(session, c, exact_c)
    experiment2["paired_c"].update(first_name="replayed_b_c", second_name="exact_b_c")
    return result
