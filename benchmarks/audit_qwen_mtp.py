from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading

import torch
from transformers import AutoTokenizer

from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.qwen_mtp import QwenMTPRunner
from freetoken.engine.speculative import Proposal, VerifyResult, SpeculativeFrontier, greedy_commit
from freetoken.engine.cache_budget import expert_bytes_per_slot
from freetoken.kvcache.base import CacheRebuildRejected
from freetoken.mm.config import MultimodalConfig
from freetoken.scheduler.config import SchedulerConfig


def state_snapshot(engine):
    pool = engine.linear_state_pool
    return pool.recurrent_states[:, 1].clone(), pool.conv_states[:, 1].clone()


def kv_snapshot(engine, layers, start, end):
    return torch.stack([
        torch.stack((engine.kv_cache.k_cache(i).flatten(0, 1)[start:end],
                     engine.kv_cache.v_cache(i).flatten(0, 1)[start:end]))
        for i in layers
    ]).clone()


def compare(name, actual, expected, *, exact=False):
    error = (actual.float() - expected.float()).abs()
    torch.testing.assert_close(actual, expected, rtol=0 if exact else 0.005, atol=0 if exact else 0.005)
    return {"state": name, "exact_required": exact, "max_abs": float(error.max().item())}


@torch.inference_mode()
def forced_acceptance(engine, runner, prompt, checks):
    pool = engine.linear_state_pool
    page = engine.config.page_size
    full_ids = engine.config.model_config.kv_cache_group_specs()[0].layer_ids
    with torch.cuda.stream(engine.stream):
        ids = torch.tensor(prompt, dtype=torch.int32)
        anchor = runner._prefill(ids, 128)
        length = len(prompt)
        initial_rec, initial_conv = state_snapshot(engine)
        before_prefix = kv_snapshot(engine, (*full_ids, engine.config.model_config.num_layers), 0, length)
        initial_seed, initial_hidden = runner.next_draft.clone(), runner.last_draft_hidden.clone()
        engine.moe_offload_cache.reset()
        outputs, hidden, states, target_kv = [], [], [], []
        token = anchor
        runner._reserve((length + 5 + page - 1) // page * page)
        for row in range(5):
            batch = runner._batch(token, length + row)
            predicted, h = runner._target(batch)
            outputs.append(predicted.clone())
            hidden.append(h.clone())
            states.append(state_snapshot(engine))
            target_kv.append(kv_snapshot(engine, full_ids, length, length + row + 1))
            token = predicted
        mtp_kv = []
        for row in range(4):
            batch = runner._batch(outputs[row], length + row)
            runner._draft(batch, hidden[row])
            mtp_kv.append(kv_snapshot(engine, (engine.config.model_config.num_layers,), length, length + row + 1))
        canonical = torch.cat(outputs)
        state = runner.states[4]
        state.enable_snapshot_reference()
        for accepted in range(4):
            engine.moe_offload_cache.reset()
            pool.recurrent_states[:, 1].copy_(initial_rec)
            pool.conv_states[:, 1].copy_(initial_conv)
            runner.next_draft, runner.last_draft_hidden = initial_seed.clone(), initial_hidden.clone()
            runner.frontier = SpeculativeFrontier(page, length, length, (length + page - 1) // page * page)
            runner.coordinator.prepare(runner.frontier, anchor, 3)
            drafts = canonical[:3].clone()
            if accepted < 3:
                drafts[accepted] = (drafts[accepted] + 1) % engine.config.model_config.vocab_size
            proposal = Proposal(anchor, drafts)
            predicted, h = runner._target(runner._batch(proposal.verify_inputs(), length, state=state))
            verified = VerifyResult(predicted, h)
            commit = greedy_commit(proposal, verified, remaining=4)
            if int(commit.accepted_drafts.item()) != accepted:
                raise AssertionError(f"forced acceptance {accepted} differs from greedy verification")
            runner.model_state.current = state
            runner.coordinator.commit(verified, commit)
            runner.last_event = torch.cuda.Event()
            runner.last_event.record(engine.stream)
            runner.last_event.synchronize()
            runner.coordinator.drain(runner.frontier, accepted + 1)
            row_checks = [
                compare("target_ssm_ar", pool.recurrent_states[:, 1], states[accepted][0]),
                compare("target_conv_ar", pool.conv_states[:, 1], states[accepted][1]),
                compare("target_kv_ar", kv_snapshot(engine, full_ids, length, length + accepted + 1), target_kv[accepted]),
                compare("mtp_kv_reconcile", kv_snapshot(engine, (engine.config.model_config.num_layers,), length, length + accepted + 1), mtp_kv[accepted]),
                compare("unchanged_prefix", kv_snapshot(engine, (*full_ids, engine.config.model_config.num_layers), 0, length), before_prefix, exact=True),
            ]
            snapshot_rec = torch.stack([state.snapshots[l.layer_id][0][accepted] for l in state.layers])
            snapshot_conv = torch.stack([state.snapshots[l.layer_id][1][accepted] for l in state.layers])
            row_checks += [compare("ssm_position_snapshot", pool.recurrent_states[:, 1], snapshot_rec, exact=True),
                           compare("conv_position_snapshot", pool.conv_states[:, 1], snapshot_conv, exact=True)]
            runner._reserve((length + accepted + 2 + page - 1) // page * page)
            next_token, _ = runner._target(runner._batch(canonical[accepted:accepted + 1], length + accepted + 1))
            assert int(next_token.item()) == int(canonical[accepted + 1].item())
            checks.append({"accepted_drafts": accepted, "committed_inputs": accepted + 1, "checks": row_checks})
            print(json.dumps({"accepted_drafts": accepted, "state_checks": "passed"}), flush=True)
        engine.stream.synchronize()
        state.snapshots = None
        engine.page_table[0].zero_()
        runner._reserved_len = 0
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("offload", "hybrid"), default="offload")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--page-size", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "complete.json").write_text('{"status":"incomplete"}\n')
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    row = json.loads(Path(args.data).read_text().splitlines()[0])
    prompt = tok.apply_chat_template([{"role": "user", "content": row["question"]}], tokenize=True,
                                     add_generation_prompt=True, enable_thinking=False, return_dict=False)
    engine = Engine(SchedulerConfig(
        model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        max_running_req=1, attention_backend="triton", moe_strategy=args.backend,
        moe_cache_auto=True, kv_reserve_tokens=4096, memory_ratio=0.9,
        max_seq_len_override=4096, gpu_memory_budget_bytes=16 << 30,
        speculative_mtp_depth=3, page_size=args.page_size,
        speculative_max_input_tokens=3840, speculative_max_output_tokens=256,
        cuda_graph_max_bs=0 if args.eager else 4, cache_type="naive", max_extend_tokens=128,
        mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))),
    ))
    runner = engine.speculative_runner
    reference = QwenMTPRunner(engine, reference_ar=True)
    evidence = {"config": vars(args), "checks": [], "strict_request_parity": False,
                "forced_window_expert_cache": "reset before reference and each candidate window"}
    try:
        evidence["forced_acceptance"] = []
        forced_acceptance(engine, runner, prompt, evidence["forced_acceptance"])
        baseline = reference.generate(prompt, max_tokens=32).token_ids
        eager_before = runner.use_graph
        runner.use_graph = False
        eager = runner.generate(prompt, max_tokens=32).token_ids
        runner.use_graph = eager_before
        actual = runner.generate(prompt, max_tokens=32).token_ids
        assert baseline == eager == actual, "baseline Engine.forward_batch/eager/native Graph differ"
        assert runner.frontier is None and runner.next_draft is None and runner.last_draft_hidden is None
        evidence["strict_request_parity"] = True
        evidence["token_ids"] = actual
        propose = runner.coordinator.proposer.propose
        rejection_round = 0
        def reject_every_round(anchor, depth):
            nonlocal rejection_round
            proposal = propose(anchor, depth)
            drafts = proposal.tokens.clone()
            if depth:
                drafts[0] = (baseline[rejection_round + 1] + 1) % engine.config.model_config.vocab_size
            rejection_round += 1
            return Proposal(anchor, drafts)
        runner.coordinator.proposer.propose = reject_every_round
        try:
            rejected = runner.generate(prompt, max_tokens=32)
            assert rejected.token_ids == baseline
            assert all(r.get("matched_drafts", r.get("accepted", 0)) == 0 for r in rejected.rounds)
            assert len(rejected.rounds) == 31
        finally:
            runner.coordinator.proposer.propose = propose
        evidence["checks"].append("31_consecutive_rejections_and_next_tokens")
        already_cancelled = threading.Event()
        already_cancelled.set()
        assert runner.generate(prompt, max_tokens=8, cancel=already_cancelled).token_ids == []
        assert not runner.active and not bool(engine.page_table[0].any().item())
        evidence["checks"].append("cancel_before_prefill")
        for cap in (1, 2, 3, 4, 5):
            assert runner.generate(prompt, max_tokens=cap).token_ids == baseline[:cap]
        for eos in baseline[:4]:
            end = baseline.index(eos) + 1
            assert runner.generate(prompt, max_tokens=8, eos_ids=(eos,)).token_ids == baseline[:end]
        for chunk in (32, 64, 128):
            ref = reference.generate(prompt, max_tokens=8, prompt_chunk_size=chunk).token_ids
            assert runner.generate(prompt, max_tokens=8, prompt_chunk_size=chunk).token_ids == ref
        evidence["checks"] += ["engine_ar_and_native_parity_32", "output_caps_1_to_5", "eos_first_four", "chunk_32_64_128"]
        for phase in ("chain", "verify", "reconcile"):
            cancel = threading.Event()
            execute = runner._execute
            def interrupted(p, batch, hidden=None, **kwargs):
                result = execute(p, batch, hidden, **kwargs)
                if p == phase:
                    cancel.set()
                return result
            runner._execute = interrupted
            try:
                cancelled = runner.generate(prompt, max_tokens=8, cancel=cancel)
                assert len(cancelled.token_ids) == 1 and not runner.active
                assert not bool(engine.page_table[0].any().item())
                assert runner.frontier is None and runner.model_state.current is None
            finally:
                runner._execute = execute
            assert runner.generate(prompt, max_tokens=8).token_ids == baseline[:8]
        evidence["checks"].append("phase_cancel_and_slot_reuse")
        old_slots = engine.moe_offload_cache.cache_size
        pool = engine.linear_state_pool
        per_state = sum(t[:, 0].numel() * t.element_size() for t in (pool.recurrent_states, pool.conv_states))
        delta = max(0, 4 - pool.num_slots) * per_state
        per_expert = expert_bytes_per_slot(engine.moe_offload_cache.bank_sources)
        freed_slots = (delta + per_expert - 1) // per_expert
        try:
            engine.rebuild_runtime_cache(num_pages=engine.num_pages, num_mamba_slots=3)
        except CacheRebuildRejected:
            assert engine.moe_offload_cache.cache_size == old_slots
            assert runner.generate(prompt, max_tokens=8).token_ids == baseline[:8]
            evidence["checks"].append("over_budget_rebuild_rejected_without_teardown")
            engine.rebuild_runtime_cache(
                moe_cache_size=old_slots - freed_slots, num_pages=engine.num_pages, num_mamba_slots=3,
            )
        runner = engine.speculative_runner
        assert runner.generate(prompt, max_tokens=8).token_ids == baseline[:8]
        evidence["checks"].append("kv_ssm_graph_rebuild")
        (out / "audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
        (out / "complete.json").write_text('{"status":"audit_completed","performance_matrix_complete":false}\n')
        print(json.dumps({"audit_completed": True, "checks": evidence["checks"]}), flush=True)
    finally:
        (out / "partial_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        runner.invalidate()
        reference.invalidate()
        engine.shutdown()


if __name__ == "__main__":
    main()
