import argparse
import json
from pathlib import Path
import tempfile
from threading import Event

import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import _make_dummy_weight_state_dict
from freetoken.kvcache.base import CacheRebuildRejected
from freetoken.models.qwen3_5_moe.mtp import QwenNativeMTP
from freetoken.mm.config import MultimodalConfig


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('fused', 'offload', 'hybrid'), required=True)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--hybrid-window-policy', choices=('balanced', 'legacy'), default='balanced')
    parser.add_argument('--hybrid-cpu-backend', choices=('grouped', 'legacy'), default='grouped')
    parser.add_argument('--three-level', action='store_true')
    parser.add_argument('--graph-implementation', choices=('reference', 'full'), default='reference')
    parser.add_argument('--scheduler', choices=('host', 'gpu'), default='gpu')
    parser.add_argument('--phases', choices=('verify', 'both'), default='verify')
    parser.add_argument('--small-prefetch', type=int, default=4)
    parser.add_argument('--verify-prefetch', type=int, default=0)
    parser.add_argument('--prefetch-gap', type=int, default=0)
    parser.add_argument('--slots', type=int, default=0)
    parser.add_argument('--budget-gib', type=int)
    parser.add_argument('--output-limit', type=int, default=8)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--draft-method', choices=('mtp', 'eagle3'), default='mtp')
    parser.add_argument('--modern', action='store_true')
    parser.add_argument('--compatibility-export', type=Path)
    parser.add_argument('--expect-pressure-error', action='store_true')
    parser.add_argument('--audit-lifecycle', action='store_true')
    parser.add_argument('--controller-lifecycle', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(123)
    with tempfile.TemporaryDirectory(prefix='baseline_sd_tiny_') as directory:
        root = Path(directory)
        text_config = dict(
            model_type='qwen3_5_moe_text', num_hidden_layers=args.layers,
            layer_types=['linear_attention', 'full_attention'] * (args.layers // 2),
            hidden_size=256, head_dim=128, num_attention_heads=2, num_key_value_heads=1,
            linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=128,
            linear_value_head_dim=128, linear_conv_kernel_dim=4, vocab_size=256, num_experts=8,
            num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
            max_position_embeddings=256, bos_token_id=1, eos_token_id=255, rms_norm_eps=1e-6,
            hidden_act='silu', attention_bias=False, attn_output_gate=True,
            mamba_ssm_dtype='float32', mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False,
            tie_word_embeddings=False, partial_rotary_factor=0.25,
            rope_parameters=dict(mrope_interleaved=True, mrope_section=[11, 11, 10],
                                 partial_rotary_factor=0.25, rope_theta=10000000, rope_type='default'))
        (root / 'config.json').write_text(json.dumps(dict(
            architectures=['Qwen3_5MoeForConditionalGeneration'], model_type='qwen3_5_moe',
            tie_word_embeddings=False, text_config=text_config)))
        draft_path = None
        if args.draft_method == 'eagle3':
            for key in ('rope_parameters', 'layer_types', 'partial_rotary_factor'):
                text_config.pop(key, None)
            text_config.update(model_type='qwen3_moe', architectures=['Qwen3MoeForCausalLM'],
                               intermediate_size=512, rope_theta=1000000.0, norm_topk_prob=True)
            (root / 'config.json').write_text(json.dumps(text_config))
            draft_path = root / 'draft'
            draft_path.mkdir()
            (draft_path / 'config.json').write_text(json.dumps(dict(
                architectures=['Eagle3DraftModel'], speculators_model_type='eagle3',
                norm_before_residual=True, draft_vocab_size=192,
                eagle_aux_hidden_state_layer_ids=[1, 2, args.layers],
                transformer_layer_config=dict(hidden_size=256, num_attention_heads=2,
                    num_key_value_heads=1, head_dim=128, intermediate_size=512, num_hidden_layers=1,
                    vocab_size=256, rms_norm_eps=1e-6, hidden_act='silu', rope_theta=10000.0,
                    max_position_embeddings=256))))
            from freetoken.models.eagle3 import Eagle3Draft
            def load_eagle_dummy(self, path, device):
                self.load_state_dict(_make_dummy_weight_state_dict(self.state_dict(), device=device))
                self._target_ids = torch.arange(self.config.draft_vocab_size, device=device) + 17
            Eagle3Draft.load_checkpoint = load_eagle_dummy

        class Config(EngineConfig):
            @property
            def distributed_addr(self):
                return 'file://' + str(root / 'rendezvous')

        def load_dummy(self, model_path, config, device):
            from freetoken.moe.expert_banks import attach_resident_banks, build_expert_banks

            self.load_state_dict(_make_dummy_weight_state_dict(self.state_dict(), device=device))
            experts = self.layers.op_list[0].mlp.experts
            banks = build_expert_banks(experts.quant_method, 1, (), device=device, dummy=True, resident=True)
            attach_resident_banks([experts], banks)

        QwenNativeMTP.load_checkpoint = load_dummy
        engine = Engine(Config(
            model_path=str(root), tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
            max_running_req=1, attention_backend='triton', moe_strategy=args.backend,
            gpu_memory_budget_bytes=args.budget_gib << 30 if args.budget_gib else None,
            moe_cache_auto=args.backend != 'fused' and args.slots == 0, moe_cache_size=args.slots,
            moe_cpu_threads=1, moe_hybrid_max_fetch=1,
            speculative_hybrid_window_policy=args.hybrid_window_policy,
            speculative_hybrid_cpu_backend=args.hybrid_cpu_backend,
            speculative_mtp_depth=3, speculative_max_input_tokens=129, speculative_max_output_tokens=args.output_limit,
            speculative_mode=('three_level_gpu' if args.three_level else 'sd') if args.modern else None,
            speculative_draft_method=args.draft_method,
            speculative_draft_model=str(draft_path) if draft_path else None,
            num_page_override=(129 + args.output_limit) if args.modern else (129 + args.output_limit + 3) // 4,
            max_seq_len_override=129 + args.output_limit,
            speculative_three_level=args.three_level,
            speculative_expert_scheduler=args.scheduler if args.three_level else 'host',
            speculative_execution_implementation='optimized',
            speculative_graph_implementation=args.graph_implementation,
            speculative_hybrid_phases=('preverify', 'verify') if args.phases == 'both' else ('verify',),
            speculative_prefetch=bool(args.small_prefetch or args.verify_prefetch),
            speculative_prefetch_layer_experts=args.small_prefetch,
            speculative_verify_prefetch_layer_experts=args.verify_prefetch,
            speculative_prefetch_gap_capacity=args.prefetch_gap,
            cuda_graph_max_bs=0 if args.eager else 4, page_size=1 if args.modern else 4, use_dummy_weight=True,
            mm=MultimodalConfig(disabled_encoders=frozenset(('vision', 'audio')))))
        try:
            runner = engine.speculative_runner
            control_reads = [0]
            if args.three_level and args.scheduler == 'gpu':
                experts = runner.three_level.experts
                original_drain = experts.drain_control
                def drain(rows):
                    control_reads[0] += 1
                    return original_drain(rows)
                experts.drain_control = drain
            graphs = set(runner.graphs)
            if args.backend == 'hybrid' and not args.three_level:
                from freetoken.layers.moe import _HYBRID_WINDOW
                assert engine.cpu_moe_executor.grouped_compat == (args.hybrid_cpu_backend == 'grouped')
                assert (engine.moe_offload_cache.baseline_hybrid is not None) == (
                    args.hybrid_window_policy == 'balanced' and _HYBRID_WINDOW)
            assert not engine.graph_runner.graph_map
            def snapshot():
                values = [engine.kv_cache._kv_buffer, *runner.state_adapter.tensors(engine.linear_state_pool)]
                if args.three_level:
                    values.extend(runner.state_adapter.tensors(runner.three_level.private_pool))
                return [value.clone() for value in values]

            def reset():
                engine.kv_cache._kv_buffer.zero_()
                runner.state_adapter.reset(runner.three_level.private_pool if args.three_level else None)
                if engine.moe_offload_cache:
                    engine.moe_offload_cache.reset()
                torch.cuda.synchronize()

            if args.audit_lifecycle:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
                prompt = list(range(9))
                reset()
                expected = engine.generate_speculative(prompt, max_tokens=4).token_ids
                if args.three_level:
                    from bench_speculative_hybrid import teacher_forced_audit
                    experts = runner.three_level.experts
                    saved_rows = experts.valid_rows
                    assert saved_rows is not None and saved_rows.item() < len(prompt) - 2
                    original_target = runner._target
                    observed = []
                    def target(batch, **kwargs):
                        if experts.active:
                            assert experts.valid_rows is None
                            observed.append(batch.size)
                        return original_target(batch, **kwargs)
                    runner._target = target
                    try:
                        audit = teacher_forced_audit(engine, prompt)
                    finally:
                        runner._target = original_target
                    assert observed == [audit["width"]] * 2
                    assert experts.final_history[:, :audit["width"]].ge(0).all()
                    assert experts.valid_rows is saved_rows
                else:
                    from freetoken.engine.qwen_mtp import _QwenReservation
                    with torch.cuda.stream(engine.stream):
                        runner._reserve(16)
                        runner.last_event = torch.cuda.Event()
                        runner.last_event.record(engine.stream)
                        _QwenReservation(runner).release(4, 16)
                        assert runner._reserved_len == 16
                    engine.stream.synchronize()
                    reset()
                    runner.baseline = False
                    try:
                        assert engine.generate_speculative(prompt, max_tokens=4).token_ids == expected
                    finally:
                        runner.baseline = True
                capacity = engine.num_pages * engine.config.page_size
                assert engine.page_table[0, :capacity].tolist() == list(range(capacity))
                reset()
                assert engine.generate_speculative(prompt, max_tokens=4).token_ids == expected
                print(json.dumps(dict(status="passed", audit_lifecycle=True, three_level=args.three_level)))
                return

            if args.controller_lifecycle:
                from freetoken.engine.qwen_three_level import ThreeLevelCoordinator
                original = runner.three_level
                assert original is not None and runner.coarse_graphs is None
                states = dict(runner.states)
                prompt = list(range(9))
                def canonical_state():
                    pool = engine.linear_state_pool
                    values = [pool.recurrent_states[:, 1], pool.conv_states[:, 1]]
                    full_ids = engine.config.model_config.kv_cache_group_specs()[0].layer_ids
                    for layer_id in full_ids:
                        values.extend(cache(layer_id).flatten(0, 1)[:len(prompt) + 7]
                                      for cache in (engine.kv_cache.k_cache, engine.kv_cache.v_cache))
                    return [value.clone() for value in values]
                reset()
                expected = engine.generate_speculative(prompt, max_tokens=8).token_ids
                reference_state = canonical_state()
                replacement = ThreeLevelCoordinator(runner)
                assert all(runner.states[w] is state for w, state in states.items())
                runner.three_level = replacement
                for prefetch in (False, True, False):
                    replacement.experts.prefetch = prefetch
                    reset()
                    result = engine.generate_speculative(prompt, max_tokens=8)
                    assert result.token_ids == expected
                    for actual, reference in zip(canonical_state(), reference_state):
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                original.invalidate()
                print(json.dumps(dict(status="passed", controller_lifecycle=True)))
                return

            if args.expect_pressure_error:
                from freetoken.moe.speculative_cache import NoEvictableSlot
                reset()
                try:
                    engine.generate_speculative([1, 2, 3], max_tokens=args.output_limit)
                except NoEvictableSlot:
                    assert not runner.active and runner.frontier is None
                    assert engine.moe_offload_cache.speculative_controller is None
                    assert len(engine.generate_speculative([1, 2, 3], max_tokens=1).token_ids) == 1
                    print(json.dumps(dict(status='expected_pressure_error', cleanup='passed')))
                    return
                raise AssertionError('the retained low-slot admission negative case did not reproduce')

            if args.compatibility_export:
                import hashlib
                records = []
                for length in (1, 63, 64, 65, 127, 128, 129):
                    reset()
                    result = engine.generate_speculative(list(range(length)), max_tokens=args.output_limit,
                                                         prompt_chunk_size=64)
                    records.append(dict(length=length, tokens=result.token_ids,
                        state_sha256=[hashlib.sha256(value.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                                      for value in snapshot()]))
                args.compatibility_export.write_text(json.dumps(records, indent=2))
                print(json.dumps(dict(status='exported', path=str(args.compatibility_export))))
                return

            coarse = runner.coarse_graphs
            captures = coarse.captures if coarse is not None else 0
            pointers = [runner.seed_hidden.data_ptr(), runner.seed_token.data_ptr(), runner.anchor_buffer.data_ptr()]
            request_outputs = []
            for length in (1, 63, 64, 65, 127, 128, 129):
                prompt = list(range(length))
                reset()
                control_reads[0] = 0
                a = engine.generate_speculative(prompt, max_tokens=args.output_limit)
                if args.three_level and args.scheduler == 'gpu':
                    assert control_reads[0] == sum(row['loop_count'] + 1 for row in a.rounds)
                state = snapshot()
                # Compare scheduling with identical residency and recency, hence placement.
                reset()
                b = engine.generate_speculative(prompt, max_tokens=args.output_limit, prompt_chunk_size=length)
                assert a.token_ids == b.token_ids, (length, a.token_ids, b.token_ids)
                if args.three_level:
                    def trajectory(result):
                        return [(row['accepted'], row['returned'], row['submit_rows'],
                                 [(loop['accepted'], loop['returned']) for loop in row['loops']],
                                 row.get('draft_tail'), row.get('draft_loop_count'))
                                for row in result.rounds]
                    assert trajectory(a) == trajectory(b)
                for before_state, after_state in zip(state, snapshot()):
                    torch.testing.assert_close(before_state, after_state, atol=0, rtol=0)
                request_outputs.append(a.token_ids)
                assert len(a.token_ids) == args.output_limit and len(b.token_ids) == args.output_limit
                assert set(runner.graphs) == graphs
                assert coarse is None or coarse.captures == captures
                assert pointers == [runner.seed_hidden.data_ptr(), runner.seed_token.data_ptr(), runner.anchor_buffer.data_ptr()]
                if args.three_level and runner.three_level.inner_width == 0:
                    assert all(row['loop_count'] == 0 for row in a.rounds)
                    assert all(not tensor.any() for tensor in runner.state_adapter.tensors(runner.three_level.private_pool))
                if args.three_level and args.scheduler == 'gpu' and args.slots == 16 and args.layers == 10:
                    conflicts = [row for row in a.rounds if row['cache_conflict']]
                    assert conflicts and all(row['loop_count'] == 1 for row in conflicts)
                    assert all(row['loops'][-1]['stop_after_loop'] == 'cache_conflict' for row in conflicts)
                    experts = runner.three_level.experts
                    assert not experts.cache_conflict.any() and not experts.pending.any()
                    assert not experts.gap_planned and not experts.gap_active
                assert not runner.active and runner.frontier is None
            for prompt, limit in ((list(range(130)), 1), ([1], args.output_limit + 1)):
                try:
                    engine.generate_speculative(prompt, max_tokens=limit)
                except ValueError:
                    pass
                else:
                    raise AssertionError('request exceeding an initialized limit must fail')
            stop = engine.generate_speculative([1, 2, 3], max_tokens=args.output_limit).token_ids[0]
            assert engine.generate_speculative([1, 2, 3], max_tokens=args.output_limit, eos_ids=(stop,)).token_ids == [stop]
            cancelled = Event()
            cancelled.set()
            assert not engine.generate_speculative([1, 2, 3], max_tokens=args.output_limit, cancel=cancelled).token_ids
            assert len(engine.generate_speculative([1, 2, 3], max_tokens=1).token_ids) == 1
            assert set(runner.graphs) == graphs
            before = engine.kv_cache._kv_buffer.data_ptr()
            try:
                engine.rebuild_runtime_cache(num_pages=1)
            except CacheRebuildRejected:
                pass
            else:
                raise AssertionError('undersized KV rebuild must fail before teardown')
            assert engine.kv_cache._kv_buffer.data_ptr() == before
            if engine.moe_offload_cache:
                full_cache = args.three_level and args.layers == 4 and (args.backend == 'offload' or args.scheduler == 'host')
                rebuilt_slots = 32 if full_cache else 16
                engine.rebuild_runtime_cache(moe_cache_size=rebuilt_slots)
                assert len(engine.generate_speculative([1, 2, 3], max_tokens=args.output_limit).token_ids) == args.output_limit
                assert engine.moe_offload_cache.cache_size == rebuilt_slots
            print(json.dumps(dict(status='passed', backend=args.backend, eager=args.eager,
                three_level=args.three_level, graph_implementation=engine.config.speculative_graph_implementation,
                draft_method=args.draft_method, mode=engine.config.speculative_mode,
                graph_replays=a.graph_replays, graph_fallbacks=a.graph_fallbacks,
                graph_captures=a.graph_captures,
                scheduler=args.scheduler, phases=args.phases, layers=args.layers, output_limit=args.output_limit,
                hybrid_window_policy=args.hybrid_window_policy, hybrid_cpu_backend=args.hybrid_cpu_backend,
                request_outputs=request_outputs,
                memory_plan=engine.baseline_sd_memory_plan.as_dict(),
                graphs=len(graphs))))
        finally:
            engine.shutdown()


if __name__ == '__main__':
    main()
