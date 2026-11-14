import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
from deepseek_vl2_runtime import Context, Lookup, POLICIES, Policy, TableLayer, decode_mask, merge_routes, prefill_mask


class RuntimeTests(unittest.TestCase):
    def table_layer(self, policy, length):
        class Gate(nn.Module):
            def forward(self, x):
                return torch.arange(6).repeat(len(x.reshape(-1, 4)), 1), torch.tensor([[.1, .5, .2, .6, .3, .4]]).repeat(len(x.reshape(-1, 4)), 1), None

        class Expert(nn.Module):
            def __init__(self, expert):
                super().__init__()
                self.expert = expert

            def forward(self, x):
                return torch.full_like(x, self.expert + 1)

        class Cache:
            registered_experts = {(1, expert): SimpleNamespace(offloaded=expert != 0) for expert in range(6)}

            def load_experts(self, *uids, **kwargs):
                for uid in uids:
                    yield uid, Expert(uid[1])

        class EmptyLookup:
            def online(self, x, ids, reuse, outputs):
                return torch.zeros_like(reuse)

            def offline_search(self, x, ids, modality, reuse, outputs, prefill):
                return torch.zeros_like(reuse)

            def update(self, *args):
                pass

        def plan(ids, mask):
            experts = ids[mask].unique().tolist()
            return experts, {expert: torch.where((ids == expert) & mask) for expert in experts}

        context = Context()
        context.vision = torch.zeros(1, length, dtype=torch.bool)
        original = SimpleNamespace(gate=Gate(), shared_experts=lambda x: torch.zeros_like(x), experts=Cache(),
                                   layer_id=1, top_k=6, num_routed_experts=6, _plan_masked_slot_execution=plan)
        return TableLayer(original, policy, context, EmptyLookup()), context

    def test_full_expert_table_layer_matches_direct_weighted_expert_outputs(self):
        from table_moe.utils.modality import ModalityContext

        layer, context = self.table_layer(Policy('tablemoe_full', 1., 6), 3)
        ModalityContext.set_attn_weights(torch.ones(3) / 3)
        output = layer(torch.ones(1, 3, 4).bfloat16())
        expected = torch.full_like(output, sum(weight * (expert + 1) for expert, weight in enumerate((.1, .5, .2, .6, .3, .4))))
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        self.assertEqual(context.counters['prefill_exact_routes'], 18)
        self.assertEqual(context.counters['prefill_dropped_routes'], 0)
        ModalityContext.clear()

    def test_table_forward_audits_both_sides_of_the_staged_boundary(self):
        layer, context = self.table_layer(POLICIES['tablemoe_06_top4_512_top6'], 1)
        context.output_position = 512
        early = layer(torch.ones(1, 1, 4).bfloat16())
        context.output_position = 513
        late = layer(torch.ones(1, 1, 4).bfloat16())
        self.assertTrue((late > early).all())
        self.assertEqual(context.counters['decode_h_4_layer_calls'], 1)
        self.assertEqual(context.counters['decode_h_6_layer_calls'], 1)
        self.assertEqual(context.counters['decode_exact_routes'], 11)

    def test_staged_policy_switches_after_output_512(self):
        policy = POLICIES['tablemoe_06_top4_512_top6']
        self.assertEqual([policy.decode_h(position) for position in (1, 511, 512, 513, 8192)], [4, 4, 4, 6, 6])
        self.assertEqual(policy.alpha, .6)

    def test_decode_uses_router_weights_and_residency_before_load(self):
        weights = torch.tensor([[.1, .5, .2, .6, .3, .4]])
        resident = torch.tensor([[True, False, False, False, False, False]])
        keep = decode_mask(weights, 4, resident)
        self.assertEqual(keep.tolist(), [[True, True, False, True, True, True]])
        self.assertTrue(decode_mask(weights, 6, resident).all())

    def test_prefill_expert_importance_uses_actual_attention_and_floor(self):
        ids = torch.arange(6).view(3, 2)
        weights = torch.ones(3, 2)
        attention = torch.tensor([.1, .8, .1])
        keep = prefill_mask(ids, weights, attention, 6, .6)
        self.assertEqual(int(keep.sum()), 3)
        self.assertTrue(keep[1].all())
        with self.assertRaises(ValueError):
            prefill_mask(ids, weights, None, 6, .6)

    def test_lookup_matches_nearest_hidden_row_before_matching_experts(self):
        lookup = Lookup.__new__(Lookup)
        lookup.size = 2
        lookup.online_keys = torch.tensor([[1., 0.], [.99, .1]], dtype=torch.bfloat16)
        lookup.online_ids = torch.tensor([[1, 2], [3, 4]])
        lookup.online_values = torch.ones(2, 2, 2, dtype=torch.bfloat16)
        output = torch.zeros(1, 2, 2, dtype=torch.bfloat16)
        hit = lookup.online(torch.tensor([[1., 0.]], dtype=torch.bfloat16), torch.tensor([[1, 4]]),
                            torch.ones(1, 2, dtype=torch.bool), output)
        self.assertEqual(hit.tolist(), [[True, False]])
        self.assertTrue(output[0, 0].eq(1).all())
        self.assertTrue(output[0, 1].eq(0).all())

    def test_offline_vision_prefill_and_decode_use_distinct_thresholds(self):
        lookup = Lookup.__new__(Lookup)
        identity = torch.eye(2).bfloat16()
        lookup.pca = {0: (identity, torch.zeros(2).bfloat16()), 1: (identity, torch.zeros(2).bfloat16())}
        lookup.offline = {(0, 0): (torch.tensor([[.65, math.sqrt(1-.65**2)]]), torch.tensor([[3., 4.]]).bfloat16())}
        x, ids = torch.tensor([[1., 0.]]).bfloat16(), torch.tensor([[0, 1]])
        outputs = torch.zeros(1, 2, 2).bfloat16()
        reuse = torch.ones(1, 2, dtype=torch.bool)
        vision = torch.zeros(1, dtype=torch.long)
        hit = lookup.offline_search(x, ids, vision, reuse, outputs, True)
        self.assertEqual(hit.tolist(), [[True, False]])
        torch.testing.assert_close(outputs[0, 0], torch.tensor([3., 4.]).bfloat16())
        self.assertFalse(lookup.offline_search(x, ids, vision, reuse, outputs, False).any())

    def test_route_merge_preserves_original_weights(self):
        outputs = torch.tensor([[[2., 3.], [0., 0.], [4., 5.]]]).bfloat16()
        weights = torch.tensor([[.4, .5, 1.1]])
        expected = (outputs.float() * weights[:, :, None]).sum(1).bfloat16()
        torch.testing.assert_close(merge_routes(outputs, weights), expected, atol=0, rtol=0)

    def test_adapmoe_uses_native_top5_selection_and_gate_normalization(self):
        from deepseek_vl2.models.configuration_deepseek import DeepseekV2Config
        from deepseek_vl2.models.modeling_deepseek import MoEGate

        config = DeepseekV2Config(hidden_size=4, n_routed_experts=6, num_experts_per_tok=6,
            topk_method='noaux_tc', n_group=1, topk_group=1, scoring_func='sigmoid', norm_topk_prob=True,
            routed_scaling_factor=2.)
        gate = MoEGate(config).eval()
        with torch.no_grad():
            gate.weight.copy_(torch.arange(24).view(6, 4).float() / 24)
            gate.e_score_correction_bias.copy_(torch.tensor([.6, -.2, .4, -.1, .2, 0.]))
        gate.top_k = 5
        for length in (1, 3):
            x = torch.ones(1, length, 4)
            ids, weights, _ = gate(x)
            scores = torch.sigmoid(x.reshape(-1, 4) @ gate.weight.T)
            expected = (scores + gate.e_score_correction_bias).topk(5, dim=-1).indices
            self.assertEqual(ids.shape[-1], 5)
            torch.testing.assert_close(ids.sort(-1).values, expected.sort(-1).values, atol=0, rtol=0)
            torch.testing.assert_close(weights.sum(-1), torch.full((length,), 2.))


if __name__ == '__main__':
    unittest.main()
