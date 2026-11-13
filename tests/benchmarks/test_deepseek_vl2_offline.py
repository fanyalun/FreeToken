import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import build_deepseek_vl2_offline as runner
from freetoken.moe import tablemoe_artifacts as artifacts


class DeepseekOfflineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_prompt_retains_interleaving_and_empty_assistant(self):
        row = dict(images=[dict(path="a.png"), dict(path="b.png")], vlmeval_message=[
            dict(type="text", value="Write a story. "), dict(type="image", value="a.png"),
            dict(type="text", value="Then "), dict(type="image", value="b.png")])
        messages = runner.conversation(row)
        self.assertEqual(messages[0]["content"], "Write a story. <image>\nThen <image>\n")
        self.assertEqual(messages[0]["images"], ["a.png", "b.png"])
        self.assertEqual(messages[1]["content"], "")
        row["images"].reverse()
        with self.assertRaisesRegex(ValueError, "image order"):
            runner.conversation(row)

    def test_image_spans_preserve_adjacent_images_and_reject_gaps(self):
        self.assertEqual(runner.image_offsets(torch.tensor([0, 1, 1, 1, 1, 0]), [2, 2]), [(1, 3), (3, 5)])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            runner.image_offsets(torch.tensor([1, 0, 1]), [2])
        with self.assertRaisesRegex(ValueError, "coverage"):
            runner.image_offsets(torch.tensor([1, 1]), [1])

    def test_deduplication_distinguishes_native_crop_modes(self):
        one = dict(images=[dict(sha256="shared")])
        three = dict(images=[dict(sha256="shared"), dict(sha256="b"), dict(sha256="c")])
        self.assertNotEqual(runner.collection_image_ids(one)[0], runner.collection_image_ids(three)[0])

    def test_shared_image_masks_collection_without_removing_prompt_inputs(self):
        config = dict(dtype="bfloat16", sampling_policy="tablemoe_default", sample_count=2,
                      collection_image_ids=[["shared"], ["shared"]], export_inputs=False)
        collector = runner.NativeCollector(self.root, config, 4, 2, identity="test", first_moe_layer=1)
        try:
            for uid in range(2):
                sample = SimpleNamespace(input_ids=torch.tensor([10, 11, 12, 13]),
                                         images_seq_mask=torch.tensor([0, 1, 1, 0]), num_image_tokens=[2])
                batch = runner.collection_batch(sample, uid)
                for layer in (1, 3):
                    collector.snapshot(layer, torch.randn(4, 4).bfloat16(), torch.zeros(4, 1, dtype=torch.long), batch)
                collector.flush(uid)
                self.assertEqual(batch.padded_reqs[0].extend_len, 4)
            self.assertEqual(collector.tokens, dict(prefill=8, decode=0))
            self.assertEqual(collector.sample_state["counts"], dict(vision=2, text=4))
            self.assertEqual(collector.sample_state["records"][1]["duplicate_visual_tokens"], 2)
            self.assertEqual(collector.sample_state["records"][1]["selected"], dict(vision=0, text=2))
            self.assertEqual(set(collector.sampling), {"1", "3"})
            self.assertEqual(collector.sampling["1"], collector.sampling["3"])
        finally:
            collector.close()
        resumed = runner.NativeCollector(self.root, config, 4, 2, identity="test", first_moe_layer=1)
        try:
            self.assertEqual(resumed.collected_image_ids, {"shared"})
            self.assertEqual(resumed.requests, [0, 1])
        finally:
            resumed.close()

    def test_sparse_layer_pipeline_matches_native_expert_without_tensor_hashes(self):
        from deepseek_vl2.models.configuration_deepseek import DeepseekV2Config
        from deepseek_vl2.models.modeling_deepseek import DeepseekV2MLP

        model, assets = self.root / "model", self.root / "assets"
        model.mkdir()
        index, weights = {}, {}
        for layer in (1, 3):
            for expert in range(2):
                prefix = f"language.model.layers.{layer}.mlp.experts.{expert}."
                for name, shape in (("gate_proj", (3, 4)), ("up_proj", (3, 4)), ("down_proj", (4, 3))):
                    key = prefix + name + ".weight"
                    weights[key] = (torch.randn(shape) * .1).bfloat16()
                    index[key] = "model.safetensors"
        save_file(weights, model / "model.safetensors")
        artifacts.write_json(model / "model.safetensors.index.json", dict(weight_map=index))
        buckets, outputs = {}, []
        for layer in (1, 3):
            for mod in ("vision", "text"):
                key = f"L{layer}_E0_{mod}"
                folder = assets / "collect" / f"layer_{layer}"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{key}.bin"
                path.write_bytes(torch.randn(16, 4).bfloat16().view(torch.uint16).numpy().tobytes())
                outputs.append(path)
                buckets[key] = dict(layer=layer, expert=0, modality=mod, kept=16)
        artifacts.finish_stage(assets / "collect", dict(sample_count=1), outputs, hash_outputs=False,
                               buckets=buckets, hidden=4, layer_ids=[1, 3], num_layers=4, num_experts=2, dtype="bfloat16")
        config = dict(output=str(self.root), model=str(model), dtype="bfloat16", device="cpu", checkpoint={}, sources={},
                      builder=dict(dtype="bfloat16", pca_dim=2, clusters=3, iterations=3, value_chunk=4, cluster_batch_size=2))
        original = artifacts.file_hash

        def metadata_hash_only(path):
            self.assertNotIn(Path(path).suffix, (".pt", ".bin", ".phase"))
            return original(path)

        with patch.object(artifacts, "file_hash", metadata_hash_only):
            runner.cluster(config)
            runner.materialize(config)
            runner.cluster(config)
            runner.materialize(config)
        manifest = artifacts.verify_stage(assets / "cache", hash_outputs=False)
        self.assertEqual(len(manifest["pca_outputs"]), 4)
        self.assertEqual(len(manifest["missing_tables"]), 4)
        self.assertFalse((assets / "cache/layer_0").exists())
        for layer in (1, 3):
            native = DeepseekV2MLP(DeepseekV2Config(hidden_size=4, intermediate_size=3)).bfloat16()
            prefix = f"language.model.layers.{layer}.mlp.experts.0."
            native.load_state_dict({name: weights[prefix + name] for name in native.state_dict()})
            for mod in ("vision", "text"):
                name = f"L{layer}_E0_{mod}"
                centroids = torch.load(assets / "cluster" / f"layer_{layer}/{name}_clusters.pt", weights_only=True)["centroid"]
                table = torch.load(assets / "cache" / f"layer_{layer}/{name}_cache.pt", weights_only=True)
                with torch.no_grad():
                    expected = native(centroids)
                torch.testing.assert_close(table["value"], expected, rtol=.02, atol=1e-5)
                self.assertEqual(table["key"].shape[1], 2)
                self.assertEqual(table["value"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
