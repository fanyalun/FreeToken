"""A k+1-row verify step commits exactly what k+1 decode steps would: with drafts taken from a plain greedy run and wrong at a different place each step (so every accept count occurs), speculative decoding must reproduce that run token for token.

Tiny hybrid models with dummy weights, served end to end through the scheduler: qwen3_5 (GatedDeltaNet + attention) and qwen4_exp (GatedDeltaNet + QSA, PLE, hyper-connections). The drafts come from a stub proposer, so this pins the target side alone; a second run drafts with a DFlash 2 draft of random weights, whose wrong drafts must leave the target's output alone through its ring, window and graphs.
A verify row and a decode step run different GEMM and attention shapes, so their logits can differ by an ulp: the runs may part at a near tie, and the test checks they part only there. fp16 keeps such ties rare (bf16 has 3 fewer mantissa bits), and fan-in scaled dummy weights keep the activations O(1), so ulp noise does not grow through the layers.
"""

import json
import math
import os
import shutil
import subprocess
import sys

import pytest
import torch

TOKENIZER_SOURCE = os.environ.get("FREETOKEN_QWEN36_MODEL", "")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not os.path.exists(os.path.join(TOKENIZER_SOURCE, "tokenizer.json")), reason="FREETOKEN_QWEN36_MODEL not set"),
]

K = 3

_SCRIPT = """
import json, os, sys, torch
from freetoken.core import SamplingParams
from freetoken.engine import engine as E, sample as S
from freetoken.llm import LLM
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.spec import PROPOSERS
from freetoken.spec import runner as R
from freetoken.spec.proposer import Proposer, SpecProposal

model, out, reference, kwargs = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
TAPS, H = (1, 2), 256

# (uid, output index) -> top-2 token ids and logits of the row that picked it
top2 = {}
current = {}
# the tapped layers' outputs as the sink got them, and as the next layer read them
taps_got, taps_ref = {}, {}
taps = {"prefill": 0, "verify": 0, "mismatch": 0, "width": None}
steps = {"verify": 0}


def check_taps(batch):
    if not isinstance(batch.features, RecordingSink):
        return
    n, width = batch.input_ids.shape[0], taps["width"]
    taps["prefill" if batch.is_prefill else "verify"] += 1
    taps["mismatch"] += sum(not torch.equal(taps_got[i][:n, :width], taps_ref[i][:n, :width]) for i in TAPS)


def watch(cls, name):
    original = getattr(cls, name)

    def wrapped(self, batch, *args):
        current["batch"] = batch
        result = original(self, batch, *args)
        check_taps(batch)
        steps["verify"] += batch.phase == "verify"
        return result

    setattr(cls, name, wrapped)


watch(E.Engine, "forward_batch")
watch(R.SpecRunner, "verify_step")


class RecordingSink:
    # copies the tapped layers' outputs into persistent buffers, so a captured verify graph writes them too
    wanted_layers = frozenset(TAPS)

    def final_hidden(self, x):
        pass

    def layer_output(self, layer_id, x, residual=None, num_streams=1):
        if residual is not None:
            x = x + residual
        taps["width"] = x.shape[1]
        taps_got[layer_id][: x.shape[0], : x.shape[1]].copy_(x)


def tap_next_layer_inputs(model):
    # the reference for layer i's output is what layer i+1 reads: the residual its fused add forms (qwen3_5), or its input streams (qwen4_exp)
    layers = model.model.layers.op_list
    for i in TAPS:
        taps_got[i] = torch.zeros(4096, 4 * H, dtype=torch.float16, device="cuda")
        taps_ref[i] = torch.zeros(4096, 4 * H, dtype=torch.float16, device="cuda")
    if hasattr(layers[0], "input_layernorm"):
        cls, by_id = type(layers[0].input_layernorm), {id(layers[i + 1].input_layernorm): i for i in TAPS}
        original = cls.forward

        def wrapped(self, x, residual=None):
            x, residual = original(self, x, residual)
            if (i := by_id.get(id(self))) is not None:
                taps_ref[i][: x.shape[0], : x.shape[1]].copy_(residual)
            return x, residual
    else:
        cls, by_id = type(layers[0]), {id(layers[i + 1]): i for i in TAPS}
        original = cls.forward

        def wrapped(self, hidden, batch):
            if (i := by_id.get(id(self))) is not None:
                taps_ref[i][: hidden.shape[0], : hidden.shape[1]].copy_(hidden)
            return original(self, hidden, batch)

    cls.forward = wrapped


def record(rows, logits):
    top = logits.float().topk(2, dim=-1)
    for key, ids, vals in zip(rows, top.indices.tolist(), top.values.tolist()):
        if key is not None:
            top2[key] = ids + vals


def decode_rows(batch):
    # sampling runs after complete_one: the token lands at device_len - 1
    return [None if isinstance(r, ChunkedReq) else (r.uid, r.device_len - 1 - r.prompt_len) for r in batch.reqs]


def verify_rows(batch, width):
    # row j of a request predicts position cached_len + j + 1
    return [(r.uid, r.cached_len + j + 1 - r.prompt_len) for r in batch.reqs for j in range(width)]


sample, sample_verify = S.Sampler.sample, S.Sampler.sample_verify
S.Sampler.sample = lambda self, logits, args: (record(decode_rows(current["batch"]), logits), sample(self, logits, args))[1]
S.Sampler.sample_verify = lambda self, logits, args, drafts: (
    record(verify_rows(current["batch"], drafts.shape[1] + 1), logits), sample_verify(self, logits, args, drafts))[1]


class OracleProposer(Proposer):
    @classmethod
    def num_steps(cls, config):
        return 3, 8

    def __init__(self, config, target_model):
        super().__init__(config, target_model)
        self.reference = json.load(open(reference))
        self.steps = 0

    def feature_sink(self, batch):
        return RecordingSink()

    def propose(self, reqs, rows, pending):
        drafts = []
        for req in reqs:
            done = req.cached_len - req.prompt_len  # outputs committed before the pending token at cached_len
            want = self.reference["outputs"][req.uid][done + 1 : done + 1 + self.k]
            want = want + [0] * (self.k - len(want))
            wrong = self.steps % (self.k + 1)  # k: every draft right
            if wrong < self.k:
                want[wrong] = (want[wrong] + 1) % 1000
            drafts.append(want)
        self.steps += 1
        return SpecProposal(torch.tensor(drafts, dtype=torch.int64, device=rows.device))


def fan_in_scaled(tensor):
    # unit-variance dummy matrices grow the activations by sqrt(fan_in) per layer, until fp16 noise decides the argmax
    if tensor.is_floating_point() and tensor.dim() >= 2:
        tensor.mul_(tensor.shape[-1] ** -0.5)
    return tensor


dummy_state = E._make_dummy_weight_state_dict
E._make_dummy_weight_state_dict = lambda *args, **kwargs: {
    name: fan_in_scaled(tensor) for name, tensor in dummy_state(*args, **kwargs).items()}
create_model = E.create_model


def create_model_with_ple_rows(*args, **kwargs):
    # imported here, once CUDA is up
    from freetoken.models.qwen4_exp import ple
    from freetoken.moe import expert_banks

    dummy_fill = expert_banks._dummy_fill

    def scaled_fill(role, tensor):
        dummy_fill(role, tensor)
        if not role.endswith(("_scale", "_global")):
            fan_in_scaled(tensor)

    expert_banks._dummy_fill = scaled_fill

    # seeded PLE rows in place of the dummy zeros, so the PLE path carries signal
    class RandomTable(ple.GpuResidentTable):
        def __init__(self, num_rows, head_dim, dtype=torch.bfloat16):
            generator = torch.Generator(device="cuda").manual_seed(0)
            super().__init__(torch.randn(num_rows, head_dim, generator=generator, device="cuda", dtype=dtype))

    ple.ZeroTable = RandomTable
    model = create_model(*args, **kwargs)
    if reference != "-":
        tap_next_layer_inputs(model)
    return model


E.create_model = create_model_with_ple_rows
if reference != "-":
    PROPOSERS.register("oracle")("__main__:OracleProposer")
llm = LLM(model_path=model, dtype=torch.float16, use_dummy_weight=True, max_running_req=4, **kwargs)
prompts = [llm.tokenizer.encode(p) for p in ("The capital of France is", "1, 2, 3, 4,", "def fibonacci(n):")]
params = [SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, presence_penalty=p, frequency_penalty=p)
          for n, p in ((48, 0.0), (37, 0.6), (41, 0.0))]
res = llm.generate(prompts, params)
pool = llm.engine.linear_state_pool
window = list(pool.slot_state("dflash_window").shape) if pool.has_slot_state("dflash_window") else None
json.dump({"outputs": [r["token_ids"] for r in res], "top2": [[u, i, *t] for (u, i), t in top2.items()], "verify_steps": steps["verify"],
           "taps": taps, "num_slots": pool.num_slots, "window": window}, open(out, "w"))
"""


def _tiny_model(path: str, family: str) -> str:
    os.makedirs(path, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if os.path.exists(os.path.join(TOKENIZER_SOURCE, name)):
            shutil.copy(os.path.join(TOKENIZER_SOURCE, name), path)
    text = {
        "vocab_size": 248320, "hidden_size": 256, "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "full_attention_interval": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 32, "linear_num_key_heads": 2,
        "linear_num_value_heads": 4, "linear_value_head_dim": 32, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "max_position_embeddings": 4096, "tie_word_embeddings": False, "eos_token_id": 248044,
    }
    if family == "qwen3_5":
        config = {
            "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text", **text,
            "intermediate_size": 512, "head_dim": 64, "attn_output_gate": True,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                                "mrope_section": [3, 3, 2], "mrope_interleaved": True},
        }
    else:
        # the shipping QSA geometry (head_dim 256, index head_dim 128, ratio 4), scaled-down widths
        text.update({
            "model_type": "qwen4_exp_text", "head_dim": 256,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                                "mrope_section": [11, 11, 10], "mrope_interleaved": True},
            "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64,
            "output_gate_type": "sigmoid", "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128,
            "indexer_budget": 2048, "indexer_compress_ratio": 4, "hc_count": 4, "hc_lowrank": 16,
            "ple_layer_ids": [2], "ple_embed_dim": 64, "ple_conv_kernel_size": 4, "ngram_size": 3, "heads_per_ngram": 2,
            "ngram_vocab_size_base": 1000, "make_ngram_vocab_size_divisible_by": 8, "split_ngram_parts": 4,
        })
        config = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
                  "tie_word_embeddings": False, "text_config": text}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    return path


def _tiny_draft(path: str) -> str:
    """A DFlash 2 draft of the tiny targets: two layers over target layers 1 and 2, a 16-position window, random fan-in scaled weights."""
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    hidden, layers, heads, kv_heads, head_dim, inter, group = 256, 2, 4, 2, 64, 512, 16
    config = {
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "is_causal": False, "hidden_size": hidden,
        "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": kv_heads, "head_dim": head_dim,
        "intermediate_size": inter, "hidden_act": "silu", "rms_norm_eps": 1e-6, "vocab_size": 248320, "num_target_layers": 4,
        "layer_types": ["sliding_attention"] * layers, "use_sliding_window": True, "sliding_window": 16, "max_position_embeddings": 4096,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "dflash_config": {"block_size": 8, "conv_kernel_size": 2, "conv_group_size": group, "mask_token_id": 248300,
                          "selector_rank": 8, "selector_top_k": 4, "target_layer_ids": [1, 2]},
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    shapes = {"fc.weight": (hidden, 2 * hidden), "hidden_norm.weight": (hidden,), "norm.weight": (hidden,),
              "candidate_selector.hidden_projection.weight": (8, hidden),
              "candidate_selector.predecessor_codebook": (248320, 8), "candidate_selector.successor_codebook": (248320, 8)}
    for i in range(layers):
        p = f"layers.{i}"
        shapes.update({
            f"{p}.self_attn.q_proj.weight": (heads * head_dim, hidden), f"{p}.self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
            f"{p}.self_attn.v_proj.weight": (kv_heads * head_dim, hidden), f"{p}.self_attn.o_proj.weight": (hidden, heads * head_dim),
            f"{p}.self_attn.q_norm.weight": (head_dim,), f"{p}.self_attn.k_norm.weight": (head_dim,),
            f"{p}.mlp.gate_proj.weight": (inter, hidden), f"{p}.mlp.up_proj.weight": (inter, hidden), f"{p}.mlp.down_proj.weight": (hidden, inter),
            f"{p}.input_layernorm.weight": (hidden,), f"{p}.post_attention_layernorm.weight": (hidden,),
        })
        for conv in ("attention_conv", "mlp_conv"):
            shapes.update({f"{p}.{conv}.base_kernel": (2, 2, hidden), f"{p}.{conv}.kernel_projection.weight": (4 * hidden // group, hidden)})
    g = torch.Generator().manual_seed(0)
    weights = {}
    for name, shape in shapes.items():
        t = torch.randn(*shape, generator=g)
        weights[name] = (t * shape[-1] ** -0.5 if len(shape) > 1 else 1 + 0.1 * t).to(torch.bfloat16)
    save_file(weights, os.path.join(path, "model.safetensors"))
    return path


def _run(model: str, out: str, reference: str, **kwargs) -> dict:
    run = subprocess.run([sys.executable, "-c", _SCRIPT, model, out, reference, json.dumps(kwargs)],
                         capture_output=True, text=True, timeout=900, env=dict(os.environ))
    assert run.returncode == 0, run.stderr[-3000:]
    with open(out) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def plain_runs(tmp_path_factory):
    # the plain greedy run of each (family, backend, graphs), shared by both algorithms: model dir, output path, result
    runs = {}

    def get(family, backend, graphs, kwargs):
        if (family, backend, graphs) not in runs:
            root = tmp_path_factory.mktemp(f"plain-{family}-{backend}-{graphs}")
            model = _tiny_model(str(root / "model"), family)
            runs[family, backend, graphs] = model, str(root / "decode.json"), _run(model, str(root / "decode.json"), "-", **kwargs)
        return runs[family, backend, graphs]

    return get


@pytest.mark.parametrize("family, backend, algorithm, graphs", [
    *[(f, b, "oracle", g) for f, b in [("qwen3_5", "fa"), ("qwen3_5", "triton"), ("qwen4_exp", "qsa_sparse")] for g in (True, False)],
    # the draft side once per family, with graphs
    ("qwen3_5", "triton", "dflash", True), ("qwen4_exp", "qsa_sparse", "dflash", True),
])
def test_verify_steps_reproduce_greedy_decode(tmp_path, plain_runs, family, backend, algorithm, graphs):
    if backend == "fa":
        pytest.importorskip("sgl_kernel")
    # resident experts take the model dtype
    kwargs = {"attention_backend": backend, "cuda_graph_max_bs": 4 if graphs else 0, "moe_strategy": "fused"}
    model, decode, reference = plain_runs(family, backend, graphs, kwargs)
    draft = {"speculative_draft_model_path": _tiny_draft(str(tmp_path / "draft"))} if algorithm == "dflash" else {}
    spec = _run(model, str(tmp_path / "spec.json"), decode, speculative_algorithm=algorithm, speculative_num_steps=K, **draft, **kwargs)
    agreed = 0
    for uid, (want, got) in enumerate(zip(reference["outputs"], spec["outputs"])):
        assert len(got) == len(want)
        split = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), len(want))
        agreed += split
        if split < len(want):
            # same prefix, so both runs must rank the two tokens top-2 and within two fp16 ulps of each other
            pair = {want[split], got[split]}
            for run in (reference, spec):
                a, b, va, vb = next(t[2:] for t in run["top2"] if t[:2] == [uid, split])
                ulp = 2.0 ** (math.floor(math.log2(abs(va) or 1.0)) - 10)
                assert {a, b} == pair and va - vb <= 2 * ulp, f"request {uid} parts at {split}: top-2 {(a, b)} gap {va - vb}"
    # most of the run must still be compared, or a tie early in every request would hide a bug
    assert agreed >= 0.5 * sum(map(len, reference["outputs"]))
    assert spec["verify_steps"] > 0
    if algorithm == "dflash":
        # the draft's context window rides the target's GDN slots: per slot, rows of the draft width covering its 16-position window and a block
        window = spec["window"]
        assert window is not None and window[0] == spec["num_slots"] and window[1] >= 16 + K + 1 and window[2] == 256
    else:
        # every prefill and verify forward handed the sink each tapped layer's output, as the next layer read it
        taps = spec["taps"]
        assert taps["prefill"] > 0 and taps["verify"] > 0 and taps["mismatch"] == 0


@pytest.mark.needs_weights
@pytest.mark.parametrize("topk, graph, strategy, skip_last, graph_widths, preverify_map", [
    (1, "eager", "offload", False, None, False),
    (4, "eager", "offload", False, None, False),
    (1, "full", "offload", False, None, False),
    (1, "full", "hybrid", False, None, False),
    (1, "eager", "hybrid", False, None, False),
    (4, "eager", "hybrid", False, None, False),
    (1, "full", "offload", True, None, False),
    (1, "full", "offload", False, [24], False),
    (1, "full", "hybrid", False, None, True),
    (4, "eager", "hybrid", False, None, True),
])
def test_online_three_level_arrival_flush_and_request_isolation(tmp_path, topk, graph, strategy, skip_last, graph_widths, preverify_map):
    model = os.environ.get("FREETOKEN_TEST_MODEL")
    if not model:
        pytest.skip("FREETOKEN_TEST_MODEL must point to a Qwen3.5/3.6 MoE checkpoint with native MTP")
    script = r'''
import json, sys, torch, numpy as np
from freetoken.core import SamplingParams
from freetoken.llm import LLM
from freetoken.message import UserMsg, AbortBackendMsg, DetokenizeMsg
from freetoken.mm.config import MultimodalConfig

token_map = None
if sys.argv[8] == "True":
    from freetoken.utils.hf import load_tokenizer
    tokenizer = load_tokenizer(sys.argv[1])
    special = set(tokenizer.get_added_vocab().values()) | set(tokenizer.all_special_ids)
    ids = sorted(special | set(range(4096)), reverse=True)
    token_map = sys.argv[2] + ".map.bin"
    np.array(ids, dtype="<i4").tofile(token_map)
llm = LLM(sys.argv[1], speculative_mode="three_level_gpu", speculative_graph=sys.argv[4],
    speculative_token_map=token_map, speculative_preverify_token_map=token_map is not None,
    speculative_skip_last_preverify=sys.argv[6] == "True",
    speculative_graph_widths=json.loads(sys.argv[7]),
    speculative_depth=2, speculative_max_loops=3, max_running_req=4, max_extend_tokens=32,
    speculative_eagle_topk=int(sys.argv[3]),
    speculative_max_input_tokens=128, speculative_max_output_tokens=24, moe_strategy=sys.argv[5],
    moe_cpu_threads=18,
    gpu_memory_budget_bytes=20 * 2**30,
    mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))))
texts = ["Repeat exactly, without explanation: alpha bravo charlie delta echo foxtrot golf hotel india juliet.",
         "Write the numbers 1 through 10, separated by commas, without explanation.",
         "Explain why the sky appears blue during the day and red near sunset in simple words.",
         "Name the colors of the rainbow, separated by commas, without explanation."]
# Distinct answers exercise state isolation beyond a shared thinking preamble.
prompts = [llm.tokenizer.apply_chat_template([{"role": "user", "content": text}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False) for text in texts]
ids = [llm._tokenize_one(p) for p in prompts]
params = SamplingParams(max_tokens=16, ignore_eos=True)
runner = llm.engine.speculative_runner
graphs, use_graph, full_graph = runner.coarse_graphs, runner.use_graph, runner.full_graph
# Compare graph execution against independent eager request runs with identical weights.
runner.coarse_graphs, runner.use_graph, runner.full_graph = None, False, False
# The offline API removes a final EOS even under ignore_eos; compare raw messages.
reference, reference_messages = [], []
send_result = llm.send_result
def capture_reference(reply):
    reference_messages.extend(reply)
    send_result(reply)
llm.send_result = capture_reference
for prompt in prompts:
    reference_messages.clear()
    llm.generate([prompt], params)
    reference.append([t for m in reference_messages if isinstance(m, DetokenizeMsg) for t in m.next_tokens])
llm.send_result = send_result
runner.coarse_graphs, runner.use_graph, runner.full_graph = graphs, use_graph, full_graph
runner.graph_fallbacks.clear()
messages = []
send_result = llm.send_result
llm.send_result = messages.extend
runtime = llm.online_spec
gap_overlap = set()
runtime_call = runtime._call
def record_runtime_call(phase, *args, **kwargs):
    if runtime.gap_pending:
        assert phase in ("draft", "reconcile"), phase
        assert runner.three_level.experts.active
        gap_overlap.add(phase)
    return runtime_call(phase, *args, **kwargs)
runtime._call = record_runtime_call
observed_batches = {}
checked_heads = set()
original_target, original_draft = runner._target, llm.engine.mtp.forward
def record_target(batch, **kwargs):
    stage = getattr(batch, "online_spec_stage", None)
    if stage is not None:
        observed_batches[stage] = max(observed_batches.get(stage, 0), batch.size)
    assert kwargs.get("preverify", False) == (stage == "preverify"), (stage, kwargs)
    result = original_target(batch, **kwargs)
    if token_map and not torch.cuda.is_current_stream_capturing():
        head = llm.engine.model.lm_head
        selected = stage == "preverify"
        if selected:
            head, mapped_ids = head.kept_rows
        with llm.engine.ctx.forward_batch(batch):
            expected = head.greedy(result[1])
            if selected:
                expected = mapped_ids[expected.long()]
        assert torch.equal(result[0], expected.to(result[0].dtype))
        checked_heads.add(stage or "prefill")
    return result
def record_draft(ids, hidden):
    from freetoken.core import get_global_ctx
    batch = get_global_ctx().batch
    stage = getattr(batch, "online_spec_stage", None)
    if stage is not None:
        observed_batches[stage] = max(observed_batches.get(stage, 0), batch.size)
    return original_draft(ids, hidden)
runner._target, llm.engine.mtp.forward = record_target, record_draft
with torch.inference_mode(), llm.engine_stream_ctx:
    for uid in (0, 1):
        llm._process_one_msg(UserMsg(uid=uid, input_ids=ids[uid], sampling_params=params))
    for _ in range(100):
        runtime.step(llm)
        if len(runtime.requests) == 2 and all(s.loop_count > 0 for s in runtime.requests.values()):
            break
    else:
        raise AssertionError("both requests must enter a provisional loop before the arrival")
    llm._process_one_msg(UserMsg(uid=2, input_ids=ids[2], sampling_params=params))
    llm._process_one_msg(UserMsg(uid=3, input_ids=ids[3], sampling_params=params))
    runtime.step(llm)
    assert set(runtime.requests) == {0, 1, 2, 3}, list(runtime.requests)
    assert all(s.loop_count == 0 and s.window is None for s in runtime.requests.values())
    resumed = set()
    for _ in range(150):
        runtime.step(llm)
        resumed.update(uid for uid, s in runtime.requests.items() if s.loop_count > 0)
        if not runtime.requests:
            break
    assert not runtime.requests and not llm.prefill_manager.runnable
    assert resumed == {0, 1, 2, 3}
    outputs = [[t for m in messages if isinstance(m, DetokenizeMsg) and m.uid == uid
                for t in m.next_tokens] for uid in (0, 1, 2, 3)]
    assert outputs == reference, (outputs, reference)
    assert sum(isinstance(m, DetokenizeMsg) and m.finished for m in messages) == 4
    for phase in ("draft", "preverify", "verify", "reconcile"):
        assert observed_batches.get(phase, 0) >= 2, (phase, observed_batches)
    assert observed_batches["preverify"] == 4, observed_batches
    assert observed_batches["verify"] == 4, observed_batches

    # Abort with unverified pages live, then reuse the slot with another prompt.
    llm._process_one_msg(UserMsg(uid=4, input_ids=ids[2], sampling_params=params))
    for _ in range(100):
        runtime.step(llm)
        if runtime.requests[4].loop_count:
            break
    assert runtime.requests[4].loop_count > 0
    llm._process_one_msg(AbortBackendMsg(uid=4))
    llm._flush_abort_acks()
    assert not runtime.requests
    llm._process_one_msg(UserMsg(uid=5, input_ids=ids[1], sampling_params=params))
    for _ in range(100):
        runtime.step(llm)
        if not runtime.requests:
            break
    assert not runtime.requests
    reused = [t for m in messages if isinstance(m, DetokenizeMsg) and m.uid == 5 for t in m.next_tokens]
    assert reused == reference[1], (reused, reference[1])
    llm.engine.stream.synchronize()
    llm.cache_manager.check_integrity()
    assert llm.table_manager.available_size == 4
    assert len(llm.cache_manager.free_slots) == llm.engine.num_pages
    # Different EOS policies must survive request switching and graph-buffer reuse.
    eos_ids = llm.eos_token_ids
    llm.eos_token_ids = {reference[0][1]}
    llm._process_one_msg(UserMsg(uid=6, input_ids=ids[0],
        sampling_params=SamplingParams(max_tokens=16, ignore_eos=False)))
    llm._process_one_msg(UserMsg(uid=7, input_ids=ids[1], sampling_params=params))
    for _ in range(150):
        runtime.step(llm)
        if not runtime.requests and not llm.prefill_manager.runnable:
            break
    stopped = [t for m in messages if isinstance(m, DetokenizeMsg) and m.uid == 6 for t in m.next_tokens]
    ignored = [t for m in messages if isinstance(m, DetokenizeMsg) and m.uid == 7 for t in m.next_tokens]
    assert stopped == reference[0][:reference[0].index(reference[0][1]) + 1]
    assert ignored == reference[1]
    llm.eos_token_ids = eos_ids
llm.send_result = send_result
# Exact-token comparisons use copy/list prompts: BF16 rounding with padded
# shapes can flip a near-tied argmax on the open-ended explanatory prompt.
short_prompts = [prompts[i] for i in (0, 1, 0, 3)]
short = llm.generate(short_prompts, [SamplingParams(max_tokens=n, ignore_eos=True) for n in (1, 2, 5, 8)])
assert [len(r["token_ids"]) for r in short] == [1, 2, 5, 8]
# Compare against ordinary AR with the same ragged prefill schedule.
prepare = llm._prepare_batch
def ordinary_prepare(batch):
    batch.three_level_runtime = True
    return prepare(batch)
llm.online_spec, llm._prepare_batch = None, ordinary_prepare
try:
    short_reference = llm.generate(short_prompts,
        [SamplingParams(max_tokens=n, ignore_eos=True) for n in (1, 2, 5, 8)])
finally:
    llm.online_spec, llm._prepare_batch = runtime, prepare
assert [r["token_ids"] for r in short] == [r["token_ids"] for r in short_reference], (short, short_reference)
if graphs is not None:
    assert graphs.ready and graphs.replays
    assert not runner.graph_fallbacks, runner.graph_fallbacks
    assert not graphs.active_lanes
if sys.argv[4] == "full":
    for phase in ("draft", "preverify", "verify", "reconcile", "commit_preverify", "commit_verify"):
        assert runner.graph_replays.get(phase, 0) > 0, runner.graph_replays
    assert not runner.graph_fallbacks, runner.graph_fallbacks
if token_map:
    assert {"preverify", "verify", "prefill"} <= checked_heads, checked_heads
experts = runner.three_level.experts
if experts.auto_gap and experts.gap_capacity > 1:
    assert int(experts.request_transfer_bytes[3].item()) > 0
    assert "reconcile" in gap_overlap, gap_overlap
    if int(sys.argv[3]) == 1:
        assert "draft" in gap_overlap, gap_overlap
json.dump({"outputs": outputs, "resumed": sorted(resumed), "batches": observed_batches, "gap_prefetch_bytes": int(runner.three_level.experts.request_transfer_bytes[3].item()),
           "gap_overlap_stages": sorted(gap_overlap), "checked_heads": sorted(checked_heads),
           "replays": runner.graph_replays, "graph_shapes": [list(k) for k,v in runtime.forwards.items() if v.graph is not None]},
          open(sys.argv[2], "w"))
llm.engine.shutdown()
'''
    log = tmp_path / "model.log"
    with log.open("w") as output:
        run = subprocess.run([sys.executable, "-u", "-c", script, model, str(tmp_path / "online.json"), str(topk),
                              graph, strategy, str(skip_last), json.dumps(graph_widths), str(preverify_map)],
                             stdout=output, stderr=subprocess.STDOUT, text=True, timeout=3600, env=dict(os.environ))
    assert run.returncode == 0, log.read_text()[-7000:]
