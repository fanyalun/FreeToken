# Vision2Web Level 1

`bench_vision2web.py` connects FreeToken model endpoints to the official Vision2Web
agent generation, screenshot capture, visual judge and result analyzer. It pins
[`zai-org/Vision2Web` at `d80a4a3d0ea7326541ad87314294269a633789e4`](https://github.com/zai-org/Vision2Web/tree/d80a4a3d0ea7326541ad87314294269a633789e4).
This is a separate benchmark entry point; it does not register a VLMEvalKit dataset.

Generation runs the official `ClaudeCodeAdapter` against FreeToken's Anthropic
Messages API. It preserves the official `WEBPAGE_PROMPT`, agent tools, retry policy
and output budgets. This measures an agent workflow, including tool calls and
multiple model requests. It is not a single HTML completion test.

The pinned official Level 1 evaluator handles empty functional workflows with
`playwright-cli` directly. It captures desktop, tablet and mobile screenshots at
each task's declared resolution and sends prototype/screenshot pairs to the
official `VisualScorer`. The official analyzer computes Visual Score. A local
judge changes the judge identity relative to the public leaderboard; comparisons
must hold that judge, prompts, agent framework and source version fixed.

## Configuration and preparation

The existing dataset is `/data/fanya/datasets/vision2web_level1`; its `webpage/`
subdirectory contains 100 projects. Use the parent directory as `datasets_dir`.
The official source is a separate, clean checkout at `/home/fanya/vision2web`.

Copy `benchmarks/vision2web_example.json` to a run-specific config and choose a
fresh `output`. The example uses all 100 tasks. For a smoke run, set
`"projects": ["abc"]` and use a separate output. Project selections, endpoint
settings, adapter source and official source are frozen for a run.

The interpreter needs the official Python requirements: `click`, `pyyaml`,
`openai`, `playwright` and `Pillow`. The existing
`/home/fanya/miniforge/envs/tablemoe/bin/python` supplies them. Preparation and
the doctor do not install packages, build images, start a model or run inference.

```bash
V2W_PYTHON=/home/fanya/miniforge/envs/tablemoe/bin/python
V2W_CONFIG=/absolute/path/to/run_config.json
"$V2W_PYTHON" benchmarks/bench_vision2web.py prepare --config "$V2W_CONFIG"
"$V2W_PYTHON" benchmarks/bench_vision2web.py doctor --config "$V2W_CONFIG"
```

Preparation copies the pinned source to `output/official_runtime/` and changes
only hard-coded temporary paths. Official prompts and scoring code are preserved;
`source_manifest.json` records both upstream and executed file fingerprints.
The sandbox binds both `/workspace` and `/v2w_runtime` to run-specific host
directories under `output/container_work/`. Host temporary files, container
temporary files, prompt files and deployment logs stay in persistent storage.
Each stage/project has a separate container name and backing directory.

The official sandbox image is required for generation and evaluation. The doctor
reports the exact build command when it is missing. Building installs Ubuntu
packages, Node, browsers and agent CLIs inside the image and consumes Docker
storage; obtain the user's installation/build authorization before running it.
The generated Dockerfile retains the official dependency versions and redirects
the Chrome package download out of `/tmp`.

## FreeToken endpoints

`generation.base_url` is an Anthropic API root, for example
`http://host.docker.internal:23840`, without `/v1`. Task containers receive the
Docker host-gateway mapping. Bind the FreeToken generation server to `0.0.0.0` so
the container can reach it. Keep the normal model tool-call parser enabled.

`evaluation.base_url` is a host-side OpenAI API root, for example
`http://127.0.0.1:23880/v1`. The visual judge can be a separate lossless FreeToken
service. The entry point does not launch, stop or reconfigure model servers.
This allows existing offload/TableMoE server settings to remain explicit.

The official Claude Code configuration uses a 200,000-token context window and
64,000 maximum output tokens; the visual scorer also requests up to 64,000 output
tokens. A server limited to 32,768 tokens is not a compatible configuration.
Provision model context and KV capacity accordingly. The runner does not silently
shorten the official prompt, context window or output budget.

Use a served model ID without `/`, `:` or spaces: the official framework uses it
as a directory component. Credentials are read from `api_key_env` and are not
written into the config or manifests. Local unauthenticated endpoints use `EMPTY`
if their key variable is unset. Set the variable explicitly for other endpoints.

## Generation, evaluation and reporting

```bash
"$V2W_PYTHON" benchmarks/bench_vision2web.py inference --config "$V2W_CONFIG"
"$V2W_PYTHON" benchmarks/bench_vision2web.py evaluate --config "$V2W_CONFIG"
"$V2W_PYTHON" benchmarks/bench_vision2web.py report --config "$V2W_CONFIG"
```

`all` runs the three stages in sequence. Generation saves the official conversation
and task result under `results/webpage/claude_code/MODEL/PROJECT/`. The root
`generation_prompt.txt` is the exact official seed prompt. `projects.json` saves
the selected workflows and asset paths/sizes. `judge_records/` retains the actual
image-bearing judge messages and the response returned after the official retry
loop. `official_analysis.json` comes from the official analyzer.

`complete.json` requires every selected task to have successful generation and
evaluation records, a deployment script, all three screenshots and valid nonempty
visual scores. Missing screenshots or empty judge responses must not disappear
from the denominator. The official analyzer output remains available for
diagnosis, but partial coverage exits nonzero and stays `complete=false`.
These records are quality results, not formal serving-throughput measurements.

Successfully completed stages can be reused with the identical configuration.
Upstream treats an existing `start.sh` as completed generation and any
`evaluation_result.json` as a finished evaluation; the wrapper checks actual
coverage and refuses to call those partial outcomes complete. Use a fresh output
for a failed/partial retry or any source, prompt, model or configuration change.
