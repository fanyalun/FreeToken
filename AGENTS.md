# Instructions for AI coding agents

Read [CONTRIBUTING.md](CONTRIBUTING.md) first. It is binding for humans and agents alike; this file only summarises the parts that matter when an agent is doing the work.

## AI policy

AI-assisted code is welcome. Submitting code the contributor does not understand is not. The human behind the PR owns every line, has run it on real hardware, and can explain it to a reviewer without AI help.

Agents may commit and push changes when the user explicitly requests or authorizes them. Authorization already given in the session remains valid within that scope; do not ask for the same permission again.

Agents must not:

- Run `gh pr create`, `gh pr comment`, or `gh issue create` without explicit user instructions.
- Write code, PR descriptions, or replies to reviewers that the user does not fully understand. The user must be able to explain and defend every line without AI help.
- Report tests or benchmarks as run when they were not.

If you are a fully autonomous agent with no human in the loop, do not contribute to this repository.

## Calibration splits and shared images

- These rules apply to all future calibration, offline-table collection, and calibration/test splits. Fix and record the random seed before sampling; use seed 42 unless the user or benchmark protocol specifies another seed. Reuse frozen split records when resuming an existing experiment.
- Default to a 6:4 split: 60% calibration and 40% test, unless the user explicitly specifies another ratio. Reuse existing frozen splits.
- If the benchmark has categories, split calibration and test samples separately within each category using the fixed seed and the requested ratio or quota. Do not substitute a single unstratified split of the entire benchmark.
- Group samples sharing a document or any input image before splitting, including transitive overlaps, and keep each group entirely in one partition. Preserve this isolation when groups span categories; record any category-quota deviation required to keep shared images out of both partitions.
- Deduplicate visual calibration collection at the image level while retaining every selected sample's text. Within the same calibration run and image-preprocessing configuration, collect each shared image's visual-token records once; later samples collect their own text-token records and any previously unseen images.
- Example: if A and B have identical images but different questions, collect A's image and text records, then collect only B's text records. Do not discard B's question or treat the entire sample as a duplicate.
- Keep the full multimodal input needed to compute each sample correctly. Text-only collection for B means masking already collected visual positions in the collector, not removing B's images from the model input. Reusing rendered image files alone does not count as deduplicating calibration collection.
- Save the seed, category definitions, split ratio or quotas, sample/group membership, image-sharing map, and visual/text collection counts. Use dataset image/document identities and saved media provenance to resolve sharing; do not add prohibited full-content checksum scans of collected tensors or offline tables.

## Offline table checks

- These rules apply to collection, construction, resuming, loading, reuse, benchmarks, evaluations, and all future tests that use offline tables.
- Do not compute or compare full-file content checksums for raw collected inputs, exported tensors, PCA, clusters, or offline cache tables. This includes SHA-256 and other content hashes for `.bin`, `.phase`, `.pt`, and equivalent large artifact files.
- Check paths, file existence, completion records, required configuration metadata, file lengths, and tensor shapes/dtypes needed for computation. Do not add a separate full-content validation pass before or after a benchmark or test.
- Historical manifests may retain checksums; those fields must not trigger content scans or comparisons when reusing or testing tables.
- Read tensor contents when the algorithm or requested test needs them, without an additional integrity scan. Hashing source code and small configuration/record files remains allowed.

## Repository layout

The main subsystems:

```
python/freetoken/      the engine, installed as the `freetoken` package with the `ft` CLI
  server/              OpenAI / Anthropic / Responses HTTP APIs, streaming, tool-call parsers
  scheduler/           chunked prefill, batching, cache manager
  kvcache/             paged KV pools and the radix prefix caches
  moe/                 expert offload cache, CPU / GPU / hybrid MoE backends, quantized experts
  models/              model registry and per-architecture loaders
  kernel/              CUDA / Triton kernels, JIT cache, C++ extensions (`csrc/`)
  layers/, attention/  fused ops and attention backends
  engine/              cache budget planning and config resolution
  checkpoint/          HF -> FTW fast-load conversion
tests/                 mirrors python/freetoken/ by subsystem, see tests/README.md
benchmarks/            end-to-end and micro benchmarks, see benchmarks/README.md
docs/                  install, quickstart, CLI and model docs
freetoken-kernel-cache/ companion wheel of prebuilt kernels, see its README
scripts/               wheel build and release scripts
```

## Development

Linux x86_64 with an NVIDIA GPU. Use `uv`, not bare `pip`:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
uv run pytest tests/ -m "not slow"
```

CUDA kernels are JIT-compiled with `nvcc` on first use unless the prebuilt `freetoken-kernel-cache` wheel is installed. The C++ extensions under `python/freetoken/kernel/csrc/` are built by `setup.py`; after changing them run `python setup.py build_ext --inplace`.

Put a new test in the `tests/` directory that mirrors the module it protects, and extend an existing file before creating a new one. Bug fixes come with a test that fails before and passes after. Performance changes come with A/B numbers against `main`.

## Issues and PRs

- Search existing issues and PRs before starting. Items on the [Roadmap](https://github.com/FlashML-org/FreeToken/issues/79) are discussed with maintainers before implementation; features not on it start as an issue.
- When helping the user draft an issue, follow the matching template in `.github/ISSUE_TEMPLATE/` (engine bug, model checkpoint, feature request) and fill in every required field: hardware, driver, FreeToken version, checkpoint ID, exact command, and the full log.
- One change per PR, linked to its issue, with the hardware, checkpoint ID and exact command it was tested with.

## Code comments

Comments explain a non-obvious "why", never restate the code. Write the code first, then add a comment only where a reader would otherwise be confused. Keep them to one or two lines. Configuration files get no comments. Use ASCII: `-` not em-dash, `->` not arrows.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/), one line, imperative, lowercase, no trailing period:

```
fix(kvcache): size the SWA radix pool for chunked prefill
```

PRs are squash-merged, so the PR title follows the same format. The subject line is usually enough; add a body only when the change needs a why that the diff does not show, and keep it to a few lines. Only commit or push when the user requests or authorizes it. Stage only the requested changes and preserve unrelated worktree changes. If the user wants attribution, use `Assisted-by: <agent name>`, not `Co-authored-by`.
