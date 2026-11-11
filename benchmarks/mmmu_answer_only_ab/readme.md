# MMMU direct-answer lookup ablation

These scripts reproduce the direct-answer comparison against the existing
TableMoE sensitivity experiment. They reuse its frozen generation runtime,
checkpoint, offline tables and 420 held-out MMMU samples. They require the
prepared experiment metadata and existing runtime environments; they do not
install dependencies or rebuild the engine.

The comparison fixes alpha at 0.8, decode at top-8, expert slots at 6144 and
router weights at their original values. `mmmu_lookup` uses approximate table
outputs for routes omitted by the TableMoE scheduler. `mmmu_schedule` replaces
those outputs with zero through `--tablemoe-skip-lookup`. Both use greedy
generation, natural EOS and a 256-token upper limit.

`prepare_answer_only.py` first checks the normal VLMEvalKit model/dataset prompt
selection against the preserved prompts. It appends a user-requested instruction
to output only the option letter for the 398 multiple-choice questions, or only
the final short answer for the 22 open questions. It preserves question text,
options, images, membership and order. The offline tables retain their original
collection prompts.

Set a persistent experiment directory before running any script:

```bash
export MMMU_AB_ROOT=/home/fanya/FreeToken_TableMoE/artifacts/mmmu_answer_only_lookup_ab_20261111
export TMPDIR="$MMMU_AB_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export LMUData=/home/fanya/tablemoe/LMUData

/home/fanya/miniforge/envs/tablemoe/bin/python \
  benchmarks/mmmu_answer_only_ab/prepare_answer_only.py

/home/fanya/local_archives/freetoken_consolidation_20261029_020126/preserved/FreeToken/.venv/bin/python \
  benchmarks/mmmu_answer_only_ab/run_mmmu_ab.py --gpu 1 --cell mmmu_lookup

/home/fanya/local_archives/freetoken_consolidation_20261029_020126/preserved/FreeToken/.venv/bin/python \
  benchmarks/mmmu_answer_only_ab/run_mmmu_ab.py --gpu 0 --cell mmmu_schedule
```

Run the generation commands in separate processes for concurrent use of the two
GPUs. Each waits for 74 GiB of available GPU memory and holds cell/GPU locks.
Coordinate with other experiment managers before launching; an availability
check alone does not reserve a GPU against another scheduler.

`priority_worker.py` implements this session's handover from the already-running
RealWorldQA matrix. It requires an explicit matching PID/script entry in that
matrix's `external_jobs.json`. Each worker runs one MMMU comparison and then its
leased RealWorldQA followup. Its runtime state is isolated from the original
manager's status file.

Scoring uses VLMEvalKit's native `exact_matching` evaluator and aggregates its
per-sample `hit` values across all 420 indices, including both dev and validation.
The primary result is accuracy over the 398 multiple-choice questions. The
overall, open-question and split results are reported separately. Full outputs,
parse failures, strict single-letter compliance and output lengths are retained
under the experiment directory. Partial approximate runs are archived and
replayed from the beginning to preserve cache order.

No full-content checksum pass is performed on offline tensors or tables. Source
code and small metadata records may be hashed. Timings from this quality
comparison are not formal performance measurements.
