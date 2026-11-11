# Deferred sensitivity jobs

This directory records the narrow controller change used to defer
`rwqa_evaluate_prefill` and its scoring job while the original MoDES comparisons
continue. Existing responses, audit records and calibrated thresholds are kept.

`defer_rwqa_hybrid.patch` applies to the experiment-local `run_experiment.py`.
Its input source SHA-256 is
`0795912c3cd3560d8eca333a3660bfa1a5ee3c109bd878df1053f1491e437548`.
The patch reads `job_policy.json`, avoids recreating deferred tasks, persists
their explicit-resume requirement, and excludes them from required completion.
The original RealWorldQA prefill+decode MoDES comparison remains required.

The policy handler does not terminate an active server. In this session only
the deferred job's verified server process group was stopped, preserving its
117 complete responses. Other generation jobs continued normally. Partial
approximate generations require a fresh cache-order replay when resumed.

`resume_after_job_defer.py` waits for the old manager to exit normally before
loading the updated controller. It sends no signals to active experiments and
checks ownership and the saved validation marker before launching `--resume`.
The controller's existing manager lock prevents duplicate active managers.

```bash
export TABLEMOE_RUN_ROOT=/home/fanya/FreeToken_TableMoE/artifacts/tablemoe_sensitivity_preserve_vlmeval_20261110
export TMPDIR="$TABLEMOE_RUN_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

python benchmarks/tablemoe_sensitivity_controls/resume_after_job_defer.py \
  --manager-pid 1523466
```

Use the actual owning manager PID. The example depends on this experiment's
prepared runtime and `protocol_changes/defer_rwqa_hybrid_20261111/policy_ready.json`.
It is not a command to interrupt or restart active requests.

Explicit restoration requires removing the target and score entries from
`deferred_jobs` and adding `rwqa_evaluate_prefill` to `resumed_jobs`. Removing
the policy file alone does not restore a job already recorded as deferred.

CPU validation covered target/score deferral, scheduler expansion, completion
requirements, explicit restoration and the wait-for-normal-exit handover.
The inference runtime, evaluation prompts and offline tables were unchanged.
