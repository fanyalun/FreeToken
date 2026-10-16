from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

from bench_sparse_verify_motivation import accepted_prefix
from bench_multiloop_sparse_verify import loop_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    checks, strict_failures, quality = [], [], []
    totals = {"windows": 0, "requests": 0, "returned_tokens": 0}
    for backend in ("offload", "hybrid"):
        directory = args.run / backend
        config = json.loads((directory / "config.json").read_text())
        complete = json.loads((directory / "complete.json").read_text())
        assert complete["status"] == "completed"
        expected = {f"s{sample}_p{prefix}_{policy}_l{count}"
                    for sample, prefix, policy, count in itertools.product(
                        config["sample_ids"], config["prefixes"], config["policies"], config["loops"])}
        observed, count = set(), 0
        for line in (directory / "windows.jsonl").open():
            row = json.loads(line)
            assert row["case_id"] not in observed
            observed.add(row["case_id"])
            assert row["loop_count"] == len(row["loops"])
            assert row["candidates"] == [token for loop in row["loops"] for token in loop["tokens"]]
            start = row["start"]
            for loop in row["loops"]:
                assert loop["start"] == start
                candidate, accepted = loop_candidates(loop["inputs"], loop["predictions"])
                assert candidate == loop["tokens"] and accepted == loop["accepted"]
                assert loop["returned"] == accepted + 1
                start += loop["returned"]
                for layer in loop["record"]["layers"]:
                    retained = set()
                    for ids, weights, hits, keep in zip(layer["ids"], layer["weights"], layer["hits"], layer["keep"]):
                        top4 = set(sorted(range(8), key=lambda i: (-weights[i], i))[:4])
                        assert keep == [True if row["policy"] == "full" else hit or i in top4 for i, hit in enumerate(hits)]
                        retained.update(expert for expert, selected in zip(ids, keep) if selected)
                    assert set(layer["fetched_ids"]) <= retained
                    assert len(layer["fetched_ids"]) == len(set(layer["fetched_ids"]))
            outer = row["outer"]
            assert outer["inputs"][1:] == row["candidates"]
            assert outer["accepted"] == accepted_prefix(row["candidates"], outer["predictions"])
            assert outer["returned"] == outer["accepted"] + 1
            assert outer["tokens"] == outer["predictions"][:outer["returned"]]
            assert row["ar_accepted"] == accepted_prefix(row["candidates"], row["reference_tokens"])
            flags = {key: row[key] for key in ("outer_reference_parity", "post_commit_next_token_parity", "matched_baseline_token_parity")}
            quality.append({"backend": backend, "case_id": row["case_id"], **flags})
            if not all(flags.values()):
                strict_failures.append({"backend": backend, "case_id": row["case_id"], "failed_greedy_checks": flags})
            count += 1
        assert observed == expected, (backend, len(observed), len(expected))
        assert count == complete["windows"]
        reference = [json.loads(line) for line in (directory / "baseline_requests.jsonl").open()]
        audit = [json.loads(line) for line in (directory / "audit_requests.jsonl").open()]
        assert len(reference) == len(audit) == config["samples"]
        for left, right in zip(reference, audit):
            assert left["sample_id"] == right["sample_id"]
            assert left["token_ids"] == right["token_ids"]
            assert len(left["token_ids"]) == config["tokens"] and right["baseline_parity"]
        totals["windows"] += count
        totals["requests"] += len(reference) + len(audit)
        totals["returned_tokens"] += sum(len(row["token_ids"]) for row in reference + audit)
        checks.append({"backend": backend, "coverage": count, "requests": len(reference) + len(audit),
                       "complete_prefix_kv_restore_assertions": True, "stable_original_top4_mask": True,
                       "same_checkpointed_request_tokens": True})
    source = json.loads((args.run / "source_manifest.json").read_text())
    production = {}
    repo = Path(json.loads((args.run / "offload/config.json").read_text())["source_root"])
    startup = {backend: json.loads((args.run / backend / "config.json").read_text())["source_sha256"]
               for backend in ("offload", "hybrid")}
    for path, digest in source.items():
        frozen = args.run / "source" / path
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == digest
        if path.startswith("python/freetoken/"):
            assert all(values[path] == digest for values in startup.values()), "backends started with different production source"
            production[path] = hashlib.sha256((repo / path).read_bytes()).hexdigest() == digest
    identity = {"backends_match_frozen_production_at_startup": True, "files": len(production),
                "current_workspace_matches_execution": all(production.values()),
                "workspace_changed_files": [path for path, same in production.items() if not same],
                "scope": "execution identity uses frozen source and startup fingerprints; current workspace may contain parallel edits"}
    (args.run / "production_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    if (args.run / "case_metrics.jsonl").exists():
        for line in (args.run / "case_metrics.jsonl").open():
            row = json.loads(line)
            assert row["accepted"] <= row["proposed"] and row["returned"] == row["accepted"] + 1
            assert row["predicted_missing"] <= row["missing"]
            assert row["useful_speculative"] <= row["speculative"]
            for key in ("serial_ms", "baseline_ms", "ideal_ms", "deadline_ms", "resource_ms"):
                assert math.isfinite(row[key]) and row[key] > 0
            assert abs(row["resource_ms"] - (row["serial_ms"] + row["resource"]["prune"]["elapsed_ms"] - row["resource"]["none"]["elapsed_ms"])) < 1e-7
            for policy in ("keep", "prune"):
                assert row["deadline"][policy]["timely_missing"] <= row["deadline"][policy]["missing"]
    result = {"valid": True, "validity_scope": "diagnostic protocol and accounting; strict greedy parity is a separate outcome",
              "checks": checks, "totals": totals, "errors": [],
              "strict_greedy_parity": not strict_failures, "strict_greedy_failures": strict_failures,
              "greedy_checks": quality, "backends_match_frozen_production_at_startup": True,
              "scope": "main real-model matrix and component accounting; simulations/overlap/figures have separate validation"}
    (args.run / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "greedy_checks"}, indent=2))


if __name__ == "__main__":
    main()
