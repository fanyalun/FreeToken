from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re


def numeric_answer(text, *, marker=False):
    pattern = r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)" if marker else r"[-+]?\d[\d,]*(?:\.\d+)?"
    found = re.findall(pattern, text)
    if not found:
        return None
    try:
        value = Decimal(found[-1].replace(",", "")).normalize()
        return "0" if value == 0 else str(value)
    except InvalidOperation:
        return None


def inspect_run(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    complete = json.loads((path / "complete.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    arguments = config["arguments"]
    expected = arguments["samples"] * arguments["repeats"]
    rows = [json.loads(line) for line in (path / "timing.jsonl").read_text().splitlines()]
    audits = [json.loads(line) for line in (path / "audit.jsonl").read_text().splitlines()]
    if len(rows) != expected or len(audits) != arguments["samples"]:
        raise ValueError(f"incomplete request/audit matrix: {path}")
    expected_ids = {(r, s) for r in range(arguments["repeats"]) for s in range(arguments["samples"])}
    if {(row["repeat"], row["sample"]) for row in rows} != expected_ids:
        raise ValueError(f"timing request identities do not match the matrix: {path}")
    if sorted(row["sample"] for row in audits) != list(range(arguments["samples"])):
        raise ValueError(f"audit request identities do not match the matrix: {path}")
    if any(len(row["token_ids"]) != arguments["tokens"] for row in rows + audits):
        raise ValueError(f"truncated request in {path}")
    if complete["status"] != "complete":
        raise ValueError(f"missing completion marker: {path}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    data_path = Path(config["data"])
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != config["data_sha256"]:
        raise ValueError(f"dataset changed since measurement: {data_path}")
    data = [json.loads(line) for line in data_path.read_text().splitlines()]
    eos = set(tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, list) else [tokenizer.eos_token_id])
    quality = []
    for row in rows:
        tokens = row["token_ids"]
        first_eos = next((i for i, token in enumerate(tokens) if token in eos), len(tokens))
        text = tokenizer.decode(tokens[:first_eos], skip_special_tokens=True)
        gold = numeric_answer(data[row["sample"]]["answer"], marker=True)
        quality.append(dict(repeat=row["repeat"], sample=row["sample"], text=text, gold=gold,
            marker_answer=numeric_answer(text, marker=True), last_number_answer=numeric_answer(text),
            marker_correct=gold is not None and numeric_answer(text, marker=True) == gold,
            last_number_correct=gold is not None and numeric_answer(text) == gold))
    totals = {}
    for row in audits:
        for window in row["rounds"]:
            for key, value in window.get("cache_stats", {}).items():
                if isinstance(value, (float, int)) and key != "expert_bytes":
                    totals[key] = totals.get(key, 0) + value
    numerical_path = path / "teacher_forced.json"
    return dict(path=str(path.resolve()), method=arguments["method"], budget_gib=arguments["budget_gib"],
        purpose=arguments["purpose"], complete=complete, slots=config["slots"], kv_pages=config["kv_pages"],
        summary=summary, audit_totals=totals, quality_protocol="numeric exact match before first EOS; marker and last-number extraction reported separately",
        independent_questions=arguments["samples"], quality=quality,
        numerical=json.loads(numerical_path.read_text()) if numerical_path.exists() else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [inspect_run(path) for path in args.runs]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "evidence.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Multi-token Hybrid measurements", "", "Calibration, smoke and formal results have separate scopes.", "",
        "| Method | Budget GiB | Purpose | Slots | Repeat | TPOT ms | tok/s | Last-number correct / questions | Replay |",
        "|---|---:|---|---:|---:|---:|---:|---|---|"]
    for result in results:
        for repeat, row in enumerate(result["summary"]["repeats"]):
            quality = [q for q in result["quality"] if q["repeat"] == repeat]
            correct = sum(q["last_number_correct"] for q in quality)
            lines.append(f"| {result['method']} | {result['budget_gib']} | {result['purpose']} | {result['slots']} | {repeat} | {row['tpot_ms']:.4f} | {row['tokens_per_second']:.4f} | {correct}/{len(quality)} | {result['summary']['independent_replay_parity']} |")
    lines += ["", "Question counts are not multiplied by timing repeats. Last-number extraction is a stated scoring heuristic, not a semantic judge.",
        "Numerical audits, state differences and raw outputs are in evidence.json. These measurements alone do not establish strict parity or default eligibility."]
    (args.output / "readme.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
