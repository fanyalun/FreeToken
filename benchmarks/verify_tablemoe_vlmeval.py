"""Compare all five suite score exports against direct native VLMEvalKit calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tablemoe_vlmeval import load_dataset, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(args.kit))
    from freetoken.moe.tablemoe_artifacts import write_json
    import pandas as pd

    results = []
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("MMBench_DEV_EN_V11", "ScienceQA_TEST", "AI2D_TEST", "POPE", "RealWorldQA"):
        dataset = load_dataset(args.kit, name, dataset_only=True)
        from vlmeval.smp import dump
        frame = dataset.data
        if name.startswith("MMBench"):
            base = int(frame.iloc[0]["index"]) % 1000000
            selected = frame[frame["index"].map(lambda i: int(i) % 1000000 == base)]
        else:
            selected = frame.iloc[:3]
        rows, predictions = [], []
        for _, line in selected.iterrows():
            metadata = json.loads(line.drop(labels=["image"], errors="ignore").to_json())
            idx = int(metadata["index"])
            rows.append(dict(id=f"{name}:{idx}", index=idx, metadata=metadata))
            answer = str(metadata["answer"])
            predictions.append(dict(id=f"{name}:{idx}", text=answer if len(rows) % 2 else "",
                                    reasoning_text="" if len(rows) % 2 else answer, natural_eos=True))
        root = args.output / name
        native_dir = root / "direct"
        native_dir.mkdir(parents=True)
        path = native_dir / "predictions.xlsx"
        dump(pd.DataFrame([{**r["metadata"], "prediction": str(r["metadata"]["answer"])} for r in rows]), str(path))
        direct = dataset.evaluate(str(path), model="exact_matching", nproc=1)
        bridged = score(dataset, rows, predictions, root / "bridge", dict(nproc=1))
        pd.testing.assert_frame_equal(direct.reset_index(drop=True), pd.DataFrame(bridged["native_results"]).reset_index(drop=True),
                                      check_dtype=False, atol=1e-8)
        if name == "POPE":
            assert bridged["accuracy_pct"] == direct[direct["split"] == "Overall"].iloc[0]["acc"]
        try:
            score(dataset, rows, predictions[:-1], root / "missing", {})
        except ValueError:
            pass
        else:
            raise AssertionError("missing predictions were accepted")
        if name.startswith("MMBench") and len(rows) > 1:
            try:
                score(dataset, rows[:1], predictions[:1], root / "partial_circular", {})
            except ValueError:
                pass
            else:
                raise AssertionError("partial circular group was accepted")
        results.append(dict(dataset=name, sample_count=len(rows), native_parity=True,
                            accuracy_pct=bridged["accuracy_pct"], scoring_denominator=bridged["scoring_denominator"]))
        print(results[-1], flush=True)
    utils = sys.modules["vlmeval.dataset.utils"]
    original = utils.build_judge
    from types import SimpleNamespace
    utils.build_judge = lambda **kwargs: SimpleNamespace(working=lambda: False)
    try:
        try:
            score(dataset, rows, predictions, args.output / "judge_unavailable", dict(model="unavailable"))
        except RuntimeError:
            pass
        else:
            raise AssertionError("unavailable judge was silently accepted")
    finally:
        utils.build_judge = original
    write_json(args.output / "results.json", dict(complete=True, native_fixture_parity=results,
                                                  reasoning_channel_predictions_checked=True,
                                                  missing_predictions_rejected=True, judge_failure_rejected=True))


if __name__ == "__main__":
    main()
