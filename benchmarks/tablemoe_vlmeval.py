"""Isolated VLMEvalKit subprocess: native prompt preparation and prediction scoring."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings
import types
import logging


def load_dataset(kit, name, dataset_only=False):
    if dataset_only:
        # Native target classes avoid eagerly importing unrelated model/GUI dependencies.
        for package, path in (("vlmeval", Path(kit) / "vlmeval"), ("vlmeval.dataset", Path(kit) / "vlmeval/dataset")):
            module = types.ModuleType(package)
            module.__path__ = [str(path)]
            sys.modules[package] = module
        from vlmeval.dataset.image_mcq import ImageMCQDataset
        from vlmeval.dataset.image_yorn import ImageYORNDataset
        classes = (ImageMCQDataset, ImageYORNDataset)
        for cls in classes:
            if name in cls.supported_datasets():
                return cls(dataset=name)
        raise ValueError(f"unsupported target dataset: {name}")
    from vlmeval.dataset import build_dataset
    return build_dataset(name)


def load_native_judge_api():
    root = sys.modules["vlmeval"]
    if getattr(root, "__spec__", None) is not None:
        return
    if "vlmeval.api" not in sys.modules:
        module = types.ModuleType("vlmeval.api")
        module.__path__ = [str(Path(root.__path__[0]) / "api")]
        sys.modules["vlmeval.api"] = module
    from vlmeval.dataset.image_base import img_root_map
    sys.modules["vlmeval.dataset"].img_root_map = img_root_map
    from vlmeval.api.gpt import OpenAIWrapper
    from vlmeval.api.hf_chat_model import HFChatModel
    from vlmeval.api.siliconflow import SiliconFlowAPI
    api = sys.modules["vlmeval.api"]
    api.OpenAIWrapper, api.HFChatModel, api.SiliconFlowAPI = OpenAIWrapper, HFChatModel, SiliconFlowAPI

from tablemoe_data import api_messages, group_rows, image_identity, read_rows, write_rows
from tablemoe_common import PREDICTION_PROTOCOL, prediction_text


def kit_fingerprint(root):
    from freetoken.moe.tablemoe_artifacts import file_hash

    root = Path(root)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    files = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*.py")
             if ".git" not in p.parts and "site-packages" not in p.parts}
    return dict(root=str(root), commit=commit, files=files, python=sys.executable)


def prepare(dataset, output, media_path_remap=None):
    frame = dataset.data
    rows = []
    for i in range(len(frame)):
        line = frame.iloc[i]
        message = dataset.build_prompt(line)
        original_message = [dict(part) for part in message]
        for part in message:
            if part["type"] == "image" and not Path(part["value"]).exists():
                for old, new in (media_path_remap or {}).items():
                    old_path = Path(old)
                    if Path(part["value"]).is_relative_to(old_path):
                        part["value"] = str(Path(new) / Path(part["value"]).relative_to(old_path))
                        break
        meta = json.loads(line.drop(labels=["image"], errors="ignore").to_json())
        idx = meta["index"]
        if isinstance(idx, float) and idx.is_integer():
            idx = int(idx)
        rows.append(dict(id=f"{dataset.dataset_name}:{idx}", index=idx, dataset=dataset.dataset_name,
                         metadata=meta, vlmeval_message=message, messages=api_messages(message),
                         native_vlmeval_message=original_message,
                         images=[image_identity(part["value"]) for part in message if part["type"] == "image"],
                         circular_group=int(idx) % 1000000 if dataset.dataset_name.startswith("MMBench") else None))
    write_rows(output, group_rows(rows))


def score(dataset, rows, predictions, root, judge_kwargs):
    import pandas as pd
    from vlmeval.smp import dump
    from freetoken.moe.tablemoe_artifacts import finish_stage

    if len(rows) != len(predictions) or [r["id"] for r in rows] != [r["id"] for r in predictions]:
        raise ValueError("missing, duplicate, or reordered predictions")
    if any(not r.get("natural_eos") for r in predictions):
        raise ValueError("accuracy requires natural outputs")
    if dataset.dataset_name.startswith("MMBench"):
        wanted = {int(r["index"]) for r in rows}
        bases = {i % 1000000 for i in wanted}
        native = {int(i) for i in dataset.data["index"] if int(i) % 1000000 in bases}
        if native != wanted:
            raise ValueError("MMBench circular group is incomplete")
    records = [{**r["metadata"], "index": r["index"], "prediction": prediction_text(p)}
               for r, p in zip(rows, predictions)]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    prediction_file = root / "predictions.xlsx"
    if prediction_file.exists():
        raise ValueError("score destination must be fresh to avoid native cached score reuse")
    dump(pd.DataFrame(records), str(prediction_file))
    judge_kwargs = {"model": "exact_matching", "nproc": 4, **judge_kwargs}
    judge = None
    if judge_kwargs["model"] != "exact_matching":
        load_native_judge_api()
        logging.getLogger("ChatAPI").addFilter(lambda record: "API Key:" not in record.getMessage())
        from vlmeval.dataset.utils import build_judge
        judge = build_judge(**{k: v for k, v in judge_kwargs.items() if k != "nproc"})
        if not judge.working():
            raise RuntimeError("configured judge is unavailable; exact-matching fallback is forbidden")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*will use exact matching.*")
        result = dataset.evaluate(str(prediction_file), **judge_kwargs)
    if judge is not None and not judge.working():
        raise RuntimeError("judge became unavailable during evaluation")
    if not isinstance(result, pd.DataFrame) or result.empty:
        raise ValueError("native scoring returned no tabular results")
    native = json.loads(result.to_json(orient="records"))
    if dataset.dataset_name == "POPE":
        overall = [r for r in native if r.get("split") == "Overall"]
        if len(overall) != 1 or overall[0].get("acc") is None:
            raise ValueError("POPE scoring lacks Overall acc")
        accuracy_pct = overall[0]["acc"]
        extras = dict(f1_pct=overall[0]["Overall"], precision_pct=overall[0]["precision"], recall_pct=overall[0]["recall"])
        denominator = sum(len(str(r["metadata"]["category"]).split(",")) for r in rows)
    else:
        overall = [r for r in native if str(r.get("split", "none")).lower() in ("none", "test", "dev", "overall")]
        if len(overall) != 1 or overall[0].get("Overall") is None:
            raise ValueError(f"ambiguous native Overall score: {native}")
        accuracy_pct = overall[0]["Overall"] * 100
        extras = {}
        denominator = len({int(r["index"]) % 1000000 for r in rows}) if dataset.dataset_name.startswith("MMBench") else len(rows)
    summary = dict(dataset=dataset.dataset_name, accuracy_pct=accuracy_pct, sample_count=len(rows),
                   prediction_protocol=PREDICTION_PROTOCOL,
                   scoring_denominator=denominator, scoring_protocol=judge_kwargs,
                   native_results=native, prediction_file=str(prediction_file), **extras)
    if judge is not None:
        summary["effective_judge_model"] = getattr(judge, "model", judge_kwargs["model"])
    from freetoken.moe.tablemoe_artifacts import write_json, digest, file_hash
    write_json(root / "score.json", summary)
    return finish_stage(root, dict(predictions=digest(predictions), rows=digest(rows), judge=judge_kwargs,
                                  prediction_protocol=PREDICTION_PROTOCOL,
                                  scorer_sources={name: file_hash(Path(__file__).parent / name) for name in
                                                  ("tablemoe_vlmeval.py", "tablemoe_common.py")}),
                        [p for p in root.iterdir() if p.is_file() and p.name not in ("manifest.json", "complete.json")], **summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "score"))
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--judge", default="{}")
    parser.add_argument("--dataset-only-import", action="store_true")
    parser.add_argument("--media-path-remap", default="{}")
    parser.add_argument("--expected-kit", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(args.kit))
    from freetoken.moe.tablemoe_artifacts import write_json

    dataset = load_dataset(args.kit, args.dataset, args.dataset_only_import)
    if dataset is None:
        raise ValueError(f"unknown VLMEvalKit dataset: {args.dataset}")
    fingerprint = kit_fingerprint(args.kit)
    if args.expected_kit and json.loads(args.expected_kit.read_text()) != fingerprint:
        raise ValueError("VLMEvalKit source/interpreter changed since prompt preparation")
    if args.stage == "prepare":
        prepare(dataset, args.output, json.loads(args.media_path_remap))
        write_json(args.output.with_suffix(".kit.json"), fingerprint)
    else:
        write_json(args.output / "kit.json", fingerprint)
        score(dataset, read_rows(args.manifest), read_rows(args.predictions), args.output, json.loads(args.judge))


if __name__ == "__main__":
    main()
