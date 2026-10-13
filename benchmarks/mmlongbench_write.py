"""VLMEvalKit dataset extension for the official MMLongBench-Write corpus."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import threading
from urllib.parse import urlparse

from tablemoe_data import api_messages, group_rows, image_identity, read_rows, verify_disjoint, write_rows

CATEGORIES = ("ppt_script", "scenery_copywriting", "story_generation", "movie_review", "medical_report", "travel_guide")
DIMENSIONS = ("Relevance", "Accuracy", "Coherence", "Clarity", "Breadth and Depth", "Reading Experience")
PARQUET_SHA256 = "5b784c89769368aae533fb2d94a0dc00e9a7bf19f96660d0515f312ce6112258"


def count_words(text):
    return len(re.findall(r"[\u4e00-\u9fff]", text)) + len(re.findall(r"\b[a-zA-Z]+\b", text))


def length_score(actual, target):
    actual, target = max(actual, 2), max(target, 2)
    return 100 * max(0., 1 - ((target / actual - 1) / 3 if target > actual else (actual / target - 1) / 2))


def parse_scores(text):
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            result, _ = decoder.raw_decode(text[start:])
            if all(type(result.get(key)) is int and 1 <= result[key] <= 5 for key in DIMENSIONS):
                return result
        except (ValueError, TypeError, AttributeError):
            pass
    raise ValueError("judge did not return all six integer scores in [1, 5]")


def prediction_file(frame, output):
    from vlmeval.smp import dump, load

    # XLSX limits a cell to 32767 characters, shorter than some 8192-token answers.
    path = Path(output) / "predictions.pkl"
    dump(frame, str(path))
    if load(str(path))["prediction"].tolist() != frame["prediction"].tolist():
        raise ValueError("native prediction serialization changed the full text")
    return path


def stratified_split(rows, seed=42):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    group_rows(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["group_id"], []).append(row)
    groups = sorted(grouped)
    strata = sorted({(r["metadata"]["category"], r["metadata"]["language"]) for r in rows})
    matrix = np.array([[sum((r["metadata"]["category"], r["metadata"]["language"]) == stratum
                           for r in grouped[group]) for group in groups] for stratum in strata], dtype=float)
    targets = matrix.sum(1) * .6
    if not np.equal(targets, np.round(targets)).all():
        raise ValueError("each category/language stratum must permit an exact 60/40 split")
    result = milp(np.random.default_rng(seed).random(len(groups)), integrality=np.ones(len(groups)),
                  bounds=Bounds(0, 1), constraints=LinearConstraint(matrix, targets, targets),
                  options={"time_limit": 60})
    if not result.success:
        raise ValueError(f"exact category split with image-group isolation is infeasible: {result.message}")
    chosen = {group for group, value in zip(groups, result.x) if value > .5}
    calibration = [r for r in rows if r["group_id"] in chosen]
    evaluation = [r for r in rows if r["group_id"] not in chosen]
    verify_disjoint(calibration, evaluation)
    for category in CATEGORIES:
        if sum(r["metadata"]["category"] == category for r in calibration) != 12 or sum(
                r["metadata"]["category"] == category for r in evaluation) != 8:
            raise ValueError("category counts do not equal 12 calibration / 8 evaluation")
    return calibration, evaluation


def prepare(source, output):
    import pandas as pd
    from PIL import Image
    from freetoken.moe.tablemoe_artifacts import file_hash, finish_stage, read_json, write_json

    source, output = Path(source), Path(output)
    if file_hash(source / "dataset.parquet") != PARQUET_SHA256:
        raise ValueError("official dataset checksum mismatch")
    frame = pd.read_parquet(source / "dataset.parquet").sort_values("id")
    if frame.id.tolist() != list(range(120)):
        raise ValueError("expected the pinned official 120-example corpus")
    rows = []
    media = output / "images"
    media.mkdir(parents=True, exist_ok=True)
    for _, item in frame.iterrows():
        index = int(item.id)
        paths = []
        for encoded in item.images:
            raw = encoded["bytes"]
            with Image.open(io.BytesIO(raw)) as image:
                suffix = ".jpg" if image.format == "JPEG" else ".png"
            path = media / (hashlib.sha256(raw).hexdigest() + suffix)
            if not path.exists():
                path.write_bytes(raw)
            paths.append(str(path.resolve()))
        message = [dict(type="text", value=item.question)] + [dict(type="image", value=p) for p in paths]
        metadata = dict(index=index, question=item.question, L=int(item.L), category=CATEGORIES[index // 20],
                        language="zh" if re.search(r"[\u4e00-\u9fff]", item.question) else "en", image_path=paths)
        rows.append(dict(id=f"MMLongBench-Write:{index}", index=index, dataset="MMLongBench-Write",
                         metadata=metadata, vlmeval_message=message, messages=api_messages(message),
                         images=[image_identity(p) for p in paths]))
    calibration, evaluation = stratified_split(rows)
    outputs = []
    for name, samples in (("all", rows), ("calibration", calibration), ("evaluation", evaluation)):
        path = output / f"{name}.jsonl"
        write_rows(path, samples)
        outputs.append(path)
    split = {category: {name: [r["index"] for r in samples if r["metadata"]["category"] == category]
                        for name, samples in (("calibration", calibration), ("evaluation", evaluation))}
             for category in CATEGORIES}
    write_json(output / "split.json", dict(seed=42, ratio=[6, 4], categories=split,
                                           category_mapping="official IDs 20 per task; checked against paper Figure 1 and prompts",
                                           image_groups=len({r["group_id"] for r in rows})))
    outputs.append(output / "split.json")
    outputs.extend(media.iterdir())
    return finish_stage(output, dict(dataset_sha256=PARQUET_SHA256, sources=read_json(source / "source_revisions.json"),
                                     seed=42, ratio=[6, 4]), outputs, calibration_samples=72, evaluation_samples=48)


def dataset_class():
    from vlmeval.dataset.image_base import ImageBaseDataset
    import pandas as pd

    class MMLongBenchWriteDataset(ImageBaseDataset):
        TYPE = "VQA"

        def __init__(self, rows, rubric):
            self.dataset_name = "MMLongBench-Write"
            self.data = pd.DataFrame([r["metadata"] for r in rows])
            self.rows = {r["index"]: r for r in rows}
            self.rubric = rubric

        def build_prompt(self, line):
            if isinstance(line, int):
                line = self.data.iloc[line]
            return self.rows[int(line["index"])]["vlmeval_message"]

        def evaluate(self, eval_file, **judge_kwargs):
            from vlmeval.dataset.utils import build_judge
            from vlmeval.smp import dump, load
            from freetoken.moe.tablemoe_artifacts import file_hash, read_json, write_json

            data = load(str(eval_file))
            if data["index"].tolist() != self.data["index"].tolist() or data.prediction.isna().any():
                raise ValueError("prediction coverage/order differs from requested evaluation rows")
            nproc = judge_kwargs.pop("nproc", 2)
            judge = build_judge(**judge_kwargs)
            if urlparse(judge_kwargs.get("api_base", "")).hostname in ("127.0.0.1", "localhost", "::1"):
                # Pin this experiment instead of inheriting the kit's .env LOCAL_LLM.
                judge.model = judge_kwargs["model"]
                for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
                    os.environ.pop(name, None)
            if not judge.working():
                raise RuntimeError("local judge unavailable; fallback scoring is forbidden")
            root = Path(eval_file).parent
            saved = read_json(root / "judge_progress.json") if (root / "judge_progress.json").exists() else {}
            if saved and (not (root / "judge_progress_hash.json").exists() or read_json(
                    root / "judge_progress_hash.json")["sha256"] != file_hash(root / "judge_progress.json")):
                raise ValueError("interrupted judge progress has no matching integrity hash")
            lock = threading.Lock()

            def evaluate_one(item):
                index = str(item["index"])
                if index in saved:
                    return saved[index]
                message = [dict(type="text", value=self.rubric.replace("$INST$", item["question"]).replace(
                    "$RESPONSE$", item["prediction"]))]
                message.extend(dict(type="image", value=p) for p in self.rows[int(index)]["metadata"]["image_path"])
                for attempt in range(3):
                    response = judge.generate(message)
                    try:
                        scores = parse_scores(response)
                        break
                    except ValueError:
                        if attempt == 2:
                            raise
                length = count_words(item["prediction"])
                quality = (sum(scores[d] for d in DIMENSIONS) - 6) * 25 / 6
                sl = length_score(length, item["L"])
                result = dict(index=item["index"], category=item["category"], language=item["language"],
                              target_length=item["L"], length=length, length_score=sl, quality_score=quality,
                              overall_score=(sl + quality) / 2, scores=scores, judge_response=response)
                with lock:
                    saved[index] = result
                    write_json(root / "judge_progress.json", saved)
                    write_json(root / "judge_progress_hash.json", dict(sha256=file_hash(root / "judge_progress.json")))
                return result

            with ThreadPoolExecutor(max_workers=nproc) as pool:
                details = list(pool.map(evaluate_one, data.to_dict("records")))
            if not judge.working() or len(details) != len(data):
                raise RuntimeError("judge failed or coverage is incomplete")
            write_rows(root / "judge_raw.jsonl", details)
            results = []
            for category in ("Overall",) + CATEGORIES:
                subset = [r for r in details if category == "Overall" or r["category"] == category]
                if not subset:
                    continue
                summary = dict(category=category, samples=len(subset))
                for field in ("length", "length_score", "quality_score", "overall_score"):
                    summary[field] = sum(r[field] for r in subset) / len(subset)
                for dimension in DIMENSIONS:
                    summary[dimension] = sum(r["scores"][dimension] for r in subset) / len(subset)
                results.append(summary)
            frame = pd.DataFrame(results)
            dump(frame, str(root / "score.csv"))
            write_json(root / "score.json", dict(complete=True, scoring_denominator=len(details),
                                                judge_protocol=judge_kwargs, results=results))
            return frame

    return MMLongBenchWriteDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kit", type=Path)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--judge", default="{}")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from freetoken.moe.tablemoe_artifacts import digest, file_hash, finish_stage, read_json, write_json
    if args.stage == "prepare":
        prepare(args.source, args.output)
        return
    from tablemoe_common import prediction_text
    from tablemoe_vlmeval import load_dataset, load_native_judge_api, kit_fingerprint
    sys.path.insert(0, str(args.kit))
    load_dataset(args.kit, "AI2D_TEST", dataset_only=True)
    load_native_judge_api()
    import logging
    logging.getLogger("ChatAPI").addFilter(lambda record: "API Key:" not in record.getMessage())
    import pandas as pd
    from vlmeval.smp import dump
    rows, predictions = read_rows(args.rows), read_rows(args.predictions)
    if [r["id"] for r in rows] != [r["id"] for r in predictions] or any(not p["natural_eos"] for p in predictions):
        raise ValueError("accuracy requires all natural predictions in manifest order")
    inputs = dict(rows=digest(rows), predictions=digest(predictions), judge=json.loads(args.judge),
                  extension_sha256=file_hash(__file__), rubric_sha256=file_hash(args.source / "eval/prompts/eval_quality.txt"),
                  kit=kit_fingerprint(args.kit))
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "stage_inputs.json").exists() and read_json(args.output / "stage_inputs.json") != inputs:
        raise ValueError("judge resume inputs changed")
    write_json(args.output / "stage_inputs.json", inputs)
    write_json(args.output / "complete.json", dict(complete=False))
    frame = pd.DataFrame([{**r["metadata"], "prediction": prediction_text(p)} for r, p in zip(rows, predictions)])
    eval_file = prediction_file(frame, args.output)
    dataset = dataset_class()(rows, (args.source / "eval/prompts/eval_quality.txt").read_text())
    dataset.evaluate(eval_file, **json.loads(args.judge))
    finish_stage(args.output, inputs, [p for p in args.output.iterdir() if p.is_file() and p.name not in
                                      ("manifest.json", "complete.json")], scoring_denominator=len(rows))


if __name__ == "__main__":
    main()
