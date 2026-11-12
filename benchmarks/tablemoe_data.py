"""Leakage-safe multimodal manifests and VLMEvalKit message conversion."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from pathlib import Path
import random


def image_identity(path):
    from PIL import Image

    path = Path(path)
    raw = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        pixels = image.convert("RGB")
        content = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
    return dict(path=str(path.resolve()), sha256=content, file_sha256=raw)


def api_messages(message):
    parts = []
    for item in message:
        if item["type"] == "text":
            parts.append(dict(type="text", text=item["value"]))
        elif item["type"] == "image":
            path = Path(item["value"])
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
            parts.append(dict(type="image_url", image_url=dict(url=url)))
        else:
            raise ValueError(f"unsupported VLMEvalKit media type: {item['type']}")
    return [dict(role="user", content=parts)]


def group_rows(rows):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate sample identity")
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, row in enumerate(rows):
        keys = [("image", image["sha256"]) for image in row["images"]]
        document = document_identity(row)
        if document is not None:
            keys.append(("document", document))
        question = row.get("metadata", {}).get("question")
        if question:
            keys.append(("question", " ".join(str(question).split()), tuple(sorted(i["sha256"] for i in row["images"]))))
        if row["dataset"].startswith("MMBench"):
            keys.append(("circular", int(row["index"]) % 1000000))
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    for members in groups.values():
        ids = sorted(r["id"] for r in members)
        group = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        for row in members:
            row["group_id"] = group
    return rows


def document_identity(row):
    metadata = row.get("metadata", {})
    for key in ("document_id", "doc_id", "document", "document_path"):
        value = metadata.get(key, row.get(key))
        if value is not None and str(value):
            return str(value)
    return None


def calibration_category(row):
    metadata = row.get("metadata", {})
    return str(metadata.get("category") or metadata.get("subject") or row.get("category") or "__all__")


def _quota_groups(groups, counts, target):
    reachable, history = 1, []
    for group in groups:
        history.append(reachable)
        reachable |= reachable << counts[group]
    target = min(max(0, target), reachable.bit_length() - 1)
    below = (reachable & ((1 << (target + 1)) - 1)).bit_length() - 1
    above_bits = reachable >> target
    above = target + (above_bits & -above_bits).bit_length() - 1
    remaining = min((below, above), key=lambda count: abs(count - target))
    chosen = set()
    for group, previous in zip(reversed(groups), reversed(history)):
        if not (previous >> remaining) & 1:
            chosen.add(group)
            remaining -= counts[group]
    return chosen


def verify_disjoint(calibration, evaluation):
    for field in ("id", "group_id"):
        overlap = {r[field] for r in calibration} & {r[field] for r in evaluation}
        if overlap:
            raise ValueError(f"calibration/evaluation {field} overlap: {sorted(overlap)[:5]}")
    for transform, label in ((lambda r: [i["sha256"] for i in r["images"]], "image content"),
                             (lambda r: [document_identity(r)] if document_identity(r) is not None else [], "document"),
                             (lambda r: [(" ".join(str(r.get("metadata", {}).get("question") or "").split()),
                                          tuple(sorted(i["sha256"] for i in r["images"])))], "question and media")):
        left = {v for r in calibration for v in transform(r) if v}
        right = {v for r in evaluation for v in transform(r) if v}
        if left & right:
            raise ValueError(f"calibration/evaluation {label} overlap")
    if not calibration or not evaluation:
        raise ValueError("calibration and evaluation must both be nonempty")


def verify_native_evaluation(evaluation, native):
    reference = {row["id"]: row for row in native}
    if len(evaluation) != len(native) or {row["id"] for row in evaluation} != set(reference):
        raise ValueError("native evaluation must cover the full requested native sample set")

    def prompt(row):
        images = iter(row["images"])
        return [(part["type"], next(images)["sha256"] if part["type"] == "image" else part["value"])
                for part in row["vlmeval_message"]]

    for row in evaluation:
        original = reference[row["id"]]
        if (row["index"] != original["index"] or row["dataset"] != original["dataset"] or
                any(row["metadata"].get(key) != value for key, value in original["metadata"].items()) or
                prompt(row) != prompt(original)):
            raise ValueError(f"native evaluation prompt, media, or labels changed: {row['id']}")


def split_rows(rows, seed=42, fraction=.7, *, stratified=False):
    group_rows(rows)
    groups = sorted({r["group_id"] for r in rows})
    if len(groups) < 2 or not 0 < fraction < 1:
        raise ValueError("need two or more groups and a fraction in (0, 1)")
    generator = random.Random(seed)
    if stratified:
        categories = {}
        for row in rows:
            counts = categories.setdefault(calibration_category(row), {})
            counts[row["group_id"]] = counts.get(row["group_id"], 0) + 1
        chosen, assigned = set(), set()
        for category, counts in sorted(categories.items()):
            target = int(sum(counts.values()) * fraction)
            current = sum(count for group, count in counts.items() if group in chosen)
            available = sorted(set(counts) - assigned)
            generator.shuffle(available)
            # Shared groups keep their first assignment when they span categories.
            chosen.update(_quota_groups(available, counts, target - current))
            assigned.update(available)
        if not chosen or len(chosen) == len(groups):
            raise ValueError("category quotas and shared groups cannot produce a nonempty holdout")
    else:
        generator.shuffle(groups)
        cut = min(len(groups) - 1, max(1, int(len(groups) * fraction)))
        chosen = set(groups[:cut])
    calibration = [r for r in rows if r["group_id"] in chosen]
    evaluation = [r for r in rows if r["group_id"] not in chosen]
    verify_disjoint(calibration, evaluation)
    return calibration, evaluation


def fixed_rows(rows, limit=100, seed=42):
    generator = random.Random(seed)
    indices = sorted(generator.sample(range(len(rows)), min(limit, len(rows))))
    return [rows[i] for i in indices]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_manifest(rows):
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError("duplicate manifest identity")
        ids.add(row["id"])
        converted = api_messages(row["vlmeval_message"])
        if converted != row["messages"]:
            raise ValueError(f"prompt/media order changed: {row['id']}")
        expected = [image_identity(part["value"]) for part in row["vlmeval_message"] if part["type"] == "image"]
        if expected != row["images"]:
            raise ValueError(f"image identity does not match prompt media: {row['id']}")
