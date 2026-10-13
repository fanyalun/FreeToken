"""Content-addressed stage markers shared by the builder and benchmark suite."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    with temp.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def checkpoint_identity(model, expected=None):
    root = Path(model).resolve()
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no raw safetensors checkpoint: {root}")
    files += [p for p in (root / "config.json", root / "model.safetensors.index.json") if p.exists()]
    identities = {}
    for p in files:
        stat = p.stat()
        stamp = dict(size=stat.st_size, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)
        previous = (expected or {}).get("files", {}).get(p.name)
        if isinstance(previous, dict) and all(previous.get(k) == v for k, v in stamp.items()):
            checksum = previous["sha256"]
        else:
            checksum = file_hash(p)
        identities[p.name] = dict(sha256=checksum, **stamp)
    return {"path": str(root), "files": identities}


def finish_stage(root, inputs, outputs, **details):
    root = Path(root)
    manifest = dict(inputs=inputs, input_sha256=digest(inputs),
                    outputs={str(Path(p).relative_to(root)): file_hash(p) for p in outputs}, **details)
    write_json(root / "manifest.json", manifest)
    write_json(root / "complete.json", dict(complete=True, manifest_sha256=file_hash(root / "manifest.json")))
    return manifest


def verify_stage(root, inputs=None):
    root = Path(root)
    marker = read_json(root / "complete.json")
    if not marker.get("complete") or marker.get("manifest_sha256") != file_hash(root / "manifest.json"):
        raise ValueError(f"incomplete or changed stage: {root}")
    manifest = read_json(root / "manifest.json")
    if manifest["input_sha256"] != digest(manifest["inputs"]):
        raise ValueError(f"changed stage inputs: {root}")
    if inputs is not None and digest(inputs) != manifest["input_sha256"]:
        raise ValueError(f"resume input mismatch: {root}")
    for name, expected in manifest["outputs"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"changed artifact: {root / name}")
    for dependency, expected in manifest.get("dependencies", {}).items():
        if file_hash(Path(dependency) / "manifest.json") != expected:
            raise ValueError(f"changed dependency: {dependency}")
        verify_stage(dependency)
    return manifest
