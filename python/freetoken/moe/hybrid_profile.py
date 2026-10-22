"""Cache Hybrid measurements by execution compatibility and measured coverage."""
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path


def pcie_identity(gpu_uuid):
    for info in Path("/proc/driver/nvidia/gpus").glob("*/information"):
        try:
            values = dict(line.split(":", 1) for line in info.read_text().splitlines() if ":" in line)
            if values.get("GPU UUID", "").strip().removeprefix("GPU-") != gpu_uuid.removeprefix("GPU-"):
                continue
            device = Path("/sys/bus/pci/devices") / values["Bus Location"].strip()
            result = {"path": str(device.resolve())}
            for name in ("max_link_speed", "max_link_width", "current_link_width", "numa_node"):
                if (device / name).exists():
                    result[name] = (device / name).read_text().strip()
            return result
        except (OSError, KeyError):
            continue
    return None


def compatible_identity(identity):
    if not isinstance(identity, dict):
        raise ValueError("invalid Hybrid profile identity")
    result = deepcopy(identity)
    result.pop("widths", None)
    result.pop("cache_slots", None)
    for bank in result.get("cache_layout", []):
        if isinstance(bank, list) and len(bank) == 4 and isinstance(bank[1], list):
            bank[1] = bank[1][1:]
    return result


def fetch_limit(identity, width):
    experts, top_k = identity["geometry"][1], identity["geometry"][-1]
    return min(width * top_k, experts, identity.get("cache_slots", experts))


def coverage(profile):
    identity = profile["identity"]
    saved = profile.get("calibration_coverage", {})
    if not isinstance(saved, dict) or not isinstance(saved.get("width_fetch_limits", {}), dict):
        raise ValueError("invalid Hybrid profile coverage")
    widths = {int(w): int(q) for w, q in saved.get("width_fetch_limits", {}).items()}
    if not saved:
        widths = {w: fetch_limit(identity, w) for w in identity["widths"]}
    io_max = saved.get("io_max", fetch_limit(identity, max(identity["widths"])))
    return widths, min(int(io_max), len(profile["io"]) - 1)


def missing_widths(profile, identity):
    measured, _ = coverage(profile) if profile else ({}, 0)
    return [w for w in identity["widths"] if measured.get(w, -1) < fetch_limit(identity, w)]


def project_profile(profile, identity, fields):
    maximum = max(identity["widths"])
    columns = maximum * identity["geometry"][-1] + 1
    result = deepcopy(profile)
    result["identity"] = deepcopy(identity)
    for name in fields:
        default = 1. if name == "contention" else 0.
        if name in ("cpu", "gpu"):
            rows = [[0.] * columns for _ in range(maximum + 1)]
            for w in (0, *identity["widths"]):
                if w < len(profile[name]):
                    values = profile[name][w][:columns]
                    rows[w][:len(values)] = values
        elif name == "io":
            rows = list(profile[name][:columns])
            rows.extend([default] * (columns - len(rows)))
        else:
            rows = [default] * (maximum + 1)
            for w in (0, *identity["widths"]):
                if w < len(profile[name]):
                    rows[w] = profile[name][w]
        result[name] = rows
    measured, io_max = coverage(profile)
    result["calibration_coverage"] = dict(
        width_fetch_limits={str(w): measured[w] for w in identity["widths"] if w in measured},
        io_max=min(io_max, columns - 1))
    return result


def merge_profile(previous, updated, fields):
    identity = deepcopy(updated["identity"])
    identity["widths"] = sorted(set(previous["identity"]["widths"]) | set(identity["widths"]))
    merged = project_profile(previous, identity, fields)
    for name in fields:
        if name == "io":
            _, io_max = coverage(updated)
            merged[name][:io_max + 1] = updated[name][:io_max + 1]
        else:
            for w in updated["identity"]["widths"]:
                if name in ("cpu", "gpu"):
                    values = updated[name][w]
                    merged[name][w][:len(values)] = values
                else:
                    merged[name][w] = updated[name][w]
    old_widths, old_io = coverage(previous)
    new_widths, new_io = coverage(updated)
    old_widths.update(new_widths)
    merged.update({k: deepcopy(v) for k, v in updated.items() if k not in fields and k not in ("identity", "calibration_coverage")})
    merged["calibration_coverage"] = dict(width_fetch_limits={str(w): q for w, q in old_widths.items()}, io_max=max(old_io, new_io))
    return merged


def initialize_profile(profile, previous, fields):
    if previous is None:
        return profile
    saved = project_profile(previous, profile["identity"], fields)
    for name in fields:
        profile[name] = saved[name]
    for name in ("measurements", "choice_audit", "raw_samples", "copy_zero_us", "calibration_coverage", "reuse"):
        if name in saved:
            profile[name] = saved[name]
    return profile


def finish_coverage(profile):
    identity = profile["identity"]
    measured, io_max = coverage(profile)
    profile["calibration_coverage"] = dict(
        width_fetch_limits={str(w): max(measured.get(w, 0), fetch_limit(identity, w)) for w in identity["widths"]},
        io_max=max(io_max, fetch_limit(identity, max(identity["widths"]))))
    return profile


def load_profile(identity, cache_dir, profile_path, validate, calibrate, fields):
    cache_dir = Path(cache_dir)
    key = hashlib.sha256(json.dumps(compatible_identity(identity), sort_keys=True).encode()).hexdigest()
    path = cache_dir / f"{key}.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        explicit = Path(profile_path) if profile_path else None
        if explicit is not None and not explicit.is_file():
            raise ValueError(f"Hybrid profile does not exist: {profile_path}")
        candidates = ([explicit] if explicit is not None else [])
        candidates += [p for p in sorted(cache_dir.glob("*.json")) if explicit is None or p.resolve() != explicit.resolve()]
        found = []
        for candidate in candidates:
            try:
                raw = candidate.read_bytes()
                saved = json.loads(raw)
                if compatible_identity(saved["identity"]) != compatible_identity(identity):
                    raise ValueError("Hybrid profile identity mismatch; recalibrate for this runtime")
                validate(saved, saved["identity"])
                measured, io_max = coverage(saved)
                stated_io = saved.get("calibration_coverage", {}).get("io_max", io_max)
                experts, top_k = saved["identity"]["geometry"][1], saved["identity"]["geometry"][-1]
                if (set(measured) - set(saved["identity"]["widths"]) or io_max < 0
                        or stated_io != io_max or io_max > min(experts, max(saved["identity"]["widths"]) * top_k)
                        or any(q < 0 or q > min(experts, w * top_k) for w, q in measured.items())):
                    raise ValueError("invalid Hybrid profile coverage")
            except (OSError, ValueError, TypeError, KeyError, IndexError):
                if explicit is not None and candidate == explicit:
                    raise ValueError(f"Hybrid profile identity mismatch or invalid measurements: {candidate}") from None
                continue
            score = (len(identity["widths"]) - len(missing_widths(saved, identity)), io_max, len(measured), candidate == path)
            found.append((score, saved, candidate, raw))
        previous = None
        if found:
            _, previous, source, raw = max(found, key=lambda item: item[0])
        required_io = fetch_limit(identity, max(identity["widths"]))
        missing = missing_widths(previous, identity)
        io_max = coverage(previous)[1] if previous else -1
        if previous is not None and not missing and io_max >= required_io:
            result = project_profile(previous, identity, fields)
            validate(result, identity)
        else:
            seed = project_profile(previous, identity, fields) if previous is not None else None
            result = calibrate(identity, seed)
            validate(result, identity)
            if missing_widths(result, identity) or coverage(result)[1] < required_io:
                raise ValueError("Hybrid calibration did not cover the requested workload")
            if previous is not None:
                result["reuse"] = dict(source_path=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
                                       source_identity=previous["identity"], calibrated_widths=missing,
                                       previous_io_max=io_max)
            stored = merge_profile(previous, result, fields) if previous is not None else result
            validate(stored, stored["identity"])
            destination = path
            if explicit is not None and explicit.resolve() == path.resolve():
                destination = path.with_name(f"{key}.extended.json")
            temporary = destination.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(stored, indent=2) + "\n")
            temporary.replace(destination)
        if previous is not None:
            result["reuse"] = dict(source_path=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
                                   source_identity=previous["identity"], calibrated_widths=missing,
                                   previous_io_max=io_max)
        return result
