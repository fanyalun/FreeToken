from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch


@dataclass(frozen=True)
class PreverifyPolicy:
    kind: str = "fixed_topk"
    top_k: int = 4
    replacement: str = "zero"
    importance: tuple[float, ...] = ()
    tau: float = 0.0
    asset_sha256: str = ""

    def __post_init__(self):
        if self.kind not in ("fixed_topk", "modes") or self.replacement not in ("zero", "identity"):
            raise ValueError("invalid Pre-Verify policy or replacement")
        if type(self.top_k) is not int or not 1 <= self.top_k <= 8:
            raise ValueError("Pre-Verify top-k must be in [1, 8]")
        if not math.isfinite(self.tau) or self.tau < 0:
            raise ValueError("Pre-Verify threshold must be finite and nonnegative")
        if self.kind == "modes" and (not self.importance or
                any(not math.isfinite(x) or x < 0 for x in self.importance) or
                not math.isclose(sum(self.importance), 1.0, abs_tol=1e-6)):
            raise ValueError("MoDES requires normalized finite layer importance")

    @classmethod
    def from_config(cls, config, *, layers, model_path):
        kind = getattr(config, "speculative_preverify_policy", "fixed_topk")
        top_k = getattr(config, "speculative_preverify_top_k", 4)
        replacement = getattr(config, "speculative_preverify_replacement", "zero")
        path = getattr(config, "speculative_modes_calibration", None)
        if kind != "modes":
            if path:
                raise ValueError("MoDES calibration requires the modes policy")
            return cls(kind=kind, top_k=top_k, replacement=replacement)
        if not path:
            raise ValueError("MoDES requires a calibration asset")
        raw = Path(path).read_bytes()
        asset = json.loads(raw)
        model_hash = hashlib.sha256((Path(model_path) / "config.json").read_bytes()).hexdigest()
        if (asset["model_config_sha256"] != model_hash or asset["dtype"] != "bfloat16"
                or asset["top_k"] != 8 or asset["replacement"] != replacement
                or asset["score_dtype"] != "float32" or asset["status"] != "validated"):
            raise ValueError("MoDES calibration model, dtype, replacement or status mismatch")
        index = Path(model_path) / "model.safetensors.index.json"
        if asset.get("weight_index_sha256") != hashlib.sha256(index.read_bytes()).hexdigest():
            raise ValueError("MoDES checkpoint weight index mismatch")
        for name, identity in asset.get("weight_shard_stat", {}).items():
            stat = (Path(model_path) / name).stat()
            if dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != identity:
                raise ValueError("MoDES checkpoint shard identity changed")
        importance = tuple(asset["importance"])
        if len(importance) != layers:
            raise ValueError("MoDES calibration layer coverage mismatch")
        return cls(kind, top_k, replacement, importance, float(asset["tau"]), hashlib.sha256(raw).hexdigest())

    def mask(self, weights, layer, ready=None):
        if self.kind == "modes":
            keep = weights.float() * self.importance[layer] >= self.tau
        else:
            rank = weights.argsort(dim=-1, descending=True, stable=True).argsort(dim=-1)
            keep = rank < self.top_k
        return keep if ready is None else keep | ready

    def identity_term(self, hidden, weights, keep):
        if self.replacement == "zero":
            return None
        # Resident GEMM can overwrite hidden, so materialize this before expert execution.
        mass = torch.where(keep, 0.0, weights.float()).sum(-1, keepdim=True)
        return hidden.float() * mass


def merge_identity(result, identity):
    return result if identity is None else (result.float() + identity).to(result.dtype)


def add_policy_arguments(parser):
    parser.add_argument("--preverify-policy", choices=("fixed_topk", "modes"), default="fixed_topk")
    parser.add_argument("--preverify-top-k", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--preverify-replacement", choices=("zero", "identity"), default="zero")
    parser.add_argument("--modes-calibration")


def policy_config(args):
    return dict(speculative_preverify_policy=args.preverify_policy,
                speculative_preverify_top_k=args.preverify_top_k,
                speculative_preverify_replacement=args.preverify_replacement,
                speculative_modes_calibration=args.modes_calibration)
