"""Import trusted upstream MoDES calibration pickles without using a GPU."""

from __future__ import annotations

import argparse
import io
import pickle
from pathlib import Path

import torch

from freetoken.moe.modes import MODES_COMMIT, validate_calibration
from freetoken.moe.tablemoe_artifacts import (
    checkpoint_identity, file_hash, finish_stage, read_json, write_json,
)


class CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def read_pickle(path):
    with Path(path).open("rb") as handle:
        return CpuUnpickler(handle).load()


def scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("MoDES importance and tau must be scalars")
        value = value.item()
    return float(value)


def import_config(importance_path, search_path, model_path, output, *, zero_tau=False,
                  checkpoint_id=None, calibration_info=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("MoDES output directory must be empty")
    hf = read_json(Path(model_path) / "config.json")
    if "Qwen3VLMoeForConditionalGeneration" not in hf.get("architectures", []):
        raise ValueError("MoDES import requires Qwen3-VL-MoE")
    text = hf.get("text_config", hf)
    layers = list(range(text["num_hidden_layers"]))
    importance = read_pickle(importance_path)
    if set(importance) != set(layers):
        raise ValueError("importance pickle must cover every physical layer exactly")
    result = read_pickle(search_path) if search_path else None
    if result is None and not zero_tau:
        raise ValueError("search result pickle is required unless exporting a zero-tau control")
    tau = {"text": 0., "visual": 0.} if zero_tau else result["tau"]
    data = dict(
        schema="freetoken_modes_v1", checkpoint=checkpoint_identity(model_path),
        checkpoint_id=checkpoint_id, checkpoint_association="declared_by_importer",
        modes_source=dict(repository="git@github.com:fanyalun/MoDES.git", commit=MODES_COMMIT),
        control=bool(zero_tau), calibration_info=calibration_info,
        layers={str(i): dict(importance={m: scalar(importance[i][m]) for m in ("text", "visual")},
                             tau={m: scalar(tau[m]) for m in ("text", "visual")}) for i in layers},
    )
    validate_calibration(data, layers)
    inputs = dict(importance=dict(path=str(Path(importance_path).resolve()), sha256=file_hash(importance_path)),
                  search=None if search_path is None else
                  dict(path=str(Path(search_path).resolve()), sha256=file_hash(search_path)),
                  source=data["modes_source"], checkpoint=data["checkpoint"],
                  checkpoint_association=data["checkpoint_association"], control=bool(zero_tau),
                  calibration_info=calibration_info)
    config_path = output / "modes_config.json"
    write_json(config_path, data)
    finish_stage(output, inputs, [config_path], physical_layer_ids=layers,
                 checkpoint_metadata_automatically_verified=False)
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance", required=True, help="trusted upstream importance pickle")
    parser.add_argument("--search-result", help="trusted upstream search result pickle containing tau")
    parser.add_argument("--model", required=True, help="local raw safetensors checkpoint")
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-info", help="optional JSON describing the original calibration")
    parser.add_argument("--zero-tau-control", action="store_true")
    args = parser.parse_args()
    path = import_config(args.importance, args.search_result, args.model, args.output,
                         zero_tau=args.zero_tau_control, checkpoint_id=args.checkpoint_id,
                         calibration_info=read_json(args.calibration_info) if args.calibration_info else None)
    print(path)


if __name__ == "__main__":
    main()
