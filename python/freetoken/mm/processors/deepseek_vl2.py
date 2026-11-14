from __future__ import annotations

import json
import math
import struct

import numpy as np
import torch
from PIL import ImageOps

from freetoken.message import MMItem
from freetoken.mm import mm_pad_value
from freetoken.mm.processor import MMProcessor, PromptReplacement, content_hash
from freetoken.utils.hf import optional_hf_file


def select_best_resolution(size, candidates):
    ow, oh = size
    best, maximum, wasted_min = None, -1, float("inf")
    for width, height in candidates:
        scale = min(width / ow, height / oh)
        effective = min(int(ow * scale) * int(oh * scale), ow * oh)
        wasted = width * height - effective
        if effective > maximum or (effective == maximum and wasted < wasted_min):
            best, maximum, wasted_min = (width, height), effective, wasted
    return best


class DeepseekVL2MMProcessor(MMProcessor):
    def __init__(self, hf_config, model_path, mm):
        super().__init__(model_path, mm)
        path = optional_hf_file(model_path, "processor_config.json")
        self._settings = json.loads(open(path).read()) if path else {}
        vc = hf_config.vision_config
        self._size = getattr(vc, "image_size", 384)
        self._patch = getattr(vc, "patch_size", 16)
        self._ratio = self._settings.get("downsample_ratio", 2)
        self._candidates = self._settings.get("candidate_resolutions", getattr(hf_config, "candidate_resolutions", [[384, 384]]))
        self.placeholder = [128815]
        if mm.image_min_tokens is not None or mm.image_max_tokens is not None or mm.processor_kwargs:
            raise ValueError("DeepSeek-VL2 currently uses the checkpoint's native image preprocessing defaults")

    def process(self, images):
        mean = self._settings.get("image_mean", [.5, .5, .5])
        std = self._settings.get("image_std", [.5, .5, .5])
        mean_t = torch.tensor(mean)[:, None, None]
        std_t = torch.tensor(std)[:, None, None]
        fill = tuple(int(x * 255) for x in mean)
        items = []
        for image in images:
            image = image.convert("RGB")
            width, height = select_best_resolution(image.size, self._candidates) if len(images) <= 2 else (self._size, self._size)
            global_view = ImageOps.pad(image, (self._size, self._size), color=fill)
            local = ImageOps.pad(image, (width, height), color=fill)
            tiles = [global_view] + [local.crop((x, y, x + self._size, y + self._size))
                                     for y in range(0, height, self._size) for x in range(0, width, self._size)]
            pixels = torch.stack([torch.from_numpy(np.array(tile, copy=True)).permute(2, 0, 1).float().div(255) for tile in tiles])
            if self._settings.get("normalize", True):
                pixels = (pixels - mean_t) / std_t
            crop = [width // self._size, height // self._size]
            item_hash = content_hash(pixels, struct.pack("<2i", *crop))
            items.append(MMItem(modality="image", hash=item_hash, pad_value=mm_pad_value(item_hash), offsets=[],
                                feature=pixels, model_specific_data={"crop": crop}))
        return items

    def prompt_replacement(self, item):
        nw, nh = item.model_specific_data["crop"]
        side = math.ceil((self._size // self._patch) / self._ratio)
        count = side * (side + 1) + 1 + nh * side * (nw * side + 1)
        return PromptReplacement([self.placeholder[0]] * count)

    def dummy_items(self, dtype, device):
        side = math.ceil((self._size // self._patch) / self._ratio)
        count = 2 * side * (side + 1) + 1
        return [MMItem(modality="image", hash=0, pad_value=0, offsets=[[0, count]],
                       feature=torch.zeros(2, 3, self._size, self._size, dtype=dtype, device=device),
                       model_specific_data={"crop": [1, 1]})]
