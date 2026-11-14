from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, LayerNorm, LinearReplicated, OPList
from freetoken.models.weight_stream import BlockWeightStreamer


class PatchProjection(BaseOP):
    def __init__(self, width, patch):
        self.weight = torch.empty(width, 3, patch, patch)
        self.bias = torch.empty(width)
        self._patch = patch

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=self._patch)


class PatchEmbed(BaseOP):
    def __init__(self, vc):
        self.proj = PatchProjection(vc.hidden_size, vc.patch_size)

    def forward(self, x):
        return self.proj.forward(x).flatten(2).transpose(1, 2)


class VisionAttention(BaseOP):
    def __init__(self, vc):
        self.qkv = LinearReplicated(vc.hidden_size, vc.hidden_size * 3, has_bias=True)
        self.proj = LinearReplicated(vc.hidden_size, vc.hidden_size, has_bias=True)
        self._heads = vc.num_heads

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv.forward(x).view(b, n, 3, self._heads, d // self._heads).permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj.forward(out.transpose(1, 2).reshape(b, n, d))


class VisionMLP(BaseOP):
    def __init__(self, width, intermediate, out_width=None, approximate="tanh"):
        self.fc1 = LinearReplicated(width, intermediate, has_bias=True)
        self.fc2 = LinearReplicated(intermediate, out_width or width, has_bias=True)
        self._approximate = approximate

    def forward(self, x):
        return self.fc2.forward(F.gelu(self.fc1.forward(x), approximate=self._approximate))


class VisionBlock(BaseOP):
    def __init__(self, vc):
        self.norm1 = LayerNorm(vc.hidden_size, 1e-6)
        self.attn = VisionAttention(vc)
        self.norm2 = LayerNorm(vc.hidden_size, 1e-6)
        self.mlp = VisionMLP(vc.hidden_size, vc.intermediate_size)

    def forward(self, x):
        x = x + self.attn.forward(self.norm1.forward(x))
        return x + self.mlp.forward(self.norm2.forward(x))


def format_image_features(features, crop, newline, separator, global_view_pos="head"):
    nw, nh = crop
    _, tokens, dim = features.shape
    side = int(tokens ** .5)
    global_view = features[0].view(side, side, dim)
    global_view = torch.cat((global_view, newline.expand(side, 1, dim)), 1).reshape(-1, dim)
    local = features[1:].view(nh, nw, side, side, dim).permute(0, 2, 1, 3, 4).reshape(nh * side, nw * side, dim)
    local = torch.cat((local, newline.expand(nh * side, 1, dim)), 1).reshape(-1, dim)
    first, second = (global_view, local) if global_view_pos == "head" else (local, global_view)
    return torch.cat((first, separator[None], second), 0)


class DeepseekVL2Vision(BaseOP):
    def __init__(self, vc):
        self.patch_embed = PatchEmbed(vc)
        self.pos_embed = torch.empty(1, (vc.image_size // vc.patch_size) ** 2, vc.hidden_size)
        self.blocks = OPList([VisionBlock(vc) for _ in range(vc.depth)])
        self.norm = LayerNorm(vc.hidden_size, 1e-6)
        self.projector = VisionMLP(vc.hidden_size * vc.downsample_ratio ** 2, vc.text_hidden_size,
                                   vc.text_hidden_size, approximate="none")
        self.image_newline = torch.empty(vc.text_hidden_size)
        self.view_seperator = torch.empty(vc.text_hidden_size)
        self._vc = vc
        self._streamer = None

    def place_weights(self, mode):
        if mode == "host" and self._streamer is None:
            self._streamer = BlockWeightStreamer(self.blocks.op_list, self.pos_embed.device)
        elif mode == "gpu" and self._streamer is not None:
            self._streamer.unstream()
            self._streamer = None
        elif mode not in ("gpu", "host"):
            raise ValueError(f"unknown encoder placement: {mode}")

    def forward(self, pixels, crop):
        x = self.patch_embed.forward(pixels.to(device=self.pos_embed.device, dtype=self.pos_embed.dtype)) + self.pos_embed
        blocks = self.blocks.op_list if self._streamer is None else (block for _, block in self._streamer.blocks(self.blocks.op_list))
        for block in blocks:
            x = block.forward(x)
        x = self.norm.forward(x)
        side = self._vc.image_size // self._vc.patch_size
        ratio = self._vc.downsample_ratio
        pad = (-side) % ratio
        x = F.pad(x.view(-1, side, side, self._vc.hidden_size), (0, 0, 0, pad, 0, pad)).permute(0, 3, 1, 2)
        x = F.unfold(x, kernel_size=ratio, stride=ratio).transpose(1, 2)
        features = self.projector.forward(x)
        return format_image_features(features, crop, self.image_newline, self.view_seperator, self._vc.global_view_pos)


class DeepseekVL2VisionMixin:
    def encode(self, item):
        return self.visual.forward(item.feature, item.model_specific_data["crop"])

    def place_encoder_weights(self, mode):
        if hasattr(self, "visual"):
            self.visual.place_weights(mode)
