"""Frozen OpenCLIP image encoder with differentiable input preprocessing."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FrozenCLIPImageEncoder(nn.Module):

    def __init__(self, cache_dir: str | None = None):
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("Install DietNeRF dependencies with `uv sync --locked --group dietnerf`.") from exc
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", force_quick_gelu=True, cache_dir=cache_dir)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.image_size = 224
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def encode(self, images: Tensor) -> Tensor:
        """Encode [B,H,W,3] or [B,3,H,W] RGB tensors in [0,1]."""
        if images.ndim != 4:
            raise ValueError("CLIP input must be a batch of RGB images")
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        elif images.shape[1] != 3:
            raise ValueError("CLIP input must have three RGB channels")
        images = images.float().clamp(0.0, 1.0)
        images = F.interpolate(images, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        images = (images - self.mean) / self.std
        visual = self.model.visual
        features = visual.transformer(visual._embeds(images))[:, 0]
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def cache_observed(self, images: Tensor, batch_size: int = 4) -> Tensor:
        features = []
        for start in range(0, images.shape[0], batch_size):
            batch = images[start:start + batch_size].to(self.mean.device)
            features.append(self.encode(batch))
        return torch.cat(features, dim=0).detach()
