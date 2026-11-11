"""Fourier feature encoding used by the NeRF model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):

    def __init__(self, input_dims: int, num_frequencies: int, include_input: bool = True,
                 convention: str = "legacy"):
        super().__init__()
        if input_dims < 1 or num_frequencies < 0:
            raise ValueError("input_dims must be positive and num_frequencies non-negative")
        self.input_dims = input_dims
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        if convention not in {"legacy", "nerf"}:
            raise ValueError("encoding convention must be legacy or nerf")
        self.convention = convention
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_frequencies), persistent=False)

    @property
    def output_dims(self) -> int:
        return self.input_dims * (int(self.include_input) + 2 * self.num_frequencies)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dims:
            raise ValueError(f"expected final dimension {self.input_dims}, got {x.shape[-1]}")
        pieces = [x] if self.include_input else []
        if self.num_frequencies:
            phase = x.unsqueeze(-2) * self.frequencies.to(dtype=x.dtype).view(-1, 1)
            if self.convention == "legacy":
                phase = phase * math.pi
                encoded = torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)
            else:
                encoded = torch.stack((torch.sin(phase), torch.cos(phase)), dim=-2)
            pieces.append(encoded.flatten(start_dim=-3))
        return torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]
