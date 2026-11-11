"""Factorized FastNeRF field and a bounded inference cache."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.encoding import PositionalEncoding


@dataclass(frozen=True)
class FastNeRFConfig:
    components: int = 8
    position_depth: int = 8
    position_width: int = 384
    direction_depth: int = 4
    direction_width: int = 128
    position_frequencies: int = 10
    direction_frequencies: int = 4
    skip_layer: int = 4
    density_activation: str = "softplus"
    encoding_convention: str = "legacy"


class FastNeRF(nn.Module):

    def __init__(self, config: FastNeRFConfig = FastNeRFConfig()):
        super().__init__()
        if min(config.components, config.position_width, config.direction_width) < 1:
            raise ValueError("components and widths must be positive")
        if config.position_depth < 1 or config.direction_depth < 1:
            raise ValueError("network depths must be positive")
        if config.density_activation not in {"softplus", "relu_logits"}:
            raise ValueError("density_activation must be softplus or relu_logits")
        self.config = config
        self.position_encoding = PositionalEncoding(3, config.position_frequencies, convention=config.encoding_convention)
        self.direction_encoding = PositionalEncoding(3, config.direction_frequencies, convention=config.encoding_convention)
        pos_dims = self.position_encoding.output_dims
        dir_dims = self.direction_encoding.output_dims
        self.position_layers = nn.ModuleList([
            nn.Linear((pos_dims if i == 0 else config.position_width) + (pos_dims if i == config.skip_layer and i > 0 else 0), config.position_width)
            for i in range(config.position_depth)
        ])
        self.position_head = nn.Linear(config.position_width, 1 + 3 * config.components)
        self.direction_layers = nn.ModuleList([
            nn.Linear(dir_dims if i == 0 else config.direction_width, config.direction_width)
            for i in range(config.direction_depth)
        ])
        self.direction_head = nn.Linear(config.direction_width, config.components)

    def position_field(self, points: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.position_encoding(points)
        hidden = encoded
        for i, layer in enumerate(self.position_layers):
            if i == self.config.skip_layer and i > 0:
                hidden = torch.cat((hidden, encoded), dim=-1)
            hidden = F.relu(layer(hidden))
        output = self.position_head(hidden)
        density_logits = output[..., :1]
        density = (F.softplus(density_logits) if self.config.density_activation == "softplus"
                   else density_logits)
        basis = torch.sigmoid(output[..., 1:].reshape(*points.shape[:-1], 3, self.config.components))
        return basis, density

    def direction_field(self, directions: Tensor) -> Tensor:
        directions = F.normalize(directions, dim=-1)
        hidden = self.direction_encoding(directions)
        for layer in self.direction_layers:
            hidden = F.relu(layer(hidden))
        return F.softmax(self.direction_head(hidden), dim=-1)

    def forward(self, points: Tensor, view_directions: Tensor) -> tuple[Tensor, Tensor]:
        basis, density = self.position_field(points)
        weights = self.direction_field(view_directions)
        rgb = (basis * weights.unsqueeze(-2)).sum(dim=-1)
        return torch.logit(rgb.clamp(1e-6, 1 - 1e-6)), density


class FastNeRFCache(nn.Module):

    def __init__(self, position_grid: Tensor, direction_grid: Tensor,
                 lower: Tensor, upper: Tensor, components: int, skip_empty: bool = False):
        super().__init__()
        self.register_buffer("position_grid", position_grid)
        self.register_buffer("direction_grid", direction_grid)
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        self.components = components
        self.skip_empty = skip_empty

    @classmethod
    @torch.inference_mode()
    def from_model(cls, model: FastNeRF, *, resolution: int = 64,
                   direction_height: int = 32, direction_width: int = 64,
                   lower: tuple[float, float, float] = (-1.5, -1.5, -1.5),
                   upper: tuple[float, float, float] = (1.5, 1.5, 1.5),
                   max_cache_mb: float = 128, chunk_size: int = 8192,
                   skip_empty: bool = False, storage_dtype: torch.dtype | None = None) -> FastNeRFCache:
        if resolution < 2 or direction_height < 2 or direction_width < 2 or chunk_size < 1:
            raise ValueError("cache dimensions and chunk size must be at least 2")
        channels = 1 + 3 * model.config.components
        storage_dtype = storage_dtype or next(model.parameters()).dtype
        bytes_per_value = torch.empty((), dtype=storage_dtype).element_size()
        bytes_needed = bytes_per_value * (channels * resolution ** 3 + model.config.components * direction_height * (direction_width + 1))
        if bytes_needed > max_cache_mb * 1024 ** 2:
            raise ValueError(f"cache needs {bytes_needed / 1024 ** 2:.1f} MiB; limit is {max_cache_mb} MiB")
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        low = torch.tensor(lower, device=device, dtype=dtype)
        high = torch.tensor(upper, device=device, dtype=dtype)
        if not torch.all(high > low):
            raise ValueError("every upper bound must exceed its lower bound")
   
        count = resolution ** 3
        position_grid = torch.empty((1, channels, resolution, resolution, resolution),
                                    device=device, dtype=storage_dtype)
        flat_grid = position_grid.view(channels, count)
        was_training = model.training
        model.eval()
        try:
            for start in range(0, count, chunk_size):
                indices = torch.arange(start, min(start + chunk_size, count), device=device)
                xyz = torch.stack((indices % resolution, (indices // resolution) % resolution,
                                   indices // (resolution ** 2)), dim=-1).to(dtype)
                part = low + xyz * (high - low) / (resolution - 1)
                basis, density = model.position_field(part)
                flat_grid[:, start:start + len(part)] = torch.cat((basis.reshape(len(part), -1), density), dim=-1).T

            theta = torch.linspace(0, math.pi, direction_height, device=device, dtype=dtype)
            phi = torch.linspace(0, 2 * math.pi, direction_width + 1, device=device, dtype=dtype)
            t, p = torch.meshgrid(theta, phi, indexing="ij")
            directions = torch.stack((t.sin() * p.cos(), t.sin() * p.sin(), t.cos()), dim=-1).reshape(-1, 3)
            weights = torch.cat([model.direction_field(part) for part in directions.split(chunk_size)])
            direction_grid = weights.T.reshape(1, model.config.components, direction_height, direction_width + 1).to(storage_dtype)
        finally:
            model.train(was_training)
        return cls(position_grid, direction_grid, low, high, model.config.components, skip_empty)

    def forward(self, points: Tensor, view_directions: Tensor) -> tuple[Tensor, Tensor]:
        if points.shape != view_directions.shape or points.shape[-1] != 3:
            raise ValueError("points and view_directions must share shape [..., 3]")
        shape = points.shape[:-1]
        flat_points = points.reshape(-1, 3)
        xyz = 2 * (flat_points - self.lower) / (self.upper - self.lower) - 1
        pos_query = xyz.reshape(1, -1, 1, 1, 3).to(self.position_grid.dtype)
        sampled = F.grid_sample(self.position_grid, pos_query, mode="bilinear", padding_mode="border", align_corners=True)
        sampled = sampled.reshape(1 + 3 * self.components, -1).T.to(points.dtype)
        basis = sampled[:, :-1].reshape(-1, 3, self.components)
        density = sampled[:, -1:]
        inside = ((xyz >= -1) & (xyz <= 1)).all(dim=-1, keepdim=True)
        density = torch.where(inside, density, torch.full_like(density, -1e6))

        directions = F.normalize(view_directions.reshape(-1, 3), dim=-1)
        if self.skip_empty:
            active = density.squeeze(-1) > 0
            rgb = torch.full((len(flat_points), 3), 0.5, device=points.device, dtype=points.dtype)
            if bool(active.any()):
                rgb[active] = self._color(basis[active], directions[active])
        else:
            rgb = self._color(basis, directions)
        return torch.logit(rgb.clamp(1e-6, 1 - 1e-6)).reshape(*shape, 3), density.reshape(*shape, 1)

    def _color(self, basis: Tensor, directions: Tensor) -> Tensor:
        theta = torch.acos(directions[:, 2].clamp(-1, 1))
        phi = torch.remainder(torch.atan2(directions[:, 1], directions[:, 0]), 2 * math.pi)
        uv = torch.stack((phi / math.pi - 1, 2 * theta / math.pi - 1), dim=-1).reshape(1, -1, 1, 2)
        weights = F.grid_sample(self.direction_grid, uv.to(self.direction_grid.dtype), mode="bilinear", padding_mode="border", align_corners=True)
        weights = weights.reshape(self.components, -1).T.to(basis.dtype)
        return (basis * weights.unsqueeze(1)).sum(dim=-1)
