"""NeRF MLP with a position trunk and view-dependent color head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoding import PositionalEncoding


class _ReferenceLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, activation: str):
        self.activation = activation
        super().__init__(in_features, out_features)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(self.activation))
        nn.init.zeros_(self.bias)


@dataclass(frozen=True)
class NeRFConfig:
    depth: int = 8
    width: int = 256
    position_frequencies: int = 10
    direction_frequencies: int = 4
    skip_layer: int = 4
    use_view_directions: bool = True
    density_activation: str = "relu_logits"
    encoding_convention: str = "legacy"
    initialization: str = "pytorch"
    skip_input_first: bool = False


class NeRF(nn.Module):

    def __init__(self, config: NeRFConfig = NeRFConfig()):
        super().__init__()
        if config.depth < 2 or config.width < 1:
            raise ValueError("depth must be at least 2 and width must be positive")
        if config.density_activation not in {"relu_logits", "softplus"}:
            raise ValueError("density_activation must be relu_logits or softplus")
        self.config = config
        self.position_encoding = PositionalEncoding(3, config.position_frequencies, convention=config.encoding_convention)
        self.direction_encoding = PositionalEncoding(3, config.direction_frequencies, convention=config.encoding_convention)
        position_dims = self.position_encoding.output_dims
        direction_dims = self.direction_encoding.output_dims if config.use_view_directions else 0
        reference_init = config.initialization == "reference"

        def linear(inputs, outputs, activation):
            return _ReferenceLinear(inputs, outputs, activation) if reference_init else nn.Linear(inputs, outputs)

        layers = []
        for layer_index in range(config.depth):
            input_dims = position_dims if layer_index == 0 else config.width
            if layer_index == config.skip_layer:
                input_dims += position_dims
            layers.append(linear(input_dims, config.width, "relu"))
        self.position_layers = nn.ModuleList(layers)
        if reference_init and config.use_view_directions:
            self.color_layer = linear(config.width + direction_dims, config.width // 2, "relu")
            self.feature_head = linear(config.width, config.width, "linear")
            self.density_head = linear(config.width, 1, "linear")
            self.color_head = linear(config.width // 2, 3, "linear")
        elif config.use_view_directions:
            self.density_head = nn.Linear(config.width, 1)
            self.feature_head = nn.Linear(config.width, config.width)
            self.color_layer = nn.Linear(config.width + direction_dims, config.width // 2)
            self.color_head = nn.Linear(config.width // 2, 3)
        else:
            self.density_head = linear(config.width, 1, "linear")
            self.color_head = nn.Linear(config.width, 3)
        if config.initialization == "dietnerf":
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    gain = nn.init.calculate_gain("relu") if layer in self.position_layers or layer is getattr(self, "color_layer", None) else 1.0
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
                    nn.init.zeros_(layer.bias)
        elif config.initialization not in {"pytorch", "reference"}:
            raise ValueError("initialization must be pytorch, dietnerf (legacy), or reference")

    def forward(self, points: Tensor, view_directions: Tensor | None = None) -> tuple[Tensor, Tensor]:
        encoded_points = self.position_encoding(points)
        hidden = encoded_points
        for layer_index, layer in enumerate(self.position_layers):
            if layer_index == self.config.skip_layer:
                parts = (encoded_points, hidden) if self.config.skip_input_first else (hidden, encoded_points)
                hidden = torch.cat(parts, dim=-1)
            hidden = F.relu(layer(hidden))

        density = self.density_head(hidden)
        if self.config.density_activation == "softplus":
            density = F.softplus(density)
        if self.config.use_view_directions:
            if view_directions is None:
                raise ValueError("view_directions are required when use_view_directions=True")
            encoded_directions = self.direction_encoding(view_directions)
            features = self.feature_head(hidden)
            color_features = F.relu(self.color_layer(torch.cat((features, encoded_directions), dim=-1)))
            rgb = self.color_head(color_features)
        else:
            rgb = self.color_head(hidden)
        return rgb, density
