"""Coarse/fine ray sampling and differentiable volume rendering."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class RenderOutput:
    rgb: Tensor
    depth: Tensor
    opacity: Tensor
    weights: Tensor
    z_values: Tensor


def sample_pdf(bins: Tensor, weights: Tensor, sample_count: int, deterministic: bool = False) -> Tensor:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cat((torch.zeros_like(pdf[..., :1]), torch.cumsum(pdf, dim=-1)), dim=-1)
    if deterministic:
        u = torch.linspace(0.0, 1.0, sample_count, device=weights.device, dtype=weights.dtype)
        u = u.expand(*cdf.shape[:-1], sample_count)
    else:
        u = torch.rand(*cdf.shape[:-1], sample_count, device=weights.device, dtype=weights.dtype)
    indices = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    below = (indices - 1).clamp(min=0)
    above = indices.clamp(max=cdf.shape[-1] - 1)
    pair_indices = torch.stack((below, above), dim=-1)
    cdf_pair = torch.gather(cdf.unsqueeze(-2).expand(*u.shape, cdf.shape[-1]), -1, pair_indices)
    bins_pair = torch.gather(bins.unsqueeze(-2).expand(*u.shape, bins.shape[-1]), -1, pair_indices)
    denominator = cdf_pair[..., 1] - cdf_pair[..., 0]
    denominator = torch.where(denominator < 1e-5, torch.ones_like(denominator), denominator)
    fraction = (u - cdf_pair[..., 0]) / denominator
    return bins_pair[..., 0] + fraction * (bins_pair[..., 1] - bins_pair[..., 0])


def _query_network(model: nn.Module, points: Tensor, directions: Tensor, chunk_size: int) -> tuple[Tensor, Tensor]:
    flat_points, flat_directions = points.reshape(-1, 3), directions.reshape(-1, 3)
    rgb_chunks, density_chunks = [], []
    for start in range(0, flat_points.shape[0], chunk_size):
        rgb, density = model(flat_points[start : start + chunk_size], flat_directions[start : start + chunk_size])
        rgb_chunks.append(rgb)
        density_chunks.append(density)
    shape = points.shape[:-1]
    return torch.cat(rgb_chunks).reshape(*shape, 3), torch.cat(density_chunks).reshape(*shape, 1)


def volume_render(rgb_logits: Tensor, density_logits: Tensor, z_values: Tensor, ray_directions: Tensor) -> RenderOutput:
    distances = z_values[..., 1:] - z_values[..., :-1]
    infinity = torch.full_like(distances[..., :1], 1e10)
    distances = torch.cat((distances, infinity), dim=-1)
    distances = distances * torch.linalg.vector_norm(ray_directions, dim=-1, keepdim=True)

    alpha = 1.0 - torch.exp(-torch.relu(density_logits.squeeze(-1)) * distances)
    transmittance = torch.cumprod(
        torch.cat((torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10), dim=-1), dim=-1
    )[..., :-1]
    weights = alpha * transmittance
    rgb = torch.sum(weights.unsqueeze(-1) * torch.sigmoid(rgb_logits), dim=-2)
    depth = torch.sum(weights * z_values, dim=-1)
    opacity = torch.sum(weights, dim=-1)
    return RenderOutput(rgb=rgb, depth=depth, opacity=opacity, weights=weights, z_values=z_values)


def render_rays(
    rays_o: Tensor,
    rays_d: Tensor,
    coarse_model: nn.Module,
    fine_model: nn.Module | None = None,
    near: float = 2.0,
    far: float = 6.0,
    coarse_samples: int = 64,
    importance_samples: int = 64,
    chunk_size: int = 32768,
    deterministic: bool = False,
) -> dict[str, RenderOutput]:
    if rays_o.shape != rays_d.shape or rays_o.shape[-1] != 3:
        raise ValueError("rays_o and rays_d must have the same [..., 3] shape")
    if not 0.0 <= near < far or coarse_samples < 2 or chunk_size < 1:
        raise ValueError("require 0 <= near < far, at least two coarse samples, and positive chunk_size")
    ray_shape = rays_o.shape[:-1]
    origins, directions = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    interpolation = torch.linspace(0.0, 1.0, coarse_samples, device=origins.device, dtype=origins.dtype)
    z_coarse = near * (1.0 - interpolation) + far * interpolation
    z_coarse = z_coarse.expand(origins.shape[0], coarse_samples)
    if not deterministic:
        mids = 0.5 * (z_coarse[..., 1:] + z_coarse[..., :-1])
        lower = torch.cat((z_coarse[..., :1], mids), dim=-1)
        upper = torch.cat((mids, z_coarse[..., -1:]), dim=-1)
        z_coarse = lower + (upper - lower) * torch.rand_like(z_coarse)

    points = origins[:, None, :] + directions[:, None, :] * z_coarse[..., None]
    viewdirs = torch.nn.functional.normalize(directions, dim=-1)[:, None, :].expand_as(points)
    coarse_rgb, coarse_density = _query_network(coarse_model, points, viewdirs, chunk_size)
    coarse = volume_render(coarse_rgb, coarse_density, z_coarse, directions)
    outputs = {
        "coarse": RenderOutput(
            rgb=coarse.rgb.reshape(*ray_shape, 3),
            depth=coarse.depth.reshape(*ray_shape),
            opacity=coarse.opacity.reshape(*ray_shape),
            weights=coarse.weights.reshape(*ray_shape, -1),
            z_values=coarse.z_values.reshape(*ray_shape, -1),
        )
    }

    if importance_samples > 0:
        z_mid = 0.5 * (z_coarse[..., 1:] + z_coarse[..., :-1])
        fine_samples = sample_pdf(z_mid, coarse.weights[..., 1:-1].detach(), importance_samples, deterministic)
        z_fine = torch.sort(torch.cat((z_coarse, fine_samples), dim=-1), dim=-1).values
        points_fine = origins[:, None, :] + directions[:, None, :] * z_fine[..., None]
        directions_fine = torch.nn.functional.normalize(directions, dim=-1)[:, None, :].expand_as(points_fine)
        model = fine_model if fine_model is not None else coarse_model
        fine_rgb, fine_density = _query_network(model, points_fine, directions_fine, chunk_size)
        fine = volume_render(fine_rgb, fine_density, z_fine, directions)
        outputs["fine"] = RenderOutput(
            rgb=fine.rgb.reshape(*ray_shape, 3), depth=fine.depth.reshape(*ray_shape),
            opacity=fine.opacity.reshape(*ray_shape), weights=fine.weights.reshape(*ray_shape, -1),
            z_values=fine.z_values.reshape(*ray_shape, -1),
        )
    return outputs


def render_rgb(origins, directions, coarse, fine, config, deterministic=False):
    outputs = render_rays(
        origins, directions, coarse, fine, near=config["near"], far=config["far"],
        coarse_samples=config["coarse_samples"], importance_samples=config["importance_samples"],
        deterministic=deterministic,
    )
    return {name: item.rgb + (1.0 - item.opacity).unsqueeze(-1) for name, item in outputs.items()}


@torch.no_grad()
def render_image(
    height: int, width: int, focal: float | Tensor, camera_to_world: Tensor,
    coarse_model: nn.Module, fine_model: nn.Module | None = None, **render_kwargs,
) -> dict[str, Tensor]:
    from .rays import get_rays

    rays_o, rays_d = get_rays(height, width, focal, camera_to_world)
    flat_o, flat_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    batch_size = int(render_kwargs.pop("ray_batch_size", 1024))
    rendered: dict[str, list[Tensor]] = {}
    for start in range(0, flat_o.shape[0], batch_size):
        result = render_rays(flat_o[start : start + batch_size], flat_d[start : start + batch_size], coarse_model, fine_model, **render_kwargs)
        for name, output in result.items():
            rendered.setdefault(name, []).append(output.rgb)
    return {name: torch.cat(chunks).reshape(height, width, 3) for name, chunks in rendered.items()}
