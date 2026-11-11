"""Camera ray generation for the Blender/Tiny-NeRF camera convention."""

from __future__ import annotations

import torch
from torch import Tensor


def get_rays(height: int, width: int, focal: float | Tensor, camera_to_world: Tensor) -> tuple[Tensor, Tensor]:

    if height < 1 or width < 1 or camera_to_world.shape != (4, 4):
        raise ValueError("height/width must be positive and camera_to_world must be 4x4")
    device, dtype = camera_to_world.device, camera_to_world.dtype
    focal = torch.as_tensor(focal, device=device, dtype=dtype)
    rows, columns = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    directions_camera = torch.stack(
        ((columns - width * 0.5) / focal, -(rows - height * 0.5) / focal, -torch.ones_like(columns)),
        dim=-1,
    )
    directions_world = directions_camera @ camera_to_world[:3, :3].T
    origins_world = camera_to_world[:3, 3].expand_as(directions_world)
    return origins_world, directions_world


def get_rays_at_pixels(
    height: int,
    width: int,
    focal: float | Tensor,
    camera_to_world: Tensor,
    rows: Tensor,
    columns: Tensor,
) -> tuple[Tensor, Tensor]:
  
    if rows.shape != columns.shape:
        raise ValueError("rows and columns must have the same shape")
    focal = torch.as_tensor(focal, device=camera_to_world.device, dtype=camera_to_world.dtype)
    directions_camera = torch.stack(
        (
            (columns.to(camera_to_world.dtype) - width * 0.5) / focal,
            -(rows.to(camera_to_world.dtype) - height * 0.5) / focal,
            -torch.ones_like(rows, dtype=camera_to_world.dtype),
        ),
        dim=-1,
    )
    directions_world = directions_camera @ camera_to_world[:3, :3].T
    origins_world = camera_to_world[:3, 3].expand_as(directions_world)
    return origins_world, directions_world
