"""A small, readable PyTorch implementation of the original NeRF pipeline."""

from .model import NeRF, NeRFConfig
from .rendering import render_rays, render_image

__all__ = ["NeRF", "NeRFConfig", "render_rays", "render_image"]
