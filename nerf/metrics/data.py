
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch


@dataclass
class EvaluationDataset:
    images: torch.Tensor  # [N,H,W,3], RGB in [0,1]
    poses: torch.Tensor  # [N,4,4], camera-to-world
    focal: float
    indices: torch.Tensor
    split: str

    @property
    def height(self) -> int:
        return self.images.shape[1]

    @property
    def width(self) -> int:
        return self.images.shape[2]


def load_blender_scene(root: str | Path, split: str = "test", device: torch.device | None = None,
                       white_background: bool = True, downsample: int = 1,
                       resize_rgba_first: bool = True) -> EvaluationDataset:
  
    root = Path(root)
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    if downsample < 1:
        raise ValueError("downsample must be a positive integer")
    device = device or torch.device("cpu")
    metadata_path = root / f"transforms_{split}.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Blender transforms file not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    angle_x = float(metadata["camera_angle_x"])
    images, poses = [], []
    for frame in metadata["frames"]:
        prefix = root / frame["file_path"]
        image_path = prefix if prefix.suffix else prefix.with_suffix(".png")
        if not image_path.is_file() and image_path.suffix.lower() == ".png":
            image_path = image_path.with_suffix(".jpg")
        if not image_path.is_file():
            raise FileNotFoundError(f"image listed in {metadata_path} was not found: {frame['file_path']}")
        image = iio.imread(image_path)
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(f"expected RGB/RGBA image at {image_path}, got shape {image.shape}")
        if np.issubdtype(image.dtype, np.integer):
            scale = float(np.iinfo(image.dtype).max)
        else:
            scale = 1.0
        image = image.astype(np.float32) / scale
        background = 1.0 if white_background else 0.0
        if image.shape[-1] == 4 and not resize_rgba_first:
            alpha = image[..., 3:4]
            image = image[..., :3] * alpha + background * (1.0 - alpha)
        if downsample > 1:
            # Area averaging gives a simple anti-aliased downsample without OpenCV.
            h, w = image.shape[:2]
            new_h, new_w = h // downsample, w // downsample
            if new_h < 1 or new_w < 1 or h % downsample or w % downsample:
                raise ValueError(f"image dimensions {w}x{h} must be divisible by downsample={downsample}")
            image = image.reshape(new_h, downsample, new_w, downsample, image.shape[-1]).mean(axis=(1, 3))
        if image.shape[-1] == 4:
            alpha = image[..., 3:4]
            image = image[..., :3] * alpha + background * (1.0 - alpha)
        images.append(image[..., :3])
        poses.append(np.asarray(frame["transform_matrix"], dtype=np.float32))
    if not images:
        raise ValueError(f"split {split} in {root} contains no frames")
    height, width = images[0].shape[:2]
    if any(im.shape != (height, width, 3) for im in images):
        raise ValueError("all images in a split must have the same dimensions")
    focal = 0.5 * width / np.tan(0.5 * angle_x)
    image_tensor = torch.from_numpy(np.stack(images)).to(device=device, dtype=torch.float32)
    pose_tensor = torch.from_numpy(np.stack(poses)).to(device=device, dtype=torch.float32)
    indices = torch.arange(len(images), device=device)
    return EvaluationDataset(image_tensor, pose_tensor, float(focal), indices, split)


def select_train_views(count: int, requested: int, seed: int) -> torch.Tensor:
    if requested == 0 or requested == count:
        return torch.arange(count)
    if not 0 < requested <= count:
        raise ValueError(f"requested {requested} training views; got {count}")
    generator = torch.Generator().manual_seed(seed)
    phase = int(torch.randint(count, (1,), generator=generator).item())
    indices = torch.linspace(0, count, requested + 1)[:-1].round().long()
    return torch.sort((indices + phase) % count).values
