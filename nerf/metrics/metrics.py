
from __future__ import annotations

from typing import Any

import numpy as np


def _as_hwc(image: Any, name: str) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D RGB image, got shape {array.shape}")
    if array.shape[-1] == 3:
        pass
    elif array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    else:
        raise ValueError(f"{name} must have 3 RGB channels, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _validate_pair(prediction: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    pred, gt = _as_hwc(prediction, "prediction"), _as_hwc(target, "target")
    if pred.shape != gt.shape:
        raise ValueError(f"prediction and target must have the same shape, got {pred.shape} and {gt.shape}")
    if pred.min() < 0.0 or pred.max() > 1.0 or gt.min() < 0.0 or gt.max() > 1.0:
        raise ValueError("images must be in [0, 1]; clamp model output before metric evaluation")
    return pred, gt


def psnr(prediction: Any, target: Any, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB; exact matches return positive infinity."""
    pred, gt = _validate_pair(prediction, target)
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    mse = float(np.mean(np.square(pred - gt), dtype=np.float64))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(data_range**2 / mse))


def ssim(prediction: Any, target: Any, data_range: float = 1.0) -> float:
   
    pred, gt = _validate_pair(prediction, target)
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError("SSIM requires the DietNeRF dependency group: uv sync --group dietnerf") from exc
   
    min_side = min(pred.shape[:2])
    if min_side < 3:
        raise ValueError("SSIM requires image height and width to be at least 3")
    win_size = min(11, min_side if min_side % 2 else min_side - 1)
    return float(structural_similarity(
        pred, gt, channel_axis=-1, data_range=data_range, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False, win_size=win_size,
    ))


def lpips_distance(prediction: Any, target: Any, model: Any, device: Any = None) -> float:
 
    import torch

    pred, gt = _validate_pair(prediction, target)
    device = device or next(model.parameters()).device
    pred_t = torch.from_numpy(np.moveaxis(pred, -1, 0)).unsqueeze(0).to(device) * 2.0 - 1.0
    gt_t = torch.from_numpy(np.moveaxis(gt, -1, 0)).unsqueeze(0).to(device) * 2.0 - 1.0
    model.eval()
    with torch.inference_mode():
        value = model(pred_t, gt_t)
    return float(value.reshape(-1).mean().item())


def compute_image_metrics(
    prediction: Any, target: Any, lpips_model: Any = None, device: Any = None,
) -> dict[str, float]:
  
    pred, gt = _validate_pair(prediction, target)
    result = {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt)}
    if lpips_model is not None:
        result["lpips"] = lpips_distance(pred, gt, lpips_model, device)
    return result
