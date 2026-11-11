"""Evaluate a NeRF/DietNeRF checkpoint on the Lego test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics.data import load_blender_scene
from metrics.metrics import compute_image_metrics
from core.model import NeRF, NeRFConfig
from core.rays import get_rays
from core.rendering import render_rgb


def load_checkpoint(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("training_config", {}).get("model_type") == "fastnerf":
        raise ValueError("evaluate FastNeRF checkpoints with fastnerf/run.py test")
    config = NeRFConfig(**saved["model_config"])
    coarse, fine = NeRF(config).to(device), NeRF(config).to(device)
    coarse.load_state_dict(saved["coarse_model"])
    fine.load_state_dict(saved["fine_model"])
    return coarse.eval(), fine.eval(), saved.get("training_config", {})


@torch.inference_mode()
def evaluate_dataset(dataset, coarse_model, fine_model, indices=None, device=None, lpips_model=None,
                     near=2.0, far=6.0, coarse_samples=64, importance_samples=128,
                     ray_batch_size=1024, previous_rows=None, on_image=None):
    device = device or next(coarse_model.parameters()).device
    indices = dataset.indices if indices is None else indices
    indices = torch.as_tensor(indices, dtype=torch.long).cpu().tolist()
    rows = list(previous_rows or [])
    finished = {row["index"] for row in rows}
    if len(finished) != len(rows) or not finished.issubset(set(indices)):
        raise ValueError("saved evaluation views do not match this evaluation")
    config = dict(near=near, far=far, coarse_samples=coarse_samples, importance_samples=importance_samples)
    for index in indices:
        if index in finished:
            continue
        rays_o, rays_d = get_rays(dataset.height, dataset.width, dataset.focal, dataset.poses[index].to(device))
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        parts = []
        for start in range(0, len(rays_o), ray_batch_size):
            outputs = render_rgb(rays_o[start:start + ray_batch_size], rays_d[start:start + ray_batch_size],
                                 coarse_model, fine_model, config, deterministic=True)
            parts.append(outputs["fine" if "fine" in outputs else "coarse"])
        prediction = torch.cat(parts).reshape(dataset.height, dataset.width, 3).clamp(0, 1)
        target = dataset.images[index].to(device)
        values = compute_image_metrics(prediction, target, lpips_model, device)
        rows.append(dict(index=index, mse=float(torch.mean((prediction - target) ** 2).item()), **values))
        if on_image is not None:
            on_image(rows)
    if not rows:
        raise ValueError("evaluation needs at least one selected image")
    squared_error = sum(row["mse"] * int(dataset.images[row["index"]].numel()) for row in rows)
    pixel_count = sum(int(dataset.images[row["index"]].numel()) for row in rows)
    mean_mse = squared_error / pixel_count
    summary = dict(
        num_images=len(rows), mean_per_image_psnr_db=float(np.mean([row["psnr"] for row in rows])),
        pooled_psnr_db=-10 * math.log10(mean_mse) if mean_mse else float("inf"),
        mean_ssim=float(np.mean([row["ssim"] for row in rows])), mean_mse=mean_mse,
    )
    if lpips_model is not None:
        summary["mean_lpips"] = float(np.mean([row["lpips"] for row in rows]))
    return dict(summary=summary, per_image=rows)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume completed test views")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    coarse, fine, training = load_checkpoint(args.checkpoint, device)
    dataset = load_blender_scene(args.data, "test", downsample=2)
    options = dict(near=float(training.get("near", 2)), far=float(training.get("far", 6)),
                   coarse_samples=int(training.get("coarse_samples", 64)),
                   importance_samples=int(training.get("importance_samples", 128)), ray_batch_size=1024)
    lpips_model = None
    if args.lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
        for parameter in lpips_model.parameters():
            parameter.requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)
    previous_rows, on_image = [], None
    if args.resume:
        identity = dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                        data=str(args.data.resolve()), split="test", downsample=2,
                        black_background=False, lpips=args.lpips, **options)
        progress = args.output / "evaluation_progress.json"
        if progress.is_file():
            saved = json.loads(progress.read_text())
            if saved["identity"] != identity:
                raise ValueError("saved evaluation belongs to a different checkpoint or configuration")
            previous_rows = saved["per_image"]

        def on_image(rows):
            temporary = progress.with_suffix(".tmp")
            temporary.write_text(json.dumps(dict(identity=identity, per_image=rows)), encoding="utf-8")
            temporary.replace(progress)

    result = evaluate_dataset(dataset, coarse, fine, device=device, lpips_model=lpips_model,
                              previous_rows=previous_rows, on_image=on_image, **options)
    payload = dict(checkpoint=str(args.checkpoint), dataset=str(args.data), dataset_type="blender",
                   split="test", resolution=[dataset.width, dataset.height],
                   metrics=result["summary"], per_image=result["per_image"])
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result["per_image"][0]))
        writer.writeheader()
        writer.writerows(result["per_image"])
    print(json.dumps(payload["metrics"], indent=2))
    print(f"Saved metrics to {args.output.resolve()}")


if __name__ == "__main__":
    main()
