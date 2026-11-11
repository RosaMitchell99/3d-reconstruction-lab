
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import signal
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics.data import load_blender_scene, select_train_views
from core.model import NeRF, NeRFConfig
from core.rays import get_rays, get_rays_at_pixels
from core.rendering import render_rgb

if __package__:
    from .semantic_loss import semantic_consistency_loss
else:
    from semantic_loss import semantic_consistency_loss


def _parse_args():
    parser = argparse.ArgumentParser(description="Train the 8-view, 50k-step DietNeRF experiment.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--semantic-weight", type=float, choices=(0.01, 0.1), default=0.01)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(
        iterations=50000, train_views=8, seed=42, width=256, depth=8,
        ray_batch_size=1024, coarse_samples=64, importance_samples=128,
        near=2.0, far=6.0, learning_rate=5e-4, evaluate_every=5000,
        val_views=5, save_every=1000, semantic_every=10, semantic_resolution=168,
        density_activation="softplus", clip_model="ViT-B-32", clip_pretrained="openai",
        clip_feature="transformer_cls", clip_cache=Path(".clip-cache"),
        consistency_theta_range=(-180.0, 180.0), consistency_phi_range=(-90.0, 0.0),
        consistency_radius_range=(3.5, 4.5), protocol="legacy", semantic_order="reference",
        lr_decay_steps=0, precrop_iters=0,
    )
    return parser.parse_args()


def _choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


class CheckpointField(torch.nn.Module):
    def __init__(self, field):
        super().__init__()
        self.field = field

    def forward(self, points, directions):
        return checkpoint(self.field, points, directions, use_reentrant=False)


def _render_pose_with_grad(pose, height, width, focal, coarse, fine, args):
    sem_h = min(args.semantic_resolution, height)
    sem_w = max(1, round(width * sem_h / height))
    start_w = np.random.uniform(0, width % sem_w)
    start_h = np.random.uniform(0, height % sem_h)
    step_w, step_h = width // sem_w, height // sem_h
    cols = torch.arange(start_w, start_w + (sem_w - 1) * step_w + 1, step_w, device=pose.device)
    rows = torch.arange(start_h, start_h + (sem_h - 1) * step_h + 1, step_h, device=pose.device)
    rows, cols = torch.meshgrid(rows, cols, indexing="ij")
    rays_o, rays_d = get_rays_at_pixels(height, width, focal, pose, rows, cols)
    result = render_rgb(
        rays_o.reshape(-1, 3), rays_d.reshape(-1, 3),
        CheckpointField(coarse), CheckpointField(fine), vars(args),
    )
    return {name: rgb.reshape(sem_h, sem_w, 3) for name, rgb in result.items()}


def _semantic_losses(clip_encoder, cached_features, height, width, focal,
                     coarse, fine, args, device):
    pose = _sample_uniform_novel_pose(args, device)
    renders = _render_pose_with_grad(pose, height, width, focal,
                                    coarse, fine, args)
    names = [name for name in ("fine", "coarse") if name in renders]
    images = torch.stack([renders[name] for name in names]).clamp(0, 1)
    features = clip_encoder.encode(images)
    target_index = torch.tensor([np.random.randint(len(cached_features))], device=device)
    feature_by_name = {name: features[i:i + 1] for i, name in enumerate(names)}
    primary = feature_by_name.get("fine", feature_by_name["coarse"])
    fine_loss = semantic_consistency_loss(cached_features, primary, target_index)
    coarse_loss = torch.zeros((), device=device)
    if "fine" in feature_by_name and "coarse" in feature_by_name:
        coarse_loss = semantic_consistency_loss(cached_features, feature_by_name["coarse"], target_index)
    return fine_loss, coarse_loss


def _sample_uniform_novel_pose(args, device):
    theta = np.deg2rad(np.random.uniform(*args.consistency_theta_range))
    phi = np.deg2rad(np.random.uniform(*args.consistency_phi_range))
    radius = np.random.uniform(*args.consistency_radius_range)
    dtype = torch.float32
    trans_z = torch.eye(4, dtype=dtype, device=device)
    trans_z[2, 3] = radius
    rot_phi = torch.eye(4, dtype=dtype, device=device)
    rot_phi[1, 1], rot_phi[1, 2] = np.cos(phi), -np.sin(phi)
    rot_phi[2, 1], rot_phi[2, 2] = np.sin(phi), np.cos(phi)
    rot_theta = torch.eye(4, dtype=dtype, device=device)
    rot_theta[0, 0], rot_theta[0, 2] = np.cos(theta), -np.sin(theta)
    rot_theta[2, 0], rot_theta[2, 2] = np.sin(theta), np.cos(theta)
    blender_basis = torch.tensor(
        [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=dtype, device=device,
    )
    # Same transform order as pose_spherical_uniform in the reference loader.
    return blender_basis @ rot_theta @ rot_phi @ trans_z


@torch.inference_mode()
def _evaluate(test_data, coarse, fine, args):
    coarse.eval(); fine.eval()
    psnrs = []
    count = len(test_data.images)
    indices = (torch.linspace(0, count - 1, args.val_views).round().long().tolist()
               if args.val_views < count else range(count))
    for index in indices:
        rays_o, rays_d = get_rays(test_data.height, test_data.width, test_data.focal,
                                  test_data.poses[index].to(next(coarse.parameters()).device))
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        rgb = []
        for start in range(0, len(rays_o), args.ray_batch_size):
            prediction = render_rgb(rays_o[start:start + args.ray_batch_size],
                                          rays_d[start:start + args.ray_batch_size],
                                          coarse, fine, vars(args), deterministic=True)
            rgb.append(prediction["fine" if "fine" in prediction else "coarse"])
        prediction = torch.cat(rgb).reshape(test_data.height, test_data.width, 3).clamp(0.0, 1.0)
        target = test_data.images[index].to(prediction.device)
        mse = torch.mean((prediction - target) ** 2).item()
        psnrs.append(float("inf") if mse == 0 else -10.0 * math.log10(mse))
    coarse.train(); fine.train()
    return float(np.mean(psnrs))


def main():
    args = _parse_args()
    device = _choose_device(args.device)
    checkpoint_path = args.output / "checkpoint_latest.pt"
    log_path = args.output / "metrics.csv"
    if args.resume:
        if not checkpoint_path.is_file() or not log_path.is_file():
            raise FileNotFoundError("--resume requires checkpoint_latest.pt and metrics.csv")
    elif checkpoint_path.exists() or log_path.exists():
        raise FileExistsError("output already contains a run; use --resume or a new output directory")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_data = load_blender_scene(args.data, "train", downsample=2, resize_rgba_first=False)
    validation_data = load_blender_scene(args.data, "val", downsample=2, resize_rgba_first=False)
    selected = select_train_views(len(train_data.images), args.train_views, args.seed)
    selected_images = train_data.images[selected]
    selected_poses = train_data.poses[selected].to(device)
    height, width = train_data.height, train_data.width
    config = NeRFConfig(depth=args.depth, width=args.width, density_activation=args.density_activation)
    with torch.device("cpu"):
        coarse, fine = NeRF(config).to(device), NeRF(config).to(device)
    optimizer = torch.optim.Adam(list(coarse.parameters()) + list(fine.parameters()), lr=args.learning_rate)

    if __package__:
        from .clip_encoder import FrozenCLIPImageEncoder
    else:
        from clip_encoder import FrozenCLIPImageEncoder
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        clip_encoder = FrozenCLIPImageEncoder(cache_dir=str(args.clip_cache)).to(device).eval()
    cached_features = clip_encoder.cache_observed(selected_images, batch_size=2)
    print(f"scene={args.data}; device={device}; train views={selected.tolist()}; semantic weight={args.semantic_weight}")

    start_step = 0
    if args.resume:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        previous = saved["training_config"]
        if NeRFConfig(**saved["model_config"]) != config:
            raise ValueError("resume model differs from checkpoint")
        keys = ("data", "train_views", "seed", "learning_rate", "ray_batch_size", "coarse_samples",
                "importance_samples", "near", "far", "semantic_weight", "semantic_every",
                "semantic_resolution", "clip_model", "clip_pretrained", "clip_feature", "protocol",
                "semantic_order", "lr_decay_steps", "precrop_iters", "consistency_theta_range",
                "consistency_phi_range", "consistency_radius_range")
        for key in keys:
            if key in previous and json.dumps(previous[key]) != json.dumps(getattr(args, key), default=str):
                raise ValueError(f"resume config mismatch: {key}")
        if previous.get("semantic_stop_step") is not None or previous.get("semantic_coarse_weight") not in (None, args.semantic_weight):
            raise ValueError("checkpoint uses a different semantic loss schedule")
        if previous.get("full_resolution", False):
            raise ValueError("checkpoint uses full-resolution images")
        coarse.load_state_dict(saved["coarse_model"])
        fine.load_state_dict(saved["fine_model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["iteration"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
        numpy_rng = saved["numpy_rng"]
        np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
        random.setstate(tuple(saved["python_rng"]))
        with log_path.open(newline="", encoding="utf-8") as file:
            rows = [row for row in csv.DictReader(file) if int(row["iteration"]) <= start_step]
        with log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Resuming from step {start_step}")
    if start_step >= args.iterations:
        print(f"Already complete at step {start_step}")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration.update(train_view_indices=selected.tolist(), initialization_device="cpu",
                         neural_profile="semantic_order_v3", white_background=True, focal=train_data.focal)
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    stop = False

    def request_stop(signum, _frame):
        nonlocal stop
        stop = True
        print(f"Received signal {signum}; saving after current iteration", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    with log_path.open("a" if args.resume else "w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=FIELDS)
        if not args.resume:
            writer.writeheader()
        for step in range(start_step + 1, args.iterations + 1):
            view_slot = int(torch.randint(len(selected), (1,)).item())
            pixel_ids = torch.randint(height * width, (args.ray_batch_size,))
            rows, cols = pixel_ids // width, pixel_ids % width
            rays_o, rays_d = get_rays_at_pixels(
                height, width, train_data.focal, selected_poses[view_slot], rows.to(device), cols.to(device),
            )
            target = selected_images[view_slot, rows, cols].to(device)
            sem_loss = torch.zeros((), device=device)
            sem_coarse_loss = torch.zeros((), device=device)
            if step % args.semantic_every == 0:
                sem_loss, sem_coarse_loss = _semantic_losses(
                    clip_encoder, cached_features, height, width, train_data.focal,
                    coarse, fine, args, device,
                )
            prediction = render_rgb(rays_o, rays_d, coarse, fine, vars(args))
            rgb_loss = F.mse_loss(prediction["coarse"], target) + F.mse_loss(prediction["fine"], target)
            total_loss = rgb_loss + args.semantic_weight * sem_loss + args.semantic_weight * sem_coarse_loss
            if not torch.isfinite(total_loss):
                raise RuntimeError(f"non-finite training loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            validation_psnr = ""
            if step % args.evaluate_every == 0 or step == args.iterations:
                validation_psnr = _evaluate(validation_data, coarse, fine, args)
                print(f"step {step:>7}/{args.iterations}: rgb={rgb_loss.item():.6f}, "
                      f"semantic={sem_loss.item():.5f}, val PSNR={validation_psnr:.2f} dB")
            writer.writerow(dict(iteration=step, total_loss=total_loss.item(), rgb_loss=rgb_loss.item(),
                                 semantic_loss=sem_loss.item(), validation_psnr=validation_psnr, test_psnr=""))
            log_file.flush()
            if step % args.save_every == 0 or step == args.iterations or stop:
                numpy_rng = np.random.get_state()
                state = dict(
                    iteration=step, coarse_model=coarse.state_dict(), fine_model=fine.state_dict(),
                    model_config=config.__dict__, optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    numpy_rng=[numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
                    python_rng=list(random.getstate()), training_config=configuration,
                )
                temporary = checkpoint_path.with_suffix(".tmp")
                torch.save(state, temporary)
                temporary.replace(checkpoint_path)
            if stop:
                raise SystemExit(75)


FIELDS = ["iteration", "total_loss", "rgb_loss", "semantic_loss", "validation_psnr", "test_psnr"]


if __name__ == "__main__":
    main()
