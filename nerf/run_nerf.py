"""Train the full-view and eight-view scratch NeRF experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
from pathlib import Path

from metrics.data import load_blender_scene, select_train_views
from core.rendering import render_rgb


def parse_args():
    parser = argparse.ArgumentParser(description="Train the published scratch NeRF experiments.")
    parser.add_argument("command", choices=("train",))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--experiment", choices=("full200k", "full50k", "sparse50k"), default="full200k")
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(
        downsample=2, ray_batch_size=1024, coarse_samples=64, importance_samples=128,
        near=2.0, far=6.0, width=256, depth=8, learning_rate=5e-4,
        lr_decay_steps=250000, log_every=100, evaluate_every=5000, save_every=1000,
        val_views=5, seed=42,
    )
    args = parser.parse_args()
    args.iterations = 200000 if args.experiment == "full200k" else 50000
    args.train_views = 8 if args.experiment == "sparse50k" else 0
    return args


def load_scene(path, split, downsample):
    return load_blender_scene(path, split=split, downsample=downsample, white_background=True)


def evaluate(scene, coarse, fine, config, batch_size, max_views=0):
    import torch
    from core.rays import get_rays

    device = next(coarse.parameters()).device
    height, width = scene.images.shape[1:3]
    count = len(scene.images)
    if max_views and max_views < count:
        indices = torch.linspace(0, count - 1, max_views).round().long().tolist()
    else:
        indices = list(range(count))
    coarse.eval()
    fine.eval()
    records = []
    with torch.inference_mode():
        for index in indices:
            origins, directions = get_rays(height, width, scene.focal, scene.poses[index].to(device))
            origins, directions = origins.reshape(-1, 3), directions.reshape(-1, 3)
            target = scene.images[index].reshape(-1, 3)
            squared_error = 0.0
            for start in range(0, len(origins), batch_size):
                stop = start + batch_size
                outputs = render_rgb(origins[start:stop], directions[start:stop], coarse, fine, config, True)
                pred = outputs["fine" if "fine" in outputs else "coarse"].clamp(0, 1)
                squared_error += (pred - target[start:stop].to(device)).square().double().sum().item()
            mse = squared_error / target.numel()
            psnr = -10.0 * math.log10(mse) if mse else float("inf")
            records.append({"index": index, "mse": mse, "psnr_db": psnr})
            print(f"  {scene.split} view {index}: PSNR={psnr:.3f} dB", flush=True)
    return records


def save_checkpoint(path, step, coarse, fine, optimizer, config):
    import torch

    temporary = path.with_suffix(".tmp")
    torch.save({
        "iteration": step, "coarse_model": coarse.state_dict(), "fine_model": fine.state_dict(),
        "model_config": vars(coarse.config), "training_config": config,
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }, temporary)
    temporary.replace(path)


def train(args, device):
    import torch
    from torch.nn import functional as F
    from core.model import NeRF, NeRFConfig
    from core.rays import get_rays_at_pixels

    checkpoint_path = args.output / "checkpoint_latest.pt"
    log_path = args.output / "training.csv"
    if args.resume:
        if not checkpoint_path.is_file() or not log_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {checkpoint_path} and {log_path}")
    elif checkpoint_path.exists() or log_path.exists():
        raise FileExistsError(f"Training output already exists in {args.output}; use --resume or choose a new --output")
    torch.manual_seed(args.seed)
    train_data = load_scene(args.data, "train", args.downsample)
    validation = load_scene(args.data, "val", args.downsample)
    selected = select_train_views(len(train_data.images), args.train_views, args.seed)
    if args.train_views:
        train_data.images = train_data.images[selected]
        train_data.poses = train_data.poses[selected]
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(white_background=True, dataset_type="blender", train_view_indices=selected.tolist())
    model_config = NeRFConfig(depth=args.depth, width=args.width)
    coarse, fine = NeRF(model_config).to(device), NeRF(model_config).to(device)
    optimizer = torch.optim.Adam(list(coarse.parameters()) + list(fine.parameters()), lr=args.learning_rate)
    height, width = train_data.images.shape[1:3]
    args.output.mkdir(parents=True, exist_ok=True)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        old_config = checkpoint["training_config"]
        comparable_keys = ("data", "downsample", "coarse_samples", "importance_samples", "near", "far",
                           "width", "depth", "learning_rate", "lr_decay_steps", "ray_batch_size", "seed")
        mismatches = [key for key in comparable_keys if config[key] != old_config[key]]
        if args.train_views != old_config.get("train_views", 0):
            mismatches.append("train_views")
        if old_config.get("precrop_iters", 0) != 0:
            mismatches.append("precrop_iters")
        if NeRFConfig(**checkpoint["model_config"]) != model_config:
            mismatches.append("model_config")
        if mismatches:
            raise ValueError(f"Resume settings differ from checkpoint: {', '.join(mismatches)}")
        coarse.load_state_dict(checkpoint["coarse_model"])
        fine.load_state_dict(checkpoint["fine_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        start_step = int(checkpoint["iteration"])
        if start_step >= args.iterations:
            print(f"Checkpoint already reached step {start_step}; nothing to train", flush=True)
            return
     
        with log_path.open(newline="", encoding="utf-8") as old_file:
            old_rows = list(csv.DictReader(old_file))
        durable_rows = [row for row in old_rows if int(row["iteration"]) <= start_step]
        temporary_log = log_path.with_suffix(".csv.tmp")
        with temporary_log.open("w", newline="", encoding="utf-8") as new_file:
            log_writer = csv.DictWriter(new_file, fieldnames=["iteration", "loss", "coarse_mse", "fine_mse", "learning_rate", "validation_psnr_db"])
            log_writer.writeheader()
            log_writer.writerows(durable_rows)
        temporary_log.replace(log_path)
        print(f"Resuming from step {start_step}", flush=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"Received signal {signum}; saving at the end of this step", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    print(f"device={device}; train views={len(train_data.images)}; resolution={width}x{height}", flush=True)
    with log_path.open("a" if args.resume else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["iteration", "loss", "coarse_mse", "fine_mse", "learning_rate", "validation_psnr_db"])
        if not args.resume:
            writer.writeheader()
        for step in range(start_step + 1, args.iterations + 1):
            coarse.train()
            fine.train()
            view = torch.randint(len(train_data.images), ()).item()
            pixels = torch.randint(height * width, (args.ray_batch_size,))
            rows, cols = pixels // width, pixels % width
            origins, directions = get_rays_at_pixels(
                height, width, train_data.focal, train_data.poses[view].to(device), rows.to(device), cols.to(device),
            )
            target = train_data.images[view, rows, cols].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = render_rgb(origins, directions, coarse, fine, config)
            coarse_loss = F.mse_loss(outputs["coarse"], target)
            fine_loss = F.mse_loss(outputs["fine"], target)
            loss = coarse_loss + fine_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at iteration {step}")
            loss.backward()
            learning_rate = args.learning_rate * 0.1 ** ((step - 1) / args.lr_decay_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            validation_psnr = ""
            should_save = step % args.save_every == 0 or step == args.iterations or stop_requested
            if should_save:
                save_checkpoint(checkpoint_path, step, coarse, fine, optimizer, config)
            if step % args.evaluate_every == 0 or step == args.iterations:
                records = evaluate(validation, coarse, fine, config, args.ray_batch_size, args.val_views)
                validation_psnr = sum(row["psnr_db"] for row in records) / len(records)
            if step == 1 or step % args.log_every == 0 or validation_psnr != "":
                print(f"step {step}/{args.iterations}: loss={loss.item():.6f}; val PSNR={validation_psnr}", flush=True)
                writer.writerow({"iteration": step, "loss": loss.item(), "coarse_mse": coarse_loss.item(),
                                 "fine_mse": fine_loss.item(),
                                 "learning_rate": learning_rate, "validation_psnr_db": validation_psnr})
                file.flush()
            if stop_requested:
                print(f"Stopped safely after step {step}; resume with --resume", flush=True)
                raise SystemExit(75)
    print(f"Training complete: {args.output / 'checkpoint_latest.pt'}")


def main():
    args = parse_args()
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch installation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    train(args, device)


if __name__ == "__main__":
    main()
