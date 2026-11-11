from __future__ import annotations

import argparse
import csv
import json
import hashlib
import math
import statistics
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from fastnerf.model import FastNeRF, FastNeRFCache, FastNeRFConfig
from core.model import NeRF, NeRFConfig
from core.rays import get_rays, get_rays_at_pixels
from run_nerf import load_scene, render_rgb, save_checkpoint


def arguments():
    parser = argparse.ArgumentParser(description="Run the published NeRF/FastNeRF comparison.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    test = commands.add_parser("test")
    for command in (train, test):
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--resume", action="store_true")
    train.set_defaults(
        iterations=200000, downsample=2, ray_batch_size=1024,
        coarse_samples=64, importance_samples=128, near=2.0, far=6.0,
        learning_rate=5e-4, lr_decay_steps=250000, components=8,
        position_width=384, direction_width=128, encoding_convention="legacy", skip_layer=4,
        evaluate_every=5000, save_every=1000, val_views=5, seed=42,
    )
    test.add_argument("--checkpoint", type=Path, required=True)
    test.add_argument("--cache-resolution", type=int, choices=(0, 64, 128, 768), default=0,
                      help="0: neural field; 64/128: diagnostic caches; 768: FP16 cache")
    test.add_argument("--quality-gate", type=Path, help="validation certificate required for the 768 cache")
    test.set_defaults(ray_batch_size=1024, split="test", max_views=0,
                      cache_bounds=(-1.5, -1.5, -1.5, 1.5, 1.5, 1.5))
    args = parser.parse_args()
    if args.command == "test":
        if args.cache_resolution == 768 and args.quality_gate is None:
            parser.error("the 768 cache requires --quality-gate from validation views")
        args.cache_dtype = "float16" if args.cache_resolution == 768 else "float32"
        args.max_cache_mb = {0: 0, 64: 64, 128: 256, 768: 22000}[args.cache_resolution]
    return args


def evaluate(scene, coarse, fine, config, device, batch_size, max_views=0, measure_time=False):
    height, width = scene.images.shape[1:3]
    count = len(scene.images)
    indices = (torch.linspace(0, count - 1, max_views).round().long().tolist()
               if 0 < max_views < count else list(range(count)))
    rows = []
    with torch.inference_mode():
        if measure_time:
            warm_o, warm_d = get_rays(height, width, scene.focal, scene.poses[indices[0]].to(device))
            render_rgb(warm_o.reshape(-1, 3)[:batch_size], warm_d.reshape(-1, 3)[:batch_size],
                       coarse, fine, config, True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        for index in indices:
            if measure_time:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
            origins, directions = get_rays(height, width, scene.focal, scene.poses[index].to(device))
            origins, directions = origins.reshape(-1, 3), directions.reshape(-1, 3)
            target = scene.images[index].reshape(-1, 3)
            predictions = []
            for start in range(0, len(origins), batch_size):
                end = start + batch_size
                output = render_rgb(origins[start:end], directions[start:end], coarse, fine, config, True)
                predictions.append(output["fine" if "fine" in output else "coarse"].clamp(0, 1))
            if measure_time:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
            prediction = torch.cat(predictions)
            mse = (prediction - target.to(device)).square().double().mean().item()
            row = {"index": index, "mse": mse, "psnr_db": -10 * math.log10(mse) if mse else float("inf")}
            if measure_time:
                row["render_seconds"] = elapsed
            rows.append(row)
    return rows


def train(args, device):
    checkpoint_path = args.output / "checkpoint_latest.pt"
    log_path = args.output / "training.csv"
    if args.resume:
        if not checkpoint_path.is_file() or not log_path.is_file():
            raise FileNotFoundError("resume requires checkpoint_latest.pt and training.csv")
    elif checkpoint_path.exists() or log_path.exists():
        raise FileExistsError(f"output already contains a run: {args.output}")
    torch.manual_seed(args.seed)
    training = load_scene(args.data, "train", args.downsample)
    validation = load_scene(args.data, "val", args.downsample)
    model_config = FastNeRFConfig(components=args.components, position_width=args.position_width,
                                   direction_width=args.direction_width,
                                   encoding_convention=args.encoding_convention, skip_layer=args.skip_layer)
    coarse = FastNeRF(model_config).to(device)
    fine = FastNeRF(model_config).to(device)
    optimizer = torch.optim.Adam((*coarse.parameters(), *fine.parameters()), lr=args.learning_rate)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(model_type="fastnerf", dataset_type="blender", white_background=True,
                  density_activation=model_config.density_activation)
    args.output.mkdir(parents=True, exist_ok=True)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        old = checkpoint["training_config"]
        previous_model = dict(checkpoint["model_config"])
        previous_model.setdefault("encoding_convention", "legacy")
        if previous_model != model_config.__dict__:
            raise ValueError("resume model differs: legacy checkpoints require their saved encoding/skip settings")
        if old.get("density_activation", "relu_logits") != model_config.density_activation:
            raise ValueError("cannot resume legacy raw-density checkpoint with corrected softplus model; use a new output directory")
        keys = ("data", "downsample", "coarse_samples", "importance_samples", "near", "far",
                "learning_rate", "lr_decay_steps", "ray_batch_size", "seed", "components",
                "position_width", "direction_width")
        mismatch = [key for key in keys if config[key] != old[key]]
        if mismatch:
            raise ValueError(f"resume configuration differs: {', '.join(mismatch)}")
        coarse.load_state_dict(checkpoint["coarse_model"])
        fine.load_state_dict(checkpoint["fine_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        start_step = int(checkpoint["iteration"])
        with log_path.open(newline="", encoding="utf-8") as file:
            durable = [row for row in csv.DictReader(file) if int(row["iteration"]) <= start_step]
        with log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=("iteration", "loss", "validation_psnr_db"))
            writer.writeheader()
            writer.writerows(durable)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if start_step >= args.iterations:
        return
    stop = False

    def request_stop(_signal, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    height, width = training.images.shape[1:3]
    with log_path.open("a" if args.resume else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("iteration", "loss", "validation_psnr_db"))
        if not args.resume:
            writer.writeheader()
        for step in range(start_step + 1, args.iterations + 1):
            view = torch.randint(len(training.images), ()).item()
            pixels = torch.randint(height * width, (args.ray_batch_size,))
            rows, cols = pixels // width, pixels % width
            origins, directions = get_rays_at_pixels(height, width, training.focal,
                                                      training.poses[view].to(device), rows.to(device), cols.to(device))
            target = training.images[view, rows, cols].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = render_rgb(origins, directions, coarse, fine, config)
            loss = F.mse_loss(output["coarse"], target)
            if "fine" in output:
                loss = loss + F.mse_loss(output["fine"], target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            rate = args.learning_rate * 0.1 ** ((step - 1) / args.lr_decay_steps)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.step()
            if step % args.save_every == 0 or step == args.iterations or stop:
                save_checkpoint(checkpoint_path, step, coarse, fine, optimizer, config)
            psnr = ""
            if step % args.evaluate_every == 0 or step == args.iterations:
                metrics = evaluate(validation, coarse, fine, config, device, args.ray_batch_size, args.val_views)
                psnr = sum(row["psnr_db"] for row in metrics) / len(metrics)
            if step == 1 or step % 100 == 0 or psnr != "":
                writer.writerow({"iteration": step, "loss": loss.item(), "validation_psnr_db": psnr})
                file.flush()
                print(f"step {step}/{args.iterations}: loss={loss.item():.6f}; val PSNR={psnr}", flush=True)
            if stop:
                raise SystemExit(75)


def test(args, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["training_config"]
    model_type = config.get("model_type")
    if model_type == "fastnerf":
        saved_model_config = dict(checkpoint["model_config"])
        saved_model_config.setdefault("density_activation", config.get("density_activation", "relu_logits"))
        model_class, model_config = FastNeRF, FastNeRFConfig(**saved_model_config)
    elif model_type is None and config.get("dataset_type") == "blender":
        model_type = "nerf"
        model_class, model_config = NeRF, NeRFConfig(**checkpoint["model_config"])
    else:
        raise ValueError("checkpoint must come from run_nerf.py or fastnerf/run.py")
    if args.cache_resolution and model_type != "fastnerf":
        raise ValueError("only FastNeRF can use the factorized cache")
    coarse = model_class(model_config).to(device).eval()
    fine = model_class(model_config).to(device).eval()
    coarse.load_state_dict(checkpoint["coarse_model"])
    fine.load_state_dict(checkpoint["fine_model"])
    iteration = checkpoint["iteration"]
    del checkpoint
    cache_build_seconds = None
    cache_mebibytes = None
    if args.cache_resolution:
        if args.quality_gate:
            gate = json.loads(args.quality_gate.read_text())
            digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            if (not gate.get("passed") or gate["checkpoint_sha256"] != digest
                    or gate["resolution"] != args.cache_resolution or gate["dtype"] != args.cache_dtype
                    or gate["bounds"] != list(args.cache_bounds)):
                raise ValueError("cache quality certificate does not match this checkpoint/cache configuration")
        low, high = tuple(args.cache_bounds[:3]), tuple(args.cache_bounds[3:])
        options = dict(resolution=args.cache_resolution, lower=low, upper=high,
                       max_cache_mb=args.max_cache_mb,
                       storage_dtype=getattr(torch, args.cache_dtype))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        coarse = FastNeRFCache.from_model(coarse, **options)
        fine = FastNeRFCache.from_model(fine, **options)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        cache_build_seconds = time.perf_counter() - started
        cache_tensors = (coarse.position_grid, coarse.direction_grid,
                         fine.position_grid, fine.direction_grid)
        cache_mebibytes = sum(t.numel() * t.element_size() for t in cache_tensors) / 1024 ** 2
    scene = load_scene(args.data, args.split, config["downsample"])
    rows = evaluate(scene, coarse, fine, config, device, args.ray_batch_size, args.max_views, measure_time=True)
    summary = save_evaluation(args.output, rows, dict(
        iteration=iteration, model_type=model_type, split=args.split,
        density_activation=model_config.density_activation if model_type == "fastnerf" else None,
        cache_resolution=args.cache_resolution, skip_empty=False,
        cache_dtype=args.cache_dtype if args.cache_resolution else None,
        cache_bounds=list(args.cache_bounds) if args.cache_resolution else None,
        validation_quality_gate=str(args.quality_gate) if args.quality_gate else None,
        cache_build_seconds=cache_build_seconds, cache_mebibytes=cache_mebibytes,
    ))
    print(json.dumps(summary, indent=2))


def save_evaluation(output, rows, metadata):
    mean_psnr = sum(row["psnr_db"] for row in rows) / len(rows)
    median_seconds = statistics.median(row["render_seconds"] for row in rows)
    summary = dict(metadata, num_images=len(rows), median_render_seconds=median_seconds,
                   median_fps=1 / median_seconds,
                   mean_per_image_psnr_db=mean_psnr if math.isfinite(mean_psnr) else "Infinity")
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("index", "mse", "psnr_db", "render_seconds"))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main():
    args = arguments()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if args.command == "train":
        train(args, device)
    else:
        test(args, device)


if __name__ == "__main__":
    main()
