"""Validate the 768³ FP16 cache, then reuse it for test-set evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from fastnerf.model import FastNeRF, FastNeRFCache, FastNeRFConfig
from fastnerf.run import evaluate, save_evaluation
from run_nerf import load_scene


class ClippedField(nn.Module):
    def __init__(self, model, extent):
        super().__init__()
        self.model, self.extent = model, extent

    def forward(self, points, directions):
        rgb, density = self.model(points, directions)
        inside = (points.abs() <= self.extent).all(-1, keepdim=True)
        return rgb, torch.where(inside, density, torch.full_like(density, -1e6))


def certify(observations):
    config = observations["configuration"]
    neural, cached = observations["neural"], observations["cache_768"]
    reference = {row["index"]: row["psnr_db"] for row in neural["views"]}
    if config["split"] != "val" or {row["index"] for row in cached["views"]} != set(reference):
        raise ValueError("cache validation requires matching held-out validation views")
    drop = neural["mean_psnr_db"] - cached["mean_psnr_db"]
    worst = max(reference[row["index"]] - row["psnr_db"] for row in cached["views"])
    clipping = neural["mean_psnr_db"] - observations["neural_clipped_1.5"]["mean_psnr_db"]
    if not (drop <= 0.5 and worst <= 0.5 and clipping <= 0.05):
        raise ValueError(f"cache quality check failed: mean={drop:.3f}, worst={worst:.3f}, clipping={clipping:.3f} dB")
    return dict(passed=True, checkpoint_sha256=config["checkpoint_sha256"], resolution=768,
                dtype="float16", bounds=[-1.5] * 3 + [1.5] * 3, validation_views=list(reference),
                validation_drop_db=drop, worst_validation_drop_db=worst, max_drop_db=0.5)


@torch.inference_mode()
def run(args, device):
    validation = args.output / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    gate_path = validation / "quality_gate.json"
    gate_path.unlink(missing_ok=True)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = saved["training_config"]
    if config.get("model_type") != "fastnerf":
        raise ValueError("expected a FastNeRF checkpoint")
    iteration = saved["iteration"]
    model_config = FastNeRFConfig(**saved["model_config"])
    fields = [FastNeRF(model_config).to(device).eval() for _ in range(2)]
    for field, key in zip(fields, ("coarse_model", "fine_model")):
        field.load_state_dict(saved[key])
    del saved
    scene = load_scene(args.data, "val", config["downsample"])
    observations = {"configuration": dict(
        checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        extent=1.5, dtype="float16", split="val", resolutions=[768],
    )}

    def measure(name, pair):
        rows = evaluate(scene, *pair, config, device, 1024, max_views=3, measure_time=True)
        observations[name] = dict(views=rows, mean_psnr_db=sum(row["psnr_db"] for row in rows) / len(rows))
        (validation / "metrics.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")

    measure("neural", fields)
    measure("neural_clipped_1.5", [ClippedField(field, 1.5) for field in fields])
    clipping = observations["neural"]["mean_psnr_db"] - observations["neural_clipped_1.5"]["mean_psnr_db"]
    if not clipping <= 0.05:
        raise ValueError(f"cache bounds remove scene content ({clipping:.3f} dB loss)")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    caches = [FastNeRFCache.from_model(
        field, resolution=768, lower=(-1.5,) * 3, upper=(1.5,) * 3,
        max_cache_mb=22000, chunk_size=32768, storage_dtype=torch.float16,
    ) for field in fields]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    build_seconds = time.perf_counter() - started
    cache_mib = sum(tensor.numel() * tensor.element_size()
                    for cache in caches for tensor in (cache.position_grid, cache.direction_grid)) / 1024**2
    del fields
    measure("cache_768", caches)
    gate = certify(observations)
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    del scene
    test_scene = load_scene(args.data, "test", config["downsample"])
    rows = evaluate(test_scene, *caches, config, device, 1024, measure_time=True)
    summary = save_evaluation(args.output / "test", rows, dict(
        iteration=iteration, model_type="fastnerf", split="test", density_activation=model_config.density_activation,
        cache_resolution=768, cache_dtype="float16", cache_bounds=gate["bounds"], skip_empty=False,
        validation_quality_gate=str(gate_path), cache_build_seconds=build_seconds, cache_mebibytes=cache_mib,
    ))
    print(f"Validation PSNR drop: {gate['validation_drop_db']:.3f} dB")
    print(f"Test PSNR: {summary['mean_per_image_psnr_db']:.3f} dB")
    print(f"Render rate: {summary['median_fps']:.3f} FPS")
    print(f"Metrics: {args.output / 'test/metrics.json'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the 768³ FP16 cache experiment requires a CUDA GPU")
    torch.set_num_threads(7)
    run(args, torch.device("cuda"))


if __name__ == "__main__":
    main()
