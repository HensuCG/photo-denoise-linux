"""Create reproducible photographic examples and exercise the real CLI on CUDA."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from photo_denoise.images import read_photo
from photo_denoise.metadata import run_exiftool
from photo_denoise.models import Denoiser, cache_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("test-results"))
    args = parser.parse_args()
    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    clean = np.asarray(Image.open(args.clean).convert("RGB"), dtype=np.float32) / 255
    noisy = np.clip(clean + np.random.default_rng(321).normal(0, 20 / 255, clean.shape), 0, 1)
    Image.fromarray(np.rint(clean * 255).astype(np.uint8)).save(target / "clean.png")
    source = target / "noisy.png"
    Image.fromarray(np.rint(noisy * 255).astype(np.uint8)).save(source)
    run_exiftool(
        "-overwrite_original",
        "-EXIF:Make=Test Camera",
        "-EXIF:ISO=1600",
        "-EXIF:DateTimeOriginal=2021:07:08 09:10:11",
        "-EXIF:Orientation#=1",
        str(source),
    )
    clean = read_photo(target / "clean.png").rgb
    noisy = read_photo(source).rgb
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "noisy_psnr_db": float(-10 * np.log10(np.mean((noisy - clean) ** 2))),
        "models": {},
    }
    panels = [("Clean reference", clean), ("Noise added (sigma 20)", noisy)]
    for name in ("scunet", "scunet-gan", "drunet"):
        destination = target / f"{name}.png"
        command = [
            str(Path(__file__).resolve().parents[1] / "photo-denoise"),
            "denoise",
            str(source),
            "-o",
            str(destination),
            "--model",
            name,
            "--device",
            "cuda",
            "--offline",
            "--overwrite",
        ]
        if name == "drunet":
            command += ["--sigma", "20"]
        started = time.monotonic()
        subprocess.run(command, check=True)
        elapsed = time.monotonic() - started
        result = read_photo(destination).rgb
        psnr = float(-10 * np.log10(np.mean((result - clean) ** 2)))
        report["models"][name] = {"cli_seconds": round(elapsed, 2), "psnr_db": round(psnr, 2)}
        panels.append((f"{name} ({psnr:.1f} dB)", result))
    panel_w = min(384, clean.shape[1])
    panel_h = round(clean.shape[0] * panel_w / clean.shape[1])
    sheet = Image.new("RGB", (panel_w * len(panels), panel_h + 36), "#242424")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    for i, (label, pixels) in enumerate(panels):
        image = Image.fromarray(np.rint(pixels * 255).astype(np.uint8)).resize((panel_w, panel_h))
        sheet.paste(image, (i * panel_w, 36))
        draw.text((i * panel_w + 6, 10), label, fill="white", font=font)
    sheet.save(target / "comparison.png")
    # Test a multi-megapixel image using the default tile settings on the GPU.
    large = np.asarray(Image.open(source).resize((2048, 1536)), dtype=np.float32) / 255
    model = Denoiser("scunet", cache_directory(), "cuda", offline=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = model.denoise(large)
    elapsed = time.monotonic() - started
    assert result.shape == large.shape and np.isfinite(result).all()
    report["large_image"] = {
        "dimensions": [2048, 1536],
        "model": "scunet",
        "tile_size": 512,
        "inference_seconds": round(elapsed, 2),
        "peak_allocated_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20),
    }
    (target / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
