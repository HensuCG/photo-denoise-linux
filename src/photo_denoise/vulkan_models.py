"""Validated Vulkan model registry and downloads, independent of ncnn."""

from pathlib import Path

from .models import checksum
from .transfer import download_file

NCNN_MODELS = {
    "drunet": {
        "param": ("5e67e478c3b763ed3ff73a85d3c3487cc864d07e66e06b898d41a1751f8f3284", 11559),
        "bin": ("c7e55ec011018ee2d4cbc6d233fe272ab723e80e146afcdf63182b33d5159505", 130564096),
    },
}
NCNN_URL = "https://github.com/HensuCG/photo-denoise/releases/download/v0.2.0/"


def ensure_model(name: str, directory: Path, *, offline=False, progress=None, cancelled=None):
    if name not in NCNN_MODELS:
        raise ValueError(f"{name} is not validated for Vulkan; choose DRUNet or CUDA/CPU")
    target = directory / "ncnn"
    target.mkdir(parents=True, exist_ok=True)
    for suffix, (digest, size) in NCNN_MODELS[name].items():
        path = target / f"{name}.ncnn.{suffix}"
        if path.exists():
            if checksum(path) != digest:
                raise ValueError(f"Model checksum mismatch: {path}")
        elif offline:
            raise ValueError(f"Vulkan model is not cached: {path}")
        else:
            download_file(
                NCNN_URL + path.name,
                path,
                digest,
                size=size,
                progress=progress,
                cancelled=cancelled,
            )
    return target
