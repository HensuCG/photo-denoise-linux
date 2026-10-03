"""Official pretrained models and bounded-memory inference."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

MODELS = {
    "scunet": ("scunet_color_real_psnr.pth", "Blind real-photo denoising (default)"),
    "scunet-gan": ("scunet_color_real_gan.pth", "Perceptual real-photo denoising"),
    "drunet": ("drunet_color.pth", "Gaussian denoising with adjustable --sigma"),
}
# Filled from the official downloads; each cached file is checked before loading.
SHA256: dict[str, str] = {
    "scunet": "fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e",
    "scunet-gan": "892c83f812c59173273b74f4f34a14ecaf57a2fdb68df056664589beb55c966e",
    "drunet": "479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4",
}
BASE_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/"


def cache_directory() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "photo-denoise"


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_model(
    name: str, directory: Path, *, offline: bool = False, progress=None, cancelled=None
) -> Path:
    filename = MODELS[name][0]
    path = directory / filename
    expected = SHA256.get(name)
    if path.is_file():
        if expected and checksum(path) != expected:
            raise ValueError(f"Model checksum mismatch: {path}. Remove it and download again.")
        return path
    if offline:
        raise ValueError(f"Model is not cached: {path}. Run 'photo-denoise download {name}' first.")
    from .transfer import download_file

    sizes = {"scunet": 71982841, "scunet-gan": 71982835, "drunet": 130579305}
    download_file(
        BASE_URL + filename,
        path,
        expected,
        size=sizes[name],
        progress=progress,
        cancelled=cancelled,
    )
    return path


def resolve_device(requested: str):
    from .torch_backend import resolve_device as resolve

    return resolve(requested)


def Denoiser(name, directory, device="cuda", *, offline=False, runtime=None, gpu_index=0):
    if runtime == "vulkan":
        from .ncnn_backend import NcnnDenoiser

        return NcnnDenoiser(name, directory, gpu_index=gpu_index, offline=offline)
    from .torch_backend import TorchDenoiser

    if device == "cuda" and gpu_index:
        device = f"cuda:{gpu_index}"
    return TorchDenoiser(name, directory, device, offline=offline)
