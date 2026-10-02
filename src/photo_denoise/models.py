"""Official pretrained models and bounded-memory inference."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from spandrel import ImageModelDescriptor, ModelLoader

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


def download_model(name: str, directory: Path, *, offline: bool = False) -> Path:
    filename = MODELS[name][0]
    path = directory / filename
    expected = SHA256.get(name)
    if path.is_file():
        if expected and checksum(path) != expected:
            raise ValueError(f"Model checksum mismatch: {path}. Remove it and download again.")
        return path
    if offline:
        raise ValueError(f"Model is not cached: {path}. Run 'photo-denoise download {name}' first.")
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{filename}.", dir=directory)
    temp = Path(temporary)
    try:
        request = urllib.request.Request(
            BASE_URL + filename, headers={"User-Agent": "photo-denoise/0.1"}
        )
        with os.fdopen(fd, "wb") as target, urllib.request.urlopen(request, timeout=120) as source:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        if expected and checksum(temp) != expected:
            raise ValueError(f"Downloaded model checksum mismatch: {name}")
        # Two concurrent downloads produce the same verified file.
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Run 'photo-denoise doctor' or choose --device cpu.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


class Denoiser:
    def __init__(self, name: str, directory: Path, device: str, *, offline: bool = False):
        self.name = name
        self.device = resolve_device(device)
        # Restrict checkpoint loading to tensors; never unpickle arbitrary model objects.
        state = torch.load(
            download_model(name, directory, offline=offline), map_location="cpu", weights_only=True
        )
        descriptor = ModelLoader().load_from_state_dict(state)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise ValueError("Expected an image restoration model")
        self.descriptor = descriptor.eval().to(self.device)
        self.model = descriptor.model

    @torch.inference_mode()
    def _predict(self, image: np.ndarray, sigma: float) -> np.ndarray:
        height, width = image.shape[:2]
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
        tensor = tensor.to(self.device)
        if self.name == "drunet":
            # Spandrel's DRUNet descriptor hardcodes sigma=15. Call the underlying model
            # with our own noise map and the same required padding instead.
            pad_h = (-height) % 8
            pad_w = (-width) % 8
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
            noise = torch.full_like(tensor[:, :1], sigma / 255.0)
            result = self.model(torch.cat((tensor, noise), dim=1))[:, :, :height, :width]
        else:
            result = self.descriptor(tensor)
        return result.squeeze(0).permute(1, 2, 0).float().cpu().numpy()

    def denoise(
        self,
        image: np.ndarray,
        *,
        sigma: float = 15,
        amount: float = 1,
        tile_size: int = 512,
        overlap: int = 32,
        progress=None,
    ) -> np.ndarray:
        if amount == 0:
            return image.copy()
        if not 0 <= amount <= 1 or not 0 <= sigma <= 100:
            raise ValueError("amount must be 0..1 and sigma must be 0..100")
        if tile_size < 64 or overlap < 0 or overlap * 2 >= tile_size:
            raise ValueError("tile size must be >=64 and overlap must be < half the tile size")
        height, width, channels = image.shape
        if channels != 3 or image.dtype != np.float32:
            raise ValueError("Expected float32 RGB pixels")
        # Each output core is surrounded by context. Only the core is retained, so
        # padding artifacts at the edge of a tile are discarded.
        core = tile_size - 2 * overlap
        result = np.empty_like(image)
        total = math.ceil(height / core) * math.ceil(width / core)
        completed = 0
        for y in range(0, height, core):
            for x in range(0, width, core):
                end_y, end_x = min(y + core, height), min(x + core, width)
                top, left = max(0, y - overlap), max(0, x - overlap)
                bottom, right = min(height, end_y + overlap), min(width, end_x + overlap)
                restored = self._predict(image[top:bottom, left:right], sigma)
                result[y:end_y, x:end_x] = restored[y - top : end_y - top, x - left : end_x - left]
                completed += 1
                if progress:
                    progress(completed, total)
        return np.clip(image + amount * (result - image), 0, 1)
