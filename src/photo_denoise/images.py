"""Pixel I/O without rotating or stripping metadata from originals."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def family(path: Path) -> str:
    return {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".tif": "tiff", ".tiff": "tiff"}[
        path.suffix.lower()
    ]


@dataclass
class Photo:
    rgb: np.ndarray
    alpha: np.ndarray | None
    grayscale: bool
    dtype: np.dtype
    icc: bytes | None = None
    dpi: tuple[float, float] | None = None


def read_photo(path: Path) -> Photo:
    if family(path) == "tiff":
        with tifffile.TiffFile(path) as source:
            if len(source.pages) != 1:
                raise ValueError("Multipage TIFF is not supported")
            page = source.pages[0]
            if page.photometric not in (1, 2):
                raise ValueError("Only grayscale MINISBLACK and RGB TIFF are supported")
            if page.planarconfig not in (None, 1):
                raise ValueError("Only contiguous TIFF pixels are supported")
            if any(int(sample) != 2 for sample in page.extrasamples):
                raise ValueError("Only unassociated alpha TIFF is supported")
            array = page.asarray()
            grayscale = page.photometric == 1
            icc_tag = page.tags.get(34675)
            icc = icc_tag.value if icc_tag else None
    else:
        if family(path) == "png":
            with path.open("rb") as stream:
                header = stream.read(29)
            if (
                len(header) >= 25
                and header[:8] == b"\x89PNG\r\n\x1a\n"
                and struct.unpack("B", header[24:25])[0] == 16
            ):
                raise ValueError(
                    "16-bit PNG is not supported; export a 16-bit TIFF to retain precision"
                )
        with Image.open(path) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("Animated/multiframe images are not supported")
            if source.mode == "P":
                source = source.convert("RGBA" if "transparency" in source.info else "RGB")
            if source.mode not in ("L", "LA", "RGB", "RGBA"):
                raise ValueError(
                    f"Unsupported pixel mode: {source.mode}; export RGB or grayscale first"
                )
            grayscale = source.mode in ("L", "LA")
            icc = source.info.get("icc_profile")
            dpi = source.info.get("dpi")
            array = np.array(source)
    if array.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError(f"Unsupported bit depth: {array.dtype}; use uint8 or uint16")
    if array.ndim == 2:
        array = array[:, :, None]
    expected = 1 if grayscale else 3
    if array.ndim != 3 or array.shape[2] not in (expected, expected + 1):
        raise ValueError("Unsupported channel layout")
    alpha = array[:, :, expected].copy() if array.shape[2] == expected + 1 else None
    rgb = array[:, :, :expected].astype(np.float32) / np.iinfo(array.dtype).max
    if grayscale:
        rgb = np.repeat(rgb, 3, axis=2)
    return Photo(rgb, alpha, grayscale, array.dtype, icc, dpi if family(path) != "tiff" else None)


def write_photo(path: Path, photo: Photo, result: np.ndarray, *, jpeg_quality: int = 95) -> None:
    if photo.grayscale:
        result = result.mean(axis=2, keepdims=True)
    pixels = np.rint(np.clip(result, 0, 1) * np.iinfo(photo.dtype).max).astype(photo.dtype)
    if photo.alpha is not None:
        pixels = np.concatenate((pixels, photo.alpha[:, :, None]), axis=2)
    if pixels.shape[2] == 1:
        pixels = pixels[:, :, 0]
    if family(path) == "tiff":
        extra = [(34675, "B", len(photo.icc), photo.icc, False)] if photo.icc else []
        tifffile.imwrite(
            path,
            pixels,
            photometric="minisblack" if photo.grayscale else "rgb",
            extrasamples=["unassalpha"] if photo.alpha is not None else None,
            metadata=None,
            extratags=extra,
        )
    else:
        kwargs = {}
        if photo.icc:
            kwargs["icc_profile"] = photo.icc
        if photo.dpi:
            kwargs["dpi"] = photo.dpi
        if family(path) == "jpeg":
            kwargs.update(quality=jpeg_quality, subsampling=0)
        Image.fromarray(pixels).save(path, **kwargs)
