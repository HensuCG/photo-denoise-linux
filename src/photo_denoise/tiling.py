"""Shared bounded-memory image processing."""

import math

import numpy as np


class TiledDenoiser:
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
