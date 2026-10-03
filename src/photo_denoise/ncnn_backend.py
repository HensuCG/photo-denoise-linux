"""Vulkan inference without a PyTorch dependency."""

import os
import tempfile

import ncnn
import numpy as np

from .tiling import TiledDenoiser
from .vulkan_models import ensure_model


def checked_native_call(function):
    # ncnn can log Vulkan allocation failures while returning success. Capture the
    # native stderr for each load/extraction so incomplete predictions are rejected.
    with tempfile.TemporaryFile() as log:
        saved = os.dup(2)
        try:
            os.dup2(log.fileno(), 2)
            result = function()
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        log.seek(0)
        message = log.read().decode(errors="replace")
    if any(word in message.lower() for word in ("failed", "out of memory", "not supported")):
        first = message.strip().splitlines()[0]
        raise ValueError(f"Vulkan failed: {first}. Reduce tile size or close other GPU workloads")
    return result


def devices():
    return [
        {"index": i, "name": ncnn.get_gpu_info(i).device_name()}
        for i in range(ncnn.get_gpu_count())
    ]


class NcnnDenoiser(TiledDenoiser):
    def __init__(self, name, directory, *, gpu_index=0, offline=False):
        folder = ensure_model(name, directory, offline=offline)
        if not 0 <= gpu_index < ncnn.get_gpu_count():
            raise ValueError("Selected Vulkan GPU is unavailable; open Settings or run doctor")
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = True
        self.net.opt.use_fp16_storage = False
        self.net.opt.use_fp16_packed = False
        self.net.opt.use_fp16_arithmetic = False
        # Winograd's transformed weights consume substantially more GPU memory.
        # Use the lower-memory path so other applications can share the card.
        self.net.opt.use_winograd_convolution = False
        self.net.opt.num_threads = 4
        self.net.set_vulkan_device(gpu_index)
        if checked_native_call(
            lambda: self.net.load_param(str(folder / f"{name}.ncnn.param"))
        ) or checked_native_call(lambda: self.net.load_model(str(folder / f"{name}.ncnn.bin"))):
            raise ValueError("Unable to load the converted Vulkan model")
        self.device = f"vulkan:{gpu_index}"

    def _predict(self, image, sigma):
        height, width = image.shape[:2]
        padded = np.pad(image, ((0, -height % 8), (0, -width % 8), (0, 0)), mode="edge")
        noise = np.full((*padded.shape[:2], 1), sigma / 255, dtype=np.float32)
        channels = np.concatenate((padded, noise), axis=2)
        tensor = np.ascontiguousarray(channels.transpose(2, 0, 1))
        with self.net.create_extractor() as extractor:
            if extractor.input("in0", ncnn.Mat(tensor).clone()):
                raise ValueError("Vulkan input failed")
            code, output = checked_native_call(lambda: extractor.extract("out0"))
            if code:
                raise ValueError(f"Vulkan inference failed ({code})")
            result = np.array(output).transpose(1, 2, 0)[:height, :width].copy()
            if result.shape != image.shape or not np.isfinite(result).all():
                raise ValueError("Vulkan returned an invalid prediction; no output will be saved")
            return result
