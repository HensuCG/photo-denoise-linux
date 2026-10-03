import numpy as np
import pytest
import torch

from photo_denoise.models import Denoiser, cache_directory


@pytest.mark.integration
@pytest.mark.parametrize("index", [0, 1])
def test_vulkan_matches_cuda(index):
    rng = np.random.default_rng(12)
    image = np.clip(
        np.full((96, 128, 3), 0.4, dtype=np.float32)
        + rng.normal(0, 20 / 255, (96, 128, 3)).astype(np.float32),
        0,
        1,
    )
    reference = Denoiser("drunet", cache_directory(), "cuda", offline=True).denoise(image, sigma=20)
    # PyTorch and Vulkan use separate processes in the GUI. Release unused CUDA
    # allocator blocks before comparing them inside this test process.
    torch.cuda.empty_cache()
    model = Denoiser("drunet", cache_directory(), runtime="vulkan", gpu_index=index, offline=True)
    result = model.denoise(image, sigma=20)
    assert np.isfinite(result).all()
    assert np.abs(result - reference).mean() < 0.001
    assert np.abs(result - reference).max() < 0.005
    assert np.mean((result - 0.4) ** 2) < np.mean((image - 0.4) ** 2) * 0.2


def test_unvalidated_model_rejected_before_loading_gpu(tmp_path):
    from photo_denoise.vulkan_models import ensure_model

    with pytest.raises(ValueError, match="not validated"):
        ensure_model("scunet", tmp_path, offline=True)


def test_native_allocation_failure_is_not_accepted_as_success():
    import os

    from photo_denoise.ncnn_backend import checked_native_call

    def native_failure():
        os.write(2, b"vkAllocateMemory failed -2\n")
        return 0

    with pytest.raises(ValueError, match="Vulkan failed"):
        checked_native_call(native_failure)
