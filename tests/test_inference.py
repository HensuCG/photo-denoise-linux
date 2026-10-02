import os

import numpy as np
import pytest
import torch

from photo_denoise.models import Denoiser, cache_directory


@pytest.fixture(scope="module")
def noisy_photo():
    y, x = np.mgrid[:192, :256].astype(np.float32)
    clean = np.stack(
        (0.25 + 0.002 * x, 0.2 + 0.0025 * y, 0.45 + 0.10 * np.sin(x / 24) * np.cos(y / 19)), axis=2
    )
    noise = np.random.default_rng(123).normal(0, 20 / 255, clean.shape).astype(np.float32)
    return clean, np.clip(clean + noise, 0, 1)


@pytest.mark.integration
@pytest.mark.parametrize("name", ["scunet", "scunet-gan", "drunet"])
def test_models_reduce_noise(name, noisy_photo):
    device = os.environ.get("PHOTO_DENOISE_TEST_DEVICE", "cpu")
    torch.set_num_threads(4)
    model = Denoiser(name, cache_directory(), device, offline=True)
    clean, noisy = noisy_photo
    output = model.denoise(noisy, sigma=20, tile_size=256, overlap=32)
    original_mse = np.mean((noisy - clean) ** 2)
    restored_mse = np.mean((output - clean) ** 2)
    print(f"{name} {device}: noisy MSE={original_mse:.6f}, restored MSE={restored_mse:.6f}")
    assert np.isfinite(output).all()
    assert output.shape == noisy.shape
    assert restored_mse < original_mse * 0.5


@pytest.mark.integration
def test_sigma_amount_and_tile_boundaries(noisy_photo):
    device = os.environ.get("PHOTO_DENOISE_TEST_DEVICE", "cpu")
    torch.set_num_threads(4)
    model = Denoiser("drunet", cache_directory(), device, offline=True)
    _, noisy = noisy_photo
    light = model.denoise(noisy, sigma=5, tile_size=512)
    strong = model.denoise(noisy, sigma=30, tile_size=512)
    partial = model.denoise(noisy, sigma=30, amount=0.4, tile_size=512)
    np.testing.assert_allclose(partial, np.clip(noisy + 0.4 * (strong - noisy), 0, 1), atol=0.002)
    assert np.mean(np.abs(strong - noisy)) > np.mean(np.abs(light - noisy))
    tiled = model.denoise(noisy, sigma=30, tile_size=192, overlap=32)
    difference = np.abs(tiled - strong)
    print(f"DRUNet tiled vs whole: mean={difference.mean():.6f}, max={difference.max():.6f}")
    assert difference.mean() < 0.003
    assert difference.max() < 0.035
    np.testing.assert_array_equal(model.denoise(noisy, amount=0), noisy)


@pytest.mark.integration
@pytest.mark.parametrize("name", ["scunet", "drunet"])
def test_tiny_odd_dimensions(name):
    device = os.environ.get("PHOTO_DENOISE_TEST_DEVICE", "cpu")
    model = Denoiser(name, cache_directory(), device, offline=True)
    for height, width in ((1, 1), (7, 13), (65, 79)):
        image = np.full((height, width, 3), 0.4, dtype=np.float32)
        result = model.denoise(image, tile_size=128)
        assert result.shape == image.shape
        assert np.isfinite(result).all()


@pytest.mark.integration
@pytest.mark.parametrize("extension", ["jpg", "png", "tif"])
def test_cli_inference_preserves_format_and_metadata(make_photo, extension):
    from photo_denoise.cli import main
    from photo_denoise.images import read_photo
    from photo_denoise.metadata import snapshot

    device = os.environ.get("PHOTO_DENOISE_TEST_DEVICE", "cpu")
    source = make_photo(extension, uint16=extension == "tif", alpha=extension == "png")
    destination = source.with_stem("restored")
    assert (
        main(
            [
                "denoise",
                str(source),
                "-o",
                str(destination),
                "--model",
                "drunet",
                "--sigma",
                "20",
                "--amount",
                "0.5",
                "--device",
                device,
                "--offline",
            ]
        )
        == 0
    )
    before, after = snapshot(source), snapshot(destination)
    assert all(after.get(key) == value for key, value in before.items())
    initial, processed = read_photo(source), read_photo(destination)
    assert initial.dtype == processed.dtype
    assert initial.rgb.shape == processed.rgb.shape
    np.testing.assert_array_equal(initial.alpha, processed.alpha)
