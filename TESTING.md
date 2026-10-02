# Validation results

Completed on 2026-10-02 on this machine: CachyOS Linux, NVIDIA GeForce RTX 3060 with 12 GB VRAM, NVIDIA driver 615.71.09, Python 3.12.13, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, Spandrel 0.4.2, ExifTool 13.59.

## Automated validation

The final complete suite passed: **47 tests in 23.45 seconds**, with real CUDA inference enabled. The JUnit report is saved in `test-results/pytest-cuda.xml`.

```bash
PHOTO_DENOISE_TEST_DEVICE=cuda .venv/bin/pytest -q --junitxml=test-results/pytest-cuda.xml
```

Coverage includes real inference with all three official checkpoints, measurable noise reduction, DRUNet sigma and blend controls, tile comparisons, tiny and odd dimensions, and the full CLI saving JPEG, PNG with alpha, and 16-bit TIFF with metadata intact. CPU inference was separately checked with the original six inference tests; all six passed in 5.33 seconds.

Metadata round trips cover EXIF capture dates, make/model, ISO, shutter speed, aperture, lens, GPS, orientation, Unicode XMP text, IPTC and ICC profiles; grayscale, alpha, 8-bit and 16-bit pixels; and authentic Canon, Nikon, Olympus, Sony, Pentax and Panasonic MakerNotes fixtures. Three further tests import camera metadata into 16-bit TIFF and verify a subsequent round trip.

Failure tests check input-file protection, hard links and symlinks, existing outputs, metadata errors, partial-batch failures, invalid options, unsupported formats, corrupt cached model files, and offline missing checkpoints. Outputs are not published when metadata verification fails.

Ruff lint and formatting checks passed. Shell syntax checks passed for the launcher and installer. The installer successfully used the pinned dependency file, rebuilt the editable CLI installation, verified cached checkpoints, and detected the RTX 3060 through `doctor`.

## Photographic example

The example starts with SCUNet's upstream `testsets/set12/09.png`, converts it to RGB, and adds deterministic Gaussian noise with sigma 20/255. The full CLI processes the noise-added PNG with each model and verifies its metadata.

| Output | PSNR against clean reference | Full CLI wall time |
|---|---|---|
| Noise-added input | 22.17 dB | — |
| SCUNet PSNR | 33.44 dB | 2.84 s |
| SCUNet GAN | 32.21 dB | 2.87 s |
| DRUNet, sigma 20 | 34.44 dB | 2.52 s |

Timings include process startup, model loading, image I/O and metadata copying/verification. This single Gaussian-noise example does not establish a general ranking for real camera photos.

The side-by-side image is `test-results/comparison.png`. Individual outputs and the noisy and clean references are in the same directory. The comparison was visually inspected: noise was reduced by all models and major photographic details were retained; each model smooths some fine texture.

## Larger image and tile behavior

A 2048 × 1536 image processed with SCUNet at the default tile size of 512 and context overlap of 32 took **5.15 seconds for inference** and reached **826 MiB peak allocated CUDA memory**. This measures PyTorch allocations, not the entire driver/process VRAM footprint. The result retained its dimensions and contained finite pixel values. The benchmark uses a resized test image and is intended to check processing and memory, not large-image restoration quality.

On the synthetic smooth-image test, DRUNet's tiled result differed from whole-image inference by mean absolute error 0.000250 and maximum absolute error 0.003346 on a 0–1 pixel scale. Tiling is an approximation and difficult textures may need greater context or a larger tile size.

Machine-readable benchmark results from this run are committed in `docs/benchmark.json`; the local original is `test-results/benchmark.json`. Generated photographic images and test reports under `test-results/` are not committed. Run `scripts/benchmark.py` as documented in README.md to repeat the photographic and larger-image exercises.
