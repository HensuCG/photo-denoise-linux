# Detailed CLI guide

This guide retains the original source-install workflow and detailed format/metadata notes. For AppImage installation, runtime selection and Vulkan usage, see [README](../README.md). CLI commands work in the AppImage as well; installed runtime settings can override the original CUDA default.

A local Linux CLI for denoising photos on an NVIDIA GPU while preserving their metadata. Tested on an RTX 3060, with a CPU option for diagnosis or use without CUDA.

## Installation guide

Requires Linux x86_64, Git, Perl, `curl`, `tar`, [uv](https://docs.astral.sh/uv/getting-started/installation/), and a working NVIDIA driver compatible with CUDA 12.8. Allow several GB of disk space for PyTorch and its CUDA runtime libraries. A separate system CUDA toolkit is not required.

Check the NVIDIA driver first:

```bash
nvidia-smi
```

Install the command-line prerequisites using your distribution's package manager. For Arch/CachyOS:

```bash
sudo pacman -S --needed git curl tar perl
```

For Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install git curl tar perl
```

If `uv` is not installed, follow its [installation instructions](https://docs.astral.sh/uv/getting-started/installation/). Then clone and install:

```bash
git clone https://github.com/HensuCG/photo-denoise-linux.git photo-denoise
cd photo-denoise
./scripts/install.sh
```

The installer selects Python 3.12, installs the tested dependencies from `requirements.lock.txt` into `.venv`, downloads the three official model checkpoints, and runs diagnostics. It uses system ExifTool if available, otherwise downloads a checksum-verified local ExifTool 13.59 into `.tools`. It does not install system packages.

The installer also links the launcher into `~/.local/bin` if that command name is unused. To use `photo-denoise` from any directory, add that directory to your PATH if needed:

```bash
export PATH="$HOME/.local/bin:$PATH"
photo-denoise doctor
```

Add the export to your shell's startup file to retain it for future terminals. Alternatively, run `./photo-denoise` from the project directory. For an existing installation, use the application directory already on your machine and skip cloning and installation.

## First photo

```bash
./photo-denoise doctor
./photo-denoise denoise "/path/to/photo.jpg"
```

The default command uses SCUNet on CUDA and creates `photo-denoised.jpg` beside the original. It never overwrites input files. After installation, all three model checkpoints are cached, so processing does not need a download. Quote paths containing spaces.

Open the original and the output in your preferred photo viewer and compare them at 100% zoom. Reduce `--amount` if fine texture is too smooth. Each successful save reports that metadata verification passed.

Reduce the amount of denoising:

```bash
./photo-denoise denoise /path/to/photo.jpg --amount 0.6
```

Use DRUNet with explicit noise-level control:

```bash
./photo-denoise denoise /path/to/photo.jpg --model drunet --sigma 20 --amount 0.8
```

Process a directory tree:

```bash
./photo-denoise denoise /path/to/photos --recursive --output-dir /path/to/results
```

Subdirectories are retained under the output directory. Files ending in the chosen suffix, and files inside a distinct output directory, are skipped during directory discovery so generated outputs do not get processed repeatedly. Explicitly naming a file allows it to be processed regardless of its suffix. Duplicate destination paths are rejected.

Choose a destination for one photo:

```bash
./photo-denoise denoise /path/to/photo.tif -o /path/to/clean.tif
```

The output must have the same format as the input. `.jpg`/`.jpeg` and `.tif`/`.tiff` are interchangeable within their format. Existing output files are protected unless `--overwrite` is supplied; input files remain protected even with that option.

For all options:

```bash
./photo-denoise denoise --help
```

## Models and strength

| Model | Best starting use | Controls |
|---|---|---|
| `scunet` (default) | Real photos; blind noise estimation; PSNR checkpoint | `--amount` |
| `scunet-gan` | Real photos; perceptual GAN checkpoint | `--amount` |
| `drunet` | Noise approximated by additive Gaussian noise | `--sigma` and `--amount` |

`--amount` ranges from 0 to 1 and blends the model output with the original decoded pixels. For example, 0.6 applies 60% of the predicted change. It can reduce denoising below the full model result. It does not ask the model to perform stronger denoising than its normal output.

`--sigma` is the assumed noise standard deviation on a 0–255 pixel scale, not ISO and not a percentage. DRUNet defaults to 15; try 5–10 for light noise, 15–25 for moderate noise, and 30–50 for stronger noise. Higher values generally remove more noise and fine texture. SCUNet estimates the noise internally and rejects `--sigma` to avoid presenting a control that has no effect.

`--amount 0` skips inference and model loading. PNG and TIFF retain their decoded pixel values; JPEG is still re-encoded, so its pixels may differ slightly even at amount zero. JPEG defaults to quality 95 and no chroma subsampling. Use `--jpeg-quality 100` for less encoding loss; saving a processed JPEG is not lossless.

## Metadata and pixels

JPEG additionally copies its complete standard/extended XMP segments intact, retaining unknown metadata schemas. Auxiliary JPEG image/trailer bytes are retained and MPF image offsets/sizes are updated for the new primary image. Original HDR gain maps are retained, not regenerated for the denoised pixels; model inference operates on the primary decoded RGB image.

JPEG and PNG copy the entire original EXIF block, including camera MakerNotes, unknown EXIF tags and embedded thumbnails. The output EXIF block is checked byte for byte before the file is published. EXIF orientation is retained and pixels are not rotated; viewers render the output using the same orientation as the input.

TIFF stores EXIF alongside its image data, so TIFF metadata is copied as tags. Pixel-storage tags such as strip offsets and compression describe the new encoding and are regenerated. Existing EXIF, MakerNotes, XMP, IPTC and ICC values are checked, excluding those storage fields. TIFFs are written uncompressed; file size may increase substantially. If a metadata value cannot be retained, processing fails for that photo and no output is published.

ICC profiles are copied and verified byte for byte for all supported formats. Pixels are processed in their existing encoded color space, without conversion into a model-specific working profile. This preserves the profile but does not guarantee equal model performance across unusual color spaces. Existing thumbnails/previews are retained from the original photo; they are not denoised. Filesystem creation dates, extended attributes and external `.xmp` sidecars are outside the embedded-metadata workflow.

| Format | Supported pixels |
|---|---|
| JPEG | 8-bit RGB and grayscale |
| PNG | 8-bit RGB, grayscale, palette images and transparency |
| TIFF | Single-page, contiguous RGB or MINISBLACK grayscale; unsigned 8-bit or 16-bit; optional unassociated alpha |

Transparency is retained without denoising the alpha channel. Grayscale images use the color model on three identical channels, then average the result back to grayscale. Sixteen-bit TIFFs retain 16-bit output precision, although these pretrained models were not specifically trained as RAW sensor denoisers.

RAW files, 16-bit PNG, CMYK, floating-point TIFF, multipage TIFF, planar TIFF and premultiplied alpha TIFF are rejected with an explanation. Export unsupported photos to a supported RGB TIFF first. Palette PNG pixels are expanded to RGB/RGBA; the original palette encoding is not retained.

## GPU memory and offline use

The default `--tile-size 512 --overlap 32` processes large images in bounded GPU memory. Each output tile receives surrounding context and discards its outer context pixels. Tiled predictions can differ slightly from processing a whole image. For a visible boundary on difficult texture, try `--overlap 64`, or increase tile size if memory allows. Increasing overlap costs time.

If CUDA memory is exhausted, reduce tile size:

```bash
./photo-denoise denoise /path/to/photo.jpg --tile-size 256
```

Images are still decoded fully in system RAM. Default compute is float32 for predictable compatibility and output quality.

Model downloads come from the official [KAIR release](https://github.com/cszn/KAIR/releases/tag/v1.0) and are verified against pinned SHA-256 checksums before loading. Checkpoints are loaded using PyTorch's tensor-only mode. The default cache is `${XDG_CACHE_HOME:-~/.cache}/photo-denoise`; override it with `--cache-dir`.

```bash
./photo-denoise download all
./photo-denoise denoise /path/to/photo.jpg --offline
./photo-denoise denoise /path/to/photo.jpg --device cpu
```

No photos are uploaded. Once checkpoints are cached, inference is entirely local. `--device auto` selects CUDA when available, otherwise CPU; the default `cuda` reports an error instead of silently switching to slower CPU inference.

Each output is written to a temporary file in the destination directory, verified, then published atomically. A failed photo produces an error and batches continue with subsequent photos. Exit codes are 0 for success, 1 for a processing/setup failure, 2 for invalid command syntax, and 130 for interruption. Some photos in a failed batch may have completed successfully. An existing output stays intact if replacement processing fails.

## Troubleshooting

| Problem | Action |
|---|---|
| `photo-denoise: command not found` | Run `./photo-denoise` from the clone directory, or add `~/.local/bin` to PATH as shown above. |
| CUDA is unavailable | Run `nvidia-smi` and `./photo-denoise doctor`. Check that the NVIDIA driver works and the installer completed. Use `--device cpu` to process without GPU access. |
| CUDA runs out of memory | Close other GPU workloads or use `--tile-size 256`. |
| Model is missing in offline mode | Run `./photo-denoise download all`, using the same `--cache-dir` if you selected a custom cache. |
| ExifTool is missing | Rerun `./scripts/install.sh` or set `PHOTO_DENOISE_EXIFTOOL=/absolute/path/to/exiftool`. Perl must be installed. |
| Metadata verification fails | No new output is published. Read the named tags in the error; keep the original and report the file format and error for investigation. |
| Unsupported input format | Export to a supported RGB JPEG, 8-bit PNG or unsigned 8/16-bit TIFF. See the format table above. |
| Output already exists | Choose another destination/suffix, or use `--overwrite` to replace an output. Inputs remain protected. |

If reporting a problem, include the exact command, error message, and `./photo-denoise doctor` output. Avoid sharing private photos or GPS metadata unless you intend to disclose them.

## Update an existing installation

From the project directory:

```bash
git pull --ff-only
./scripts/install.sh
```

The installer reuses cached model weights and installed dependencies when they already match the pinned versions.

## Tests and example output

```bash
.venv/bin/pytest -q -m 'not integration'
PHOTO_DENOISE_TEST_DEVICE=cuda .venv/bin/pytest -q -m integration
.venv/bin/ruff check src tests scripts/benchmark.py
.venv/bin/ruff format --check src tests scripts/benchmark.py
```

Integration tests require cached official weights. They default to CPU unless the environment variable above selects CUDA. Camera metadata tests use small fixtures shipped in the local ExifTool source distribution and skip if those fixtures are not installed.

Locally generated `test-results/comparison.png` adds known Gaussian noise to a clean reference image and shows all three outputs. `test-results/benchmark.json` records timings and PSNR. Generated images, weights, environments and test artifacts are excluded from Git. The measured results from the initial RTX 3060 run are included in [benchmark.json](benchmark.json). This is a reproducible example, not a broad ranking of model quality or a benchmark of camera sensor noise.

To reproduce it with a clean reference image:

```bash
.venv/bin/python scripts/benchmark.py --clean /path/to/clean.png
```

See [TESTING.md](../TESTING.md) for the measured results and validation coverage. For the full test suite with CUDA:

```bash
PHOTO_DENOISE_TEST_DEVICE=cuda .venv/bin/pytest -q
```

## Upstream attribution

Inference architectures are provided by [Spandrel](https://github.com/chaiNNer-org/spandrel). The original model projects are [SCUNet](https://github.com/cszn/SCUNet) and [DRUNet / DPIR](https://github.com/cszn/DPIR). Metadata handling uses [ExifTool](https://exiftool.org/). These dependencies and weights retain their upstream licensing terms; see `THIRD_PARTY.md`.
