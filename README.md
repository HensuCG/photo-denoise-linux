# Photo Denoise

Local Linux photo denoising with a GUI, a backup CLI, and verified preservation of embedded metadata. Your photos stay on your computer; original files are protected.

## AppImage installation

Download **PhotoDenoise-0.2.0-x86_64.AppImage** from the [release page](https://github.com/HensuCG/photo-denoise-linux/releases/tag/v0.2.0), then:

```bash
chmod +x PhotoDenoise-0.2.0-x86_64.AppImage
./PhotoDenoise-0.2.0-x86_64.AppImage
```

Requires Linux x86_64 with glibc 2.34 or newer, a working desktop, and Perl for the bundled ExifTool. Install your GPU's NVIDIA or Vulkan drivers through your distribution. A system CUDA toolkit and system Python are not required. `curl` is recommended for downloads. If FUSE mounting fails, run with `--appimage-extract-and-run` before the application arguments.

The AppImage is approximately **129 MB**. On first startup, choose a runtime and GPU; the app downloads and checks only the selected runtime and its default model. A progress bar shows downloaded bytes. Setup can be cancelled and retried; completed downloads are reused. Downloads require internet access; processing after setup is local.

| Runtime | Recommended hardware | Runtime download | Models |
|---|---|---:|---|
| CUDA / PyTorch | NVIDIA, including RTX 3060 | 3.86 GB | SCUNet, SCUNet GAN, DRUNet |
| Vulkan / ncnn | AMD, NVIDIA; Intel is untested | 8.4 MB | DRUNet |
| CPU | Any compatible x86_64 CPU | 189 MB | SCUNet, SCUNet GAN, DRUNet |

SCUNet models are approximately **72 MB each**; DRUNet is approximately **131 MB**. CUDA setup requires about **16 GB free disk space** for downloads and unpacking; CPU needs 1.5 GB plus model space. GPU drivers must already work. CUDA uses CUDA 12.8 runtime libraries and needs a compatible NVIDIA driver. CPU is generally much slower; relative CUDA/Vulkan speed depends on the GPU and model. Vulkan was tested on RX 6900 XT and RTX 3060. SCUNet conversion to ncnn is not validated and is not offered with Vulkan.

## GUI usage

Add photos or a folder, choose a model, adjust **Denoising amount**, and click **Denoise photos**. Outputs default to the originals' folder with a `-denoised` suffix; choose an output folder to keep results together. Folder structure is retained. Existing outputs require the explicit replacement checkbox. You can cancel processing and change runtime/GPU later in **Settings**.

Amount blends the prediction with the original: 60% applies 60% of the predicted change. DRUNet also has a **noise level** control on a 0–255 scale: start at 15, lower it for light noise and raise it for stronger noise. It is not ISO. SCUNet estimates noise internally. If memory is limited, choose tile size 256 or 128. Original/result previews show the selected photo; assess fine detail in your photo viewer at 100% zoom.

Settings, runtime libraries and verified wheel downloads live in `${XDG_DATA_HOME:-~/.local/share}/photo-denoise`. Model weights live in `${XDG_CACHE_HOME:-~/.cache}/photo-denoise`. Settings can remove a runtime and its unshared downloaded wheels while keeping model weights and photos. `PHOTO_DENOISE_HOME` overrides the settings/runtime directory.

## CLI backup

The same AppImage accepts CLI commands. Runtime libraries installed through the GUI are reused; ordinary `denoise` commands use the selected runtime/GPU.

```bash
./PhotoDenoise-0.2.0-x86_64.AppImage doctor
./PhotoDenoise-0.2.0-x86_64.AppImage denoise /path/to/photo.jpg --amount 0.6
./PhotoDenoise-0.2.0-x86_64.AppImage denoise /path/to/photo.tif --runtime vulkan --gpu-index 0 --model drunet --sigma 20
./PhotoDenoise-0.2.0-x86_64.AppImage denoise /path/to/photos --recursive --output-dir /path/to/results
./PhotoDenoise-0.2.0-x86_64.AppImage denoise --help
```

For Vulkan, specify `--model drunet` in CLI commands; the CLI model default remains SCUNet. `--runtime cpu` and `--runtime cuda` select an already installed runtime explicitly. `--offline` requires cached weights. `--amount 0` skips inference. Quote paths containing spaces. See [the detailed CLI guide](docs/CLI.md) for metadata details, advanced options and original benchmarks.

## Formats and metadata

Supports 8-bit JPEG/PNG and single-page unsigned 8/16-bit RGB or grayscale TIFF, including supported transparency. RAW, 16-bit PNG, CMYK, floating-point/multipage/planar TIFF and premultiplied TIFF alpha are rejected. TIFF output is uncompressed and may be much larger. JPEG is re-encoded, even at amount zero.

JPEG/PNG EXIF is preserved and verified byte for byte, including MakerNotes, unknown tags and original thumbnails. TIFF metadata values are verified while storage tags are regenerated. ICC profiles are copied and verified. Orientation is retained without rotating processing pixels. Existing embedded previews retain the original image; filesystem attributes and external XMP sidecars are outside this workflow. Unsupported metadata produces an error instead of publishing a damaged output. Output writes are atomic.

Pixels remain in their original encoded color space. These pretrained models are not RAW sensor denoisers. Tiled inference can differ slightly from whole-image processing. See [the format and preservation details](docs/CLI.md#metadata-and-pixels).

## Source installation and development

Requires Git, Perl, curl, tar and [uv](https://docs.astral.sh/uv/getting-started/installation/). The source installer creates Python 3.12 `.venv`, installs the pinned CUDA developer environment plus GUI/ncnn, downloads official PyTorch checkpoints, and sets up local ExifTool. Use the AppImage for a small installation that downloads only your chosen runtime.

```bash
git clone https://github.com/HensuCG/photo-denoise-linux.git photo-denoise
cd photo-denoise
./scripts/install.sh
./photo-denoise                # GUI
./photo-denoise denoise /path/to/photo.jpg
```

For an existing clone, run `git pull --ff-only` and `./scripts/install.sh` to update. To run tests and build:

```bash
uv pip install --python .venv/bin/python -e '.[dev,gui]'
QT_QPA_PLATFORM=offscreen PHOTO_DENOISE_TEST_DEVICE=cuda .venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/python scripts/build_appimage.py
```

The AppImage builder requires a relocatable uv Python 3.12 and project-local ExifTool. Runtime wheel URLs/checksums are pinned in `src/photo_denoise/runtime-manifest.json`; `scripts/runtime_manifest.py` regenerates that manifest. `scripts/export_ncnn.py --help` describes model conversion; only the shipped DRUNet artifacts are validated.

## Status and troubleshooting

Version 0.2.0 is a preview pending hands-on desktop feedback. Automated Qt tests, real inference, isolated runtime installation and AppImage checks are recorded in [TESTING.md](TESTING.md). Follow [the GUI feedback checklist](docs/GUI_TESTING.md) for manual testing.

For GPU errors, check `nvidia-smi` or `vulkaninfo`, try a smaller tile, and run `doctor`. For setup failures, keep the error text and retry; completed downloads are retained. Metadata errors leave originals and existing outputs intact. Report your runtime/GPU, exact error and `doctor` output; private photos are not required.

## Attribution

MIT application code. Models: [SCUNet](https://github.com/cszn/SCUNet), [DRUNet/DPIR](https://github.com/cszn/DPIR); inference: Spandrel, PyTorch and ncnn; GUI: PySide6/Qt; metadata: ExifTool. Dependencies and weights retain their upstream licenses; see [THIRD_PARTY.md](THIRD_PARTY.md).
