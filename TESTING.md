# Validation results

## Extended-XMP fix (0.2.1)

**83 tests passed in 47.50 seconds** on 2026-10-03, with real CUDA and Vulkan inference enabled. Report: `test-results/pytest-xmp-fix.xml`.

The reported photo reproduced a tag-copy failure for XMPToolkit, HasExtendedXMP and an unknown GCamera schema. The fix preserves JPEG standard/extended XMP segments byte for byte, copies original trailer/auxiliary image bytes, and recalculates MPF pointers and primary image size. PNG/TIFF copy the complete XMP block so unknown schemas survive. Strict metadata/EXIF/ICC checks remain enabled.

Eleven new regression cases cover standard and multi-segment extended XMP, custom namespaces, progressive JPEG scans, little/big-endian MPF directories, exact auxiliary-image extraction, truncated scans, and unknown XMP in PNG/TIFF. Full denoising of the reported 4080 × 3072 photo passed on CUDA and AMD Vulkan with all source metadata verified. The rebuilt AppImage GUI also processed that photo using the user's installed CUDA runtime, with a result preview and metadata verification. No private photo or generated output is uploaded.

HDR gain maps are preserved unchanged as auxiliary image data; they are not regenerated from model output. Inference uses the primary decoded RGB image. Subjective HDR rendering/denoising quality still needs user review.


## GUI/AppImage milestone (0.2.0)

Completed on 2026-10-03 on CachyOS, Python 3.12.13, NVIDIA RTX 3060 12 GB (driver 615.71.09), and AMD RX 6900 XT 16 GB. PySide6/Qt 6.11.2, ncnn 1.0.20260526, PyTorch 2.7.1+cu128 / 2.7.1+cpu. Hands-on user acceptance remains pending; this is a preview release.

**72 automated tests passed in 43.50 seconds**, including real CUDA, both Vulkan devices and CPU inference. Report: `test-results/pytest-gui-runtimes.xml`.

```bash
QT_QPA_PLATFORM=offscreen PHOTO_DENOISE_TEST_DEVICE=cuda .venv/bin/pytest -q --junitxml=test-results/pytest-gui-runtimes.xml
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
```

Coverage adds GUI runtime/GPU selection, restore/settings filtering, actual GUI-to-worker processing and previews on all four runtime/device combinations, cancellation without partial outputs, setup cancellation, 16-bit TIFF preview normalization and orientation, checked downloads, archive traversal rejection, retry after failed probes, shared-cache removal, CLI runtime dispatch and Vulkan numerical agreement. All original pixel/metadata/input-protection tests remain in the suite.

Vulkan DRUNet outputs are compared against CUDA on odd-sized inputs with mean absolute error below 0.002 on the 0–1 scale. Native Vulkan allocation failures are checked even if ncnn returns success, preventing invalid saves. Winograd convolution is disabled to reduce weight memory under competing GPU workloads. SCUNet conversion was attempted but unsupported attention operations prevented reliable conversion; it is not shipped for Vulkan.

### Packaging and fresh installation checks

The final AppImage is approximately 128.9 MB and contains Python/Qt/image libraries/ExifTool, with no PyTorch, CUDA or ncnn inference libraries. The actual FUSE-mounted executable launches CLI commands. A controlled native GUI launch selected the Wayland platform and produced a screenshot before closing automatically.

Fresh profiles using the bundled interpreter downloaded and checksum-verified the real runtime wheels: Vulkan 8.4 MB, CPU 189.4 MB and CUDA 3.86 GB. The CPU probe reported `2.7.1+cpu` with no CUDA runtime; CUDA reported `2.7.1+cu128`, CUDA 12.8 and the RTX 3060. Each isolated runtime processed a photo successfully through the GUI subprocess path, with metadata verification. Separate actual AppImage CLI checks exercised installed CUDA and CPU libraries. The fresh Vulkan setup smoke also checked settings persistence and absence of PyTorch in the GUI environment.

```bash
QT_QPA_PLATFORM=offscreen PYTHONNOUSERSITE=1 \
  build/appimage/PhotoDenoise.AppDir/usr/python/bin/python3.12 \
  scripts/smoke_appimage.py --app-dir build/appimage/PhotoDenoise.AppDir \
  --model-dir build/ncnn --photo test-results/noisy.png --results test-results
```

The prepublication smoke used exact converted artifacts through local file URLs. The final build also passed a fresh-profile smoke with `--model-dir` omitted: ncnn and both model files were downloaded from their public release URLs, checksum-verified, and used for successful GUI processing. The repository rename to `photo-denoise-linux` redirects the original model download URLs correctly. GitHub asset digests match the local AppImage/model checksums. Generated screenshots, JUnit reports and photos remain local under `test-results/`; private input images are not uploaded. Lint, formatting and shell syntax checks passed.

### Remaining manual checks

The automated checks establish processing and metadata behavior on this machine. They do not establish desktop usability across distributions or denoising quality on the user's real camera photos. Intel Vulkan and other GPUs are untested. Use [the GUI feedback checklist](docs/GUI_TESTING.md) for display/layout/file-dialog behavior, subjective photo quality, and actual interaction with progress/cancellation/settings.

## Original CLI validation (0.1.0)

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
