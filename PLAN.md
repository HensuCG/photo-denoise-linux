# Photo denoise CLI implementation plan

1. Inspect the machine and choose a CUDA-compatible Python environment.
2. Implement SCUNet real-photo denoising and DRUNet noise-level control, tiled inference, and output blending.
3. Preserve metadata using ExifTool; retain orientation, ICC profiles, transparency and TIFF bit depth. Write atomically and protect original files.
4. Provide single-file and batch commands, model download/cache management, and a diagnostics command.
5. Test real RTX 3060 inference, synthetic noise reduction, tile boundaries, and metadata round trips across JPEG, PNG and TIFF. Test failures and input protection.
6. Document installation, CLI examples, test results and format limitations.

## Completion

All six steps completed. The CLI is installed in the project environment, all three official models are cached and checksum-verified, and the final suite passed 47 tests with CUDA inference on the RTX 3060. CPU inference was also exercised. README.md documents usage and limitations; TESTING.md records the validation and benchmark results. Photographic example outputs are saved in test-results/.

## Environment findings

- CachyOS; RTX 3060 with 12 GB VRAM; NVIDIA driver 615.71.09.
- GPU access works outside the tool sandbox.
- Use existing Python 3.12 in a project virtual environment.
- ExifTool 13.59 is installed locally under `.tools/` (no system package changes).
