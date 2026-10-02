# Third-party components

Application-specific code in this repository is licensed under MIT (see LICENSE). Third-party components are installed or downloaded separately and retain their respective licenses.

| Component | Source and attribution | License reference |
|---|---|---|
| SCUNet checkpoints | Kai Zhang et al., Practical Blind Image Denoising via Swin-Conv-UNet and Data Synthesis; official KAIR v1.0 downloads | https://github.com/cszn/SCUNet/blob/main/LICENSE (Apache 2.0) |
| DRUNet checkpoint | Kai Zhang et al., Plug-and-Play Image Restoration with Deep Denoiser Prior; official KAIR v1.0 download | https://github.com/cszn/DPIR/blob/master/LICENSE |
| Spandrel | chaiNNer contributors; architecture loading and inference | https://github.com/chaiNNer-org/spandrel |
| ExifTool 13.59 | Phil Harvey; metadata preservation and camera test fixtures | https://exiftool.org/ (same terms as Perl: Artistic License or GPL) |
| PyTorch / torchvision | PyTorch contributors; tensor operations and CUDA inference | https://github.com/pytorch/pytorch/blob/main/LICENSE ; https://github.com/pytorch/vision/blob/main/LICENSE |
| Photographic test image | `testsets/set12/09.png` from SCUNet's upstream test set; used locally for a noise-added benchmark | https://github.com/cszn/SCUNet/tree/main/testsets/set12 |

Additional Python and NVIDIA runtime dependencies are recorded in `requirements.lock.txt`; their installed distributions contain the corresponding licensing notices. This repository does not relicense third-party weights or benchmark images.
