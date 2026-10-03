# Third-party components

Application code is MIT licensed. Dependencies and pretrained weights retain their upstream terms. The AppImage contains unmodified Python, NumPy, Pillow, tifffile, packaging, PySide6/Qt, shiboken6 and ExifTool. Inference libraries and weights are downloaded separately on request.

| Component | Attribution/source | License |
|---|---|---|
| SCUNet checkpoints | Kai Zhang et al.; [SCUNet](https://github.com/cszn/SCUNet), official KAIR v1.0 weights | Apache 2.0; [included notice](licenses/SCUNet.txt) |
| DRUNet checkpoint and converted ncnn weights | Kai Zhang et al.; [DPIR](https://github.com/cszn/DPIR), official KAIR v1.0 weights converted with pnnx | MIT; [included notice](licenses/DRUNet.txt) |
| Spandrel | [chaiNNer contributors](https://github.com/chaiNNer-org/spandrel) | MIT |
| PyTorch / torchvision | [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE), [torchvision](https://github.com/pytorch/vision/blob/main/LICENSE) | BSD-style; notices in downloaded distributions |
| ncnn / pnnx | [Tencent](https://github.com/Tencent/ncnn) | BSD 3-Clause; [included notice](licenses/ncnn.txt); pnnx is a build tool |
| PySide6, shiboken6 and Qt 6.11.2 | The Qt Company and contributors; [matching source](https://github.com/qt/pyside-setup/tree/v6.11.2), [Qt source](https://download.qt.io/official_releases/qt/6.11/6.11.2/submodules/) | LGPL 3; [distribution notice](licenses/Qt-NOTICE.txt), [LGPL](licenses/LGPL-3.0.txt), [GPL](licenses/GPL-3.0.txt) |
| ExifTool 13.59 | [Phil Harvey](https://exiftool.org/), metadata preservation | Same terms as Perl: Artistic License or GPL; LICENSE/README included alongside bundled ExifTool |
| Python | [Python Software Foundation](https://www.python.org/downloads/source/) | PSF; LICENSE included in bundled Python |
| NumPy, Pillow, tifffile, packaging | Respective upstream contributors | Notices in bundled Python distribution metadata |
| AppImage runtime | [AppImage type2-runtime](https://github.com/AppImage/type2-runtime) | MIT; appimagetool is a build tool |
| Local photographic benchmark | SCUNet `testsets/set12/09.png`, used locally for a noise-added benchmark | Upstream test-set attribution; image is not shipped in the app/repository |

The AppImage's LGPL libraries are dynamically loaded and can be replaced after `--appimage-extract`; see the Qt notice. No Qt library changes are made. Additional Python/NVIDIA runtime packages and pinned download URLs are recorded in `requirements.lock.txt` and `src/photo_denoise/runtime-manifest.json`; their wheel distributions include upstream notices. NVIDIA runtime components retain NVIDIA's license terms. This repository does not relicense models, dependencies or benchmark images.
