"""Test an extracted AppImage with a fresh profile and real Vulkan setup/UI work."""

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument(
        "--model-dir", type=Path, help="Local artifacts; omit to test release downloads"
    )
    parser.add_argument("--photo", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="photo-denoise-appimage-smoke-"))
    os.environ["PHOTO_DENOISE_HOME"] = str(root / "profile")
    os.environ["XDG_CACHE_HOME"] = str(root / "cache")
    os.environ["PHOTO_DENOISE_BUNDLED"] = "1"
    sys.path.insert(0, str(args.app_dir / "usr/share/photo-denoise/src"))
    assert importlib.util.find_spec("torch") is None, "AppImage unexpectedly bundles PyTorch"
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from photo_denoise import vulkan_models
    from photo_denoise.gui import MainWindow, RuntimeDialog
    from photo_denoise.metadata import snapshot
    from photo_denoise.settings import load_settings

    # The release asset is published after validation; use the exact checked build
    # artifacts locally for this prepublication first-launch test.
    if args.model_dir:
        vulkan_models.NCNN_URL = args.model_dir.resolve().as_uri() + "/"
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    setup = RuntimeDialog({})
    setup.runtime.setCurrentIndex(setup.runtime.findData("vulkan"))
    setup.show()
    setup.begin()
    timer = QTimer()
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(lambda: (print("Smoke test timed out", flush=True), app.exit(1)))
    deadline.start(180000)
    windows = []
    original = root / "photo.png"
    shutil.copy2(args.photo, original)
    state = {"processing": False, "finished": False}

    def poll():
        if not state["processing"]:
            if load_settings():
                print("Fresh Vulkan download/install and settings persistence passed", flush=True)
                window = MainWindow(first_start=False)
                windows.append(window)
                window.add_paths([original])
                window.show()
                window.start_processing()
                state["processing"] = True
            elif "Setup failed" in setup.status.text():
                print(setup.status.text(), flush=True)
                app.exit(1)
        else:
            window = windows[0]
            if window.status.text().startswith("Finished"):
                output = original.with_stem("photo-denoised")
                assert output.exists()
                before, after = snapshot(original), snapshot(output)
                assert all(after.get(key) == value for key, value in before.items())
                args.results.mkdir(parents=True, exist_ok=True)
                window.grab().save(str(args.results / "appimage-gui.png"))
                print("AppImage GUI Vulkan processing, previews and metadata passed", flush=True)
                print(f"Fresh test profile: {root}", flush=True)
                state["finished"] = True
                app.exit(0)
            elif window.status.text().startswith("Processing failed"):
                print(window.status.text(), flush=True)
                app.exit(1)

    timer.timeout.connect(poll)
    timer.start(100)
    code = app.exec()
    assert code or state["finished"], "GUI smoke test exited before processing completed"
    return code


if __name__ == "__main__":
    raise SystemExit(main())
