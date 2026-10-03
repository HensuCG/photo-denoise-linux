"""Build a GUI/Python AppImage; inference libraries are installed on first launch."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL_SHA = "a6d71e2b6cd66f8e8d16c37ad164658985e0cf5fcaa950c90a482890cb9d13e0"
RUNTIME_SHA = "156f4bdbde9c52d01814600013e0a273f0118dc2de98975f3c8c63427ec79074"


def main():
    os.chdir(ROOT)
    for name, url, expected in (
        (
            "appimagetool-x86_64.AppImage",
            "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage",
            TOOL_SHA,
        ),
        (
            "appimage-runtime-x86_64",
            "https://github.com/AppImage/type2-runtime/releases/download/continuous/runtime-x86_64",
            RUNTIME_SHA,
        ),
    ):
        target = ROOT / ".tools" / name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            subprocess.run(
                ["curl", "-fsSL", "--connect-timeout", "10", url, "-o", str(target)], check=True
            )
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Packaging tool hash mismatch: {name}")
        target.chmod(0o755)
    build = ROOT / "build/appimage"
    build.mkdir(parents=True, exist_ok=True)
    app = build / "PhotoDenoise.AppDir"
    if app.exists():
        shutil.rmtree(app)
    python = app / "usr/python"
    # uv's Python build is relocatable. Copy the interpreter/stdlib, not this
    # developer environment's GPU libraries or virtual-environment links.
    original = Path(sys.base_prefix)
    shutil.copytree(
        original, python, ignore=shutil.ignore_patterns("__pycache__", "*.a", "test", "tests")
    )
    packages = python / "lib/python3.12/site-packages"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python / "bin/python3.12"),
            "--target",
            str(packages),
            "numpy==2.5.2",
            "Pillow==12.3.0",
            "tifffile==2026.9.20",
            "PySide6-Essentials==6.11.2",
            "shiboken6==6.11.2",
            "packaging==26.3",
        ],
        check=True,
    )
    source = app / "usr/share/photo-denoise"
    shutil.copytree(ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__"))
    exiftool = ROOT / ".tools/exiftool-13.59"
    if not (exiftool / "exiftool").exists():
        raise ValueError("Install project-local ExifTool with scripts/install.sh before packaging")
    tool_target = source / ".tools/exiftool-13.59"
    tool_target.mkdir(parents=True)
    shutil.copy2(exiftool / "exiftool", tool_target / "exiftool")
    shutil.copytree(exiftool / "lib", tool_target / "lib")
    for name in ("LICENSE", "README"):
        shutil.copy2(exiftool / name, tool_target / name)
    shutil.copytree(ROOT / "licenses", source / "licenses")
    for name in ("LICENSE", "THIRD_PARTY.md"):
        shutil.copy2(ROOT / name, source / name)
    for name in ("photo-denoise.svg", "photo-denoise.desktop"):
        shutil.copy2(ROOT / "assets" / name, app / name)
    launcher = app / "AppRun"
    launcher.write_text("""#!/bin/sh
set -eu
APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PHOTO_DENOISE_BUNDLED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$APP_ROOT/usr/share/photo-denoise/src"
export QT_PLUGIN_PATH="$APP_ROOT/usr/python/lib/python3.12/site-packages/PySide6/Qt/plugins"
exec "$APP_ROOT/usr/python/bin/python3.12" -m photo_denoise.launch "$@"
""")
    launcher.chmod(0o755)
    tool_dir = build / "tool"
    tool_dir.mkdir(exist_ok=True)
    subprocess.run(
        [str(ROOT / ".tools/appimagetool-x86_64.AppImage"), "--appimage-extract"],
        cwd=tool_dir,
        stdout=subprocess.DEVNULL,
        check=True,
    )
    output = ROOT / "dist/PhotoDenoise-0.2.0-x86_64.AppImage"
    output.parent.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["ARCH"] = "x86_64"
    subprocess.run(
        [
            str(tool_dir / "squashfs-root/AppRun"),
            "--runtime-file",
            str(ROOT / ".tools/appimage-runtime-x86_64"),
            str(app),
            str(output),
        ],
        env=env,
        check=True,
    )
    output.chmod(0o755)
    print(f"Built {output} ({output.stat().st_size / 10**6:.1f} MB)")


if __name__ == "__main__":
    main()
