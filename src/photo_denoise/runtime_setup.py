"""Install only the chosen inference runtime using verified wheel downloads."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .models import checksum
from .settings import data_directory
from .transfer import check_cancelled, download_file


def manifest():
    return json.loads(Path(__file__).with_name("runtime-manifest.json").read_text())


def runtime_directory(runtime):
    return data_directory() / "runtimes" / runtime


def runtime_environment(runtime, directory=None):
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1])
    folder = str(directory or runtime_directory(runtime))
    env["PYTHONPATH"] = os.pathsep.join((folder, source))
    env["PYTHONNOUSERSITE"] = "1"
    env["PHOTO_DENOISE_ACTIVE_RUNTIME"] = runtime
    env["PYTHONUNBUFFERED"] = "1"
    return env


def probe_runtime(runtime, *, directory=None, timeout=60):
    completed = subprocess.run(
        [sys.executable, "-m", "photo_denoise.runtime_probe", runtime],
        env=runtime_environment(runtime, directory),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode:
        raise ValueError(
            completed.stderr.strip().splitlines()[-1]
            if completed.stderr.strip()
            else f"{runtime} runtime probe failed"
        )
    return json.loads(completed.stdout)


def runtime_installed(runtime):
    return (runtime_directory(runtime) / "ready.json").is_file()


def install_runtime(runtime, *, progress=None, cancelled=None, allow_existing=True):
    spec = manifest()[runtime]
    target = runtime_directory(runtime)
    if runtime_installed(runtime):
        return probe_runtime(runtime)
    if allow_existing and not os.environ.get("PHOTO_DENOISE_BUNDLED"):
        try:
            result = probe_runtime(runtime)
            target.mkdir(parents=True, exist_ok=True)
            result["source"] = "existing installation"
            (target / "ready.json").write_text(json.dumps(result) + "\n")
            return result
        except (ValueError, subprocess.SubprocessError):
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    cache = data_directory() / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    # Keep verified wheels for retry. Ensure room for downloads plus unpacked packages.
    required = max(
        spec["download_bytes"] * 4, {"cuda": 0, "cpu": 1500 * 10**6, "vulkan": 200 * 10**6}[runtime]
    )
    if shutil.disk_usage(target.parent).free < required:
        raise ValueError("Not enough disk space for the selected runtime; free space and retry")
    staging = Path(tempfile.mkdtemp(prefix=f".{runtime}-", dir=target.parent))
    done = 0
    try:
        for wheel in spec["wheels"]:
            check_cancelled(cancelled)
            path = cache / wheel["filename"]
            if not path.exists() or checksum(path) != wheel["sha256"]:
                download_file(
                    wheel["url"],
                    path,
                    wheel["sha256"],
                    size=wheel["bytes"],
                    cancelled=cancelled,
                    progress=lambda current, total: (
                        progress(
                            f"Downloading {wheel['name']}", done + current, spec["download_bytes"]
                        )
                        if progress
                        else None
                    ),
                )
            done += wheel["bytes"]
            if progress:
                progress(f"Installing {wheel['name']}", done, spec["download_bytes"])
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    check_cancelled(cancelled)
                    name = PurePosixPath(item.filename)
                    if name.is_absolute() or ".." in name.parts:
                        raise ValueError("Unsafe wheel archive path")
                    parts = name.parts
                    if parts and parts[0].endswith(".data"):
                        if len(parts) < 3 or parts[1] not in ("purelib", "platlib"):
                            continue
                        parts = parts[2:]
                    output = staging.joinpath(*parts)
                    if item.is_dir():
                        output.mkdir(parents=True, exist_ok=True)
                    else:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(item) as source, output.open("wb") as stream:
                            while chunk := source.read(1024 * 1024):
                                check_cancelled(cancelled)
                                stream.write(chunk)
                        mode = item.external_attr >> 16
                        if mode & 0o111:
                            output.chmod(0o755)
        if progress:
            progress("Checking runtime and GPU", 0, 0)
        check_cancelled(cancelled)
        result = probe_runtime(runtime, directory=staging)
        check_cancelled(cancelled)
        (staging / "ready.json").write_text(json.dumps(result) + "\n")
        if target.exists():
            # Only this application's incomplete runtime directory is replaceable.
            shutil.rmtree(target)
        os.replace(staging, target)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def remove_runtime(runtime):
    target = runtime_directory(runtime)
    if target.is_symlink():
        raise ValueError("Refusing a symlink runtime directory")
    if target.exists():
        shutil.rmtree(target)
    specifications = manifest()
    shared = {
        wheel["filename"]
        for other, spec in specifications.items()
        if other != runtime and runtime_installed(other)
        for wheel in spec["wheels"]
    }
    for wheel in specifications[runtime]["wheels"]:
        if wheel["filename"] not in shared:
            (data_directory() / "downloads" / wheel["filename"]).unlink(missing_ok=True)
