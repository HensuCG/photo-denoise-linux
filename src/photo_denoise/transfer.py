"""Checked, cancellable streaming downloads with byte progress."""

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from .network import open_url


class Cancelled(Exception):
    pass


def check_cancelled(cancelled):
    if cancelled and cancelled():
        raise Cancelled("Cancelled")


def download_file(url, destination: Path, digest, *, size=0, progress=None, cancelled=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
    temporary = Path(name)
    try:
        curl = shutil.which("curl")
        if curl and url.startswith(("https://", "http://")):
            os.close(fd)
            # Curl handles dual-stack networking consistently across Linux distros.
            process = subprocess.Popen(
                [
                    curl,
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "10",
                    "--speed-time",
                    "30",
                    "--speed-limit",
                    "1024",
                    "--output",
                    str(temporary),
                    url,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                while process.poll() is None:
                    check_cancelled(cancelled)
                    if progress:
                        progress(temporary.stat().st_size, size)
                    time.sleep(0.08)
                if process.returncode:
                    raise ValueError("Download failed: " + process.stderr.read().strip())
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stderr.close()
            check_cancelled(cancelled)
            with temporary.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest and actual != digest:
                raise ValueError(f"Download checksum mismatch: {destination.name}")
            if size and temporary.stat().st_size != size:
                raise ValueError(f"Incomplete download: {destination.name}")
            if progress:
                progress(temporary.stat().st_size, size)
            os.replace(temporary, destination)
            return
        request = urllib.request.Request(url, headers={"User-Agent": "photo-denoise/0.2"})
        with os.fdopen(fd, "wb") as output, open_url(request) as source:
            total = int(source.headers.get("Content-Length") or size)
            done = 0
            hasher = hashlib.sha256()
            while True:
                check_cancelled(cancelled)
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
                output.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        check_cancelled(cancelled)
        if digest and hasher.hexdigest() != digest:
            raise ValueError(f"Download checksum mismatch: {destination.name}")
        if size and done != size:
            raise ValueError(f"Incomplete download: {destination.name}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
