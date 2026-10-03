"""Persistent user choices, independent of optional inference packages."""

import json
import os
import tempfile
from pathlib import Path


def data_directory():
    override = os.environ.get("PHOTO_DENOISE_HOME")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "photo-denoise"


def load_settings():
    path = data_directory() / "settings.json"
    try:
        value = json.loads(path.read_text())
        if isinstance(value, dict) and value.get("runtime") in ("cuda", "vulkan", "cpu"):
            return value
    except (OSError, ValueError):
        pass
    return {}


def save_settings(value):
    folder = data_directory()
    folder.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".settings-", dir=folder)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(name, folder / "settings.json")
    finally:
        Path(name).unlink(missing_ok=True)
