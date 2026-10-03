import hashlib
import json
import threading
import zipfile

import pytest

from photo_denoise.runtime_setup import install_runtime, runtime_directory
from photo_denoise.settings import load_settings, save_settings
from photo_denoise.transfer import Cancelled, download_file


def test_cancelled_download_removes_partial_file(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"a" * (3 * 1024 * 1024))
    target = tmp_path / "destination"
    stop = threading.Event()
    with pytest.raises(Cancelled):
        download_file(
            source.as_uri(),
            target,
            hashlib.sha256(source.read_bytes()).hexdigest(),
            progress=lambda *_: stop.set(),
            cancelled=stop.is_set,
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".download-*"))


def test_bad_download_does_not_replace_existing_file(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"corrupted data")
    target = tmp_path / "destination"
    target.write_bytes(b"previous version")
    with pytest.raises(ValueError, match="checksum"):
        download_file(source.as_uri(), target, "0" * 64)
    assert target.read_bytes() == b"previous version"


@pytest.mark.parametrize("bad_path", ["../outside.txt", "/outside.txt"])
def test_runtime_rejects_archive_traversal(tmp_path, monkeypatch, bad_path):
    monkeypatch.setenv("PHOTO_DENOISE_HOME", str(tmp_path / "app-data"))
    source = tmp_path / "malicious.whl"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(bad_path, b"bad")
    wheel = {
        "name": "fixture",
        "filename": "fixture.whl",
        "url": source.as_uri(),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "bytes": source.stat().st_size,
    }
    monkeypatch.setattr(
        "photo_denoise.runtime_setup.manifest",
        lambda: {"vulkan": {"wheels": [wheel], "download_bytes": wheel["bytes"]}},
    )
    with pytest.raises(ValueError, match="Unsafe wheel"):
        install_runtime("vulkan", allow_existing=False)
    assert not runtime_directory("vulkan").exists()
    assert not list((tmp_path / "app-data/runtimes").glob(".vulkan-*"))
    assert not (tmp_path / "outside.txt").exists()


def test_failed_probe_retains_verified_download_for_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTO_DENOISE_HOME", str(tmp_path / "app-data"))
    source = tmp_path / "fixture.whl"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("fixture/__init__.py", b"value = 1\n")
    wheel = {
        "name": "fixture",
        "filename": "fixture.whl",
        "url": source.as_uri(),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "bytes": source.stat().st_size,
    }
    monkeypatch.setattr(
        "photo_denoise.runtime_setup.manifest",
        lambda: {"cpu": {"wheels": [wheel], "download_bytes": wheel["bytes"]}},
    )

    def fail(*args, **kwargs):
        raise ValueError("probe failed")

    monkeypatch.setattr("photo_denoise.runtime_setup.probe_runtime", fail)
    with pytest.raises(ValueError, match="probe failed"):
        install_runtime("cpu", allow_existing=False)
    assert not runtime_directory("cpu").exists()
    assert (tmp_path / "app-data/downloads/fixture.whl").exists()
    monkeypatch.setattr(
        "photo_denoise.runtime_setup.probe_runtime", lambda *a, **kw: {"runtime": "cpu", "gpus": []}
    )
    source.unlink()
    result = install_runtime("cpu", allow_existing=False)
    assert result["runtime"] == "cpu"
    assert (runtime_directory("cpu") / "ready.json").exists()


def test_settings_round_trip_and_invalid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTO_DENOISE_HOME", str(tmp_path))
    save_settings({"runtime": "vulkan", "gpu_index": 1})
    assert load_settings()["gpu_index"] == 1
    (tmp_path / "settings.json").write_text("invalid")
    assert load_settings() == {}
    (tmp_path / "settings.json").write_text(json.dumps({"runtime": "unknown"}))
    assert load_settings() == {}


def test_remove_runtime_keeps_shared_wheels_and_models(tmp_path, monkeypatch):
    from photo_denoise.runtime_setup import remove_runtime

    monkeypatch.setenv("PHOTO_DENOISE_HOME", str(tmp_path))
    specifications = {
        "cpu": {"wheels": [{"filename": "shared.whl"}, {"filename": "cpu.whl"}]},
        "cuda": {"wheels": [{"filename": "shared.whl"}, {"filename": "cuda.whl"}]},
    }
    monkeypatch.setattr("photo_denoise.runtime_setup.manifest", lambda: specifications)
    for runtime in specifications:
        folder = tmp_path / "runtimes" / runtime
        folder.mkdir(parents=True)
        (folder / "ready.json").write_text("{}")
    cache = tmp_path / "downloads"
    cache.mkdir()
    for name in ("shared.whl", "cpu.whl", "cuda.whl", "model.pth"):
        (cache / name).touch()
    remove_runtime("cpu")
    assert not (tmp_path / "runtimes/cpu").exists()
    assert (cache / "shared.whl").exists()
    assert (cache / "cuda.whl").exists()
    assert (cache / "model.pth").exists()
    assert not (cache / "cpu.whl").exists()
