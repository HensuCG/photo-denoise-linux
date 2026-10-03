"""Exercise real UI controls and the subprocess processing path."""

import json

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QProcess

from photo_denoise.gui import MainWindow, RuntimeDialog
from photo_denoise.metadata import snapshot
from photo_denoise.settings import data_directory, save_settings


@pytest.fixture
def isolated_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOTO_DENOISE_HOME", str(tmp_path / "app-data"))
    return tmp_path / "app-data"


def test_runtime_selection_filters_devices_and_models(qtbot, monkeypatch, isolated_settings):
    gpus = [
        {"index": 0, "name": "AMD test card", "vendor": "AMD"},
        {"index": 1, "name": "NVIDIA test card", "vendor": "NVIDIA"},
    ]
    monkeypatch.setattr("photo_denoise.gui.detect_gpus", lambda: gpus)
    dialog = RuntimeDialog({})
    qtbot.addWidget(dialog)
    assert dialog.runtime.currentData() == "cuda"
    assert dialog.gpu.count() == 1
    assert dialog.gpu.currentText() == "NVIDIA test card"
    dialog.runtime.setCurrentIndex(dialog.runtime.findData("vulkan"))
    assert dialog.gpu.count() == 2
    assert "Models: DRUNet." in dialog.details.text()
    dialog.runtime.setCurrentIndex(dialog.runtime.findData("cpu"))
    assert dialog.gpu.currentText() == "CPU"


def test_gui_restores_settings_and_disallows_unvalidated_model(qtbot, isolated_settings):
    save_settings({"runtime": "vulkan", "gpu_index": 0, "gpu_name": "AMD"})
    window = MainWindow(first_start=False)
    qtbot.addWidget(window)
    assert window.model.count() == 1
    assert window.model.currentData() == "drunet"
    assert window.sigma.isEnabled()


def test_no_runtime_or_inputs_does_not_start(qtbot, isolated_settings):
    window = MainWindow(first_start=False)
    qtbot.addWidget(window)
    assert not window.start_button.isEnabled()
    window.start_processing()
    assert "Add at least one" in window.status.text()
    assert window.process is None


@pytest.mark.integration
@pytest.mark.parametrize("runtime,index", [("cuda", 0), ("vulkan", 0), ("vulkan", 1), ("cpu", 0)])
def test_gui_processes_photo(qtbot, make_photo, isolated_settings, runtime, index):
    folder = data_directory() / "runtimes" / runtime
    folder.mkdir(parents=True)
    (folder / "ready.json").write_text(json.dumps({"runtime": runtime}))
    save_settings({"runtime": runtime, "gpu_index": index, "gpu_name": "Test GPU"})
    source = make_photo("png", alpha=True)
    window = MainWindow(first_start=False)
    qtbot.addWidget(window)
    window.show()
    window.add_paths([source])
    window.model.setCurrentIndex(window.model.findData("drunet"))
    window.amount.setValue(65)
    window.sigma.setValue(20)
    window.start_processing()
    qtbot.waitUntil(
        lambda: window.status.text().startswith(("Finished", "Processing failed")), timeout=60000
    )
    assert window.status.text().startswith("Finished"), window.status.text()
    output = source.with_stem(source.stem + "-denoised")
    assert output.exists()
    before, after = snapshot(source), snapshot(output)
    assert all(after.get(key) == value for key, value in before.items())
    assert window.progress_bar.value() == 1000
    assert window.result.pixmap() is not None
    assert window.open_output_button.isEnabled()


@pytest.mark.integration
def test_gui_cancels_without_publishing_partial_output(qtbot, tmp_path, isolated_settings):
    folder = data_directory() / "runtimes/cuda"
    folder.mkdir(parents=True)
    (folder / "ready.json").write_text("{}")
    save_settings({"runtime": "cuda", "gpu_index": 0, "gpu_name": "RTX 3060"})
    source = tmp_path / "large.png"
    pixels = np.random.default_rng(1).integers(0, 256, (1536, 2048, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(source)
    window = MainWindow(first_start=False)
    qtbot.addWidget(window)
    window.add_paths([source])
    window.tile.setCurrentText("128")
    window.start_processing()
    qtbot.waitUntil(
        lambda: (
            window.process is not None and window.process.state() == QProcess.ProcessState.Running
        ),
        timeout=30000,
    )
    qtbot.waitUntil(
        lambda: window.progress_bar.maximum() == 1000 and window.progress_bar.value() > 0,
        timeout=30000,
    )
    window.cancel_processing()
    qtbot.waitUntil(lambda: window.status.text().startswith("Cancelled"), timeout=30000)
    assert not source.with_stem("large-denoised").exists()
    assert not list(tmp_path.glob(".photo-denoise-*"))


def test_photo_without_runtime_reports_settings(qtbot, isolated_settings, tmp_path):
    window = MainWindow(first_start=False)
    qtbot.addWidget(window)
    window.add_paths([tmp_path / "photo.png"])
    window.start_processing()
    assert "Settings first" in window.status.text()
    assert window.process is None


@pytest.mark.parametrize("extension,uint16", [("png", False), ("tif", True)])
def test_preview_normalizes_and_applies_orientation(
    qtbot, make_photo, isolated_settings, extension, uint16
):
    source = make_photo(extension, uint16=uint16)
    window = MainWindow(first_start=False)
    qtbot.addWidget(window)
    window.show_image(window.original, source)
    pixmap = window.preview_images[window.original]
    assert pixmap.width() == 96
    assert pixmap.height() == 128
    assert not pixmap.isNull()


def test_setup_cancels_cleanly(qtbot, monkeypatch, isolated_settings):
    import time

    from photo_denoise.transfer import Cancelled

    started = []

    def install(*args, cancelled, **kwargs):
        started.append(True)
        while not cancelled():
            time.sleep(0.01)
        raise Cancelled()

    monkeypatch.setattr("photo_denoise.gui.install_runtime", install)
    dialog = RuntimeDialog({})
    qtbot.addWidget(dialog)
    dialog.runtime.setCurrentIndex(dialog.runtime.findData("cpu"))
    dialog.show()
    dialog.begin()
    qtbot.waitUntil(lambda: bool(started))
    dialog.reject()
    qtbot.waitUntil(lambda: not dialog.worker.isRunning())
    qtbot.waitUntil(lambda: not dialog.isVisible())
    assert not (isolated_settings / "settings.json").exists()
