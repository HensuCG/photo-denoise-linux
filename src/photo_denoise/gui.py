"""Desktop frontend; inference runs in a separate, selected runtime process."""

from __future__ import annotations

import json
import sys
import threading
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .hardware import detect_gpus, recommended_runtime
from .images import read_photo
from .models import MODELS, cache_directory, download_model
from .runtime_setup import (
    install_runtime,
    manifest,
    probe_runtime,
    remove_runtime,
    runtime_environment,
    runtime_installed,
)
from .settings import load_settings, save_settings
from .transfer import Cancelled

LABELS = {
    "vulkan": "Vulkan / ncnn — AMD, NVIDIA and Intel GPUs",
    "cuda": "CUDA / PyTorch — recommended for NVIDIA; largest installation",
    "cpu": "CPU — no supported GPU required; usually much slower",
}


def readable_bytes(count):
    return f"{count / 10**9:.2f} GB" if count >= 10**9 else f"{count / 10**6:.1f} MB"


class SetupWorker(QThread):
    progress = Signal(str, int, int)
    succeeded = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, runtime, model, *, install=True, parent=None):
        super().__init__(parent)
        self.runtime, self.model, self.install = runtime, model, install
        self.stop = threading.Event()

    def report(self, label, done, total):
        # Qt's integer signal is 32-bit, so report percentages for multi-GB downloads.
        percent = round(done * 1000 / total) if total else 0
        self.progress.emit(
            label + (f" — {readable_bytes(done)} / {readable_bytes(total)}" if total else ""),
            percent,
            1000 if total else 0,
        )

    def run(self):
        try:
            result = (
                install_runtime(self.runtime, progress=self.report, cancelled=self.stop.is_set)
                if self.install
                else probe_runtime(self.runtime)
            )
            if self.stop.is_set():
                raise Cancelled()
            self.report("Preparing selected model", 0, 0)
            if self.runtime == "vulkan":
                # Import ncnn only inside the selected runtime process. Model downloading
                # itself needs no ncnn library in the GUI environment.
                from .vulkan_models import ensure_model

                ensure_model(
                    self.model,
                    cache_directory(),
                    progress=lambda a, b: self.report("Downloading model", a, b),
                    cancelled=self.stop.is_set,
                )
            else:
                download_model(
                    self.model,
                    cache_directory(),
                    progress=lambda a, b: self.report("Downloading model", a, b),
                    cancelled=self.stop.is_set,
                )
            self.succeeded.emit(result)
        except Cancelled:
            self.cancelled.emit()
        except Exception as error:
            self.failed.emit(str(error))


class RuntimeDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Runtime settings")
        self.setMinimumWidth(650)
        self.settings = dict(settings)
        self.worker = None
        self.gpus = detect_gpus()
        layout = QVBoxLayout(self)
        title = QLabel("Choose how photos are processed")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(title)
        detected = ", ".join(gpu["name"] for gpu in self.gpus) or "No compatible GPU detected"
        devices = QLabel("Detected: " + detected)
        devices.setWordWrap(True)
        layout.addWidget(devices)
        self.runtime = QComboBox()
        self.runtime.setObjectName("runtimeChoice")
        for key, label in LABELS.items():
            self.runtime.addItem(label, key)
        selected = settings.get("runtime", recommended_runtime(self.gpus))
        self.runtime.setCurrentIndex(self.runtime.findData(selected))
        self.gpu = QComboBox()
        self.gpu.setObjectName("gpuChoice")
        form = QFormLayout()
        form.addRow("Runtime", self.runtime)
        form.addRow("Device", self.gpu)
        layout.addLayout(form)
        self.details = QLabel()
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.status = QLabel("Dependencies are downloaded once. Photos stay on this computer.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        layout.addWidget(self.bar)
        buttons = QHBoxLayout()
        self.remove = QPushButton("Remove selected runtime")
        self.remove.clicked.connect(self.remove_selected)
        buttons.addWidget(self.remove)
        buttons.addStretch()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.cancel_button)
        self.install_button = QPushButton("Install / use runtime")
        self.install_button.setObjectName("installRuntime")
        self.install_button.clicked.connect(self.begin)
        buttons.addWidget(self.install_button)
        layout.addLayout(buttons)
        self.runtime.currentIndexChanged.connect(self.refresh)
        self.refresh()
        wanted = settings.get("gpu_name")
        for index in range(self.gpu.count()):
            if wanted and self.gpu.itemText(index) == wanted:
                self.gpu.setCurrentIndex(index)

    def refresh(self):
        key = self.runtime.currentData()
        self.gpu.clear()
        if key == "cpu":
            self.gpu.addItem("CPU", 0)
        else:
            candidates = [gpu for gpu in self.gpus if key != "cuda" or gpu["vendor"] == "NVIDIA"]
            for index, gpu in enumerate(candidates):
                # CUDA has its own numbering; Vulkan's actual numbering is confirmed
                # after installation by the runtime probe.
                self.gpu.addItem(gpu["name"], index if key == "cuda" else gpu["index"])
        supported = "DRUNet" if key == "vulkan" else "SCUNet, SCUNet GAN and DRUNet"
        size = readable_bytes(manifest()[key]["download_bytes"])
        state = "Installed" if runtime_installed(key) else "Not installed"
        self.details.setText(
            f"{state}. Runtime download: {size}, plus the selected model. Models: {supported}.\n"
            "Vulkan has been tested on RX 6900 XT and RTX 3060; Intel support is not tested. "
            "Runtime speed depends on the card and model."
        )
        self.remove.setEnabled(runtime_installed(key))
        self.install_button.setEnabled(self.gpu.count() > 0)

    def begin(self):
        key = self.runtime.currentData()
        self.chosen_name = self.gpu.currentText()
        self.chosen_index = self.gpu.currentData() or 0
        self.runtime.setEnabled(False)
        self.gpu.setEnabled(False)
        self.install_button.setEnabled(False)
        self.remove.setEnabled(False)
        self.worker = SetupWorker(key, "drunet" if key == "vulkan" else "scunet", parent=self)
        self.worker.progress.connect(self.update_progress)
        self.worker.succeeded.connect(self.success)
        self.worker.failed.connect(self.failure)
        self.worker.cancelled.connect(self.was_cancelled)
        self.worker.start()

    def update_progress(self, label, done, total):
        self.status.setText(label)
        self.bar.setRange(0, total)
        self.bar.setValue(done)

    def success(self, result):
        key = self.runtime.currentData()
        if key != "cpu":
            matching = next(
                (gpu for gpu in result["gpus"] if gpu["name"] == self.chosen_name), None
            )
            if not matching:
                self.failure(
                    "Selected GPU was not found by the runtime. Select a detected device and retry."
                )
                return
            self.chosen_index = matching["index"]
        self.settings.update(runtime=key, gpu_index=self.chosen_index, gpu_name=self.chosen_name)
        save_settings(self.settings)
        self.worker.finished.connect(self.accept)
        if not self.worker.isRunning():
            self.accept()

    def failure(self, message):
        self.status.setText(
            "Setup failed: " + message + "\nCompleted downloads are retained; retry when ready."
        )
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.runtime.setEnabled(True)
        self.gpu.setEnabled(True)
        self.install_button.setText("Retry setup")
        self.install_button.setEnabled(True)
        self.cancel_button.setEnabled(True)

    def was_cancelled(self):
        self.worker.finished.connect(lambda: super(RuntimeDialog, self).reject())
        if not self.worker.isRunning():
            super().reject()

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop.set()
            self.status.setText("Cancelling setup…")
            self.cancel_button.setEnabled(False)
        else:
            super().reject()

    def remove_selected(self):
        key = self.runtime.currentData()
        if (
            QMessageBox.question(
                self,
                "Remove runtime",
                "Remove this runtime's installed libraries? Model weights and photos are kept.",
            )
            == QMessageBox.StandardButton.Yes
        ):
            try:
                remove_runtime(key)
            except (OSError, ValueError) as error:
                self.status.setText("Unable to remove runtime: " + str(error))
                return
            if self.settings.get("runtime") == key:
                self.settings = {}
                save_settings({})
            self.refresh()


class MainWindow(QMainWindow):
    def __init__(self, *, first_start=True):
        super().__init__()
        self.setWindowTitle("Photo Denoise")
        self.resize(1120, 780)
        self.settings = load_settings()
        self.process = None
        self.setup_worker = None
        self.stdout_buffer = ""
        self.errors = []
        self.outputs = {}
        self.preview_images = {}
        self.cancel_requested = False
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(24, 20, 24, 20)
        heading = QHBoxLayout()
        title = QLabel("Photo Denoise")
        title.setStyleSheet("font-size: 26px; font-weight: 600;")
        heading.addWidget(title)
        heading.addStretch()
        self.runtime_label = QLabel()
        heading.addWidget(self.runtime_label)
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings)
        heading.addWidget(self.settings_button)
        layout.addLayout(heading)
        layout.addWidget(
            QLabel("Denoise locally. Preserve EXIF and color profiles. Keep your originals.")
        )
        split = QSplitter(Qt.Orientation.Horizontal)
        queue_widget = QWidget()
        queue_layout = QVBoxLayout(queue_widget)
        row = QHBoxLayout()
        self.add_button = QPushButton("Add photos")
        self.add_button.clicked.connect(self.add_photos)
        self.folder_button = QPushButton("Add folder")
        self.folder_button.clicked.connect(self.add_folder)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_queue)
        for button in (self.add_button, self.folder_button, self.clear_button):
            row.addWidget(button)
        queue_layout.addLayout(row)
        self.queue = QListWidget()
        self.queue.setObjectName("photoQueue")
        self.queue.currentTextChanged.connect(self.preview_selection)
        queue_layout.addWidget(self.queue)
        self.recursive = QCheckBox("Include subfolders")
        self.recursive.setChecked(True)
        queue_layout.addWidget(self.recursive)
        split.addWidget(queue_widget)
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        previews = QHBoxLayout()
        self.original = QLabel("Select a photo to preview")
        self.result = QLabel("Denoised photo appears here")
        for label, name in ((self.original, "Original"), (self.result, "Result")):
            column = QVBoxLayout()
            column.addWidget(QLabel(name))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(180, 220)
            label.setStyleSheet("background: #20262e; color: #c5ccd6; border-radius: 8px;")
            column.addWidget(label, 1)
            previews.addLayout(column)
        preview_layout.addLayout(previews)
        self.open_output_button = QPushButton("Open output folder")
        self.open_output_button.clicked.connect(self.open_output)
        self.open_output_button.setEnabled(False)
        preview_layout.addWidget(self.open_output_button)
        split.addWidget(preview_widget)
        split.setSizes([360, 740])
        layout.addWidget(split, 1)
        form = QFormLayout()
        self.model = QComboBox()
        self.model.setObjectName("modelChoice")
        self.model.currentIndexChanged.connect(self.model_changed)
        form.addRow("Model", self.model)
        amount_row = QHBoxLayout()
        self.amount = QSlider(Qt.Orientation.Horizontal)
        self.amount.setRange(0, 100)
        self.amount.setValue(80)
        self.amount_label = QLabel("80%")
        self.amount.valueChanged.connect(lambda value: self.amount_label.setText(f"{value}%"))
        amount_row.addWidget(self.amount)
        amount_row.addWidget(self.amount_label)
        form.addRow("Denoising amount", amount_row)
        controls = QHBoxLayout()
        self.sigma = QSpinBox()
        self.sigma.setRange(0, 100)
        self.sigma.setValue(15)
        self.sigma.setToolTip(
            "DRUNet noise standard deviation on a 0–255 scale; not ISO or a percentage"
        )
        controls.addWidget(QLabel("DRUNet noise level"))
        controls.addWidget(self.sigma)
        controls.addSpacing(20)
        self.tile = QComboBox()
        self.tile.addItems(["512", "256", "128"])
        controls.addWidget(QLabel("Tile size"))
        controls.addWidget(self.tile)
        controls.addStretch()
        form.addRow("Processing", controls)
        output_row = QHBoxLayout()
        self.output = QLineEdit()
        self.output.setPlaceholderText("Next to originals, with -denoised suffix")
        self.output_button = QPushButton("Choose folder")
        self.output_button.clicked.connect(self.choose_output)
        output_row.addWidget(self.output)
        output_row.addWidget(self.output_button)
        form.addRow("Output", output_row)
        layout.addLayout(form)
        self.overwrite = QCheckBox("Replace existing output files (originals are always protected)")
        layout.addWidget(self.overwrite)
        self.status = QLabel("Add photos or a folder to start.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        actions = QHBoxLayout()
        actions.addStretch()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_processing)
        actions.addWidget(self.cancel_button)
        self.start_button = QPushButton("Denoise photos")
        self.start_button.setObjectName("startDenoising")
        self.start_button.clicked.connect(self.start_processing)
        actions.addWidget(self.start_button)
        layout.addLayout(actions)
        self.refresh_runtime()
        if first_start and not self.settings:
            QTimer.singleShot(0, self.open_settings)

    def refresh_runtime(self):
        key = self.settings.get("runtime")
        self.runtime_label.setText(
            f"{key.upper()} · {self.settings.get('gpu_name', '')}" if key else "Choose a runtime"
        )
        selected = self.model.currentData()
        self.model.clear()
        for name, (_, description) in MODELS.items():
            if key == "vulkan" and name != "drunet":
                continue
            self.model.addItem(f"{name} — {description.replace('--sigma', 'noise level')}", name)
        index = self.model.findData(selected)
        if index >= 0:
            self.model.setCurrentIndex(index)
        self.start_button.setEnabled(bool(key) and runtime_installed(key))
        self.model_changed()

    def model_changed(self):
        self.sigma.setEnabled(self.model.currentData() == "drunet")

    def open_settings(self):
        dialog = RuntimeDialog(self.settings, self)
        dialog.exec()
        self.settings = load_settings()
        self.refresh_runtime()

    def add_paths(self, paths):
        existing = {self.queue.item(index).text() for index in range(self.queue.count())}
        for path in paths:
            path = str(Path(path).expanduser().resolve())
            if path not in existing:
                self.queue.addItem(path)
                existing.add(path)
        if self.queue.currentRow() < 0 and self.queue.count():
            self.queue.setCurrentRow(0)

    def add_photos(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add photos", "", "Photos (*.jpg *.jpeg *.png *.tif *.tiff)"
        )
        self.add_paths(paths)

    def add_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Add photo folder")
        if path:
            self.add_paths([path])

    def clear_queue(self):
        self.queue.clear()
        self.original.clear()
        self.result.clear()
        self.preview_images.clear()

    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder", self.output.text())
        if folder:
            self.output.setText(folder)

    def show_image(self, label, path):
        try:
            photo = read_photo(Path(path))
            pixels = np.rint(photo.rgb * 255).astype(np.uint8)
            image = Image.fromarray(pixels[:, :, 0] if photo.grayscale else pixels)
            if photo.icc:
                try:
                    image = ImageCms.profileToProfile(
                        image,
                        ImageCms.ImageCmsProfile(BytesIO(photo.icc)),
                        ImageCms.createProfile("sRGB"),
                        outputMode="RGB",
                    )
                except (ImageCms.PyCMSError, OSError, ValueError):
                    image = image.convert("RGB")
            else:
                image = image.convert("RGB")
            if photo.alpha is not None:
                alpha = Image.fromarray(
                    np.rint(
                        photo.alpha.astype(np.float32) / np.iinfo(photo.dtype).max * 255
                    ).astype(np.uint8)
                )
                background = Image.new("RGB", image.size, "#e6e6e6")
                background.paste(image, mask=alpha)
                image = background
            with Image.open(path) as source:
                image.getexif()[274] = source.getexif().get(274, 1)
            image = ImageOps.exif_transpose(image)
            image.thumbnail((1600, 1600))
            pixels = np.ascontiguousarray(np.asarray(image))
            picture = QImage(
                pixels.data,
                pixels.shape[1],
                pixels.shape[0],
                pixels.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()
            pixmap = QPixmap.fromImage(picture)
            self.preview_images[label] = pixmap
            label.setPixmap(
                pixmap.scaled(
                    label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        except Exception as error:
            label.setText("Preview unavailable: " + str(error))

    def preview_selection(self, path):
        if path and Path(path).is_file():
            self.show_image(self.original, path)
            if path in self.outputs:
                self.show_image(self.result, self.outputs[path])
            else:
                self.result.setText("Denoised photo appears here")
                self.preview_images.pop(self.result, None)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for label, pixmap in self.preview_images.items():
            label.setPixmap(
                pixmap.scaled(
                    label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def set_busy(self, busy):
        for item in (
            self.start_button,
            self.settings_button,
            self.add_button,
            self.folder_button,
            self.clear_button,
            self.model,
            self.amount,
            self.tile,
            self.output,
            self.output_button,
            self.overwrite,
            self.recursive,
        ):
            item.setEnabled(not busy)
        self.sigma.setEnabled(not busy and self.model.currentData() == "drunet")
        self.cancel_button.setEnabled(busy)

    def start_processing(self):
        if not self.queue.count():
            self.status.setText("Add at least one photo or folder.")
            return
        runtime = self.settings.get("runtime")
        if not runtime or not runtime_installed(runtime):
            self.status.setText("Choose and install a runtime in Settings first.")
            return
        self.cancel_requested = False
        self.errors = []
        self.set_busy(True)
        self.setup_worker = SetupWorker(
            self.settings["runtime"], self.model.currentData(), install=False, parent=self
        )
        self.setup_worker.progress.connect(self.update_progress)
        self.setup_worker.succeeded.connect(lambda _: self.start_process())
        self.setup_worker.failed.connect(self.setup_failed)
        self.setup_worker.cancelled.connect(lambda: self.finish_processing(130))
        self.setup_worker.start()

    def update_progress(self, label, done, total):
        self.status.setText(label)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)

    def setup_failed(self, message):
        self.errors.append(message)
        self.finish_processing(1)

    def start_process(self):
        if self.cancel_requested:
            self.finish_processing(130)
            return
        key = self.settings["runtime"]
        arguments = ["-m", "photo_denoise.cli", "denoise"]
        arguments += [self.queue.item(index).text() for index in range(self.queue.count())]
        arguments += [
            "--runtime",
            key,
            "--gpu-index",
            str(self.settings.get("gpu_index", 0)),
            "--model",
            self.model.currentData(),
            "--amount",
            str(self.amount.value() / 100),
            "--tile-size",
            self.tile.currentText(),
            "--json-events",
            "--offline",
        ]
        if self.model.currentData() == "drunet":
            arguments += ["--sigma", str(self.sigma.value())]
        if self.output.text().strip():
            arguments += ["--output-dir", self.output.text().strip()]
        if self.recursive.isChecked():
            arguments.append("--recursive")
        if self.overwrite.isChecked():
            arguments.append("--overwrite")
        self.process = QProcess(self)
        environment = QProcessEnvironment()
        for name, value in runtime_environment(key).items():
            environment.insert(name, value)
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self.read_events)
        self.process.readyReadStandardError.connect(self.read_errors)
        self.process.finished.connect(lambda code, _: self.finish_processing(code))
        self.process.errorOccurred.connect(lambda _: self.setup_failed(self.process.errorString()))
        self.stdout_buffer = ""
        self.progress_bar.setRange(0, 0)
        self.status.setText("Loading model…")
        self.process.start(sys.executable, arguments)

    def read_events(self):
        self.stdout_buffer += bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        while "\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split("\n", 1)
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("event")
            if kind == "file":
                self.status.setText(
                    f"[{event['index']}/{event['total']}] {Path(event['source']).name}"
                )
                self.show_image(self.original, event["source"])
            elif kind == "progress":
                self.progress_bar.setRange(0, 1000)
                value = ((event["index"] - 1) + event["done"] / event["total"]) / event["files"]
                self.progress_bar.setValue(round(value * 1000))
            elif kind == "saved":
                self.outputs[event["source"]] = event["output"]
                self.show_image(self.result, event["output"])
                self.last_output = event["output"]
                self.open_output_button.setEnabled(True)
            elif kind == "error":
                self.errors.append(event["message"])

    def read_errors(self):
        text = bytes(self.process.readAllStandardError()).decode(errors="replace").strip()
        if text:
            self.last_stderr = text

    def finish_processing(self, code):
        self.set_busy(False)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(1000 if code == 0 else 0)
        if self.cancel_requested or code == 130:
            self.status.setText("Cancelled. Already completed outputs are kept.")
        elif code:
            self.status.setText(
                "Processing failed: "
                + (
                    self.errors[-1]
                    if self.errors
                    else getattr(self, "last_stderr", "Unknown error")
                )
            )
        else:
            self.status.setText(
                "Finished. Outputs saved with metadata verified; originals are unchanged."
            )

    def cancel_processing(self):
        self.cancel_requested = True
        self.cancel_button.setEnabled(False)
        self.status.setText("Cancelling…")
        if self.setup_worker and self.setup_worker.isRunning():
            self.setup_worker.stop.set()
        elif self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()

    def open_output(self):
        if getattr(self, "last_output", None):
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.last_output).parent)))

    def closeEvent(self, event: QCloseEvent):
        busy = (self.setup_worker and self.setup_worker.isRunning()) or (
            self.process and self.process.state() != QProcess.ProcessState.NotRunning
        )
        if busy:
            self.cancel_processing()
            self.status.setText("Cancelling active work. Close the window again after it stops.")
            event.ignore()
        else:
            event.accept()


def main(paths=None):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Photo Denoise")
    app.setOrganizationName("HensuCG")
    window = MainWindow()
    if paths:
        window.add_paths(paths)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
