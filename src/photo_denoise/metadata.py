"""Copy photo metadata with ExifTool and check it before publishing output."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .images import family
from .jpeg_metadata import preserve_jpeg_packets

# These describe the encoded file's layout, which changes when pixels are saved.
# Camera information, GPS, orientation, capture times, profiles, and previews
# remain subject to verification.
STORAGE_TAGS = {
    "StripOffsets",
    "StripByteCounts",
    "TileOffsets",
    "TileByteCounts",
    "RowsPerStrip",
    "Compression",
    "Predictor",
    "BitsPerSample",
    "SampleFormat",
    "PhotometricInterpretation",
    "SamplesPerPixel",
    "PlanarConfiguration",
    "ExtraSamples",
    "YCbCrSubSampling",
    "YCbCrPositioning",
    "ReferenceBlackWhite",
    "JPEGProc",
    "ThumbnailOffset",
    "ThumbnailLength",
    "PreviewImageStart",
    "PreviewImageLength",
    "JpgFromRawStart",
    "JpgFromRawLength",
    "OtherImageStart",
    "OtherImageLength",
    "ExifOffset",
    "GPSInfo",
    "InteropOffset",
    "SubIFD",
    "ImageOffset",
    "ImageByteCount",
}


def exiftool_command() -> list[str]:
    override = os.environ.get("PHOTO_DENOISE_EXIFTOOL")
    candidates = [
        override,
        shutil.which("exiftool"),
        str(Path(__file__).resolve().parents[2] / ".tools/exiftool-13.59/exiftool"),
        str(Path.home() / ".local/share/photo-denoise/exiftool-13.59/exiftool"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ["perl", str(Path(candidate).resolve())]
    raise ValueError(
        "ExifTool is required. Run ./scripts/install.sh or install the exiftool package."
    )


def run_exiftool(*args: str) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        exiftool_command() + list(args), capture_output=True, text=True, timeout=120, check=False
    )
    if completed.returncode:
        raise ValueError(f"ExifTool failed: {completed.stderr.strip() or completed.stdout.strip()}")
    return completed


def snapshot(path: Path) -> dict:
    data = json.loads(
        run_exiftool(
            "-json",
            "-G1",
            "-s",
            "-n",
            "-EXIF:all",
            "-MakerNotes:all",
            "-XMP:all",
            "-IPTC:all",
            "-ICC_Profile:all",
            str(path),
        ).stdout
    )[0]
    return {
        key: value
        for key, value in data.items()
        if key != "SourceFile" and key.rsplit(":", 1)[-1] not in STORAGE_TAGS
    }


def copy_metadata(source: Path, destination: Path) -> None:
    # JPEG and PNG carry a separable EXIF block. Copy it intact to retain unknown
    # tags, exact rational values, interoperability fields and embedded previews.
    # In TIFF, EXIF shares the image's container, so copy metadata tags instead.
    exif = (
        ["-EXIF"]
        if family(source) in ("jpeg", "png")
        else ["-unsafe", "-ThumbnailImage", "-PreviewImage"]
    )
    run_exiftool(
        "-overwrite_original",
        "-TagsFromFile",
        str(source),
        "-all:all",
        "-MakerNotes",
        "-ICC_Profile",
        "-XMP",
        *exif,
        str(destination),
    )
    if family(source) == "jpeg":
        preserve_jpeg_packets(source, destination)
    before, after = snapshot(source), snapshot(destination)
    changed = [key for key, value in before.items() if after.get(key) != value]
    if changed:
        raise ValueError(
            "Metadata verification failed for "
            + ", ".join(changed[:12])
            + "; no output was published"
        )
    # Compare the actual profile bytes in addition to decoded profile fields.
    command = exiftool_command()
    blocks = ["ICC_Profile"]
    if family(source) in ("jpeg", "png"):
        blocks.append("EXIF")
    for block in blocks:
        values = [
            subprocess.run(
                command + ["-b", f"-{block}", str(path)],
                capture_output=True,
                check=True,
                timeout=120,
            ).stdout
            for path in (source, destination)
        ]
        if values[0] != values[1]:
            raise ValueError(f"{block} block verification failed; no output was published")
