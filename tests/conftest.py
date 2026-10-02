from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from photo_denoise.metadata import run_exiftool


@pytest.fixture
def rgb():
    y, x = np.mgrid[:96, :128]
    return np.stack((40 + x, 60 + y, 80 + (x + y) // 2), axis=2).astype(np.uint8)


@pytest.fixture
def profile():
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


@pytest.fixture
def make_photo(tmp_path, rgb, profile):
    def make(extension="png", *, uint16=False, alpha=False, grayscale=False, name="photo") -> Path:
        path = tmp_path / f"{name}.{extension}"
        pixels = rgb[:, :, 0] if grayscale else rgb.copy()
        if uint16:
            pixels = pixels.astype(np.uint16) * 257 + 7
        if alpha:
            pixels = np.dstack((pixels, np.full(pixels.shape[:2], 123, dtype=pixels.dtype)))
        if extension in ("tif", "tiff"):
            tifffile.imwrite(
                path,
                pixels,
                photometric="minisblack" if grayscale else "rgb",
                extrasamples=["unassalpha"] if alpha else None,
                metadata=None,
                extratags=[(34675, "B", len(profile), profile, False)] if not grayscale else [],
            )
        else:
            Image.fromarray(pixels).save(path, icc_profile=profile if not grayscale else None)
        run_exiftool(
            "-overwrite_original",
            "-Make=Test Camera",
            "-Model=Metadata Fixture",
            "-EXIF:DateTimeOriginal=2021:07:08 09:10:11",
            "-EXIF:ISO=1600",
            "-EXIF:ExposureTime=1/125",
            "-EXIF:FNumber=2.8",
            "-EXIF:LensModel=Test Lens",
            "-EXIF:Orientation#=6",
            "-EXIF:GPSLatitude=59.437",
            "-EXIF:GPSLatitudeRef=N",
            "-EXIF:GPSLongitude=24.7536",
            "-EXIF:GPSLongitudeRef=E",
            "-XMP-dc:Title=Preserve this title",
            "-XMP-dc:Description=Unicode: Tallinn – test",
            str(path),
        )
        if extension in ("jpg", "jpeg", "tif", "tiff"):
            run_exiftool(
                "-overwrite_original",
                "-IPTC:Keywords=metadata-test",
                "-IPTC:CopyrightNotice=Metadata preservation fixture",
                str(path),
            )
        return path

    return make
