import hashlib
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from photo_denoise.cli import main
from photo_denoise.images import read_photo, write_photo
from photo_denoise.metadata import copy_metadata, snapshot


@pytest.mark.parametrize(
    "extension,uint16,alpha,grayscale",
    [
        ("jpg", False, False, False),
        ("png", False, False, False),
        ("png", False, True, False),
        ("tif", False, False, False),
        ("tif", True, False, False),
        ("tif", True, True, False),
        ("png", False, False, True),
        ("tif", True, False, True),
        ("tif", True, True, True),
    ],
)
def test_metadata_round_trip(make_photo, extension, uint16, alpha, grayscale):
    source = make_photo(extension, uint16=uint16, alpha=alpha, grayscale=grayscale)
    original_hash = hashlib.sha256(source.read_bytes()).digest()
    output = source.with_stem("output")
    assert main(["denoise", str(source), "-o", str(output), "--amount", "0", "--quiet"]) == 0
    assert output.exists()
    assert hashlib.sha256(source.read_bytes()).digest() == original_hash
    before, after = snapshot(source), snapshot(output)
    assert before
    assert all(after.get(key) == value for key, value in before.items())
    initial, processed = read_photo(source), read_photo(output)
    assert processed.rgb.shape == initial.rgb.shape
    assert initial.dtype == processed.dtype
    assert initial.icc == processed.icc
    assert np.array_equal(initial.alpha, processed.alpha)
    if extension != "jpg":
        np.testing.assert_array_equal(initial.rgb, processed.rgb)


def test_changed_metadata_fails(make_photo):
    source = make_photo()
    output = source.with_stem("output")
    photo = read_photo(source)
    write_photo(output, photo, photo.rgb)
    from photo_denoise.metadata import run_exiftool

    copy_metadata(source, output)
    run_exiftool("-overwrite_original", "-Make=Different Camera", str(output))
    assert snapshot(source)["IFD0:Make"] != snapshot(output)["IFD0:Make"]


def test_rejects_multipage_tiff(tmp_path):
    path = tmp_path / "multi.tif"
    tifffile.imwrite(path, np.zeros((2, 16, 16), np.uint8), photometric="minisblack")
    with pytest.raises(ValueError, match="Multipage"):
        read_photo(path)


def test_rejects_16_bit_png(tmp_path):
    path = tmp_path / "high.png"
    Image.fromarray(np.arange(256, dtype=np.uint16).reshape(16, 16)).save(path)
    with pytest.raises(ValueError, match="16-bit PNG"):
        read_photo(path)


def test_rejects_cmyk(tmp_path):
    path = tmp_path / "cmyk.jpg"
    Image.new("CMYK", (16, 16)).save(path)
    with pytest.raises(ValueError, match="CMYK"):
        read_photo(path)


def test_metadata_failure_does_not_publish(make_photo, monkeypatch):
    source = make_photo()
    output = source.with_stem("output")

    def fail(*args):
        raise ValueError("metadata problem")

    monkeypatch.setattr("photo_denoise.cli.copy_metadata", fail)
    assert main(["denoise", str(source), "-o", str(output), "--amount", "0"]) == 1
    assert not output.exists()
    assert not list(source.parent.glob(".photo-denoise-*"))


@pytest.mark.parametrize("camera", ["Canon", "Nikon", "Olympus", "Sony", "Pentax", "Panasonic"])
def test_camera_makernotes_round_trip(tmp_path, camera):
    # The upstream ExifTool distribution includes small, genuine camera metadata fixtures.
    source = Path(__file__).resolve().parents[1] / f".tools/exiftool-13.59/t/images/{camera}.jpg"
    if not source.exists():
        pytest.skip("ExifTool's upstream camera fixture is not installed")
    output = tmp_path / f"{camera}.jpg"
    assert main(["denoise", str(source), "-o", str(output), "--amount", "0", "--quiet"]) == 0
    before, after = snapshot(source), snapshot(output)
    assert len(before) > 20
    assert all(after.get(key) == value for key, value in before.items())


@pytest.mark.parametrize("camera", ["Canon", "Nikon", "Sony"])
def test_tiff_camera_metadata_round_trip(make_photo, camera):
    from photo_denoise.metadata import run_exiftool

    fixture = Path(__file__).resolve().parents[1] / f".tools/exiftool-13.59/t/images/{camera}.jpg"
    if not fixture.exists():
        pytest.skip("ExifTool's upstream camera fixture is not installed")
    source = make_photo("tif", uint16=True)
    run_exiftool(
        "-overwrite_original",
        "-TagsFromFile",
        str(fixture),
        "-all:all",
        "-MakerNotes",
        "-unsafe",
        "-ThumbnailImage",
        "-PreviewImage",
        str(source),
    )
    output = source.with_stem("output")
    assert main(["denoise", str(source), "-o", str(output), "--amount", "0", "--quiet"]) == 0
    assert len(snapshot(output)) > 20
    np.testing.assert_array_equal(read_photo(source).rgb, read_photo(output).rgb)
