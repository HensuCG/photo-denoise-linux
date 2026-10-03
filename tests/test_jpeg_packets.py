"""Synthetic extended-XMP and auxiliary JPEGs, containing no private photos."""

import hashlib
import struct
import subprocess
from io import BytesIO

import pytest
from PIL import Image

from photo_denoise.jpeg_metadata import segments, xmp_segments
from photo_denoise.metadata import copy_metadata, exiftool_command, snapshot


def jpeg(color="gray", progressive=False):
    stream = BytesIO()
    Image.new("RGB", (48, 32), color).save(stream, format="JPEG", progressive=progressive)
    return stream.getvalue()


def app(marker, payload):
    return bytes((255, marker)) + struct.pack(">H", len(payload) + 2) + payload


def packet(properties):
    return (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Original camera toolkit">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:GCamera="http://ns.google.com/photos/1.0/camera/" '
        'xmlns:xmpNote="http://ns.adobe.com/xmp/note/">'
        + properties
        + "</rdf:Description></rdf:RDF></x:xmpmeta>"
    ).encode()


@pytest.mark.parametrize("progressive", [False, True])
@pytest.mark.parametrize("extended", [False, True])
@pytest.mark.parametrize("order", ["<", ">"])
def test_jpeg_xmp_and_auxiliary_images_round_trip(tmp_path, progressive, extended, order):
    unknown = (
        "<GCamera:GFileMetadata>"
        + "camera-private-schema" * (4000 if extended else 1)
        + "</GCamera:GFileMetadata>"
    )
    extension = packet(unknown)
    guid = hashlib.md5(extension).hexdigest().upper().encode()
    main = (
        packet("<xmpNote:HasExtendedXMP>" + guid.decode() + "</xmpNote:HasExtendedXMP>")
        if extended
        else extension
    )
    xmp = app(0xE1, b"http://ns.adobe.com/xap/1.0/\0" + main)
    if extended:
        for start in range(0, len(extension), 60000):
            xmp += app(
                0xE1,
                b"http://ns.adobe.com/xmp/extension/\0"
                + guid
                + struct.pack(">II", len(extension), start)
                + extension[start : start + 60000],
            )
    primary = jpeg(progressive=progressive)
    auxiliary = jpeg("red")
    # Two-entry MPF index, with both byte orders covered.
    tiff = bytearray((b"II" if order == "<" else b"MM") + struct.pack(order + "HIH", 42, 8, 3))
    tiff += struct.pack(order + "HHI4s", 0xB000, 7, 4, b"0100")
    tiff += struct.pack(order + "HHII", 0xB001, 4, 1, 2)
    tiff += struct.pack(order + "HHII", 0xB002, 7, 32, 50)
    tiff += struct.pack(order + "I", 0)
    primary_length = len(primary) + len(xmp) + 4 + 4 + 82
    mpf_base = 2 + len(xmp) + 8
    tiff += struct.pack(order + "IIIHH", 0, primary_length, 0, 0, 0)
    tiff += struct.pack(order + "IIIHH", 0, len(auxiliary), primary_length - mpf_base, 0, 0)
    source = tmp_path / "camera.jpg"
    original = primary[:2] + xmp + app(0xE2, b"MPF\0" + tiff) + primary[2:] + auxiliary
    source.write_bytes(original)
    output = tmp_path / "output.jpg"
    output.write_bytes(jpeg("blue", progressive=progressive))
    copy_metadata(source, output)
    data = output.read_bytes()
    assert source.read_bytes() == original
    assert xmp_segments(data) == xmp_segments(original)
    before, after = snapshot(source), snapshot(output)
    assert before["XMP-x:XMPToolkit"] == "Original camera toolkit"
    assert "XMP-GCamera:GFileMetadata" in before
    assert all(after.get(key) == value for key, value in before.items())
    _, end = segments(data)
    assert data[end:] == auxiliary
    extracted = subprocess.run(
        exiftool_command() + ["-b", "-MPImage2", str(output)], capture_output=True, check=True
    ).stdout
    assert extracted == auxiliary


def test_truncated_scan_rejected():
    with pytest.raises(ValueError, match="end marker"):
        segments(jpeg()[:-2])


@pytest.mark.parametrize("extension", ["png", "tif"])
def test_unknown_xmp_preserved_in_other_formats(tmp_path, extension):
    from photo_denoise.metadata import run_exiftool

    source = tmp_path / ("source." + extension)
    output = tmp_path / ("output." + extension)
    Image.new("RGB", (48, 32), "gray").save(source)
    Image.new("RGB", (48, 32), "blue").save(output)
    xml = tmp_path / "metadata.xmp"
    xml.write_bytes(packet("<GCamera:GFileMetadata>unknown schema data</GCamera:GFileMetadata>"))
    run_exiftool("-overwrite_original", "-XMP<=" + str(xml), str(source))
    before = snapshot(source)
    copy_metadata(source, output)
    after = snapshot(output)
    assert before["XMP-GCamera:GFileMetadata"] == "unknown schema data"
    assert all(after.get(key) == value for key, value in before.items())
