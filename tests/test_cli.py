import hashlib
import os

import pytest
from PIL import Image

from photo_denoise.cli import main


def test_refuses_input_overwrite(make_photo):
    source = make_photo()
    digest = hashlib.sha256(source.read_bytes()).digest()
    assert main(["denoise", str(source), "-o", str(source), "--overwrite", "--amount", "0"]) == 1
    assert hashlib.sha256(source.read_bytes()).digest() == digest


def test_refuses_hard_link(make_photo):
    source = make_photo()
    linked = source.with_stem("linked")
    os.link(source, linked)
    assert main(["denoise", str(source), "-o", str(linked), "--overwrite", "--amount", "0"]) == 1


def test_refuses_symlink(make_photo):
    source = make_photo()
    linked = source.with_stem("linked")
    linked.symlink_to(source)
    assert main(["denoise", str(source), "-o", str(linked), "--overwrite", "--amount", "0"]) == 1


def test_existing_output(make_photo):
    source = make_photo()
    output = source.with_stem("out")
    output.write_bytes(b"existing data")
    assert main(["denoise", str(source), "-o", str(output), "--amount", "0"]) == 1
    assert output.read_bytes() == b"existing data"
    assert main(["denoise", str(source), "-o", str(output), "--amount", "0", "--overwrite"]) == 0


def test_nested_batch(tmp_path):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    (inputs / "nested").mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(inputs / "one.png")
    Image.new("RGB", (16, 16)).save(inputs / "nested" / "two.png")
    Image.new("RGB", (16, 16)).save(inputs / "skip-denoised.png")
    assert (
        main(["denoise", str(inputs), "--output-dir", str(outputs), "--recursive", "--amount", "0"])
        == 0
    )
    assert (outputs / "one-denoised.png").exists()
    assert (outputs / "nested" / "two-denoised.png").exists()
    assert not (outputs / "skip-denoised-denoised.png").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--sigma", "20"],
        ["--tile-size", "32"],
        ["--overlap", "256"],
        ["--amount", "nan"],
        ["--suffix", "../escape"],
    ],
)
def test_invalid_options_fail(make_photo, arguments):
    source = make_photo()
    with pytest.raises(SystemExit) as error:
        main(["denoise", str(source), *arguments])
    assert error.value.code == 2


def test_conversion_is_rejected(make_photo):
    source = make_photo()
    assert (
        main(["denoise", str(source), "-o", str(source.with_suffix(".jpg")), "--amount", "0"]) == 1
    )


def test_batch_continues_after_bad_file(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"not an image")
    Image.new("RGB", (16, 16)).save(tmp_path / "good.png")
    assert main(["denoise", str(tmp_path), "--amount", "0"]) == 1
    assert (tmp_path / "good-denoised.png").exists()
    assert not (tmp_path / "bad-denoised.png").exists()


def test_output_directory_can_equal_input_directory(tmp_path):
    Image.new("RGB", (16, 16)).save(tmp_path / "one.png")
    assert main(["denoise", str(tmp_path), "--output-dir", str(tmp_path), "--amount", "0"]) == 0
    assert (tmp_path / "one-denoised.png").exists()
