import pytest

from photo_denoise.models import download_model


def test_corrupted_checkpoint_is_rejected(tmp_path):
    (tmp_path / "scunet_color_real_psnr.pth").write_bytes(b"partial download")
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_model("scunet", tmp_path, offline=True)


def test_missing_offline_checkpoint_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not cached"):
        download_model("scunet", tmp_path, offline=True)
