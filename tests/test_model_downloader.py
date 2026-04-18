import hashlib

import pytest

from src.services.model_downloader import ModelDownloadError, ensure_model_asset, verify_checksum


def test_verify_checksum_accepts_algorithm_prefix(tmp_path):
    target = tmp_path / "asset.bin"
    target.write_bytes(b"model")

    checksum = hashlib.sha256(b"model").hexdigest()

    verify_checksum(target, f"sha256:{checksum}")


def test_verify_checksum_rejects_mismatch(tmp_path):
    target = tmp_path / "asset.bin"
    target.write_bytes(b"model")

    with pytest.raises(ModelDownloadError, match="Checksum mismatch"):
        verify_checksum(target, f"sha256:{hashlib.sha256(b'other').hexdigest()}")


def test_ensure_model_asset_downloads_missing_file(tmp_path):
    source = tmp_path / "source.bin"
    target = tmp_path / "nested" / "target.bin"
    source.write_bytes(b"model")
    checksum = hashlib.sha256(b"model").hexdigest()

    result = ensure_model_asset(
        target,
        url=source.as_uri(),
        checksum=f"sha256:{checksum}",
        description="test model",
    )

    assert result == target
    assert target.read_bytes() == b"model"


def test_ensure_model_asset_can_disable_download(tmp_path):
    target = tmp_path / "missing.bin"

    with pytest.raises(ModelDownloadError, match="missing"):
        ensure_model_asset(target, url="", auto_download=False, description="test model")
