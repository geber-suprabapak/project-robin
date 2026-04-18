"""Utilities for downloading runtime model assets."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

_CHECKSUM_ALGORITHMS = {
    64: "sha256",
    96: "sha384",
    128: "sha512",
}


class ModelDownloadError(Exception):
    """Raised when a model asset cannot be downloaded or verified."""


def _parse_checksum(checksum: str) -> tuple[str, str] | None:
    value = checksum.strip()
    if not value:
        return None

    if ":" in value:
        algorithm, expected = value.split(":", 1)
        return algorithm.lower(), expected.lower()

    algorithm = _CHECKSUM_ALGORITHMS.get(len(value))
    if not algorithm:
        raise ModelDownloadError(
            "Checksum must include an algorithm prefix or use sha256/sha384/sha512 hex length"
        )
    return algorithm, value.lower()


def _file_hexdigest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: str | Path, checksum: str) -> None:
    """Verify a file checksum when a checksum value is configured."""
    parsed = _parse_checksum(checksum)
    if parsed is None:
        return

    algorithm, expected = parsed
    actual = _file_hexdigest(Path(path), algorithm)
    if actual != expected:
        raise ModelDownloadError(
            f"Checksum mismatch for {path}: expected {algorithm}:{expected}, got {algorithm}:{actual}"
        )


def ensure_model_asset(
    path: str | Path,
    *,
    url: str,
    checksum: str = "",
    auto_download: bool = True,
    description: str = "model asset",
) -> Path:
    """Return an asset path, downloading it first when enabled and missing."""
    target = Path(path)
    if target.is_file():
        verify_checksum(target, checksum)
        return target

    if not auto_download:
        raise ModelDownloadError(f"{description} is missing: {target}")
    if not url:
        raise ModelDownloadError(f"{description} is missing and no download URL is configured: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_name(f".{target.name}.download")

    logger.info("Downloading %s from %s to %s", description, url, target)
    try:
        urlretrieve(url, temp_target)
        verify_checksum(temp_target, checksum)
        os.replace(temp_target, target)
    except (OSError, URLError, ValueError) as error:
        temp_target.unlink(missing_ok=True)
        raise ModelDownloadError(f"Failed to download {description} from {url}: {error}") from error
    except ModelDownloadError:
        temp_target.unlink(missing_ok=True)
        raise

    logger.info("Downloaded %s to %s", description, target)
    return target
