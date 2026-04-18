"""Shared helpers for validating base64 payloads without full decoding."""

import re

BASE64_RE = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")


def estimate_base64_decoded_size(value: str) -> int:
    """Return the decoded byte size implied by a compact base64 string."""
    padding = len(value) - len(value.rstrip("="))
    return (len(value) * 3 // 4) - padding
