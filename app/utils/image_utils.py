"""
ASTRA – Image validation and encoding utilities

Supports three image sources:
  1. File upload  (multipart/form-data UploadFile)
  2. Camera capture sent as base64 data URI  (e.g. data:image/jpeg;base64,/9j/...)
  3. Raw bytes passed directly from any source
"""

import base64
import re
from typing import Tuple

from fastapi import UploadFile

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
    "image/heic",
    "image/heif",
}

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024   # 15 MB  (camera photos can be large)

# Regex to parse a data URI:  data:<mime>;base64,<data>
_DATA_URI_RE = re.compile(
    r"^data:(image/[a-zA-Z0-9.+\-]+);base64,(.+)$",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# File upload path
# ---------------------------------------------------------------------------

async def validate_and_read_image(file: UploadFile) -> Tuple[bytes, str]:
    """
    Read and validate a multipart-uploaded image file.

    Returns:
        (raw_bytes, mime_type)

    Raises:
        ValueError: on unsupported type, empty file, or oversized file
    """
    content_type = (file.content_type or "").lower().split(";")[0].strip()

    if content_type not in ALLOWED_MIME_TYPES:
        raise ValueError(
            f"Unsupported file type '{content_type}'. "
            f"Accepted formats: JPEG, PNG, WEBP, GIF, BMP, TIFF, HEIC."
        )

    raw = await file.read()

    if len(raw) == 0:
        raise ValueError("The uploaded file is empty.")

    if len(raw) > MAX_FILE_SIZE_BYTES:
        mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        raise ValueError(
            f"File is too large ({len(raw) // (1024*1024)} MB). "
            f"Maximum allowed size is {mb} MB."
        )

    return raw, content_type


# ---------------------------------------------------------------------------
# Camera / base64 data URI path
# ---------------------------------------------------------------------------

def decode_data_uri(data_uri: str) -> Tuple[bytes, str]:
    """
    Decode a base64 data URI string (as sent by a browser camera capture).

    Example input:
        "data:image/jpeg;base64,/9j/4AAQSkZJRgAB..."

    Returns:
        (raw_bytes, mime_type)

    Raises:
        ValueError: on invalid format or unsupported MIME type
    """
    data_uri = data_uri.strip()
    match = _DATA_URI_RE.match(data_uri)
    if not match:
        raise ValueError(
            "Invalid image data URI format. "
            "Expected: data:<mime_type>;base64,<base64_data>"
        )

    mime_type = match.group(1).lower()
    b64_data  = match.group(2)

    if mime_type not in ALLOWED_MIME_TYPES:
        raise ValueError(
            f"Unsupported image type '{mime_type}' in data URI. "
            f"Accepted: JPEG, PNG, WEBP, GIF, BMP, TIFF."
        )

    try:
        raw = base64.b64decode(b64_data, validate=True)
    except Exception:
        raise ValueError(
            "The base64 image data is malformed and could not be decoded. "
            "Ensure the camera capture is encoded correctly."
        )

    if len(raw) == 0:
        raise ValueError("Decoded image data is empty.")

    if len(raw) > MAX_FILE_SIZE_BYTES:
        mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        raise ValueError(f"Camera image exceeds {mb} MB limit after decoding.")

    return raw, mime_type


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def encode_image_base64(image_bytes: bytes) -> str:
    """Return a base64-encoded string of raw image bytes (for Gemini API)."""
    return base64.b64encode(image_bytes).decode("utf-8")


def get_image_size_kb(image_bytes: bytes) -> float:
    """Return image size in kilobytes, rounded to 1 decimal place."""
    return round(len(image_bytes) / 1024, 1)
