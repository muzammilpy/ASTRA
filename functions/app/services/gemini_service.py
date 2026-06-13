"""
ASTRA – Gemini Vision Service

Accepts images from any source:
  - Direct camera capture (base64 data URI from frontend)
  - File upload (multipart form)
  - URL reference (optional future use)

Sends the image to Google Gemini and returns a deep, structured waste analysis
including a rich narrative description that powers Groq's DIY generation.
"""

import json
import re
import time
from typing import Any, Dict

import httpx

from core.config import settings
from core.logging import get_logger
from utils.image_utils import encode_image_base64
from utils.prompt_templates import GEMINI_ANALYSIS_PROMPT, GEMINI_SYSTEM_INSTRUCTION

logger = get_logger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


# ---------------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------------

def _build_request_body(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """
    Construct the Gemini generateContent payload.
    Uses inline_data (base64) which works for both file uploads and camera captures.
    Increases maxOutputTokens to 2048 to allow rich gemini_description field.
    """
    b64 = encode_image_base64(image_bytes)
    return {
        "system_instruction": {
            "parts": [{"text": GEMINI_SYSTEM_INSTRUCTION}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64,
                        }
                    },
                    {"text": GEMINI_ANALYSIS_PROMPT},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.15,        # Low temperature = more accurate, less hallucination
            "maxOutputTokens": 2048,    # Enough for rich gemini_description
            "responseMimeType": "application/json",
        },
    }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from Gemini's text response.
    Handles markdown code fences, leading/trailing whitespace, and partial wrapping.
    """
    # Remove markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.strip().rstrip("`").strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting just the JSON object (find first { ... } block)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise json.JSONDecodeError("No valid JSON object found in Gemini response", cleaned, 0)


def _normalize_gemini_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize Gemini's raw JSON into a guaranteed-structure dict.
    Provides safe defaults for every field so downstream services never KeyError.
    """
    return {
        "item_name":          raw.get("item_name", "Unknown Item"),
        "waste_category":     raw.get("waste_category", "Unknown / Needs Review"),
        "sub_category":       raw.get("sub_category", ""),
        "materials":          raw.get("materials") or [],
        "hazards":            raw.get("hazards") or [],
        "visible_labels":     raw.get("visible_labels") or [],
        "condition":          raw.get("condition", "Unknown"),
        "diy_verdict":        raw.get("diy_verdict", "Suitable with Caution"),
        "recycle_verdict":    raw.get("recycle_verdict", "Standard Recycling"),
        "confidence":         float(raw.get("confidence", 0.5)),
        "gemini_description": raw.get("gemini_description", raw.get("gemini_notes", "")),
        # Keep gemini_notes as alias for backward compat
        "gemini_notes":       raw.get("gemini_description", raw.get("gemini_notes", "")),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def analyze_image(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """
    Send image bytes to Gemini Vision and return normalized structured analysis.

    Works with images from:
    - Camera capture (bytes passed from multipart upload or base64 decode)
    - File upload (standard multipart)
    - Any image source converted to bytes

    Args:
        image_bytes : raw bytes of the image
        mime_type   : MIME type string (e.g. 'image/jpeg', 'image/png')

    Returns:
        Normalized analysis dict with rich gemini_description field

    Raises:
        RuntimeError: on API failure, timeout, or unparseable response
    """
    if not settings.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set — returning mock analysis for development")
        return _mock_analysis()

    url = (
        f"{GEMINI_API_BASE}/models/{settings.GEMINI_MODEL}"
        f":generateContent?key={settings.GEMINI_API_KEY}"
    )
    body = _build_request_body(image_bytes, mime_type)

    logger.info(
        f"Sending {len(image_bytes) / 1024:.1f} KB image to Gemini "
        f"(model: {settings.GEMINI_MODEL}, type: {mime_type})"
    )
    t0 = time.perf_counter()

    async with httpx.AsyncClient(timeout=settings.GEMINI_TIMEOUT) as client:
        try:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Gemini API timed out after {settings.GEMINI_TIMEOUT}s. "
                "Try a smaller image or increase GEMINI_TIMEOUT."
            )
        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text[:600]
            raise RuntimeError(
                f"Gemini API returned HTTP {exc.response.status_code}. "
                f"Details: {error_body}"
            )
        except httpx.RequestError as exc:
            raise RuntimeError(f"Network error contacting Gemini: {exc}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Gemini responded in {elapsed_ms:.0f} ms")

    data = resp.json()

    # ── Parse Gemini response structure ──────────────────────────────────────
    try:
        candidates = data.get("candidates", [])
        if not candidates:
            # Check for prompt blocks
            block_reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
            raise ValueError(
                f"Gemini returned no candidates. Block reason: {block_reason}. "
                "This may mean the image was flagged. Try a different image."
            )

        finish_reason = candidates[0].get("finishReason", "STOP")
        if finish_reason not in ("STOP", "MAX_TOKENS"):
            raise ValueError(
                f"Gemini generation stopped unexpectedly: finishReason={finish_reason}"
            )

        content_parts = candidates[0]["content"]["parts"]
        # Response may be in text part or direct JSON part
        raw_text = content_parts[0].get("text", "")
        raw_json = _extract_json_from_text(raw_text)

    except (KeyError, IndexError) as exc:
        logger.error(f"Unexpected Gemini response structure: {exc}\nRaw response: {data}")
        raise RuntimeError(
            f"Gemini response structure was unexpected: {exc}. "
            "This may be a temporary API issue."
        )
    except json.JSONDecodeError as exc:
        logger.error(f"JSON parse failed from Gemini output: {exc}")
        raise RuntimeError(
            "Gemini returned text that could not be parsed as JSON. "
            "This is a temporary model behaviour issue — please retry."
        )

    normalized = _normalize_gemini_output(raw_json)

    logger.info(
        f"✓ Gemini analysis complete | item='{normalized['item_name']}' | "
        f"category='{normalized['waste_category']}' | "
        f"diy_verdict='{normalized['diy_verdict']}' | "
        f"confidence={normalized['confidence']:.0%} | "
        f"description_length={len(normalized['gemini_description'])} chars"
    )

    return normalized


# ---------------------------------------------------------------------------
# Development mock (used when GEMINI_API_KEY is absent)
# ---------------------------------------------------------------------------

def _mock_analysis() -> Dict[str, Any]:
    """Realistic mock that exercises the full downstream pipeline."""
    return {
        "item_name": "Samsung Galaxy S8 Smartphone",
        "waste_category": "E-Waste",
        "sub_category": "Android Smartphone",
        "materials": ["plastic", "gorilla glass", "circuit board", "lithium battery",
                      "copper", "aluminium frame", "rubber seals"],
        "hazards": ["battery leakage risk", "electronic contamination"],
        "visible_labels": ["Samsung", "Galaxy S8", "SM-G950F", "CE mark", "WEEE symbol"],
        "condition": "Heavily Damaged",
        "diy_verdict": "Suitable with Caution",
        "recycle_verdict": "E-Waste Recycling Required",
        "confidence": 0.92,
        "gemini_description": (
            "The image shows a Samsung Galaxy S8 smartphone with severe screen damage — "
            "the AMOLED display is shattered across approximately 70% of its surface with "
            "visible spider-web crack patterns originating from the top-right corner. "
            "The aluminium frame is largely intact with minor scuffs along the bottom edge. "
            "The back glass panel is cracked but still in place, and the Samsung branding "
            "and model number SM-G950F are clearly legible. The USB-C port appears unobstructed. "
            "Given the cracked screen and battery damage risk, the device should not be used "
            "as a phone, but the outer housing, frame, and non-battery components offer "
            "significant creative DIY potential for tech art, organisers, and display projects."
        ),
        "gemini_notes": (
            "Samsung Galaxy S8 with severe screen damage. Battery intact but cracked screen "
            "makes standard use impossible. E-waste recycling recommended for battery and board."
        ),
    }
