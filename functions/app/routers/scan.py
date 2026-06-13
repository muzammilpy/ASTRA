"""
ASTRA – /scan router

Endpoints:
  POST /scan          – file upload  (multipart form, works with <input type="file">)
  POST /scan/camera   – base64 data URI  (sent from browser camera capture / webcam)

Both endpoints go through the same pipeline:
  image bytes → Gemini Vision → Groq LLM → Agency Lookup → unified JSON response
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile, status

from core.config import settings
from core.logging import get_logger
from models.schemas import (
    AdvisorRecommendation,
    DIYImpact,
    DIYProject,
    ErrorDetail,
    ErrorResponse,
    ItemAnalysis,
    RawAIOutput,
    RecyclingAgency,
    SafetyInfo,
    ScanInput,
    ScanMeta,
    ScanResponse,
)
from services import agency_service, gemini_service, groq_service
from services.sustainability_service import enrich_with_impact
from utils.image_utils import (
    decode_data_uri,
    get_image_size_kb,
    validate_and_read_image,
)
from utils.validation import validate_city, validate_coordinates

router = APIRouter()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------

async def _run_pipeline(
    image_bytes: bytes,
    mime_type: str,
    city: str,
    latitude: Optional[float],
    longitude: Optional[float],
) -> ScanResponse:
    """
    Core ASTRA pipeline.
    Called by both /scan (file upload) and /scan/camera (base64 data URI).

    Steps:
      1  Gemini Vision   – deep image analysis + rich description
      2  Groq LLM        – 6 DIY projects + safety + advisor
      3  Impact enrichment – fill any zero CO₂/landfill values from baselines
      4  Agency lookup   – phone, website, maps URL, contact instructions
      5  Assemble response
    """
    scan_id = str(uuid.uuid4())
    t_start = time.perf_counter()
    size_kb = get_image_size_kb(image_bytes)

    logger.info(
        f"[{scan_id}] Pipeline start | city='{city}' | "
        f"mime={mime_type} | size={size_kb} KB"
    )

    # ── Step 1: Gemini Vision ────────────────────────────────────────────────
    try:
        gemini_data = await gemini_service.analyze_image(image_bytes, mime_type)
    except RuntimeError as exc:
        logger.error(f"[{scan_id}] Gemini error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                code="GEMINI_ERROR",
                message=str(exc),
            ).model_dump(),
        )

    logger.info(
        f"[{scan_id}] Gemini done | "
        f"item='{gemini_data['item_name']}' | "
        f"category='{gemini_data['waste_category']}' | "
        f"diy_verdict='{gemini_data['diy_verdict']}'"
    )

    # ── Step 2: Groq LLM ────────────────────────────────────────────────────
    try:
        groq_data = await groq_service.reason_about_waste(gemini_data)
    except RuntimeError as exc:
        logger.error(f"[{scan_id}] Groq error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                code="GROQ_ERROR",
                message=str(exc),
            ).model_dump(),
        )

    logger.info(
        f"[{scan_id}] Groq done | "
        f"best_action='{groq_data['best_action']}' | "
        f"diy_projects={len(groq_data['diy_projects'])}"
    )

    # ── Step 3: Impact enrichment ────────────────────────────────────────────
    groq_data = enrich_with_impact(groq_data, gemini_data["waste_category"])

    # ── Step 4: Agency lookup ────────────────────────────────────────────────
    try:
        raw_agencies = await agency_service.get_agencies(
            waste_category=gemini_data["waste_category"],
            materials=gemini_data["materials"],
            city=city,
            lat=latitude,
            lon=longitude,
        )
    except Exception as exc:
        logger.warning(f"[{scan_id}] Agency lookup failed: {exc} – continuing without agencies")
        raw_agencies = []

    # ── Step 5: Assemble response ────────────────────────────────────────────
    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    timestamp  = datetime.now(tz=timezone.utc).isoformat()

    diy_projects = [
        DIYProject(
            title          = p["title"],
            description    = p["description"],
            difficulty     = p["difficulty"],
            estimated_time = p["estimated_time"],
            tools          = p["tools"],
            materials      = p["materials"],
            steps          = p["steps"],
            safety_notes   = p["safety_notes"],
            pro_tip        = p.get("pro_tip", ""),
            impact=DIYImpact(
                co2_saved_kg      = p["co2_saved_kg"],
                landfill_saved_kg = p["landfill_saved_kg"],
            ),
        )
        for p in groq_data["diy_projects"]
    ]

    agencies = [
        RecyclingAgency(
            name                 = a["name"],
            address              = a["address"],
            phone                = a.get("phone", ""),
            website              = a.get("website", ""),
            accepted_waste_types = a.get("accepted_waste_types", []),
            distance_km          = a.get("distance_km"),
            contact_instructions = a.get("contact_instructions", ""),
            maps_url             = a.get("maps_url", ""),
        )
        for a in raw_agencies
    ]

    response = ScanResponse(
        scan_id = scan_id,
        input   = ScanInput(city=city, latitude=latitude, longitude=longitude),
        item_analysis = ItemAnalysis(
            item_name          = gemini_data["item_name"],
            waste_category     = gemini_data["waste_category"],
            sub_category       = gemini_data["sub_category"],
            materials          = gemini_data["materials"],
            hazards            = gemini_data["hazards"],
            visible_labels     = gemini_data["visible_labels"],
            condition          = gemini_data["condition"],
            diy_verdict        = gemini_data["diy_verdict"],
            recycle_verdict    = gemini_data["recycle_verdict"],
            confidence         = gemini_data["confidence"],
            gemini_description = gemini_data["gemini_description"],
        ),
        safety = SafetyInfo(
            handling_level           = groq_data["handling_level"],
            is_diy_safe              = groq_data["is_diy_safe"],
            requires_special_recycling = groq_data["requires_special_recycling"],
            reason                   = groq_data["safety_reason"],
        ),
        advisor = AdvisorRecommendation(
            summary            = groq_data["summary"],
            best_action        = groq_data["best_action"],
            why                = groq_data["why"],
            recycling_guidance = groq_data.get("recycling_guidance", ""),
            co2_saved_kg       = groq_data["advisor_co2_saved_kg"],
            landfill_saved_kg  = groq_data["advisor_landfill_saved_kg"],
        ),
        diy_projects       = diy_projects,
        recycling_agencies = agencies,
        raw_ai = RawAIOutput(
            gemini_description = gemini_data.get("gemini_description", ""),
            groq_notes         = groq_data.get("groq_notes", ""),
        ),
        meta = ScanMeta(
            processing_time_ms = elapsed_ms,
            timestamp          = timestamp,
            image_size_kb      = size_kb,
            gemini_model       = settings.GEMINI_MODEL,
            groq_model         = settings.GROQ_MODEL,
        ),
    )

    logger.info(
        f"[{scan_id}] ✓ Scan complete | {elapsed_ms} ms | "
        f"{gemini_data['waste_category']} → {groq_data['best_action']} | "
        f"{len(diy_projects)} DIY projects | {len(agencies)} agencies"
    )
    return response


# ---------------------------------------------------------------------------
# Input validation helper
# ---------------------------------------------------------------------------

def _validate_inputs(city: str, latitude: Optional[float], longitude: Optional[float]) -> str:
    """Validate city and coordinates. Returns normalised city string."""
    from utils.validation import validate_city, validate_coordinates
    try:
        city = validate_city(city)
        validate_coordinates(latitude, longitude)
        return city
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(code="INVALID_INPUT", message=str(exc)).model_dump(),
        )


# ---------------------------------------------------------------------------
# POST /scan   — file upload (standard multipart)
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ScanResponse,
    summary="Scan a waste item (file upload)",
    description=(
        "Upload an image file of a waste or junk item together with your city. "
        "Supports JPEG, PNG, WEBP, HEIC, GIF, BMP, TIFF up to 15 MB. "
        "ASTRA will deeply analyse the image with Gemini Vision, generate 6 detailed "
        "DIY reuse projects with Groq, and locate nearby recycling agencies."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input or unsupported image"},
        422: {"model": ErrorResponse, "description": "Missing required fields"},
        503: {"model": ErrorResponse, "description": "Gemini or Groq API error"},
    },
)
async def scan_from_upload(
    image: UploadFile = File(
        ...,
        description="Waste item image — JPEG, PNG, WEBP, HEIC, GIF, BMP, TIFF (max 15 MB)",
    ),
    city: str = Form(
        ...,
        description="Your city name for recycling agency lookup",
        examples=["Lagos", "London", "Mumbai", "Nairobi"],
    ),
    latitude: Optional[float]  = Form(None, description="GPS latitude  (improves distance accuracy)"),
    longitude: Optional[float] = Form(None, description="GPS longitude (improves distance accuracy)"),
) -> ScanResponse:

    city = _validate_inputs(city, latitude, longitude)

    try:
        image_bytes, mime_type = await validate_and_read_image(image)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(code="INVALID_IMAGE", message=str(exc)).model_dump(),
        )

    return await _run_pipeline(image_bytes, mime_type, city, latitude, longitude)


# ---------------------------------------------------------------------------
# POST /scan/camera   — base64 data URI from browser camera
# ---------------------------------------------------------------------------

@router.post(
    "/camera",
    response_model=ScanResponse,
    summary="Scan a waste item (camera capture)",
    description=(
        "Send a base64-encoded image captured directly from a device camera or webcam. "
        "The frontend should encode the camera frame as a data URI string "
        "(e.g. `data:image/jpeg;base64,...`) and send it in the JSON body. "
        "Same analysis pipeline as the file-upload endpoint."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid base64 data URI or city"},
        503: {"model": ErrorResponse, "description": "Gemini or Groq API error"},
    },
)
async def scan_from_camera(
    image_data_uri: str = Body(
        ...,
        embed=True,
        description="Base64 data URI of the camera image — format: data:image/jpeg;base64,<data>",
        examples=["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."],
    ),
    city: str = Body(
        ...,
        embed=True,
        description="Your city name for recycling agency lookup",
        examples=["Lagos", "London", "Mumbai"],
    ),
    latitude: Optional[float]  = Body(None, embed=True, description="GPS latitude"),
    longitude: Optional[float] = Body(None, embed=True, description="GPS longitude"),
) -> ScanResponse:

    city = _validate_inputs(city, latitude, longitude)

    try:
        image_bytes, mime_type = decode_data_uri(image_data_uri)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDetail(code="INVALID_IMAGE", message=str(exc)).model_dump(),
        )

    return await _run_pipeline(image_bytes, mime_type, city, latitude, longitude)
