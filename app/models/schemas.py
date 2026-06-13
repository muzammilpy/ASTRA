"""
ASTRA – Pydantic v2 schemas for all request / response models
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

class ScanInput(BaseModel):
    city: str = Field(..., description="User's city for agency lookup")
    latitude: Optional[float]  = Field(None, description="Optional GPS latitude")
    longitude: Optional[float] = Field(None, description="Optional GPS longitude")


# ---------------------------------------------------------------------------
# Item Analysis  (Gemini output)
# ---------------------------------------------------------------------------

class ItemAnalysis(BaseModel):
    item_name: str
    waste_category: str = Field(
        ...,
        description=(
            "E-Waste | Plastic Waste | Metal Waste | Glass Waste | "
            "Paper/Cardboard Waste | Organic Waste | Mixed Waste | "
            "Hazardous Waste | Reusable Junk | Unknown / Needs Review"
        ),
    )
    sub_category: str          = Field(default="")
    materials: List[str]       = Field(default_factory=list)
    hazards: List[str]         = Field(default_factory=list)
    visible_labels: List[str]  = Field(default_factory=list)
    condition: str             = Field(default="Unknown")
    diy_verdict: str           = Field(default="")
    recycle_verdict: str       = Field(default="")
    confidence: float          = Field(0.0, ge=0.0, le=1.0)
    # Rich paragraph description written by Gemini — feeds Groq's DIY generation
    gemini_description: str    = Field(default="")


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

class SafetyInfo(BaseModel):
    handling_level: str = Field(
        ...,
        description=(
            "DIY Safe | DIY with Caution | Send to Agency | "
            "Requires Specialized Recycling | Unsafe for Home Handling"
        ),
    )
    is_diy_safe: bool
    requires_special_recycling: bool
    reason: str


# ---------------------------------------------------------------------------
# Sustainability Advisor
# ---------------------------------------------------------------------------

class AdvisorRecommendation(BaseModel):
    summary: str
    best_action: str = Field(
        ...,
        description=(
            "Reuse at home | Convert into DIY product | Repair | "
            "Donate | Recycle at agency | Dispose safely"
        ),
    )
    why: str
    recycling_guidance: str = Field(default="")
    co2_saved_kg: float     = Field(0.0, ge=0.0)
    landfill_saved_kg: float = Field(0.0, ge=0.0)


# ---------------------------------------------------------------------------
# DIY Projects
# ---------------------------------------------------------------------------

class DIYImpact(BaseModel):
    co2_saved_kg: float      = Field(0.0, ge=0.0)
    landfill_saved_kg: float = Field(0.0, ge=0.0)


class DIYProject(BaseModel):
    title: str
    description: str
    difficulty: str        = Field(..., description="Beginner | Intermediate | Advanced")
    estimated_time: str
    tools: List[str]       = Field(default_factory=list)
    materials: List[str]   = Field(default_factory=list)
    steps: List[str]       = Field(default_factory=list)
    safety_notes: List[str] = Field(default_factory=list)
    pro_tip: str           = Field(default="")
    impact: DIYImpact      = Field(default_factory=DIYImpact)


# ---------------------------------------------------------------------------
# Recycling Agencies
# ---------------------------------------------------------------------------

class RecyclingAgency(BaseModel):
    name: str
    address: str
    phone: str                      = Field(default="")
    website: str                    = Field(default="")
    accepted_waste_types: List[str] = Field(default_factory=list)
    distance_km: Optional[float]    = None
    contact_instructions: str       = Field(default="")
    maps_url: str                   = Field(default="")


# ---------------------------------------------------------------------------
# Raw AI output  (debug / transparency)
# ---------------------------------------------------------------------------

class RawAIOutput(BaseModel):
    gemini_description: str = Field(default="")
    groq_notes: str         = Field(default="")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class ScanMeta(BaseModel):
    processing_time_ms: int
    timestamp: str          # ISO-8601
    image_size_kb: float    = Field(default=0.0)
    gemini_model: str       = Field(default="")
    groq_model: str         = Field(default="")


# ---------------------------------------------------------------------------
# Full Scan Response
# ---------------------------------------------------------------------------

class ScanResponse(BaseModel):
    success: bool                            = True
    scan_id: str
    input: ScanInput
    item_analysis: ItemAnalysis
    safety: SafetyInfo
    advisor: AdvisorRecommendation
    diy_projects: List[DIYProject]           = Field(default_factory=list)
    recycling_agencies: List[RecyclingAgency] = Field(default_factory=list)
    raw_ai: RawAIOutput                      = Field(default_factory=RawAIOutput)
    meta: ScanMeta


# ---------------------------------------------------------------------------
# Error Response
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str
