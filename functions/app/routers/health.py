"""
ASTRA – Health check router
GET /health
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from core.config import settings
from models.schemas import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service status, name, version, and current UTC timestamp.",
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )
