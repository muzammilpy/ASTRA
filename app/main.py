"""
ASTRA – AI-Powered Sustainability Platform
Main FastAPI application entry point

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

# ── Path bootstrap ────────────────────────────────────────────────────────────
# When uvicorn loads `app.main`, Python's sys.path contains the project root
# but NOT the `app/` subdirectory.  All internal imports (core, routers, etc.)
# are written as top-level imports, so we add `app/` to sys.path here once.
import sys
import os

_APP_DIR = os.path.dirname(__file__)   # .../project1/app
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
# ─────────────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logging import setup_logging, get_logger
from routers import scan, health

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle handler."""
    logger.info(f"🌍 ASTRA backend v{settings.APP_VERSION} starting up")
    logger.info(f"   Environment : {settings.ENVIRONMENT}")
    logger.info(f"   Gemini model: {settings.GEMINI_MODEL}")
    logger.info(f"   Groq model  : {settings.GROQ_MODEL}")
    logger.info(f"   Gemini key  : {'✓ configured' if settings.GEMINI_API_KEY else '✗ NOT SET – mock mode'}")
    logger.info(f"   Groq key    : {'✓ configured' if settings.GROQ_API_KEY else '✗ NOT SET – mock mode'}")
    logger.info(f"   Places key  : {'✓ configured' if settings.GOOGLE_PLACES_API_KEY else '— not set (mock agencies)'}")
    yield
    logger.info("ASTRA backend shutting down ✓")


app = FastAPI(
    title="ASTRA – AI Sustainability Platform",
    description=(
        "## 🌍 ASTRA – Turn Waste Into Value\n\n"
        "ASTRA is an AI-powered sustainability advisor that:\n"
        "- **Scans** any waste item from a photo or live camera\n"
        "- **Identifies** the item, materials, hazards, and waste category using Gemini Vision\n"
        "- **Generates 6 detailed DIY reuse projects** with step-by-step instructions using Groq LLM\n"
        "- **Locates nearby recycling agencies** with phone numbers, websites, and directions\n\n"
        "### Endpoints\n"
        "| Endpoint | Method | Purpose |\n"
        "|---|---|---|\n"
        "| `/scan` | POST | File upload scan (JPEG, PNG, WEBP, HEIC…) |\n"
        "| `/scan/camera` | POST | Camera capture scan (base64 data URI) |\n"
        "| `/health` | GET | Service health check |\n"
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(health.router, tags=["Health"])
app.include_router(scan.router,   prefix="/scan", tags=["Scan"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service":  "ASTRA",
        "version":  settings.APP_VERSION,
        "status":   "running",
        "docs":     "/docs",
        "endpoints": {
            "scan_upload": "POST /scan",
            "scan_camera": "POST /scan/camera",
            "health":      "GET  /health",
        },
    }
