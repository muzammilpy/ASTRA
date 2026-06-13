"""
ASTRA – Full test suite

Covers:
  GET  /health
  POST /scan          (file upload)
  POST /scan/camera   (base64 data URI)
  Error handling: bad image, bad city, Gemini/Groq failures, agency failure

All external API calls are mocked — no real keys required.
"""

import base64
import io
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ── Path setup ───────────────────────────────────────────────────────────────
_app_dir = os.path.join(os.path.dirname(__file__), "..", "app")
if os.path.abspath(_app_dir) not in sys.path:
    sys.path.insert(0, os.path.abspath(_app_dir))

from main import app  # noqa: E402

client = TestClient(app)


# ── Image helpers ─────────────────────────────────────────────────────────────

def _make_png_bytes() -> bytes:
    """Create a minimal valid PNG in memory."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(80, 160, 80)).save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


def _make_data_uri(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime};base64,{b64}"


# ── Mock payloads ─────────────────────────────────────────────────────────────

MOCK_GEMINI = {
    "item_name":          "Samsung Galaxy S8 Smartphone",
    "waste_category":     "E-Waste",
    "sub_category":       "Android Smartphone",
    "materials":          ["plastic", "gorilla glass", "circuit board", "lithium battery", "copper"],
    "hazards":            ["battery leakage risk"],
    "visible_labels":     ["Samsung", "SM-G950F"],
    "condition":          "Heavily Damaged",
    "diy_verdict":        "Suitable with Caution",
    "recycle_verdict":    "E-Waste Recycling Required",
    "confidence":         0.92,
    "gemini_description": (
        "The image shows a Samsung Galaxy S8 smartphone with a severely cracked AMOLED screen. "
        "The aluminium frame is intact. Model number SM-G950F is visible on the rear. "
        "Battery may pose a leakage risk. Non-battery components have DIY potential."
    ),
    "gemini_notes": "Samsung Galaxy S8 with cracked screen.",
}

MOCK_GROQ_SAFE = {
    "handling_level":             "DIY with Caution",
    "is_diy_safe":                True,
    "requires_special_recycling": False,
    "safety_reason":              "Outer casing is plastic and safe to handle with gloves.",
    "summary":                    "This smartphone has strong DIY reuse potential.",
    "best_action":                "Convert into DIY product",
    "why":                        "The durable plastic casing can be repurposed many ways.",
    "advisor_co2_saved_kg":       6.0,
    "advisor_landfill_saved_kg":  0.35,
    "recycling_guidance":         "Send battery separately to an e-waste centre.",
    "diy_projects": [
        {
            "title":          f"Project {i}",
            "description":    f"Description for project {i}.",
            "difficulty":     "Beginner",
            "estimated_time": "30-45 minutes",
            "tools":          ["scissors", "glue"],
            "materials":      ["casing", "paint"],
            "steps":          [f"Step {s} of project {i}." for s in range(1, 9)],
            "safety_notes":   ["Wear gloves."],
            "pro_tip":        "Take your time.",
            "co2_saved_kg":   0.5,
            "landfill_saved_kg": 0.1,
        }
        for i in range(1, 7)   # exactly 6 projects
    ],
    "groq_notes": "6 projects generated.",
}

MOCK_GROQ_HAZARDOUS = {
    "handling_level":             "Send to Agency",
    "is_diy_safe":                False,
    "requires_special_recycling": True,
    "safety_reason":              "Contains lithium battery leakage risk.",
    "summary":                    "This item requires professional e-waste recycling.",
    "best_action":                "Recycle at agency",
    "why":                        "Hazardous materials must be handled by certified facilities.",
    "advisor_co2_saved_kg":       6.0,
    "advisor_landfill_saved_kg":  0.35,
    "recycling_guidance":         "Find a WEEE-certified e-waste drop-off in your city.",
    "diy_projects":               [],
    "groq_notes":                 "No DIY — hazardous item.",
}

MOCK_AGENCIES = [
    {
        "name":                 "GreenCycle E-Waste Solutions",
        "address":              "123 Recycling Road, Industrial Zone",
        "phone":                "+1-800-473-3601",
        "website":              "https://greencycle-ewaste.example.com",
        "accepted_waste_types": ["E-Waste", "Batteries", "Mobile Phones"],
        "distance_km":          3.2,
        "contact_instructions": "Call Monday–Friday 8 AM–5 PM.",
        "maps_url":             "https://www.google.com/maps/search/?api=1&query=GreenCycle",
    }
]


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"]  == "ok"
        assert data["service"] == "ASTRA"
        assert "version"   in data
        assert "timestamp" in data

    def test_root_running(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"
        assert "endpoints" in resp.json()


# ── File upload scan ──────────────────────────────────────────────────────────

class TestScanUpload:

    @patch("routers.scan.gemini_service.analyze_image",  new_callable=AsyncMock, return_value=MOCK_GEMINI)
    @patch("routers.scan.groq_service.reason_about_waste", new_callable=AsyncMock, return_value=MOCK_GROQ_SAFE)
    @patch("routers.scan.agency_service.get_agencies",   new_callable=AsyncMock, return_value=MOCK_AGENCIES)
    def test_successful_scan_full_response(self, _ag, _gr, _ge):
        png = _make_png_bytes()
        resp = client.post(
            "/scan",
            data={"city": "Lagos"},
            files={"image": ("photo.png", io.BytesIO(png), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()

        # Top-level shape
        assert data["success"] is True
        assert "scan_id" in data
        assert data["input"]["city"] == "Lagos"

        # Item analysis — new fields
        ia = data["item_analysis"]
        assert ia["item_name"]       == "Samsung Galaxy S8 Smartphone"
        assert ia["waste_category"]  == "E-Waste"
        assert "lithium battery"     in ia["materials"]
        assert "gemini_description"  in ia
        assert len(ia["gemini_description"]) > 10
        assert "diy_verdict"    in ia
        assert "recycle_verdict" in ia
        assert "visible_labels" in ia

        # Safety
        safety = data["safety"]
        assert safety["is_diy_safe"]              is True
        assert safety["requires_special_recycling"] is False
        assert safety["handling_level"]           == "DIY with Caution"

        # Advisor — recycling_guidance field
        advisor = data["advisor"]
        assert advisor["best_action"]        == "Convert into DIY product"
        assert advisor["co2_saved_kg"]       > 0
        assert "recycling_guidance" in advisor

        # 6 DIY projects with full detail
        assert len(data["diy_projects"]) == 6
        proj = data["diy_projects"][0]
        assert "title"       in proj
        assert "steps"       in proj
        assert len(proj["steps"]) >= 8
        assert "pro_tip"     in proj
        assert "impact"      in proj

        # Agencies — with phone, website, maps_url, contact_instructions
        assert len(data["recycling_agencies"]) > 0
        ag = data["recycling_agencies"][0]
        assert ag["phone"]                != ""
        assert ag["website"]              != ""
        assert "maps_url"                 in ag
        assert "contact_instructions"     in ag

        # Meta — new fields
        meta = data["meta"]
        assert meta["processing_time_ms"] >= 0
        assert "image_size_kb" in meta
        assert "gemini_model"  in meta
        assert "groq_model"    in meta

    @patch("routers.scan.gemini_service.analyze_image",  new_callable=AsyncMock, return_value=MOCK_GEMINI)
    @patch("routers.scan.groq_service.reason_about_waste", new_callable=AsyncMock, return_value=MOCK_GROQ_HAZARDOUS)
    @patch("routers.scan.agency_service.get_agencies",   new_callable=AsyncMock, return_value=MOCK_AGENCIES)
    def test_hazardous_item_no_diy_projects(self, _ag, _gr, _ge):
        png = _make_png_bytes()
        resp = client.post(
            "/scan",
            data={"city": "Nairobi"},
            files={"image": ("battery.png", io.BytesIO(png), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety"]["is_diy_safe"]               is False
        assert data["safety"]["requires_special_recycling"] is True
        assert data["advisor"]["best_action"]               == "Recycle at agency"
        assert data["diy_projects"]                         == []

    def test_missing_image_returns_422(self):
        resp = client.post("/scan", data={"city": "London"})
        assert resp.status_code == 422

    def test_missing_city_returns_422(self):
        png = _make_png_bytes()
        resp = client.post(
            "/scan",
            files={"image": ("x.png", io.BytesIO(png), "image/png")},
        )
        assert resp.status_code == 422

    def test_invalid_file_type_returns_400(self):
        resp = client.post(
            "/scan",
            data={"city": "Cairo"},
            files={"image": ("doc.pdf", io.BytesIO(b"%PDF fake"), "application/pdf")},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["code"] == "INVALID_IMAGE"

    def test_empty_city_string_returns_400(self):
        png = _make_png_bytes()
        resp = client.post(
            "/scan",
            data={"city": "   "},
            files={"image": ("x.png", io.BytesIO(png), "image/png")},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["code"] == "INVALID_INPUT"

    def test_invalid_latitude_returns_400(self):
        png = _make_png_bytes()
        resp = client.post(
            "/scan",
            data={"city": "Tokyo", "latitude": "999"},
            files={"image": ("x.png", io.BytesIO(png), "image/png")},
        )
        assert resp.status_code == 400

    @patch("routers.scan.gemini_service.analyze_image",
           new_callable=AsyncMock, side_effect=RuntimeError("Gemini is down"))
    def test_gemini_failure_returns_503(self, _ge):
        png = _make_png_bytes()
        resp = client.post(
            "/scan",
            data={"city": "Accra"},
            files={"image": ("x.png", io.BytesIO(png), "image/png")},
        )
        assert resp.status_code == 503
        assert resp.json()["detail"]["code"] == "GEMINI_ERROR"

    @patch("routers.scan.gemini_service.analyze_image",  new_callable=AsyncMock, return_value=MOCK_GEMINI)
    @patch("routers.scan.groq_service.reason_about_waste",
           new_callable=AsyncMock, side_effect=RuntimeError("Groq is down"))
    def test_groq_failure_returns_503(self, _gr, _ge):
        png = _make_png_bytes()
        resp = client.post(
            "/scan",
            data={"city": "Abuja"},
            files={"image": ("x.png", io.BytesIO(png), "image/png")},
        )
        assert resp.status_code == 503
        assert resp.json()["detail"]["code"] == "GROQ_ERROR"

    @patch("routers.scan.gemini_service.analyze_image",  new_callable=AsyncMock, return_value=MOCK_GEMINI)
    @patch("routers.scan.groq_service.reason_about_waste", new_callable=AsyncMock, return_value=MOCK_GROQ_SAFE)
    @patch("routers.scan.agency_service.get_agencies",
           new_callable=AsyncMock, side_effect=Exception("Agency service offline"))
    def test_agency_failure_degrades_gracefully(self, _ag, _gr, _ge):
        """Agency lookup failure must NOT crash the scan — just return empty list."""
        png = _make_png_bytes()
        resp = client.post(
            "/scan",
            data={"city": "Kigali"},
            files={"image": ("x.png", io.BytesIO(png), "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["recycling_agencies"] == []

    @patch("routers.scan.gemini_service.analyze_image",  new_callable=AsyncMock, return_value=MOCK_GEMINI)
    @patch("routers.scan.groq_service.reason_about_waste", new_callable=AsyncMock, return_value=MOCK_GROQ_SAFE)
    @patch("routers.scan.agency_service.get_agencies",   new_callable=AsyncMock, return_value=MOCK_AGENCIES)
    def test_jpeg_image_accepted(self, _ag, _gr, _ge):
        jpg = _make_jpeg_bytes()
        resp = client.post(
            "/scan",
            data={"city": "Dubai"},
            files={"image": ("photo.jpg", io.BytesIO(jpg), "image/jpeg")},
        )
        assert resp.status_code == 200

    @patch("routers.scan.gemini_service.analyze_image",  new_callable=AsyncMock, return_value=MOCK_GEMINI)
    @patch("routers.scan.groq_service.reason_about_waste", new_callable=AsyncMock, return_value=MOCK_GROQ_SAFE)
    @patch("routers.scan.agency_service.get_agencies",   new_callable=AsyncMock, return_value=MOCK_AGENCIES)
    def test_coordinates_forwarded_to_pipeline(self, _ag, _gr, _ge):
        png = _make_png_bytes()
        resp = client.post(
            "/scan",
            data={"city": "Mumbai", "latitude": "19.076", "longitude": "72.877"},
            files={"image": ("x.png", io.BytesIO(png), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["input"]["latitude"]  == pytest.approx(19.076, rel=1e-3)
        assert data["input"]["longitude"] == pytest.approx(72.877, rel=1e-3)


# ── Camera capture scan ───────────────────────────────────────────────────────

class TestScanCamera:

    @patch("routers.scan.gemini_service.analyze_image",  new_callable=AsyncMock, return_value=MOCK_GEMINI)
    @patch("routers.scan.groq_service.reason_about_waste", new_callable=AsyncMock, return_value=MOCK_GROQ_SAFE)
    @patch("routers.scan.agency_service.get_agencies",   new_callable=AsyncMock, return_value=MOCK_AGENCIES)
    def test_camera_endpoint_success(self, _ag, _gr, _ge):
        jpg      = _make_jpeg_bytes()
        data_uri = _make_data_uri(jpg, "image/jpeg")

        resp = client.post(
            "/scan/camera",
            json={"image_data_uri": data_uri, "city": "Lagos"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"]                is True
        assert data["input"]["city"]          == "Lagos"
        assert len(data["diy_projects"])      == 6
        assert len(data["recycling_agencies"]) > 0

    @patch("routers.scan.gemini_service.analyze_image",  new_callable=AsyncMock, return_value=MOCK_GEMINI)
    @patch("routers.scan.groq_service.reason_about_waste", new_callable=AsyncMock, return_value=MOCK_GROQ_SAFE)
    @patch("routers.scan.agency_service.get_agencies",   new_callable=AsyncMock, return_value=MOCK_AGENCIES)
    def test_camera_png_data_uri(self, _ag, _gr, _ge):
        png      = _make_png_bytes()
        data_uri = _make_data_uri(png, "image/png")

        resp = client.post(
            "/scan/camera",
            json={"image_data_uri": data_uri, "city": "Nairobi"},
        )
        assert resp.status_code == 200

    def test_camera_invalid_data_uri_returns_400(self):
        resp = client.post(
            "/scan/camera",
            json={"image_data_uri": "not-a-valid-data-uri", "city": "Cairo"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["code"] == "INVALID_IMAGE"

    def test_camera_wrong_mime_type_returns_400(self):
        fake_b64 = base64.b64encode(b"fake pdf content").decode()
        resp = client.post(
            "/scan/camera",
            json={
                "image_data_uri": f"data:application/pdf;base64,{fake_b64}",
                "city": "Accra",
            },
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["code"] == "INVALID_IMAGE"

    def test_camera_missing_city_returns_422(self):
        jpg      = _make_jpeg_bytes()
        data_uri = _make_data_uri(jpg)
        resp = client.post(
            "/scan/camera",
            json={"image_data_uri": data_uri},   # city is missing
        )
        assert resp.status_code == 422

    def test_camera_empty_city_returns_400(self):
        jpg      = _make_jpeg_bytes()
        data_uri = _make_data_uri(jpg)
        resp = client.post(
            "/scan/camera",
            json={"image_data_uri": data_uri, "city": "  "},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["code"] == "INVALID_INPUT"

    @patch("routers.scan.gemini_service.analyze_image",
           new_callable=AsyncMock, side_effect=RuntimeError("Gemini API error"))
    def test_camera_gemini_failure_returns_503(self, _ge):
        jpg      = _make_jpeg_bytes()
        data_uri = _make_data_uri(jpg)
        resp = client.post(
            "/scan/camera",
            json={"image_data_uri": data_uri, "city": "London"},
        )
        assert resp.status_code == 503
        assert resp.json()["detail"]["code"] == "GEMINI_ERROR"

    @patch("routers.scan.gemini_service.analyze_image",  new_callable=AsyncMock, return_value=MOCK_GEMINI)
    @patch("routers.scan.groq_service.reason_about_waste", new_callable=AsyncMock, return_value=MOCK_GROQ_SAFE)
    @patch("routers.scan.agency_service.get_agencies",   new_callable=AsyncMock, return_value=MOCK_AGENCIES)
    def test_camera_with_gps_coordinates(self, _ag, _gr, _ge):
        jpg      = _make_jpeg_bytes()
        data_uri = _make_data_uri(jpg)
        resp = client.post(
            "/scan/camera",
            json={
                "image_data_uri": data_uri,
                "city":           "Johannesburg",
                "latitude":       -26.2041,
                "longitude":       28.0473,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["input"]["latitude"]  == pytest.approx(-26.2041, rel=1e-3)
        assert data["input"]["longitude"] == pytest.approx(28.0473,  rel=1e-3)
