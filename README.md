# 🌍 ASTRA – AI Sustainability Platform (Backend)

ASTRA is an AI-powered sustainability advisor that scans waste items through a single image upload and instantly returns:

- **What it is** – item name, waste category, detected materials
- **Whether it's safe** – DIY-safe or requires professional recycling
- **What you can make** – step-by-step DIY reuse projects
- **Where to recycle** – nearby recycling agencies based on your city
- **Environmental impact** – CO₂ and landfill savings for each choice

---

## 🏗️ Project Structure

```
project1/
├── app/
│   ├── main.py                    # FastAPI app + CORS + router registration
│   ├── core/
│   │   ├── config.py              # Pydantic Settings (reads .env)
│   │   └── logging.py             # Structured logging setup
│   ├── routers/
│   │   ├── scan.py                # POST /scan – main endpoint
│   │   └── health.py              # GET /health
│   ├── services/
│   │   ├── gemini_service.py      # Google Gemini Vision API integration
│   │   ├── groq_service.py        # Groq LLM reasoning + DIY generation
│   │   ├── agency_service.py      # Agency lookup (Google Places / mock)
│   │   └── sustainability_service.py  # Environmental impact enrichment
│   ├── models/
│   │   └── schemas.py             # All Pydantic request/response models
│   └── utils/
│       ├── image_utils.py         # Image validation + base64 encoding
│       ├── prompt_templates.py    # Gemini and Groq prompt strings
│       └── validation.py          # City and coordinate validators
├── tests/
│   └── test_scan.py               # Pytest test suite
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd project1

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required keys:
| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `GROQ_API_KEY` | [Groq Console](https://console.groq.com/keys) |
| `GOOGLE_PLACES_API_KEY` | *(optional)* [Google Cloud Console](https://console.cloud.google.com/) |

> **No API keys?** ASTRA automatically falls back to realistic mock data for all services, so you can demo it without any keys.

### 3. Run the server

```bash
# From the project root
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **Docs:** http://localhost:8000/docs
- **Health:** http://localhost:8000/health
- **Scan:** `POST http://localhost:8000/scan`

---

## 📡 API Reference

### `POST /scan`

Analyze a waste item from an uploaded image.

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | ✅ | JPEG, PNG, WEBP, GIF, BMP, TIFF (max 10 MB) |
| `city` | string | ✅ | User's city for agency lookup |
| `latitude` | float | ❌ | GPS latitude (improves agency distance) |
| `longitude` | float | ❌ | GPS longitude |

**Example (curl):**

```bash
curl -X POST http://localhost:8000/scan \
  -F "image=@/path/to/old_phone.jpg" \
  -F "city=Lagos" \
  -F "latitude=6.5244" \
  -F "longitude=3.3792"
```

**Response (200 OK):**

```json
{
  "success": true,
  "scan_id": "uuid",
  "input": { "city": "Lagos", "latitude": 6.5244, "longitude": 3.3792 },
  "item_analysis": {
    "item_name": "Old Smartphone",
    "waste_category": "E-Waste",
    "sub_category": "Mobile Phone",
    "materials": ["plastic", "glass", "circuit board", "battery"],
    "hazards": ["battery leakage"],
    "confidence": 0.87
  },
  "safety": {
    "handling_level": "Send to Agency",
    "is_diy_safe": false,
    "requires_special_recycling": true,
    "reason": "Contains lithium battery requiring certified recycling."
  },
  "advisor": {
    "summary": "...",
    "best_action": "Recycle at agency",
    "why": "...",
    "co2_saved_kg": 6.0,
    "landfill_saved_kg": 0.3
  },
  "diy_projects": [],
  "recycling_agencies": [
    {
      "name": "GreenCycle E-Waste Solutions",
      "address": "123 Recycling Road",
      "phone": "+1-800-GREEN-01",
      "website": "https://greencycle.example.com",
      "accepted_waste_types": ["E-Waste"],
      "distance_km": 3.2
    }
  ],
  "raw_ai": { "gemini_summary": "...", "groq_summary": "..." },
  "meta": { "processing_time_ms": 1843, "timestamp": "2025-01-01T12:00:00+00:00" }
}
```

---

### `GET /health`

```json
{ "status": "ok", "service": "ASTRA", "version": "1.0.0", "timestamp": "..." }
```

---

## 🧪 Running Tests

```bash
# From project root
pytest tests/test_scan.py -v
```

Tests mock all external API calls (Gemini, Groq, Agencies) — no real keys needed.

---

## 🔄 Processing Flow

```
User uploads image + city
         │
         ▼
   [1] Validate image & inputs
         │
         ▼
   [2] Gemini Vision API
       → item name, waste category
       → materials, hazards, confidence
         │
         ▼
   [3] Groq LLM
       → safety classification
       → DIY projects + steps
       → sustainability advisor
       → environmental impact
         │
         ▼
   [4] Agency Lookup
       → Google Places (if configured)
       → Mock data fallback
         │
         ▼
   [5] Impact Enrichment
       → fill zero impact values from baselines
         │
         ▼
   [6] Unified JSON Response
```

---

## ♻️ Waste Categories

| Category | Handling Levels |
|---|---|
| E-Waste | Send to Agency / Requires Specialized Recycling |
| Plastic Waste | DIY Safe / DIY with Caution |
| Metal Waste | DIY Safe / DIY with Caution |
| Glass Waste | DIY with Caution |
| Paper/Cardboard Waste | DIY Safe |
| Organic Waste | DIY Safe |
| Hazardous Waste | Unsafe for Home Handling |
| Mixed Waste | DIY with Caution / Send to Agency |
| Reusable Junk | DIY Safe |
| Unknown / Needs Review | Send to Agency |

---

## 🛡️ Error Codes

| Code | HTTP | Cause |
|---|---|---|
| `INVALID_IMAGE` | 400 | Unsupported file type or empty file |
| `INVALID_INPUT` | 400 | Invalid city, latitude, or longitude |
| `GEMINI_ERROR` | 503 | Gemini API failure or timeout |
| `GROQ_ERROR` | 503 | Groq API failure or timeout |

---

## 🌱 Built With

- **FastAPI** – web framework
- **Google Gemini** – multimodal image understanding
- **Groq (LLaMA 3)** – waste reasoning and DIY generation
- **Pydantic v2** – data validation
- **httpx** – async HTTP client
- **Uvicorn** – ASGI server

---

*ASTRA – Turning waste into wisdom.* 🌍♻️
