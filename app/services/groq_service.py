"""
ASTRA – Groq LLM Reasoning Service

Takes Gemini's deep visual analysis and produces:
  - Safety classification with full reasoning
  - Sustainability advisory with environmental impact
  - EXACTLY 6 detailed DIY projects with 8+ steps each
  - Beginner-friendly language throughout
  - Recycling guidance with agency contact recommendations
"""

import json
import re
import time
from typing import Any, Dict, List

import httpx

from core.config import settings
from core.logging import get_logger
from utils.prompt_templates import build_groq_prompt

logger = get_logger(__name__)

GROQ_API_BASE = "https://api.groq.com/openai/v1"


# ---------------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------------

def _build_request_body(prompt: str) -> Dict[str, Any]:
    return {
        "model": settings.GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are ASTRA, an expert sustainability advisor and DIY coach. "
                    "You respond ONLY with valid, complete JSON. "
                    "You never include markdown fences, code blocks, or any text outside the JSON. "
                    "You always generate EXACTLY 6 DIY projects unless the item is hazardous."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,       # Slight creativity for DIY variety
        "max_tokens": 6000,       # Enough for 6 detailed projects
        "top_p": 0.9,
    }


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Robustly extract JSON from Groq's output.
    Handles markdown fences, stray text prefix/suffix.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.strip().rstrip("`").strip()

    # Direct parse attempt
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find outermost JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found in Groq response", cleaned, 0)


def _parse_diy_projects(raw_projects: List[Dict]) -> List[Dict[str, Any]]:
    """
    Normalise each DIY project into a guaranteed structure.
    Ensures steps list is never empty and fields always exist.
    """
    projects = []
    for p in (raw_projects or []):
        steps = p.get("steps") or []
        # Ensure minimum 3 steps even if model was stingy
        if len(steps) < 3:
            steps += [f"Continue with step {i+1}." for i in range(3 - len(steps))]

        projects.append({
            "title":             p.get("title", "Untitled Project"),
            "description":       p.get("description", ""),
            "difficulty":        p.get("difficulty", "Beginner"),
            "estimated_time":    p.get("estimated_time", "30-60 minutes"),
            "tools":             p.get("tools") or [],
            "materials":         p.get("materials") or [],
            "steps":             steps,
            "safety_notes":      p.get("safety_notes") or [],
            "pro_tip":           p.get("pro_tip", ""),
            "co2_saved_kg":      float(p.get("co2_saved_kg", 0.0)),
            "landfill_saved_kg": float(p.get("landfill_saved_kg", 0.0)),
        })
    return projects


def _normalize_groq_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return a fully guaranteed-structure dict from Groq's raw JSON."""
    return {
        "handling_level":           raw.get("handling_level", "DIY with Caution"),
        "is_diy_safe":              bool(raw.get("is_diy_safe", False)),
        "requires_special_recycling": bool(raw.get("requires_special_recycling", False)),
        "safety_reason":            raw.get("safety_reason", ""),
        "summary":                  raw.get("summary", ""),
        "best_action":              raw.get("best_action", "Recycle at agency"),
        "why":                      raw.get("why", ""),
        "advisor_co2_saved_kg":     float(raw.get("advisor_co2_saved_kg", 0.0)),
        "advisor_landfill_saved_kg": float(raw.get("advisor_landfill_saved_kg", 0.0)),
        "recycling_guidance":       raw.get("recycling_guidance", ""),
        "diy_projects":             _parse_diy_projects(raw.get("diy_projects", [])),
        "groq_notes":               raw.get("groq_notes", ""),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def reason_about_waste(gemini_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send Gemini's deep analysis to Groq and return a full sustainability advisory
    including 6 detailed DIY projects (or recycling guidance for hazardous items).

    Args:
        gemini_data : normalized dict from gemini_service.analyze_image()

    Returns:
        Normalized advisory dict

    Raises:
        RuntimeError: on API failure, timeout, or JSON parse error
    """
    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — returning rich mock advisory for development")
        return _mock_advisory(gemini_data)

    prompt = build_groq_prompt(gemini_data)
    body   = _build_request_body(prompt)

    logger.info(
        f"Sending Gemini analysis to Groq (model: {settings.GROQ_MODEL}) | "
        f"item='{gemini_data.get('item_name')}'"
    )
    t0 = time.perf_counter()

    async with httpx.AsyncClient(timeout=settings.GROQ_TIMEOUT) as client:
        try:
            resp = await client.post(
                f"{GROQ_API_BASE}/chat/completions",
                json=body,
                headers={
                    "Authorization":  f"Bearer {settings.GROQ_API_KEY}",
                    "Content-Type":   "application/json",
                },
            )
            resp.raise_for_status()
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Groq API timed out after {settings.GROQ_TIMEOUT}s. "
                "Try increasing GROQ_TIMEOUT or using a smaller model."
            )
        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text[:600]
            raise RuntimeError(
                f"Groq API returned HTTP {exc.response.status_code}. "
                f"Details: {error_body}"
            )
        except httpx.RequestError as exc:
            raise RuntimeError(f"Network error contacting Groq: {exc}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Groq responded in {elapsed_ms:.0f} ms")

    data = resp.json()

    try:
        content = data["choices"][0]["message"]["content"]
        raw_json = _extract_json(content)
    except (KeyError, IndexError) as exc:
        logger.error(f"Unexpected Groq response structure: {exc}\nRaw: {data}")
        raise RuntimeError(f"Groq response structure was unexpected: {exc}")
    except json.JSONDecodeError as exc:
        logger.error(f"JSON parse failed from Groq output: {exc}")
        raise RuntimeError(
            "Groq returned text that could not be parsed as JSON. Please retry."
        )

    normalized = _normalize_groq_output(raw_json)

    project_count = len(normalized["diy_projects"])
    logger.info(
        f"✓ Groq advisory complete | best_action='{normalized['best_action']}' | "
        f"is_diy_safe={normalized['is_diy_safe']} | "
        f"diy_projects={project_count}"
    )

    return normalized


# ---------------------------------------------------------------------------
# Development mock (used when GROQ_API_KEY is absent)
# ---------------------------------------------------------------------------

def _mock_advisory(gemini_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Realistic mock that returns 6 DIY projects for safe items
    and 0 DIY projects with recycling guidance for hazardous items.
    """
    item     = gemini_data.get("item_name", "Unknown item")
    hazards  = gemini_data.get("hazards", [])
    category = gemini_data.get("waste_category", "Mixed Waste")
    desc     = gemini_data.get("gemini_description", "")

    dangerous_keywords = ["battery leakage", "toxic", "asbestos", "biohazard"]
    is_hazardous = any(k in " ".join(hazards).lower() for k in dangerous_keywords)

    base_projects = [
        {
            "title": "Desktop Organiser Tray",
            "description": (
                f"Transform the outer casing of the {item} into a stylish desk organiser "
                "that holds pens, paper clips, and small items. Perfect for any workspace."
            ),
            "difficulty": "Beginner",
            "estimated_time": "30-45 minutes",
            "tools": ["scissors or craft knife", "sandpaper (120 grit)", "ruler", "hot glue gun"],
            "materials": ["item outer casing", "felt or cork sheet", "paint (optional)", "hot glue sticks"],
            "steps": [
                "Step 1: Clean the item thoroughly with a damp cloth and allow it to dry completely.",
                "Step 2: Using sandpaper, lightly rough up any smooth plastic surfaces so paint will adhere.",
                "Step 3: If the item has sharp edges, carefully file them smooth with sandpaper.",
                "Step 4: Decide on the compartment layout and mark divisions with a pencil.",
                "Step 5: Cut a piece of felt or cork to fit the base interior and glue it in place.",
                "Step 6: Apply your chosen paint colour in two thin coats, allowing 20 minutes drying time between coats.",
                "Step 7: Add any decorative elements — washi tape borders, stickers, or a stencilled pattern.",
                "Step 8: Allow 1 hour for full drying, then fill with your desk items and enjoy.",
            ],
            "safety_notes": [
                "Wear gloves when handling sharp edges.",
                "Use paint in a well-ventilated area.",
            ],
            "pro_tip": "Apply a coat of clear varnish at the end for a professional, durable finish.",
            "co2_saved_kg": 0.8,
            "landfill_saved_kg": 0.2,
        },
        {
            "title": "Wall-Mounted Key & Mail Holder",
            "description": (
                "Repurpose the item as a wall-mounted organiser for keys, letters, and small notes. "
                "A practical upcycle that adds character to your hallway or entrance."
            ),
            "difficulty": "Beginner",
            "estimated_time": "45-60 minutes",
            "tools": ["drill or strong adhesive", "screwdriver", "level", "pencil"],
            "materials": ["item body", "small adhesive hooks", "mounting screws or Command strips"],
            "steps": [
                "Step 1: Clean the item and remove any loose components safely.",
                "Step 2: Decide on the mounting orientation — landscape or portrait.",
                "Step 3: Mark the wall position at eye level using a pencil and level.",
                "Step 4: Attach Command strips or drill pilot holes for mounting screws.",
                "Step 5: Mount the item securely to the wall and check it is level.",
                "Step 6: Attach small adhesive hooks inside or below the item for hanging keys.",
                "Step 7: Add a small pocket or envelope holder for mail using folded card and glue.",
                "Step 8: Label sections with printed or handwritten labels for a neat, organised look.",
            ],
            "safety_notes": [
                "Ensure wall fixings are appropriate for your wall type (drywall vs masonry).",
                "Do not exceed the weight rating of adhesive strips.",
            ],
            "pro_tip": "Use a stud finder before drilling for a more secure mounting.",
            "co2_saved_kg": 0.6,
            "landfill_saved_kg": 0.15,
        },
        {
            "title": "Mini Indoor Herb Garden Planter",
            "description": (
                "Convert the item into a charming mini planter for growing herbs like basil, "
                "mint, or chives on your windowsill. Great for kitchens with limited space."
            ),
            "difficulty": "Beginner",
            "estimated_time": "20-30 minutes",
            "tools": ["hand drill or nail and hammer", "marker pen", "small spade or spoon"],
            "materials": ["item casing", "potting soil", "small plant or seeds", "pebbles for drainage"],
            "steps": [
                "Step 1: Clean the item thoroughly, removing any residue inside.",
                "Step 2: Mark 3-5 drainage holes in the bottom using a marker pen.",
                "Step 3: Carefully drill or punch the drainage holes — 5mm diameter is ideal.",
                "Step 4: Add a 1cm layer of small pebbles or gravel at the base for drainage.",
                "Step 5: Fill two-thirds of the space with potting compost.",
                "Step 6: Create a small hole in the centre of the soil with your finger.",
                "Step 7: Place your seedling or plant the seeds according to packet instructions.",
                "Step 8: Water lightly, place on a bright windowsill, and water every 2-3 days.",
            ],
            "safety_notes": [
                "Ensure drainage holes prevent waterlogging — standing water kills most herbs.",
                "Avoid using items that previously contained chemicals or toxic materials.",
            ],
            "pro_tip": "Paint the outside with chalk paint and write the herb name directly on it.",
            "co2_saved_kg": 0.4,
            "landfill_saved_kg": 0.1,
        },
        {
            "title": "Upcycled Bookend or Paperweight",
            "description": (
                "Fill and seal the item to create a satisfyingly heavy, decorative bookend "
                "or paperweight. A 10-minute project with zero waste."
            ),
            "difficulty": "Beginner",
            "estimated_time": "15-25 minutes",
            "tools": ["hot glue gun", "funnel (optional)", "paint or spray can"],
            "materials": ["item", "sand or small pebbles for weight", "strong glue", "felt for base"],
            "steps": [
                "Step 1: Open or access the interior of the item if possible.",
                "Step 2: Using a funnel, fill the interior with sand or small pebbles for weight.",
                "Step 3: Seal the item permanently using strong epoxy glue or hot glue.",
                "Step 4: Allow the glue to cure for at least 30 minutes.",
                "Step 5: Sand the exterior lightly to create a good surface for painting.",
                "Step 6: Apply 2 coats of spray paint in your chosen colour.",
                "Step 7: Cut a piece of felt to match the base footprint and glue it on — this protects surfaces.",
                "Step 8: Once fully dry, place between books or use as a stylish desk paperweight.",
            ],
            "safety_notes": [
                "Use spray paint outdoors or in a very well-ventilated space.",
                "Allow full curing time before using to avoid sticky surfaces.",
            ],
            "pro_tip": "Metallic spray paint gives a premium look — gold or copper works especially well.",
            "co2_saved_kg": 0.3,
            "landfill_saved_kg": 0.1,
        },
        {
            "title": "Charging Station Cable Organiser",
            "description": (
                "Transform the item into a cable management station that keeps your desk "
                "free of tangled wires. Works great beside a bed or on a desk."
            ),
            "difficulty": "Intermediate",
            "estimated_time": "60-90 minutes",
            "tools": ["craft knife", "drill", "ruler", "marker pen", "sandpaper"],
            "materials": ["item", "cable clips or zip ties", "foam padding", "paint"],
            "steps": [
                "Step 1: Clean the item and plan where cable entry/exit holes will go.",
                "Step 2: Mark the hole positions on the sides with a marker — typically 2-3 holes.",
                "Step 3: Carefully drill or cut the holes, starting with a small pilot hole.",
                "Step 4: Smooth all hole edges with sandpaper to prevent cable damage.",
                "Step 5: Line the interior with foam padding to protect cables from rattling.",
                "Step 6: Route your charging cables through the holes and organise them inside.",
                "Step 7: Use small cable clips inside the item to separate and label each cable.",
                "Step 8: Paint or cover the exterior as desired, then place your charger power strip inside.",
            ],
            "safety_notes": [
                "Ensure cables are not pinched at any entry holes — this is a fire risk.",
                "Do not enclose a power strip fully — it needs airflow to avoid overheating.",
            ],
            "pro_tip": "Cut a small ventilation slot in the back to ensure adequate airflow around electronics.",
            "co2_saved_kg": 0.5,
            "landfill_saved_kg": 0.12,
        },
        {
            "title": "Decorative Photo Frame Display",
            "description": (
                "Use the item as a unique frame or mounting base for a favourite photograph, "
                "artwork print, or motivational quote. A conversation piece for any room."
            ),
            "difficulty": "Beginner",
            "estimated_time": "30-45 minutes",
            "tools": ["scissors", "ruler", "craft knife", "hot glue gun"],
            "materials": ["item", "printed photo or artwork", "clear acetate sheet (optional)", "backing card"],
            "steps": [
                "Step 1: Clean the item thoroughly and decide which face will display the photo.",
                "Step 2: Measure the display area and note the exact dimensions.",
                "Step 3: Print or cut your chosen photo to match those dimensions exactly.",
                "Step 4: Cut a piece of backing card slightly smaller than the display area.",
                "Step 5: Mount the photo onto the backing card using glue stick — smooth out any bubbles.",
                "Step 6: Optionally, cut a clear acetate sheet to size and place over the photo as a 'glass' effect.",
                "Step 7: Secure the photo assembly into the display area using small drops of hot glue at corners.",
                "Step 8: Attach a hanging hook or stand to the back, and display proudly.",
            ],
            "safety_notes": [
                "Test hot glue on a small hidden area first — some plastics warp under heat.",
                "Use UV-resistant print if displaying in direct sunlight to prevent fading.",
            ],
            "pro_tip": "Add a border of washi tape or gold leaf paint around the photo for an elegant gallery look.",
            "co2_saved_kg": 0.3,
            "landfill_saved_kg": 0.08,
        },
    ]

    if is_hazardous:
        return {
            "handling_level":             "Send to Agency",
            "is_diy_safe":                False,
            "requires_special_recycling": True,
            "safety_reason": (
                f"The {item} contains hazardous components including {', '.join(hazards)}. "
                "These require certified e-waste or hazardous waste handling. "
                "Do not attempt to dismantle or repurpose this item at home."
            ),
            "summary": (
                f"This {item} has been identified as requiring professional recycling "
                "due to hazardous materials detected in its construction. "
                "While it may be tempting to reuse parts, the risks to your health and "
                "the environment are too significant to attempt home DIY. "
                "By sending this to a certified recycler, you ensure valuable materials "
                "are recovered safely and toxic compounds are disposed of properly. "
                "This is the most responsible and impactful choice you can make."
            ),
            "best_action":                "Recycle at agency",
            "why": (
                f"The detected hazards ({', '.join(hazards)}) make this item unsafe for home handling. "
                "A certified e-waste facility has the equipment to safely extract valuable materials "
                "like copper, gold, and rare earth metals while neutralising toxic compounds. "
                "Proper recycling of this item prevents heavy metals from entering soil and groundwater."
            ),
            "advisor_co2_saved_kg":       2.5,
            "advisor_landfill_saved_kg":  0.4,
            "recycling_guidance": (
                "Look for a certified e-waste or hazardous waste facility in your city. "
                "Search for 'WEEE certified recycler' or 'e-waste drop-off' near you. "
                "When you arrive, tell the staff exactly what the item is and mention the hazards detected — "
                "they will handle it accordingly. Many facilities offer free drop-off for small electronics."
            ),
            "diy_projects": [],
            "groq_notes": (
                "Mock advisory (GROQ_API_KEY not configured). "
                f"Item flagged as hazardous: {', '.join(hazards)}."
            ),
        }

    return {
        "handling_level":             "DIY Safe",
        "is_diy_safe":                True,
        "requires_special_recycling": False,
        "safety_reason": (
            f"The {item} is made of materials that are safe to handle at home with basic precautions. "
            "No hazardous substances were detected. Standard safety measures like gloves and eye "
            "protection are sufficient for all suggested projects."
        ),
        "summary": (
            f"Your {item} is in {gemini_data.get('condition', 'used')} condition "
            f"and is an excellent candidate for creative reuse. "
            "Its materials are safe to work with and versatile enough for a wide range of DIY projects. "
            "Rather than sending this to landfill, you can transform it into something useful and beautiful. "
            "Choosing to reuse this item prevents new raw materials from being extracted and "
            "saves significant CO₂ compared to buying an equivalent new product. "
            "Below are 6 projects you can start today — no special skills required."
        ),
        "best_action":     "Convert into DIY product",
        "why": (
            f"The {item}'s materials — {', '.join(gemini_data.get('materials', ['various']))} — "
            "are durable and versatile, making them ideal for a wide range of upcycling projects. "
            "Converting this item into a useful product avoids the carbon cost of manufacturing "
            "a replacement, and keeps valuable materials in circulation longer."
        ),
        "advisor_co2_saved_kg":      1.8,
        "advisor_landfill_saved_kg": 0.35,
        "recycling_guidance": (
            "If none of the DIY projects appeal to you, this item can be recycled through standard "
            "municipal collection or a local recycling centre. "
            "Separate any metal components before dropping off — metal recyclers often pay for aluminium and copper. "
            "Check your city's recycling portal for the nearest drop-off point."
        ),
        "diy_projects": base_projects,
        "groq_notes": (
            "Mock advisory generated (GROQ_API_KEY not configured). "
            "In production, Groq will generate 6 custom projects based on the specific item detected."
        ),
    }
