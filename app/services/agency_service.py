"""
ASTRA – Agency Lookup Service

Real-place lookup strategy (priority order):
  1. Google Places API   — if GOOGLE_PLACES_API_KEY is configured (richest results globally)
  2. Overpass API        — real OSM data, works best in Europe/North America
  3. Curated smart links — always generated for any city, zero API needed:
                           direct Google Maps search links pre-filled with the
                           right query and city so users open their browser and
                           see real nearby results immediately.

Every returned agency always has:
  name, address, phone, website, accepted_waste_types,
  distance_km, contact_instructions, maps_url
"""

import asyncio
import math
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import httpx

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Category maps
# ─────────────────────────────────────────────────────────────────────────────

# Human-readable search keyword per category (used in Places + Maps links)
CATEGORY_KEYWORD: Dict[str, str] = {
    "E-Waste":                "e-waste recycling",
    "Plastic Waste":          "plastic recycling centre",
    "Metal Waste":            "scrap metal recycling",
    "Glass Waste":            "glass recycling centre",
    "Paper/Cardboard Waste":  "paper cardboard recycling",
    "Organic Waste":          "composting organic waste facility",
    "Hazardous Waste":        "hazardous waste disposal facility",
    "Mixed Waste":            "waste recycling centre",
    "Reusable Junk":          "charity shop donation centre",
    "Unknown / Needs Review": "recycling centre waste disposal",
}

# What a real agency at each category likely accepts
CATEGORY_ACCEPTED: Dict[str, List[str]] = {
    "E-Waste":                ["E-Waste", "Electronics", "Batteries", "Mobile Phones", "Computers"],
    "Plastic Waste":          ["Plastic", "PET Bottles", "HDPE", "Mixed Plastic"],
    "Metal Waste":            ["Scrap Metal", "Aluminium", "Copper", "Steel", "Iron"],
    "Glass Waste":            ["Glass Bottles", "Window Glass", "Jars"],
    "Paper/Cardboard Waste":  ["Paper", "Cardboard", "Newspapers", "Books"],
    "Organic Waste":          ["Organic Waste", "Food Scraps", "Garden Waste"],
    "Hazardous Waste":        ["Hazardous Waste", "Batteries", "Chemicals", "Paints", "Solvents"],
    "Mixed Waste":            ["General Waste", "Mixed Recyclables", "Bulk Waste"],
    "Reusable Junk":          ["Furniture", "Clothing", "Electronics", "Household Items"],
    "Unknown / Needs Review": ["General Waste", "Recyclables"],
}

# OSM Overpass tags per category
CATEGORY_OSM_TAGS: Dict[str, List[str]] = {
    "E-Waste":                ['"recycling:electronics"="yes"', '"recycling:electrical_appliances"="yes"'],
    "Plastic Waste":          ['"recycling:plastic"="yes"', '"recycling:plastic_bottles"="yes"'],
    "Metal Waste":            ['"recycling:scrap_metal"="yes"', '"recycling:metal"="yes"'],
    "Glass Waste":            ['"recycling:glass"="yes"', '"recycling:glass_bottles"="yes"'],
    "Paper/Cardboard Waste":  ['"recycling:paper"="yes"', '"recycling:cardboard"="yes"'],
    "Organic Waste":          ['"recycling:organic"="yes"'],
    "Hazardous Waste":        ['"recycling:hazardous_waste"="yes"'],
    "Mixed Waste":            [],
    "Reusable Junk":          [],
    "Unknown / Needs Review": [],
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float,
               lat2: Optional[float], lon2: Optional[float]) -> Optional[float]:
    if lat2 is None or lon2 is None:
        return None
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(p1) * math.cos(p2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)


def _gmaps_search_url(query: str, lat: Optional[float] = None,
                       lon: Optional[float] = None) -> str:
    """Deep-link to Google Maps that opens a live search for the query."""
    if lat is not None and lon is not None:
        # Near coordinates — shows pins on real map immediately
        return (
            f"https://www.google.com/maps/search/{quote_plus(query)}"
            f"/@{lat},{lon},13z"
        )
    return f"https://www.google.com/maps/search/{quote_plus(query)}"


def _gmaps_coord_url(lat: float, lon: float, name: str = "") -> str:
    return f"https://www.google.com/maps?q={lat},{lon}"


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Nominatim geocoding
# ─────────────────────────────────────────────────────────────────────────────

async def _geocode_city(city: str) -> Optional[Dict[str, float]]:
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": city, "format": "json", "limit": 1},
                headers={"User-Agent": "ASTRA-SustainabilityApp/1.0",
                         "Accept-Language": "en"},
            )
            r.raise_for_status()
        data = r.json()
        if data:
            return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
    except Exception as exc:
        logger.warning(f"Nominatim geocoding failed for '{city}': {exc}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Google Places  (requires GOOGLE_PLACES_API_KEY)
# ─────────────────────────────────────────────────────────────────────────────

async def _places_details(place_id: str, api_key: str) -> Dict[str, str]:
    try:
        async with httpx.AsyncClient(timeout=settings.AGENCY_TIMEOUT) as c:
            r = await c.get(
                "https://maps.googleapis.com/maps/api/place/details/json",
                params={
                    "place_id": place_id,
                    "fields":   "formatted_phone_number,website,opening_hours",
                    "key":      api_key,
                },
            )
            r.raise_for_status()
        res = r.json().get("result", {})
        hours_list = res.get("opening_hours", {}).get("weekday_text", [])
        return {
            "phone":         res.get("formatted_phone_number", ""),
            "website":       res.get("website", ""),
            "opening_hours": "; ".join(hours_list[:3]),
        }
    except Exception:
        return {"phone": "", "website": "", "opening_hours": ""}


async def _fetch_google_places(
    waste_category: str, city: str,
    lat: Optional[float], lon: Optional[float],
) -> List[Dict[str, Any]]:
    key     = settings.GOOGLE_PLACES_API_KEY
    keyword = CATEGORY_KEYWORD.get(waste_category, "recycling centre")
    query   = f"{keyword} near {city}"

    params: Dict[str, Any] = {"query": query, "key": key}
    if lat and lon:
        params["location"] = f"{lat},{lon}"
        params["radius"]   = 20000

    try:
        async with httpx.AsyncClient(timeout=settings.AGENCY_TIMEOUT) as c:
            r = await c.get(
                "https://maps.googleapis.com/maps/api/place/textsearch/json",
                params=params,
            )
            r.raise_for_status()
    except Exception as exc:
        logger.warning(f"Google Places search failed: {exc}")
        return []

    raw = r.json().get("results", [])[:5]
    if not raw:
        return []

    # Fetch phone + website concurrently
    details_list = await asyncio.gather(
        *[_places_details(item.get("place_id", ""), key) for item in raw]
    )

    agencies = []
    for item, det in zip(raw, details_list):
        geo     = item.get("geometry", {}).get("location", {})
        r_lat   = geo.get("lat")
        r_lon   = geo.get("lng")
        name    = item.get("name", "Recycling Facility")
        address = item.get("formatted_address", city)
        phone   = det["phone"]
        website = det["website"]
        hours   = det["opening_hours"]

        contact = f"Drop off your {waste_category} here. "
        if hours:
            contact += f"Hours: {hours}. "
        if phone:
            contact += "Call ahead to confirm they accept your item type."
        else:
            contact += "Visit in person or check their website for the latest information."

        agencies.append({
            "name":                 name,
            "address":              address,
            "phone":                phone,
            "website":              website,
            "accepted_waste_types": CATEGORY_ACCEPTED.get(waste_category, ["Recyclables"]),
            "distance_km":          _haversine(lat, lon, r_lat, r_lon) if lat and lon else None,
            "contact_instructions": contact,
            "maps_url":             _gmaps_coord_url(r_lat, r_lon, name) if r_lat and r_lon
                                    else _gmaps_search_url(f"{name} {city}"),
        })

    logger.info(f"✓ Google Places: {len(agencies)} real agencies for '{city}'")
    return agencies


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — Overpass / OpenStreetMap  (free, no key, best in EU/NA)
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_overpass(
    waste_category: str, lat: float, lon: float,
    radius_m: int = 20000,
) -> List[Dict[str, Any]]:
    """Query Overpass for real recycling nodes/ways near lat/lon."""
    specific_tags = CATEGORY_OSM_TAGS.get(waste_category, [])

    # Build union of tag filters
    union: List[str] = []
    for tag in specific_tags:
        for typ in ("node", "way"):
            union.append(f'  {typ}["amenity"="recycling"][{tag}](around:{radius_m},{lat},{lon});')

    # Always add generic recycling + waste_transfer_station
    for typ in ("node", "way"):
        union.append(f'  {typ}["amenity"="recycling"](around:{radius_m},{lat},{lon});')
        union.append(f'  {typ}["amenity"="waste_transfer_station"](around:{radius_m},{lat},{lon});')
        union.append(f'  {typ}["shop"="charity"](around:{radius_m},{lat},{lon});')
        union.append(f'  {typ}["shop"="scrap_metal"](around:{radius_m},{lat},{lon});')

    # Also search by name keywords
    kw = CATEGORY_KEYWORD.get(waste_category, "recycling").split()[0]
    for typ in ("node", "way"):
        union.append(f'  {typ}["name"~"{kw}|recycle|recycling|scrap|waste",i](around:{radius_m},{lat},{lon});')

    query = (
        "[out:json][timeout:25];\n(\n"
        + "\n".join(union)
        + "\n);\nout center tags 15;"
    )

    try:
        async with httpx.AsyncClient(timeout=28) as c:
            r = await c.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                headers={"User-Agent": "ASTRA/1.0"},
            )
            r.raise_for_status()
    except Exception as exc:
        logger.warning(f"Overpass failed: {exc}")
        return []

    elements = r.json().get("elements", [])
    logger.info(f"Overpass returned {len(elements)} raw elements (radius={radius_m}m)")

    agencies: List[Dict[str, Any]] = []
    seen: set = set()

    for el in elements:
        tags = el.get("tags", {})
        name = (tags.get("name") or tags.get("operator")
                or tags.get("brand") or None)
        if not name or name in seen:
            continue
        seen.add(name)

        # Coordinates
        if el["type"] == "node":
            e_lat, e_lon = el.get("lat"), el.get("lon")
        else:
            ctr = el.get("center", {})
            e_lat, e_lon = ctr.get("lat"), ctr.get("lon")

        # Address
        addr_parts = [
            tags.get("addr:housenumber", ""),
            tags.get("addr:street", ""),
            tags.get("addr:suburb") or tags.get("addr:district", ""),
            tags.get("addr:city", ""),
        ]
        address = ", ".join(p for p in addr_parts if p) or "See map for exact location"

        phone   = tags.get("phone") or tags.get("contact:phone") or ""
        website = tags.get("website") or tags.get("contact:website") or ""
        hours   = tags.get("opening_hours") or ""

        # Detect accepted types from OSM recycling:* tags
        accepted = [
            label for k, label in {
                "recycling:electronics":           "E-Waste / Electronics",
                "recycling:electrical_appliances": "Electrical Appliances",
                "recycling:plastic":               "Plastic",
                "recycling:glass":                 "Glass",
                "recycling:metal":                 "Metal",
                "recycling:scrap_metal":           "Scrap Metal",
                "recycling:paper":                 "Paper",
                "recycling:cardboard":             "Cardboard",
                "recycling:batteries":             "Batteries",
                "recycling:organic":               "Organic",
                "recycling:hazardous_waste":       "Hazardous Waste",
                "recycling:clothes":               "Clothing",
            }.items()
            if tags.get(k) in ("yes", "1", "true")
        ] or CATEGORY_ACCEPTED.get(waste_category, ["Recyclables"])

        contact = f"Drop off your {waste_category} here."
        if hours:
            contact += f" Open: {hours}."
        if phone:
            contact += " Call ahead to confirm accepted item types."
        else:
            contact += " Visit in person to confirm opening hours."

        agencies.append({
            "name":                 name,
            "address":              address,
            "phone":                phone,
            "website":              website,
            "accepted_waste_types": accepted,
            "distance_km":          _haversine(lat, lon, e_lat, e_lon),
            "contact_instructions": contact,
            "maps_url":             _gmaps_coord_url(e_lat, e_lon, name)
                                    if e_lat and e_lon
                                    else _gmaps_search_url(f"{name}", lat, lon),
        })

    agencies.sort(key=lambda a: a["distance_km"] or 9999)
    return agencies[:5]


# ─────────────────────────────────────────────────────────────────────────────
# Tier 4 — Smart curated links  (works for ANY city globally, zero API needed)
# ─────────────────────────────────────────────────────────────────────────────

def _build_smart_links(
    waste_category: str,
    city: str,
    lat: Optional[float],
    lon: Optional[float],
) -> List[Dict[str, Any]]:
    """
    Generate 4-5 pre-built Google Maps search links for real places near the user.
    These open live Maps search results — users see real pins immediately.
    This works for 100% of cities worldwide with zero API key or database needed.
    """
    keyword  = CATEGORY_KEYWORD.get(waste_category, "recycling centre")
    accepted = CATEGORY_ACCEPTED.get(waste_category, ["Recyclables"])

    # Build 4 search link variants so user has multiple options
    searches = [
        {
            "label":   keyword,
            "query":   f"{keyword} near {city}",
        },
        {
            "label":   "recycling centre",
            "query":   f"recycling centre near {city}",
        },
        {
            "label":   "waste management facility",
            "query":   f"waste management facility {city}",
        },
        {
            "label":   "municipal waste authority",
            "query":   f"municipal waste authority {city}",
        },
    ]

    # Add a donation-centre entry for reusable items
    if waste_category in ("Reusable Junk", "Mixed Waste"):
        searches.insert(1, {
            "label": "charity donation centre",
            "query": f"charity shop donation centre {city}",
        })

    agencies = []
    for s in searches[:5]:
        maps_url = _gmaps_search_url(s["query"], lat, lon)
        agencies.append({
            "name":    f"Find: {s['label'].title()} near {city}",
            "address": f"{city} — click 'Open in Maps' to see real nearby locations",
            "phone":   "",
            "website": "",
            "accepted_waste_types": accepted,
            "distance_km": None,
            "contact_instructions": (
                f"Click the Maps link below to open a live Google Maps search for "
                f"'{s['query']}'. You will see real facilities near you with their "
                f"addresses, phone numbers, ratings, and opening hours."
            ),
            "maps_url": maps_url,
        })

    return agencies


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def get_agencies(
    waste_category: str,
    materials: List[str],
    city: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Return up to 5 recycling agencies/links for the user's city.

    Tier 1 → Google Places (real data + phone + website, needs key)
    Tier 2 → OpenStreetMap Overpass (real data, free, best in EU/NA)
    Tier 3 → Smart Google Maps search links (works globally, zero key)

    Result always includes real-place Google Maps deep-links so users
    can tap and immediately see facilities on a live map.
    """

    # ── Geocode city if no GPS coords provided ────────────────────────────────
    resolved_lat, resolved_lon = lat, lon
    if resolved_lat is None or resolved_lon is None:
        logger.info(f"Geocoding '{city}' via Nominatim")
        geo = await _geocode_city(city)
        if geo:
            resolved_lat, resolved_lon = geo["lat"], geo["lon"]
            logger.info(f"  → ({resolved_lat:.4f}, {resolved_lon:.4f})")

    # ── Tier 1: Google Places ─────────────────────────────────────────────────
    if settings.GOOGLE_PLACES_API_KEY:
        try:
            agencies = await _fetch_google_places(
                waste_category, city, resolved_lat, resolved_lon
            )
            if agencies:
                # Append a "search more" link at the end
                agencies.append(_search_more_link(waste_category, city, resolved_lat, resolved_lon))
                return agencies
        except Exception as exc:
            logger.warning(f"Google Places failed: {exc}")

    # ── Tier 2: Overpass / OSM ────────────────────────────────────────────────
    if resolved_lat is not None and resolved_lon is not None:
        try:
            agencies = await _fetch_overpass(
                waste_category, resolved_lat, resolved_lon, radius_m=20000
            )
            if not agencies:
                # Widen to 40 km for cities with sparse OSM data
                agencies = await _fetch_overpass(
                    waste_category, resolved_lat, resolved_lon, radius_m=40000
                )
            if agencies:
                agencies.append(_search_more_link(waste_category, city, resolved_lat, resolved_lon))
                return agencies
            logger.info("Overpass found 0 agencies — using smart link fallback")
        except Exception as exc:
            logger.warning(f"Overpass failed: {exc}")

    # ── Tier 3: Smart Maps Links (works for all cities globally) ─────────────
    logger.info(f"Building smart Maps search links for '{city}'")
    return _build_smart_links(waste_category, city, resolved_lat, resolved_lon)


def _search_more_link(
    waste_category: str, city: str,
    lat: Optional[float], lon: Optional[float],
) -> Dict[str, Any]:
    """Append a 'search more on Google Maps' entry to real results."""
    keyword = CATEGORY_KEYWORD.get(waste_category, "recycling centre")
    return {
        "name":    f"🔍 Search more: {keyword} near {city}",
        "address": "Tap to open live Google Maps search",
        "phone":   "",
        "website": "",
        "accepted_waste_types": CATEGORY_ACCEPTED.get(waste_category, ["Recyclables"]),
        "distance_km": None,
        "contact_instructions": (
            "Click this link to see all recycling facilities near you on Google Maps, "
            "including their phone numbers, opening hours, and directions."
        ),
        "maps_url": _gmaps_search_url(f"{keyword} near {city}", lat, lon),
    }
