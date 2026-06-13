"""
ASTRA – Sustainability Impact Service

Provides per-category CO₂ and landfill savings baselines.
Used to fill zero values when Groq doesn't return explicit impact numbers.
Values are conservative estimates derived from lifecycle analysis literature.
"""

from typing import Dict, Any


# ---------------------------------------------------------------------------
# Per-category CO₂ and landfill baselines
# (kg CO₂ saved per kg of material recycled vs. virgin production)
# ---------------------------------------------------------------------------

CATEGORY_BASELINES: Dict[str, Dict[str, float]] = {
    "E-Waste":                {"co2_kg_per_kg": 20.0, "landfill_kg_per_kg": 1.0},
    "Plastic Waste":          {"co2_kg_per_kg":  3.5, "landfill_kg_per_kg": 1.0},
    "Metal Waste":            {"co2_kg_per_kg":  9.0, "landfill_kg_per_kg": 1.0},
    "Glass Waste":            {"co2_kg_per_kg":  0.6, "landfill_kg_per_kg": 1.0},
    "Paper/Cardboard Waste":  {"co2_kg_per_kg":  1.1, "landfill_kg_per_kg": 1.0},
    "Organic Waste":          {"co2_kg_per_kg":  0.5, "landfill_kg_per_kg": 1.0},
    "Hazardous Waste":        {"co2_kg_per_kg":  5.0, "landfill_kg_per_kg": 1.0},
    "Mixed Waste":            {"co2_kg_per_kg":  2.0, "landfill_kg_per_kg": 1.0},
    "Reusable Junk":          {"co2_kg_per_kg":  4.0, "landfill_kg_per_kg": 1.0},
    "Unknown / Needs Review": {"co2_kg_per_kg":  1.0, "landfill_kg_per_kg": 1.0},
}

# Approximate item weight in kg per category (used when exact weight is unknown)
CATEGORY_WEIGHT_KG: Dict[str, float] = {
    "E-Waste":                0.35,
    "Plastic Waste":          0.20,
    "Metal Waste":            0.60,
    "Glass Waste":            0.40,
    "Paper/Cardboard Waste":  0.30,
    "Organic Waste":          0.50,
    "Hazardous Waste":        0.20,
    "Mixed Waste":            0.40,
    "Reusable Junk":          1.00,
    "Unknown / Needs Review": 0.30,
}


def estimate_impact(waste_category: str) -> Dict[str, float]:
    """
    Return estimated CO₂ and landfill savings for recycling one typical item
    of the given category.
    """
    baseline = CATEGORY_BASELINES.get(
        waste_category,
        CATEGORY_BASELINES["Unknown / Needs Review"],
    )
    weight = CATEGORY_WEIGHT_KG.get(waste_category, 0.30)

    return {
        "co2_saved_kg":      round(baseline["co2_kg_per_kg"] * weight, 3),
        "landfill_saved_kg": round(baseline["landfill_kg_per_kg"] * weight, 3),
    }


def enrich_with_impact(groq_data: Dict[str, Any], waste_category: str) -> Dict[str, Any]:
    """
    If Groq did not return non-zero impact values, fill them from category baselines.
    Also enriches individual DIY project impacts.

    Modifies groq_data in-place and returns it.
    """
    baseline = estimate_impact(waste_category)

    if groq_data.get("advisor_co2_saved_kg", 0.0) == 0.0:
        groq_data["advisor_co2_saved_kg"] = baseline["co2_saved_kg"]

    if groq_data.get("advisor_landfill_saved_kg", 0.0) == 0.0:
        groq_data["advisor_landfill_saved_kg"] = baseline["landfill_saved_kg"]

    for project in groq_data.get("diy_projects", []):
        if project.get("co2_saved_kg", 0.0) == 0.0:
            project["co2_saved_kg"] = round(baseline["co2_saved_kg"] * 0.6, 3)
        if project.get("landfill_saved_kg", 0.0) == 0.0:
            project["landfill_saved_kg"] = round(baseline["landfill_saved_kg"] * 0.6, 3)

    return groq_data
