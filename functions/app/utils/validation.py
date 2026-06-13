"""
ASTRA – Input validation helpers
"""

import re
from typing import Optional


def validate_city(city: str) -> str:
    """
    Ensure city is a non-empty, reasonably-sized string.
    Returns the stripped city name or raises ValueError.
    """
    city = city.strip()
    if not city:
        raise ValueError("City name cannot be empty.")
    if len(city) > 100:
        raise ValueError("City name is too long (max 100 characters).")
    # Allow letters, digits, spaces, hyphens, periods, apostrophes, commas
    # (covers international names like "São Paulo", "N'Djamena", "St. Louis")
    if not re.match(r"^[\w\s\-\.\'\,]+$", city, re.UNICODE):
        raise ValueError(
            "City name contains invalid characters. "
            "Only letters, numbers, spaces, hyphens, and periods are allowed."
        )
    return city


def validate_coordinates(lat: Optional[float], lon: Optional[float]) -> None:
    """Validate latitude and longitude ranges if provided."""
    if lat is not None and not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude must be between -90 and 90 (got {lat}).")
    if lon is not None and not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude must be between -180 and 180 (got {lon}).")
