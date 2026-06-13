"""
ASTRA – Core configuration
Reads all settings from environment variables (or .env file via python-dotenv).
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -----------------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------------
    APP_NAME: str = "ASTRA"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # -----------------------------------------------------------------------
    # External AI APIs
    # -----------------------------------------------------------------------
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # -----------------------------------------------------------------------
    # Optional: Google Places for live agency lookup
    # -----------------------------------------------------------------------
    GOOGLE_PLACES_API_KEY: str = ""

    # -----------------------------------------------------------------------
    # CORS
    # -----------------------------------------------------------------------
    ALLOWED_ORIGINS: List[str] = [
        "https://astra-aiplatform.web.app",
        "https://astra-aiplatform.firebaseapp.com",
        "http://localhost:5173",
        "http://localhost:8001",
        "*",
    ]

    # -----------------------------------------------------------------------
    # Timeouts (seconds)
    # -----------------------------------------------------------------------
    GEMINI_TIMEOUT: int = 60
    GROQ_TIMEOUT: int = 60
    AGENCY_TIMEOUT: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
