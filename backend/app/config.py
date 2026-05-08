from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field, field_validator
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    supabase_url: str = ""
    supabase_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "SUPABASE_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
        ),
    )
    serpapi_key: str = ""

    clip_device: str = "cpu"

    catalog_size: int = 500
    max_upload_images: int = 10
    max_pinterest_images: int = 50

    faiss_index_path: str = "data/faiss_index.bin"
    catalog_cache_path: str = "data/catalog_cache.json"

    confidence_threshold: float = 0.20
    price_band_tolerance: float = 0.30

    rank_weight_taste: float = 0.45
    rank_weight_trend: float = 0.15
    rank_weight_utility: float = 0.30
    rank_weight_anti: float = 0.10
    rank_weight_skip: float = 0.12

    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
    )

    anthropic_api_key: str = Field(
        default="",
        validation_alias="ANTHROPIC_API_KEY",
    )

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v


@lru_cache()
def get_settings() -> Settings:
    return Settings()
