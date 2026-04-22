"""Runtime configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAUD_", env_file=".env", extra="ignore")

    env: str = "local"
    database_url: str | None = None
    model_dir: Path = Path("/models")
    policy_dir: Path = Path("policy")
    allow_demo_model: bool = False

    vllm_base_url: str = "http://vllm.open-webui.svc.cluster.local:8000/v1"
    vllm_api_key: str = "EMPTY"
    vllm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"

    tei_base_url: str | None = None
    qdrant_url: str | None = None
    qdrant_collection: str = "fraud_policy"
    nats_url: str | None = None

    api_cors_origins: str = Field(default="http://localhost:5173,http://localhost:3000")
    batch_limit: int = 500

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.api_cors_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
