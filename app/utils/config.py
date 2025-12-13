"""Configuration utilities for the inference service."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os


@dataclass(frozen=True)
class AppSettings:
    """Application configuration loaded from environment variables."""

    env: str
    model_registry_path: Path
    default_model_version: str
    service_name: str
    batch_max_workers: int
    drift_window: int
    drift_threshold: float
    request_timeout_seconds: float

    @staticmethod
    def from_env() -> "AppSettings":
        return AppSettings(
            env=os.getenv("APP_ENV", "dev"),
            model_registry_path=Path(os.getenv("MODEL_REGISTRY_PATH", "config/model_store")),
            default_model_version=os.getenv("DEFAULT_MODEL_VERSION", "v1"),
            service_name=os.getenv("SERVICE_NAME", "production-ml-inference-system"),
            batch_max_workers=int(os.getenv("BATCH_MAX_WORKERS", "2")),
            drift_window=int(os.getenv("DRIFT_WINDOW", "200")),
            drift_threshold=float(os.getenv("DRIFT_THRESHOLD", "0.15")),
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "2.0")),
        )


@lru_cache()
def get_settings() -> AppSettings:
    """Return cached application settings."""

    return AppSettings.from_env()


__all__ = ["AppSettings", "get_settings"]
