"""Dependency wiring for FastAPI routes."""
from functools import lru_cache

from app.models.registry import ModelRegistry
from app.monitoring.drift import DriftTracker
from app.monitoring.logger import configure_logging
from app.monitoring.metrics import MetricsCollector
from app.services.inference_service import InferenceService
from app.services.job_manager import JobManager
from app.utils.config import get_settings


@lru_cache()
def get_registry() -> ModelRegistry:
    settings = get_settings()
    return ModelRegistry(registry_path=settings.model_registry_path)


@lru_cache()
def get_metrics() -> MetricsCollector:
    return MetricsCollector()


@lru_cache()
def get_drift_tracker() -> DriftTracker:
    settings = get_settings()
    return DriftTracker(window_size=settings.drift_window, threshold=settings.drift_threshold)


@lru_cache()
def get_job_manager() -> JobManager:
    settings = get_settings()
    return JobManager(max_workers=settings.batch_max_workers)


@lru_cache()
def get_inference_service() -> InferenceService:
    settings = get_settings()
    configure_logging(settings.service_name)
    return InferenceService(
        settings=settings,
        registry=get_registry(),
        metrics=get_metrics(),
        drift_tracker=get_drift_tracker(),
        job_manager=get_job_manager(),
    )


__all__ = [
    "get_registry",
    "get_metrics",
    "get_drift_tracker",
    "get_job_manager",
    "get_inference_service",
]
