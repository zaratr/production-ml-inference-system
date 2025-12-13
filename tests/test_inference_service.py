from pathlib import Path

from app.models.registry import ModelRegistry
from app.monitoring.drift import DriftTracker
from app.monitoring.metrics import MetricsCollector
from app.services.inference_service import InferenceService
from app.services.job_manager import JobManager
from app.utils.config import AppSettings


def build_service() -> InferenceService:
    settings = AppSettings.from_env()
    settings = AppSettings(
        env=settings.env,
        model_registry_path=Path("config/model_store"),
        default_model_version=settings.default_model_version,
        service_name=settings.service_name,
        batch_max_workers=1,
        drift_window=5,
        drift_threshold=0.1,
        request_timeout_seconds=settings.request_timeout_seconds,
    )
    registry = ModelRegistry(settings.model_registry_path)
    metrics = MetricsCollector()
    drift = DriftTracker(window_size=settings.drift_window, threshold=settings.drift_threshold)
    jobs = JobManager(max_workers=settings.batch_max_workers)
    return InferenceService(settings=settings, registry=registry, metrics=metrics, drift_tracker=drift, job_manager=jobs)


def test_predict_returns_probabilities() -> None:
    service = build_service()
    response = service.predict([{"feature_a": 1.0, "feature_b": 0.2}])
    assert "predictions" in response
    prediction = response["predictions"][0]
    assert 0.0 <= prediction["probability"] <= 1.0
    assert prediction["version"] == "v1"


def test_batch_submission_and_retrieval() -> None:
    service = build_service()
    job_id = service.enqueue_batch([{"feature_a": 0.1, "feature_b": 0.2}])
    status = service.batch_status(job_id)
    assert status["status"] in {"running", "completed", "pending"}
    result = service.batch_status(job_id)
    if result.get("result"):
        assert "predictions" in result["result"]


def test_drift_tracker_signals_change() -> None:
    tracker = DriftTracker(window_size=3, threshold=0.2)
    signals = []
    signals.extend(tracker.update({"feature_a": 1.0}))
    signals.extend(tracker.update({"feature_a": 1.0}))
    signals.extend(tracker.update({"feature_a": 1.0}))
    signals.extend(tracker.update({"feature_a": 2.0}))
    assert any(signal.feature == "feature_a" for signal in signals)
