"""Inference service orchestrating model execution and monitoring."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from app.models.registry import ModelRegistry
from app.monitoring.drift import DriftTracker
from app.monitoring.logger import logger
from app.monitoring.metrics import MetricsCollector
from app.services.job_manager import JobManager
from app.utils.config import AppSettings


class InferenceService:
    def __init__(
        self,
        settings: AppSettings,
        registry: ModelRegistry,
        metrics: MetricsCollector,
        drift_tracker: DriftTracker,
        job_manager: JobManager,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.metrics = metrics
        self.drift_tracker = drift_tracker
        self.job_manager = job_manager

    def predict(self, features: List[Dict[str, Any]], model_version: Optional[str] = None) -> Dict[str, Any]:
        version = model_version or self.settings.default_model_version
        start = time.monotonic()
        self.metrics.increment("request_total")
        try:
            model = self.registry.load(version)
        except Exception as exc:  # noqa: BLE001
            self.metrics.increment("errors")
            logger.exception("Failed to load model", extra={"ctx_version": version})
            raise

        predictions = model.predict(features)
        latency = time.monotonic() - start
        self.metrics.observe_latency("inference_latency", latency)
        logger.info(
            "prediction completed",
            extra={
                "ctx_version": version,
                "ctx_latency_ms": int(latency * 1000),
                "ctx_num_records": len(features),
            },
        )
        for row in features:
            signals = self.drift_tracker.update({k: float(v) for k, v in row.items() if isinstance(v, (int, float))})
            for signal in signals:
                logger.warning(
                    "drift detected",
                    extra={
                        "ctx_feature": signal.feature,
                        "ctx_drift_score": signal.drift_score,
                        "ctx_baseline_mean": signal.baseline_mean,
                        "ctx_current_mean": signal.current_mean,
                    },
                )
        return {"predictions": predictions, "version": version, "latency_ms": latency * 1000}

    def enqueue_batch(self, features: List[Dict[str, Any]], model_version: Optional[str] = None) -> str:
        return self.job_manager.submit(self.predict, features, model_version)

    def batch_status(self, job_id: str) -> Dict[str, Any]:
        status = self.job_manager.status(job_id)
        result = self.job_manager.result(job_id)
        payload: Dict[str, Any] = {"job_id": job_id, "status": status}
        if result is not None:
            payload["result"] = result
        return payload

    def health(self) -> Dict[str, Any]:
        try:
            self.registry.load(self.settings.default_model_version)
            status = "ready"
        except Exception:
            status = "degraded"
        return {
            "status": status,
            "default_model": self.settings.default_model_version,
            "env": self.settings.env,
        }


__all__ = ["InferenceService"]
