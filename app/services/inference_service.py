"""Inference service orchestrating model execution and monitoring."""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from app.models.registry import ModelRegistry
from app.monitoring.drift import DriftTracker
from app.monitoring.logger import logger
from app.monitoring.metrics import MetricsCollector
from app.services.batch_scheduler import BatchScheduler
from app.services.circuit_breaker import CircuitBreaker
from app.services.job_manager import JobManager
from app.utils.config import AppSettings




class InferenceService:
    def load_model(self, version: str) -> None:
        """Load a model version into memory."""
        self.registry.load(version)

    def unload_model(self, version: str) -> None:
        """Unload a model version from memory."""
        self.registry.unload(version)

    def promote_model(self, version: str) -> None:
        """Set a model version as the default."""
        self.registry.set_default_version(version)

    def list_models(self) -> Dict[str, Any]:
        """List loaded models and current default."""
        return {
            "loaded_versions": self.registry.list_loaded_versions(),
            "default_version": self.registry.default_version,
        }

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
        

        # Circuit breaker for model protection
        self.breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=5.0)
        
        # Batch scheduler for the default model
        self.scheduler = BatchScheduler(
            prediction_fn=self._run_batch_prediction,
            max_batch_size=32,
            max_latency_ms=10.0,
            max_queue_size=1024,
        )

    def start(self) -> None:
        """Start the batch scheduler."""
        self.scheduler.start()

    async def stop(self) -> None:
        """Stop the batch scheduler."""
        await self.scheduler.stop()

    def _run_batch_prediction(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Internal callback for batch scheduler to run prediction on a batch."""
        # Use circuit breaker to protect the model call
        with self.breaker:
            return self._unsafe_run_batch_prediction(features)

    def _unsafe_run_batch_prediction(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # This runs in a thread pool managed by the scheduler (via to_thread)
        # Use dynamic default version from registry
        version = self.registry.default_version
        try:
            model = self.registry.load(version)
            predictions = model.predict(features)
            return predictions
        except Exception:
            logger.exception("Batch prediction execution failed")
            raise

    async def predict(self, features: List[Dict[str, Any]], model_version: Optional[str] = None) -> Dict[str, Any]:
        version = model_version or self.registry.default_version
        start = time.monotonic()
        self.metrics.increment("request_total")
        
        # Use batch scheduler ONLY if:
        # 1. It is the default model (our scheduler is single-model for now)
        # 2. We are not forcing a specific version that differs
        # Note: If default_version changes while we are here, we might have a race,
        # but 'version' variable captures it for this request.
        use_batching = (model_version is None) or (model_version == self.registry.default_version)

        try:
            if use_batching:
                # Disaggregate input: The current API accepts a LIST of features (batch request).
                # But the scheduler accepts single items or we change scheduler to accept chunks?
                # My BatchScheduler.predict accepts 'features: Dict[str, Any]'.
                # But the API input 'features' is List[Dict].
                # If the user sends a batch of 10 items, we should ideally keep them together 
                # or split them.
                # If we split them, we have to wait for all futures.
                
                # For simplicity, let's map the input list to multiple scheduler calls.
                tasks = [self.scheduler.predict(f) for f in features]
                predictions = await asyncio.gather(*tasks)
            else:
                # Legacy/Specific version path (sync -> thread pool via fastapi or local)
                # Since this function is async, we shouldn't block loop.
                # Run in executor.
                model = self.registry.load(version)
                predictions = await asyncio.to_thread(model.predict, features)

        except Exception as exc:  # noqa: BLE001
            self.metrics.increment("errors")
            logger.exception("Failed to load/run model", extra={"ctx_version": version})
            raise

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
        
        # Async drift tracking (fire and forget or await?)
        # Logging/metrics are fast. Drift tracking might compute things.
        # Let's keep it inline for now or make it a background task.
        for row in features:
            signals = self.drift_tracker.update({k: float(v) for k, v in row.items() if isinstance(v, (int, float))})
            for signal in signals:
                logger.warning(
                    "drift detected",
                    extra={
                        "ctx_feature": signal.feature,
                        "ctx_drift_score": signal.drift_score,
                    },
                )
        return {"predictions": predictions, "version": version, "latency_ms": latency * 1000}

    def enqueue_batch(self, features: List[Dict[str, Any]], model_version: Optional[str] = None) -> str:
        # job_manager is sync (ThreadPollExecutor). 
        # But now predict is async. JobManager.submit takes a function.
        # We need an async-to-sync bridge or update JobManager to support async functions?
        # Or just have JobManager run a sync wrapper "run_async(predict...)"?
        # Easier: Keep JobManager for "offline batch" which bypasses the online scheduler.
        # So we define a new sync helper for the job manager.
        return self.job_manager.submit(self._predict_sync, features, model_version)
        
    def _predict_sync(self, features: List[Dict[str, Any]], model_version: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous wrapper for job manager."""
        # For batch jobs, strictly speaking, we might NOT want the micro-batch scheduler
        # because the job IS a big batch. We can just run it directly.
        # Also JobManager runs in a separate thread.
        version = model_version or self.registry.default_version
        model = self.registry.load(version)
        
        # Chunking: Split large batch into smaller chunks to release lock periodically
        # This prevents locking the GPU for too long, allowing online requests to interleave.
        chunk_size = 8
        all_predictions = []
        
        start = time.monotonic()
        for i in range(0, len(features), chunk_size):
            chunk = features[i : i + chunk_size]
            # model.predict acquires the lock for (Base + ChunkSize) duration
            # With chunk_size=32, this is ~42ms.
            # Between chunks, the lock is released, allowing online traffic to grab it.
            chunk_predictions = model.predict(chunk)
            all_predictions.extend(chunk_predictions)
            
            # Optional: Short sleep to force yield? 
            # In threading, releasing lock is usually enough for others to grab it 
            # if they are waiting.
            # UPDATE: Lock is not fair. We must sleep to let the scheduler thread grab it.
            time.sleep(0.05)
        
        latency = time.monotonic() - start
        return {"predictions": all_predictions, "version": version, "latency_ms": latency * 1000}

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
