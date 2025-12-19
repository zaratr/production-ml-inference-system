"""Example deterministic model used for inference scaffolding."""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from app.models.base import Model


class ExampleModel(Model):
    def __init__(self, version: str, model_path: Path) -> None:
        self.version = version
        with open(model_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.bias: float = float(payload.get("bias", 0.0))
        self.weights: Dict[str, float] = {
            key: float(value) for key, value in payload.get("weights", {}).items()
        }
        # Simulate GPU lock (only one inference at a time per model instance)
        self._lock = threading.Lock()

    def predict(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Simulate GPU latency: processing a batch takes roughly fixed time
        # regardless of size (up to a limit), unlike CPU loops.
        # Enforce serial execution to simulate single GPU resource.
        with self._lock:
            # Simulate GPU latency: Base overhead + per-item processing time
            # e.g., 10ms base + 1ms per item.
            # Batch of 32: ~42ms. Batch of 1000: ~1.01s.
            latency = 0.01 + (0.001 * len(features))
            time.sleep(latency)

        predictions: List[Dict[str, Any]] = []
        for row in features:
            score = self.bias
            for name, weight in self.weights.items():
                score += float(row.get(name, 0.0)) * weight
            probability = 1 / (1 + math.exp(-score))
            predictions.append(
                {
                    "probability": probability,
                    "label": int(probability >= 0.5),
                    "version": self.version,
                    "confidence": float(abs(probability - 0.5) * 2),
                }
            )
        return predictions

    def metadata(self) -> Dict[str, Any]:
        return {"version": self.version, "features": sorted(self.weights.keys())}


__all__ = ["ExampleModel"]
