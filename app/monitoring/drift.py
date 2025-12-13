"""Simple statistical drift detection utilities."""
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List


@dataclass
class DriftSignal:
    feature: str
    baseline_mean: float
    current_mean: float
    drift_score: float


class DriftTracker:
    """Tracks mean statistics for features and signals drift when they diverge."""

    def __init__(self, window_size: int, threshold: float) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self._buffers: Dict[str, Deque[float]] = {}
        self._baselines: Dict[str, float] = {}

    def _mean(self, values: Deque[float]) -> float:
        return sum(values) / len(values)

    def update(self, features: Dict[str, float]) -> List[DriftSignal]:
        """Update tracked features and return any drift signals."""

        signals: List[DriftSignal] = []
        for name, value in features.items():
            buffer = self._buffers.setdefault(name, deque(maxlen=self.window_size))
            buffer.append(value)
            if name not in self._baselines and len(buffer) == buffer.maxlen:
                self._baselines[name] = self._mean(buffer)
                continue
            if name in self._baselines and len(buffer) == buffer.maxlen:
                baseline_mean = self._baselines[name]
                current_mean = self._mean(buffer)
                if baseline_mean == 0:
                    continue
                drift_score = abs(current_mean - baseline_mean) / abs(baseline_mean)
                if drift_score >= self.threshold:
                    signals.append(
                        DriftSignal(
                            feature=name,
                            baseline_mean=baseline_mean,
                            current_mean=current_mean,
                            drift_score=drift_score,
                        )
                    )
        return signals


__all__ = ["DriftTracker", "DriftSignal"]
