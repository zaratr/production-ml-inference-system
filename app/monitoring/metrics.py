"""Lightweight in-memory metrics collector."""
import statistics
import threading
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List


class MetricsCollector:
    """Tracks counters and latency histograms for the service."""

    def __init__(self, latency_window: int = 500) -> None:
        self._lock = threading.Lock()
        self.counters: Dict[str, int] = defaultdict(int)
        self.latencies: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=latency_window)
        )

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self.counters[name] += value

    def observe_latency(self, name: str, value: float) -> None:
        with self._lock:
            self.latencies[name].append(value)

    def summary(self, name: str) -> Dict[str, float]:
        with self._lock:
            values: Iterable[float] = list(self.latencies[name])
        if not values:
            return {"count": 0, "p50": 0.0, "p95": 0.0}
        sorted_values: List[float] = sorted(values)
        return {
            "count": len(sorted_values),
            "p50": percentile(sorted_values, 50),
            "p95": percentile(sorted_values, 95),
        }


def percentile(values: List[float], q: float) -> float:
    """Return percentile value from a sorted list."""

    if not values:
        return 0.0
    k = (len(values) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[int(k)]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


__all__ = ["MetricsCollector", "percentile"]
