"""
Circuit breaker pattern to protect against cascading failures.
"""
import time
from enum import Enum
from threading import RLock
from typing import Optional

from app.monitoring.logger import logger


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpen(Exception):
    """Raised when the circuit is open and requests are blocked."""
    pass


class CircuitBreaker:
    """monitor failures and cut off access when threshold exceeded."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 5.0,
        expected_exception_types: tuple = (Exception,),
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception_types = expected_exception_types

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = RLock()

    @property
    def state(self) -> str:
        return self._state.value

    def __enter__(self) -> None:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    raise CircuitBreakerOpen("Circuit is open")
            
            # In HALF_OPEN, we allow the request to proceed (testing the waters)
            # If it fails, we go back to OPEN immediately.
            # If it succeeds, we reset to CLOSED.
            # (Logic handled in __exit__)

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        with self._lock:
            if exc_type is not None:
                # A failure occurred
                if issubclass(exc_type, self.expected_exception_types):
                    self._handle_failure()
                # Do not suppress the exception
                return False
            
            # Success
            self._handle_success()
            return False

    def _handle_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            # Failed trial -> Re-open immediately
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def _handle_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            # Trial succeeded -> Close circuit
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success? 
            # Classic pattern: simple count reset, or rolling window.
            # For simplicity: reset on success is generous (good for intermittent).
            # Alternatively: only reset if failure_count > 0.
            if self._failure_count > 0:
                self._failure_count = 0

    def _transition_to(self, new_state: CircuitState) -> None:
        if self._state != new_state:
            logger.warning(
                "circuit breaker state change",
                extra={
                    "ctx_from": self._state.value,
                    "ctx_to": new_state.value,
                    "ctx_failures": self._failure_count,
                },
            )
            self._state = new_state
