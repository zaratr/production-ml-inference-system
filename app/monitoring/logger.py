"""Structured logging utilities."""
import json
import logging
from typing import Any, Dict


def configure_logging(service_name: str) -> None:
    """Configure a structured JSON logger for the service."""

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
            payload: Dict[str, Any] = {
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
                "service": service_name,
            }
            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)
            for key, value in getattr(record, "__dict__", {}).items():
                if key.startswith("ctx_"):
                    payload[key[4:]] = value
            return json.dumps(payload)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


logger = logging.getLogger("inference-service")

__all__ = ["configure_logging", "logger"]
