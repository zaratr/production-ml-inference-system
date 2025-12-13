"""Model registry abstraction for loading versioned models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from app.models.base import Model
from app.models.example_model import ExampleModel


class ModelRegistry:
    """Loads models from a versioned local registry."""

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path
        self._cache: Dict[str, Model] = {}

    def load(self, version: str) -> Model:
        if version in self._cache:
            return self._cache[version]
        version_dir = self.registry_path / version
        model_file = version_dir / "model.json"
        if not model_file.exists():
            raise FileNotFoundError(f"Model artifact not found for version {version}")
        model = ExampleModel(version=version, model_path=model_file)
        self._cache[version] = model
        return model


__all__ = ["ModelRegistry"]
