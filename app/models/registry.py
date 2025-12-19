"""Model registry abstraction for loading versioned models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from app.models.base import Model
from app.models.example_model import ExampleModel



import threading
from typing import Dict, Optional, List

class ModelRegistry:
    """Loads models from a versioned local registry.
    
    Supports dynamic loading, unloading, and promoting/rolling back default versions.
    """

    def __init__(self, registry_path: Path, default_version: str = "v1") -> None:
        self.registry_path = registry_path
        self._loaded_models: Dict[str, Model] = {}
        self._default_version = default_version
        self._lock = threading.RLock()

    @property
    def default_version(self) -> str:
        with self._lock:
            return self._default_version

    def list_loaded_versions(self) -> List[str]:
        with self._lock:
            return list(self._loaded_models.keys())
            
    def set_default_version(self, version: str) -> None:
        """Promote a specific version to be the default.
        
        The version must be loaded first.
        """
        with self._lock:
            if version not in self._loaded_models:
                # Attempt auto-load
                self.load(version)
            self._default_version = version

    def load(self, version: str) -> Model:
        """Load a model version into memory."""
        with self._lock:
            if version in self._loaded_models:
                return self._loaded_models[version]
            
            version_dir = self.registry_path / version
            model_file = version_dir / "model.json"
            if not model_file.exists():
                raise FileNotFoundError(f"Model artifact not found for version {version}")
            
            model = ExampleModel(version=version, model_path=model_file)
            self._loaded_models[version] = model
            return model

    def unload(self, version: str) -> None:
        """Unload a model version from memory."""
        with self._lock:
            if version == self._default_version:
                raise ValueError(f"Cannot unload the default version ({version})")
            if version in self._loaded_models:
                del self._loaded_models[version]

__all__ = ["ModelRegistry"]
