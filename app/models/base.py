"""Model abstraction for inference."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Model(ABC):
    """Abstract base class for inference models."""

    version: str

    @abstractmethod
    def predict(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return predictions for a list of feature dictionaries."""

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return model metadata such as version and feature schema."""
