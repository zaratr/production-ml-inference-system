"""Admin routes for model management."""
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from app.deps import get_inference_service
from app.services.inference_service import InferenceService

router = APIRouter(prefix="/admin/models", tags=["admin"])


@router.get("")
def list_models(
    service: InferenceService = Depends(get_inference_service),
) -> Dict[str, Any]:
    return service.list_models()


@router.post("/{version}/load")
def load_model(
    version: str,
    service: InferenceService = Depends(get_inference_service),
) -> Dict[str, str]:
    try:
        service.load_model(version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "loaded", "version": version}


@router.post("/{version}/promote")
def promote_model(
    version: str,
    service: InferenceService = Depends(get_inference_service),
) -> Dict[str, str]:
    try:
        service.promote_model(version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "promoted", "version": version}


@router.delete("/{version}")
def unload_model(
    version: str,
    service: InferenceService = Depends(get_inference_service),
) -> Dict[str, str]:
    try:
        service.unload_model(version)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "unloaded", "version": version}
