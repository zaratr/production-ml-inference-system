"""API routes for inference service."""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from app.services.inference_service import InferenceService
from app.utils.config import get_settings
from app.deps import get_inference_service

router = APIRouter()


@router.get("/health")
def health(service: InferenceService = Depends(get_inference_service)) -> Dict[str, Any]:
    return service.health()


@router.post("/predict")
def predict(
    payload: Dict[str, List[Dict[str, Any]]],
    version: Optional[str] = None,
    service: InferenceService = Depends(get_inference_service),
) -> Dict[str, Any]:
    features = payload.get("instances")
    if features is None:
        raise HTTPException(status_code=400, detail="instances field is required")
    return service.predict(features=features, model_version=version)


@router.post("/batch")
def submit_batch(
    payload: Dict[str, List[Dict[str, Any]]],
    version: Optional[str] = None,
    service: InferenceService = Depends(get_inference_service),
) -> Dict[str, Any]:
    features = payload.get("instances")
    if features is None:
        raise HTTPException(status_code=400, detail="instances field is required")
    job_id = service.enqueue_batch(features, model_version=version)
    return {"job_id": job_id, "status": "submitted"}


@router.get("/batch/{job_id}")
def batch_status(job_id: str, service: InferenceService = Depends(get_inference_service)) -> Dict[str, Any]:
    result = service.batch_status(job_id)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    return result


__all__ = ["router"]
