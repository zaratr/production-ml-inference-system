import concurrent.futures
import json
import uuid
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from app.monitoring.logger import logger


class JobManager:
    def __init__(self, max_workers: int = 2, storage_dir: str = "data/jobs") -> None:
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, concurrent.futures.Future] = {} 
        # Note: _jobs only tracks in-memory futures for the current process. 
        # Persistence allows querying old jobs.

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> str:
        job_id = str(uuid.uuid4())
        
        # Save initial PENDING state
        self._save_job_state(job_id, {"status": "pending", "submitted_at": time.time()})

        # Wrap the function to handle status updates
        def wrapped_fn():
            self._save_job_state(job_id, {"status": "running", "started_at": time.time()})
            try:
                result = fn(*args, **kwargs)
                self._save_job_state(job_id, {
                    "status": "completed", 
                    "completed_at": time.time(),
                    "result": result
                })
                return result
            except Exception as e:
                logger.exception("Batch job failed", extra={"ctx_job_id": job_id})
                self._save_job_state(job_id, {
                    "status": "failed", 
                    "completed_at": time.time(),
                    "error": str(e)
                })
                raise

        future = self.executor.submit(wrapped_fn)
        self._jobs[job_id] = future
        return job_id

    def status(self, job_id: str) -> str:
        # Check in-memory first for active jobs? 
        # Or just read disk for SSoT? Disk is safer for consistency.
        state = self._load_job_state(job_id)
        if state:
            return state.get("status", "unknown")
        return "not_found"

    def result(self, job_id: str) -> Optional[Any]:
        state = self._load_job_state(job_id)
        if state and state.get("status") == "completed":
            return state.get("result")
        return None

    def _save_job_state(self, job_id: str, updates: Dict[str, Any]) -> None:
        """Update job state on disk."""
        file_path = self.storage_dir / f"{job_id}.json"
        
        # Read existing to merge updates (simple lock-free approach, risky if high concurrency on same job, but jobs are sequential)
        current_state = {}
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    current_state = json.load(f)
            except Exception:
                pass
        
        current_state.update(updates)
        current_state["updated_at"] = time.time()
        
        try:
            with open(file_path, "w") as f:
                json.dump(current_state, f, default=str)
        except Exception as e:
            logger.error(f"Failed to save job state: {e}")

    def _load_job_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        file_path = self.storage_dir / f"{job_id}.json"
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception:
            return None


__all__ = ["JobManager"]
