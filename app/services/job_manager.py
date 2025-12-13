"""Simple asynchronous job manager for batch inference."""
import concurrent.futures
import uuid
from typing import Any, Dict, List, Optional


class JobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, concurrent.futures.Future] = {}

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> str:
        job_id = str(uuid.uuid4())
        future = self.executor.submit(fn, *args, **kwargs)
        self._jobs[job_id] = future
        return job_id

    def status(self, job_id: str) -> str:
        future = self._jobs.get(job_id)
        if future is None:
            return "not_found"
        if future.running():
            return "running"
        if future.done():
            return "completed"
        return "pending"

    def result(self, job_id: str) -> Optional[Any]:
        future = self._jobs.get(job_id)
        if future is None:
            return None
        if future.done():
            return future.result()
        return None


__all__ = ["JobManager"]
