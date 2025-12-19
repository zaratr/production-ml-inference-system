"""
Async batch scheduling for high-throughput inference.
Aggregates individual requests into batches to maximize GPU utilization.
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from app.monitoring.logger import logger


@dataclass
class _QueueItem:
    features: Dict[str, Any]
    future: asyncio.Future
    received_at: float


class BatchScheduler:
    """Aggregates individual requests into batches for efficient processing."""

    def __init__(
        self,
        prediction_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
        max_batch_size: int = 32,
        max_latency_ms: float = 10.0,
        max_queue_size: int = 1024,
    ) -> None:
        self.prediction_fn = prediction_fn
        self.max_batch_size = max_batch_size
        self.max_latency = max_latency_ms / 1000.0  # Convert to seconds
        self.queue: asyncio.Queue[_QueueItem] = asyncio.Queue(maxsize=max_queue_size)
        self._shutdown = False
        self._worker_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Start the background worker loop."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("BatchScheduler started", extra={"ctx_batch_size": self.max_batch_size})

    async def stop(self) -> None:
        """Stop the background worker."""
        self._shutdown = True
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            logger.info("BatchScheduler stopped")

    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a single request to the batch queue and await result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = _QueueItem(features=features, future=future, received_at=time.time())

        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            future.cancel()
            raise  # Caller must handle asyncio.QueueFull
        try:
            return await future
        except Exception:
            # If the future failed or was cancelled
            raise

    async def _worker_loop(self) -> None:
        """Continuously process batches from the queue."""
        while not self._shutdown:
            batch_items: List[_QueueItem] = []

            # 1. Wait for the first item (blocking)
            try:
                first_item = await self.queue.get()
                batch_items.append(first_item)
            except asyncio.CancelledError:
                break

            # 2. Collect more items up to max_batch_size, within time window
            # We enforce the deadline based on the OLDEST item in the batch
            # BUT if we are already behind (queue is backed up), we should prioritize throughput
            # and fill the batch via non-blocking gets.
            deadline = first_item.received_at + self.max_latency
            
            while len(batch_items) < self.max_batch_size:
                now = time.time()
                remaining = deadline - now
                
                try:
                    if remaining > 0:
                        # Wait for next item with timeout
                        item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                        batch_items.append(item)
                    else:
                        # Deadline exceeded: Check if we can grab items immediately to fill batch
                        # This improves throughput under load.
                        item = self.queue.get_nowait()
                        batch_items.append(item)
                except (asyncio.TimeoutError, asyncio.CancelledError, asyncio.QueueEmpty):
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in batch loop: {e}")
                    break

            # 3. Process the batch
            if batch_items:
                await self._process_batch(batch_items)

    async def _process_batch(self, items: List[_QueueItem]) -> None:
        """Run prediction and resolve futures."""
        features = [item.features for item in items]
        try:
            # Offload blocking prediction_fn to thread
            predictions = await asyncio.to_thread(self.prediction_fn, features)
            
            # Map results back to futures
            for item, prediction in zip(items, predictions):
                if not item.future.done():
                    item.future.set_result(prediction)
        except Exception as e:
            logger.exception("Batch prediction failed")
            # Fail all futures in this batch
            for item in items:
                if not item.future.done():
                    item.future.set_exception(e)
