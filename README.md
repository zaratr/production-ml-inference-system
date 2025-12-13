# Production ML Inference System

This repository implements a production-style machine learning inference service with built-in monitoring, drift detection, and support for real-time and batch workloads. The system is designed to run continuously and expose operational signals required for reliability and observability.

## Architecture

```
/app
  /api            # FastAPI application and routing
  /models         # Model abstractions and registry
  /services       # Inference orchestration and batch jobs
  /monitoring     # Logging, metrics, and drift detection
  /utils          # Configuration utilities
/config
  /model_store    # Versioned local model artifacts
/tests            # Unit tests
```

### Components
- **API layer** (`app/api`): FastAPI app exposing health, prediction, and batch job endpoints with request validation and error handling.
- **Model layer** (`app/models`): Clean `Model` interface with a pluggable `ModelRegistry` that loads versioned artifacts from a local registry. Swapping models does not require API changes.
- **Services layer** (`app/services`): `InferenceService` orchestrates model loading, inference, metrics, and drift tracking. `JobManager` supports asynchronous batch inference using a worker pool.
- **Monitoring** (`app/monitoring`): Structured JSON logging, in-memory metrics collector (requests, errors, latency percentiles), and a simple drift detector that compares rolling feature means against a baseline.
- **Configuration** (`app/utils/config.py`): Environment-driven settings parsed from environment variables with sensible defaults for development and production.

## Inference Flows

### Real-time
1. Client sends `POST /predict` with an `instances` list of feature dictionaries and optional `version` query parameter.
2. `InferenceService` loads the requested model version from the registry, runs inference, records latency metrics, logs structured output, and updates drift statistics.
3. Response returns predictions with model version and latency budget in milliseconds.

### Batch
1. Client submits `POST /batch` with `instances` and optional `version` to enqueue work.
2. `JobManager` schedules processing on a background thread pool and returns a `job_id` immediately.
3. Client polls `GET /batch/{job_id}` to retrieve job status and results when available.

## Monitoring and Drift Detection
- **Logging**: JSON-formatted logs include request counts, latency, model version, and drift events to make troubleshooting and aggregation straightforward.
- **Metrics**: The in-memory collector tracks request totals, errors, and latency distributions (p50, p95) for quick inspection or export to an external system.
- **Drift**: `DriftTracker` maintains rolling windows of numeric feature means. Once a baseline is established, deviations above the configured relative threshold emit drift signals and logs.

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the API server:
   ```bash
   uvicorn app.api.server:app --reload --host 0.0.0.0 --port 8000
   ```
3. Example request:
   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"instances": [{"feature_a": 1.0, "feature_b": 0.5}]}'
   ```
4. Batch job example:
   ```bash
   curl -X POST "http://localhost:8000/batch" -H "Content-Type: application/json" \
     -d '{"instances": [{"feature_a": 0.2, "feature_b": 0.1}]}'
   curl "http://localhost:8000/batch/<job_id>"
   ```

## Production Considerations
- Deploy behind a process manager (e.g., systemd, Kubernetes) with health and readiness probes hitting `/health`.
- Export metrics and logs to centralized observability stacks (Prometheus, OpenTelemetry, or vendor solutions).
- Backed model registry can be replaced with object storage or a feature store by extending `ModelRegistry`.
- Use GPU acceleration by implementing model subclasses that leverage frameworks like PyTorch or TensorFlow.
- Configure request timeouts, worker counts, and drift thresholds through environment variables for each environment.

## Tests

Run the unit test suite:

```bash
pytest
```
