"""FastAPI application setup."""
from fastapi import FastAPI

from app.api.routes import router
from app.monitoring.logger import configure_logging
from app.utils.config import get_settings

settings = get_settings()
configure_logging(settings.service_name)

from contextlib import asynccontextmanager
from app.deps import get_inference_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start up services
    service = get_inference_service()
    service.start()
    yield
    # Shut down services
    await service.stop()

from app.api.admin_routes import router as admin_router

app = FastAPI(title=settings.service_name, version="0.1.0", lifespan=lifespan)
app.include_router(router)
app.include_router(admin_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": settings.service_name, "env": settings.env}


__all__ = ["app"]
