"""FastAPI application setup."""
from fastapi import FastAPI

from app.api.routes import router
from app.monitoring.logger import configure_logging
from app.utils.config import get_settings

settings = get_settings()
configure_logging(settings.service_name)

app = FastAPI(title=settings.service_name, version="0.1.0")
app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": settings.service_name, "env": settings.env}


__all__ = ["app"]
