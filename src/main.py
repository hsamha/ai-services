
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from src.api import api_router
from src.core.vectorstores.registry import close_connections, open_connections
from src.middleware.auth import AuthMiddleware
from src.middleware.context import ContextMiddleware


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str
    service: str
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Hold the database connections open for as long as the service runs."""
    await open_connections()
    try:
        yield
    finally:
        await close_connections()


app = FastAPI(title="ai-services", version="0.1.0", lifespan=lifespan)

app.add_middleware(ContextMiddleware)
app.add_middleware(AuthMiddleware)

app.include_router(api_router)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=app.title, version=app.version)
