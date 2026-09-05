"""FastAPI application factory.

Builds the app, registers the auth middleware and error handlers, mounts the
API router, and manages the Qdrant client's lifespan (open on startup, close on
shutdown).
"""

from fastapi import FastAPI
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Liveness payload returned by GET /health."""

    status: str
    service: str
    version: str


app = FastAPI(title="rag-chatbot", version="0.1.0")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Report that the service is up. Unauthenticated, no external calls."""
    return HealthResponse(status="ok", service=app.title, version=app.version)
