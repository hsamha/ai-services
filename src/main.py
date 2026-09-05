"""Service entry point.

Creates the application, wires up its startup and shutdown, and exposes the
liveness check.
"""

from fastapi import FastAPI
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str
    service: str
    version: str


app = FastAPI(title="rag-chatbot", version="0.1.0")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Report that the service is up. Unauthenticated, no external calls."""
    return HealthResponse(status="ok", service=app.title, version=app.version)
