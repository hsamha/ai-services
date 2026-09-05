import hmac
import os
from functools import lru_cache

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

SERVICE_KEY_HEADER = "X-API-Key"

API_KEY_ENV = "API_KEY"

# Reachable without a key.
OPEN_PATHS: frozenset[str] = frozenset(
    {"/health", "/docs", "/openapi.json"}
)


def is_open(path: str) -> bool:
    return path in OPEN_PATHS


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if is_open(request.url.path):
            return await call_next(request)

        key = request.headers.get(SERVICE_KEY_HEADER)
        if key is None or not self._accepted(key):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": f"Invalid or missing {SERVICE_KEY_HEADER}."},
            )

        return await call_next(request)

    @staticmethod
    def _accepted(key: str) -> bool:
        expected = _service_key()
        if expected is None:
            return False
        return hmac.compare_digest(key, expected)


@lru_cache
def _service_key() -> str | None:
    return os.environ.get(API_KEY_ENV) or None
