"""Building the request context.

Reads the values a caller sends with each request -- their key and, optionally,
which models and store to use -- and puts them where the rest of the code can
reach them. A caller names a model; which provider serves it is decided deeper
in. Anything they leave out falls back to the configured default.

Runs after authentication, so only accepted callers get this far.
"""

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.context import RequestContext, reset_context, set_context
from src.middleware.auth import SERVICE_KEY_HEADER, is_open

PROVIDER_KEY_HEADER = "X-AI-Provider-Key"
LLM_MODEL_HEADER = "X-LLM-Model"
EMBEDDING_MODEL_HEADER = "X-Embedding-Model"


class ContextMiddleware(BaseHTTPMiddleware):
    """Makes the caller's request values available for the life of the request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Fill the context, handle the request, then clear it again."""
        if is_open(request.url.path):
            return await call_next(request)

        provider_key = request.headers.get(PROVIDER_KEY_HEADER)
        if not provider_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": f"Missing {PROVIDER_KEY_HEADER}."},
            )

        context = RequestContext(
            # Already checked by the authentication middleware, which runs first.
            api_key=request.headers.get(SERVICE_KEY_HEADER, ""),
            provider_key=provider_key,
            llm_model=request.headers.get(LLM_MODEL_HEADER),
            embedding_model=request.headers.get(EMBEDDING_MODEL_HEADER),
        )

        token = set_context(context)
        try:
            return await call_next(request)
        finally:
            reset_context(token)
