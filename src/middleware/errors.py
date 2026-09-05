import logging

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger("api")


class ErrorMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await call_next(request)

        except HTTPException as exc:
            logger.warning(
                "%s %s -> %s: %s",
                request.method,
                request.url.path,
                exc.status_code,
                exc.detail,
            )
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

        except Exception as exc:
            logger.exception("%s %s failed", request.method, request.url.path)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"{type(exc).__name__}: {exc}"},
            )
