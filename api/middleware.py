"""API middleware — logging, rate limiting, request tracking."""

import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from core.logger import setup_logger

logger = setup_logger("api.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with timing and request ID."""

    async def dispatch(self, request: Request, call_next):
        # Let CORS preflight pass through without interference
        if request.method == "OPTIONS":
            return await call_next(request)

        request_id = str(uuid.uuid4())[:8]
        start = time.time()

        # Attach request_id for downstream use
        request.state.request_id = request_id

        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown",
            },
        )

        response = await call_next(request)

        elapsed_ms = int((time.time() - start) * 1000)
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "status": response.status_code,
                "elapsed_ms": elapsed_ms,
            },
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(elapsed_ms)

        return response
