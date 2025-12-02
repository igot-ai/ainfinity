"""
Authentication middleware
"""

from typing import Callable, Optional

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Simple API key authentication middleware"""

    def __init__(self, app, api_key: Optional[str] = None, exclude_paths: Optional[list[str]] = None):
        super().__init__(app)
        self.api_key = api_key
        self.exclude_paths = exclude_paths or ["/", "/docs", "/redoc", "/openapi.json"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip authentication for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # If no API key is configured, allow all requests
        if not self.api_key:
            return await call_next(request)

        # Check for API key in header
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")

        if api_key:
            # Remove "Bearer " prefix if present
            if api_key.startswith("Bearer "):
                api_key = api_key[7:]

            if api_key == self.api_key:
                return await call_next(request)

        # Unauthorized
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"success": False, "error": "Unauthorized", "message": "Invalid or missing API key"},
        )
