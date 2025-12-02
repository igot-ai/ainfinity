"""
CORS middleware for cross-origin requests
"""

from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class CORSMiddleware(BaseHTTPMiddleware):
    """Handle CORS for API access"""

    def __init__(self, app, allow_origins=None, allow_methods=None, allow_headers=None):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["Content-Type", "Authorization"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response(status_code=200)
        else:
            response = await call_next(request)

        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = ", ".join(self.allow_origins)
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
        response.headers["Access-Control-Allow-Credentials"] = "true"

        return response
