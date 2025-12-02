"""
Rate limiting middleware
"""

import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"

        # Clean old requests
        current_time = time.time()
        self.requests[client_ip] = [req_time for req_time in self.requests[client_ip] if current_time - req_time < 60]

        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "success": False,
                    "error": "RateLimitExceeded",
                    "message": f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute.",
                },
            )

        # Add current request
        self.requests[client_ip].append(current_time)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(self.requests_per_minute - len(self.requests[client_ip]))

        return response
