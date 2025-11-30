"""
Logging middleware for request/response tracking
"""
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log request and response information"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log (you can replace with proper logging)
        print(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
        
        # Add custom headers
        response.headers["X-Process-Time"] = str(duration)
        
        return response
