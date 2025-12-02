"""
Middleware package for application
"""

from ainfinity.app.middleware.auth import AuthenticationMiddleware
from ainfinity.app.middleware.cors import CORSMiddleware
from ainfinity.app.middleware.logging import LoggingMiddleware
from ainfinity.app.middleware.rate_limit import RateLimitMiddleware

__all__ = [
    "AuthenticationMiddleware",
    "CORSMiddleware",
    "LoggingMiddleware",
    "RateLimitMiddleware",
]
