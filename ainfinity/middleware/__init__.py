"""
Middleware package for application
"""

from ainfinity.middleware.auth import AuthenticationMiddleware
from ainfinity.middleware.cors import CORSMiddleware
from ainfinity.middleware.logging import LoggingMiddleware
from ainfinity.middleware.rate_limit import RateLimitMiddleware

__all__ = [
    "AuthenticationMiddleware",
    "CORSMiddleware",
    "LoggingMiddleware",
    "RateLimitMiddleware",
]
