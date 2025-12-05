from collections.abc import Callable
from contextlib import _AsyncGeneratorContextManager
from typing import Any

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ainfinity.app.api.v1 import api_router as api_v1_router
from ainfinity.app.exceptions import (
    JobAlreadyExistsException,
    JobNotFoundException,
    ServiceException,
    ValidationException,
    generic_exception_handler,
    job_exists_handler,
    job_not_found_handler,
    service_exception_handler,
    validation_exception_handler,
    value_error_handler,
)
from ainfinity.app.middleware import LoggingMiddleware
from ainfinity.utils.config import serving_settings


def create_application(
    router: APIRouter,
    lifespan: Callable[[FastAPI], _AsyncGeneratorContextManager[Any]] | None = None,
) -> FastAPI:

    app = FastAPI(lifespan=lifespan)
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=serving_settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=serving_settings.CORS_METHODS,
        allow_headers=serving_settings.CORS_HEADERS,
    )

    # Add custom middlewares
    app.add_middleware(LoggingMiddleware)

    # Optional: Add rate limiting (uncomment to enable)
    # app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

    # Optional: Add authentication (uncomment and set API key to enable)
    # app.add_middleware(
    #     AuthenticationMiddleware,
    #     api_key=None,  # Set your API key here or from environment
    #     exclude_paths=["/", "/health", "/docs", "/redoc", "/openapi.json"]
    # )

    # Add exception handlers
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(JobNotFoundException, job_not_found_handler)
    app.add_exception_handler(JobAlreadyExistsException, job_exists_handler)
    app.add_exception_handler(ValidationException, validation_exception_handler)
    app.add_exception_handler(ServiceException, service_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Include API v1 router
    app.include_router(api_v1_router, prefix=serving_settings.API_V1_PREFIX)

    return app
