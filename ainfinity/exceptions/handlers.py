"""
HTTP exception handlers for FastAPI
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse

from ainfinity.exceptions.base import (
    JobAlreadyExistsException,
    JobNotFoundException,
    ServiceException,
    ValidationException,
)


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle ValueError exceptions"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error": "ValidationError",
            "message": str(exc),
            "path": request.url.path,
        },
    )


async def job_not_found_handler(request: Request, exc: JobNotFoundException) -> JSONResponse:
    """Handle JobNotFoundException"""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "success": False,
            "error": "JobNotFound",
            "message": str(exc),
            "path": request.url.path,
        },
    )


async def job_exists_handler(request: Request, exc: JobAlreadyExistsException) -> JSONResponse:
    """Handle JobAlreadyExistsException"""
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "success": False,
            "error": "JobAlreadyExists",
            "message": str(exc),
            "path": request.url.path,
        },
    )


async def service_exception_handler(request: Request, exc: ServiceException) -> JSONResponse:
    """Handle ServiceException"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "ServiceError",
            "message": str(exc),
            "path": request.url.path,
        },
    )


async def validation_exception_handler(request: Request, exc: ValidationException) -> JSONResponse:
    """Handle ValidationException"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "ValidationError",
            "message": str(exc),
            "path": request.url.path,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle generic exceptions"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "path": request.url.path,
        },
    )
