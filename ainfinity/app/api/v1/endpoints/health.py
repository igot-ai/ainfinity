"""
Health check endpoints
"""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from ainfinity.app.api.dependencies import get_training_service
from ainfinity.utils.config import serving_settings

router = APIRouter()


@router.get("", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Basic health check endpoint

    Returns:
        Health status with service information
    """
    try:
        service = get_training_service()

        # Check if service is accessible
        workspace_info = {
            "workspace_root": str(service.workspace_root),
            "jobs_db_exists": service.jobs_db_path.exists(),
        }

        return {
            "status": "healthy",
            "service": serving_settings.API_TITLE,
            "version": serving_settings.API_VERSION,
            "workspace": workspace_info,
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "service": serving_settings.API_TITLE,
                "version": serving_settings.API_VERSION,
            },
        )


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    """
    Readiness check - indicates if service is ready to accept requests

    Returns:
        Readiness status
    """
    try:
        service = get_training_service()

        # Verify service can perform basic operations
        _ = service._load_jobs_db()

        return {"status": "ready", "service": serving_settings.API_TITLE}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "not_ready", "error": str(e)}
        )


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check():
    """
    Liveness check - indicates if service is running

    Returns:
        Simple liveness status
    """
    return {"status": "alive", "service": serving_settings.API_TITLE}
