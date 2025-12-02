"""
Main FastAPI application for AIFininity Training Service
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ainfinity.app.api.dependencies import get_training_service
from ainfinity.app.api.v1 import api_router as api_v1_router

from ainfinity.utils.config import serving_settings
from ainfinity.app.api.setup import create_application


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print(f"Starting {serving_settings.API_TITLE} v{serving_settings.API_VERSION}")
    # Initialize training service
    _ = get_training_service()
    yield
    # Shutdown
    print("Shutting down application")


app = create_application(router=api_v1_router, lifespan=lifespan)

@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "service": serving_settings.API_TITLE,
        "version": serving_settings.API_VERSION,
        "docs": "/docs",
        "health": f"{serving_settings.API_V1_PREFIX}/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ainfinity.app.main:app",
        host=serving_settings.HOST,
        port=serving_settings.PORT,
        reload=serving_settings.RELOAD,
        log_level=serving_settings.LOG_LEVEL.lower(),
    )
