"""
Main FastAPI application for AIFininity Training Service
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ainfinity.api.dependencies import get_training_service
from ainfinity.api.v1 import api_router as api_v1_router
from ainfinity.exceptions import (JobAlreadyExistsException,
                                  JobNotFoundException, ServiceException,
                                  ValidationException,
                                  generic_exception_handler,
                                  job_exists_handler, job_not_found_handler,
                                  service_exception_handler,
                                  validation_exception_handler,
                                  value_error_handler)
from ainfinity.middleware import (AuthenticationMiddleware, LoggingMiddleware,
                                  RateLimitMiddleware)
from ainfinity.utils.config import serving_settings


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


# Create FastAPI app
app = FastAPI(
    title=serving_settings.API_TITLE,
    description=serving_settings.API_DESCRIPTION,
    version=serving_settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

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


@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "service": serving_settings.API_TITLE,
        "version": serving_settings.API_VERSION,
        "docs": "/docs",
        "health": f"{serving_settings.API_V1_PREFIX}/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ainfinity.main:app",
        host=serving_settings.HOST,
        port=serving_settings.PORT,
        reload=serving_settings.RELOAD,
        log_level=serving_settings.LOG_LEVEL.lower()
    )
