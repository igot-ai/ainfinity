"""
Exceptions package
Centralized exception definitions and handlers
"""

from ainfinity.app.exceptions.base import (
    AIFinityException,
    ConfigurationError,
    JobAlreadyExistsException,
    JobNotFoundException,
    ServiceException,
    SkyPilotException,
    ValidationException,
)
from ainfinity.app.exceptions.handlers import (
    generic_exception_handler,
    job_exists_handler,
    job_not_found_handler,
    service_exception_handler,
    validation_exception_handler,
    value_error_handler,
)

__all__ = [
    # Base exceptions
    "AIFinityException",
    "ServiceException",
    "ConfigurationError",
    "JobNotFoundException",
    "JobAlreadyExistsException",
    "SkyPilotException",
    "ValidationException",
    # Handlers
    "generic_exception_handler",
    "job_exists_handler",
    "job_not_found_handler",
    "service_exception_handler",
    "validation_exception_handler",
    "value_error_handler",
]
